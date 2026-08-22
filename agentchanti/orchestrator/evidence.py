"""Who wrote the evidence the run is claiming success on?

WHY THIS EXISTS
---------------
Measured over six benchmark runs of one task, against ground-truth probes
the pipeline never saw:

    iter  loop  ground truth  pipeline claim  its own `unittest`
    1     on    FAIL          "success"       passed
    2     on    FAIL          "success"       passed
    1     off   FAIL          reported failed failed
    2     off   FAIL          reported failed failed

Both agent-loop failures printed ``All tasks completed successfully!``
over a Pac-Man whose player could not move at 1/60 — the loop iterated
until the gate went green, and the gate was a suite it had written in the
same run. The classic path's failures failed their own tests and said so.

The pipeline was not lying; it had simply marked its own homework. Every
declared postcondition genuinely held: the files existed, parsed, exported
what the plan promised, and the suite passed. Nothing in the run was in a
position to notice that all of it was self-authored.

WHAT THIS DOES, AND DOES NOT, DO
--------------------------------
It does not decide whether the code works — nothing inside a run can. It
separates two verdicts that were previously one:

  * **completed** — the pipeline executed its plan without failing a step
  * **verified**  — something the agent did not write in this run agreed

A greenfield build with no pre-existing suite and no user acceptance
command is honestly *unverified*, and saying so is the whole point. That
is not a failure, and it must not be reported as one — which is why this
never flips ``pipeline_success`` on its own. It changes what the run is
allowed to *claim*, and offers two ways to earn the stronger claim:
supply acceptance commands, or keep a test the agent did not touch.

WHY MODIFIED PRE-EXISTING TESTS DO NOT COUNT
--------------------------------------------
A seeded test the agent rewrote is not independent evidence, it is the
oldest cheat in the book: the run that "fixes" a failing test by editing
the assertion. Provenance is checked by content hash, so a test file that
existed before the run counts only while its bytes are unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

log = logging.getLogger("agentchanti")

# Directories that never hold a project's own tests.
_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env",
    "__pycache__", ".agentchanti", ".pytest_cache", ".mypy_cache",
    "dist", "build", ".tox", ".next", "site-packages",
}

_logger = logging.getLogger(__name__)

INDEPENDENT_ACCEPTANCE = "acceptance-commands"
INDEPENDENT_PRE_EXISTING = "pre-existing-tests"
SELF_AUTHORED = "self-authored"
PRE_EXISTING_FAILED = "pre-existing-tests-failed"
ACCEPTANCE_FAILED = "acceptance-commands-failed"
ACCEPTANCE_NOT_RUN = "acceptance-commands-not-run"
NO_TESTS = "no-tests"


@dataclass(frozen=True)
class Evidence:
    """What the run's success rests on.

    ``shallow`` separates two things that used to read identically: a
    check that could have failed and didn't, and one that could not have
    failed at all. Measured 2026-08-18 across two consecutive runs — one
    verified by a contract with seven discriminating assertions, the
    next by one asserting only that the script existed and its process
    had not exited — and both printed the same banner. A WARNING in the
    log was the whole difference, which is the "advisory nobody acts on"
    shape this project has been bitten by before.

    It deliberately does NOT flip ``independent``. The shallow check is
    still independent and still ran, so calling it self-authored would
    be a different lie; and failing every run whose task admits only a
    simple contract would be false precision. It changes what the run
    may *claim*, not whether it passed.
    """

    independent: bool
    kind: str
    detail: str
    shallow: bool = False
    # Repair rounds spent getting the acceptance commands green. Modelled
    # on `shallow`: it does NOT flip `independent`, because the check was
    # never edited (agent_tools refuses that) and it really did run and
    # really did pass -- so the evidence is genuinely independent. What
    # changes is the CLAIM. A run that passed on the third attempt is not
    # the same result as one that passed clean, and averaging them in a
    # benchmark hides exactly the variance worth measuring.
    repaired: int = 0

    @property
    def headline(self) -> str:
        if not self.independent:
            return "Tasks completed — but nothing independent verified them."
        if self.shallow:
            return ("Tasks completed — verified only that it runs, not that "
                    "it works.")
        if self.repaired:
            return (f"All tasks completed — verified after {self.repaired} "
                    f"repair round(s).")
        return "All tasks completed successfully!"

    def log_line(self) -> str:
        verdict = "independent" if self.independent else "self-authored"
        if self.shallow:
            verdict += " but SHALLOW"
        if self.repaired:
            verdict += f" after {self.repaired} repair round(s)"
        return f"Evidence: {verdict} ({self.kind}) — {self.detail}"


def _norm(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/").lstrip("./")


def _digest(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def acceptance_instrument_files(cmds, root: str = ".") -> set[str]:
    """Existing files the acceptance commands invoke, relative to *root*.

    `acceptance_cmds` is the only check in a run that the model did not
    write. That holds absolutely for the command STRING, which lives in
    config the model cannot reach — and did not hold at all for a command
    that invokes a file, which sits in the project root like any other
    source. This names those files so the agent's tools can refuse to
    write them.

    Deliberately conservative: a token counts only when it resolves to a
    file that already exists, so `npm --prefix frontend run build`
    contributes nothing while `node acceptance_check.cjs` contributes the
    script. Protecting a path that does not exist would block the run from
    creating an ordinary file that merely shares a name with a token.
    """
    found: set[str] = set()
    for cmd in cmds or ():
        for tok in re.split(r"[\s;|&<>()]+", str(cmd or "")):
            tok = tok.strip("\"'")
            if not tok or tok.startswith("-"):
                continue
            norm = tok.replace("\\", "/")
            while norm.startswith("./"):
                norm = norm[2:]
            norm = norm.lstrip("/")
            if not norm or norm in (".", ".."):
                continue
            try:
                if os.path.isfile(os.path.join(root, norm)):
                    found.add(norm)
            except (OSError, ValueError):
                continue
    return found


def snapshot_test_files(root: str) -> dict[str, str]:
    """``relative path -> sha256`` for every test file present *now*.

    Called before the first step runs, so "now" means "before the agent
    touched anything". Hashes rather than mere presence, because the
    question this answers later is not "was there a test file" but "is
    this the same test file".
    """
    from .pipeline import _is_test_file

    out: dict[str, str] = {}
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                full = os.path.join(dirpath, name)
                rel = _norm(os.path.relpath(full, root))
                if not _is_test_file(rel):
                    continue
                digest = _digest(full)
                if digest:
                    out[rel] = digest
    except OSError as exc:
        log.debug(f"[Evidence] could not scan {root}: {exc}")
    return out


def surviving_pre_existing_tests(root: str,
                                 snapshot: dict[str, str]) -> list[str]:
    """Pre-existing test files whose bytes the run left alone.

    Deliberately re-hashed from disk rather than trusting the run's own
    record of what it wrote: a file can be changed by a shell command, a
    formatter, or a plugin, none of which pass through `FileMemory`.
    """
    survivors: list[str] = []
    for rel, before in sorted(snapshot.items()):
        now = _digest(os.path.join(root, rel.replace("/", os.sep)))
        if now is not None and now == before:
            survivors.append(rel)
    return survivors



# A suite can fail for reasons that say nothing about the code. The one
# measured here (2026-08-18 08:56) is a contract whose `finally:` called
# `game.userExit()`; Panda3D's userExit calls sys.exit(), unittest reports
# the resulting SystemExit as an ERROR, and a run whose every assertion
# had PASSED exited non-zero under require_independent_evidence. The
# instrument broke during cleanup, after judging the code favourably.
_INCONCLUSIVE_MARKERS = (
    ("SystemExit", "the suite called sys.exit()/userExit() during cleanup, "
                   "so unittest recorded an error after its assertions ran"),
    ("KeyboardInterrupt", "the suite was interrupted"),
    ("ModuleNotFoundError: No module named 'unittest",
     "the test runner itself is unavailable"),
    # A framework that permits one instance per process refuses the
    # second, and that refusal is a property of the framework, not of the
    # code under test. Measured 2026-08-18 09:54: a contract built a
    # fresh ShowBase in each of its five tests, so only the first could
    # ever run and the other four errored no matter what the artifact
    # did — while an external probe scored that artifact 20/20.
    ("Attempt to spawn multiple ShowBase instances",
     "the suite builds a second instance of a framework singleton, which "
     "only its first test can ever do"),
    ("QApplication instance already exists",
     "the suite builds a second instance of a framework singleton"),
)


def inconclusive_failure_reason(output: str) -> Optional[str]:
    """Why this red suite produced no verdict about the code, or None.

    Only markers that are unambiguous about the *instrument* breaking.
    Anything else — an assertion, an AttributeError in the code under
    test, a collection error naming a project module — is a real failure
    and must stay one, or this becomes a machine for explaining away
    disagreement.
    """
    text = output or ""
    for marker, reason in _INCONCLUSIVE_MARKERS:
        if marker in text:
            return reason
    return None


def shallow_survivors(root: str, survivors: Iterable[str]) -> Optional[str]:
    """Describe the survivors that cannot fail on wrong behaviour.

    Read from source at verdict time rather than remembered from the
    seeding step, so it covers a *user's* thin suite exactly as it covers
    a generated one — the question "could this have failed?" does not
    care who wrote the file.

    None when at least one survivor is substantive: a run with one real
    contract alongside a smoke test is verified, not shallow.
    """
    from .seed_strength import weak_contract_reason

    files = [f for f in (survivors or ()) if f.endswith(".py")]
    if not files:
        return None
    weak: list[str] = []
    for rel in files:
        try:
            with open(os.path.join(root, rel.replace("/", os.sep)),
                      encoding="utf-8") as fh:
                src = fh.read()
        except OSError:
            return None                   # unreadable: do not guess
        reason = weak_contract_reason(src)
        if reason is None:
            return None                   # one real check is enough
        weak.append(rel)
    return (f"{', '.join(weak[:3])} asserts nothing that could fail on wrong "
            f"behaviour, so this only shows the build runs")


def _was_seeded(root: str, rel: str) -> bool:
    """Did this pipeline generate *rel*, rather than find it?

    Decided from the file's own stamped header, so a suite the user
    wrote keeps every bit of its authority — including the authority to
    fail the run — while one the pipeline generated does not.
    """
    try:
        from .acceptance_seed import seed_state
        return seed_state(os.path.join(root, rel.replace("/", os.sep))) is not None
    except Exception:
        return False


def run_pre_existing_tests(executor, root: str,
                           survivors: Iterable[str]) -> tuple[bool | None, str]:
    """Actually run the surviving pre-existing tests. ``(passed, detail)``.

    ``None`` means the question could not be answered — no survivors, no
    executor, or nothing runnable — and must never be read as a pass.

    This exists because `classify` used to report those files as having
    "passed" on the strength of `tests_ran`, a flag about whether the
    pipeline ran ANY tests of its own. Measured twice, 2026-08-17 and
    2026-08-18: both runs logged `Evidence: independent (pre-existing-
    tests) ... passed: test_acceptance_contract.py` over a contract that
    errored on every test it had. The word "passed" was asserted, never
    measured, in the one layer whose entire job is to check that
    something independent agreed.
    """
    files = [f for f in (survivors or ()) if f.endswith(".py")]
    if not files or executor is None:
        return None, "no runnable pre-existing test file"
    failures: list[str] = []
    inconclusive: list[str] = []
    ran = 0
    for rel in files:
        cmd = "python -m unittest " + rel.replace("/", os.sep)
        try:
            ok, out = executor.run_command(cmd, timeout=300)
        except Exception as exc:
            _logger.debug("[Evidence] %s could not run: %s", rel, exc)
            continue
        ran += 1
        if ok:
            continue
        tail = " | ".join((out or "").strip().splitlines()[-3:])
        reason = inconclusive_failure_reason(out or "")
        if reason:
            inconclusive.append(f"{rel}: {reason}")
        elif _was_seeded(root, rel):
            # A contract this pipeline generated is not the instrument
            # CLAUDE.md grants the power to fail a run — that is reserved
            # for user-supplied `acceptance_cmds`, "the one instrument the
            # model neither wrote nor can edit". A seeded contract was
            # written by a model, before the code and in good faith, but
            # still by a model: measured across three of four consecutive
            # runs it failed artifacts that scored 20/20 externally, once
            # by demanding the snake accept a reversal its own test name
            # said must be refused. It can establish evidence when green;
            # it may not convict when red.
            inconclusive.append(
                f"{rel}: the seeded contract failed ({tail[:120]}) — it was "
                f"written by a model in this run, so it establishes evidence "
                f"when it passes but cannot convict the code when it fails")
        else:
            failures.append(f"{rel}: {tail[:200]}")
    if ran == 0:
        return None, "no pre-existing test file could be run"
    if failures:
        return False, "; ".join(failures[:3])
    if inconclusive:
        # The suite broke itself rather than judging the code, so it
        # produced no verdict — the same distinction `GateLedger`
        # already draws between a crash and a real failure, and the
        # reason `verify_dt_invariance` reserves an exit code for
        # "could not verify". Reporting it as a failure would
        # manufacture a regression out of silence.
        return None, "; ".join(inconclusive[:3])
    return True, f"{ran} pre-existing test file(s) ran and passed"


def classify(root: str,
             snapshot: dict[str, str],
             *,
             tests_ran: bool,
             acceptance_passed: Optional[bool] = None,
             acceptance_cmds: Iterable[str] = (),
             survivors_passed: Optional[bool] = None,
             survivors_detail: str = "") -> Evidence:
    """What kind of evidence does this run's success actually rest on?

    ``acceptance_passed`` is tri-state: ``None`` when the user supplied no
    acceptance commands, so their absence is never mistaken for a failure.
    """
    cmds = list(acceptance_cmds or ())
    if acceptance_passed:
        shown = "; ".join(cmds[:2]) + (f" (+{len(cmds) - 2})"
                                       if len(cmds) > 2 else "")
        return Evidence(True, INDEPENDENT_ACCEPTANCE,
                        f"{len(cmds)} user-supplied acceptance command(s) "
                        f"passed: {shown}")

    if acceptance_passed is False:
        # Louder than "unverified", and the mirror of PRE_EXISTING_FAILED
        # below: the one instrument the model neither wrote nor can edit
        # ran and DISAGREED. That says strictly more than absence of
        # evidence, and it is the strongest negative signal this module
        # can report.
        #
        # `False` used to fall through to the self-authored branch, whose
        # advice is "Supply `acceptance_cmds` in .agentchanti.yaml" --
        # addressed to a user who had already supplied them, about a run
        # those very commands had just failed. Measured 2026-08-19 run
        # 29: the acceptance check correctly failed an artifact whose
        # login issued a `randomUUID()` token that no route ever
        # verified, and the closing line read `Evidence: self-authored
        # ... the run marked its own homework. Supply acceptance_cmds`.
        # The verdict was right and the explanation of it was wrong in
        # both halves.
        shown = "; ".join(cmds[:2]) + (f" (+{len(cmds) - 2})"
                                       if len(cmds) > 2 else "")
        return Evidence(False, ACCEPTANCE_FAILED,
                        f"{len(cmds)} user-supplied acceptance command(s) "
                        f"ran and FAILED: {shown} - the one instrument this "
                        f"run neither wrote nor can edit disagrees with it")

    if acceptance_passed is None and cmds:
        # Supplied, but never reached: the caller only runs them when the
        # pipeline is otherwise green. `None` therefore means two very
        # different things, and the self-authored branch below gives the
        # wrong advice for one of them -- "Supply `acceptance_cmds` in
        # .agentchanti.yaml" to a user who supplied them, about a run that
        # never got as far as executing them. Measured 2026-08-22 run 31,
        # which failed on a gate conflict and then printed exactly that.
        #
        # The evidence genuinely IS self-authored (nothing independent
        # ran), so `independent` stays False; only the explanation and the
        # advice change. Its own kind, so a benchmark can separate "no
        # contract" from "contract never got to run".
        shown = "; ".join(cmds[:2]) + (f" (+{len(cmds) - 2})"
                                       if len(cmds) > 2 else "")
        return Evidence(False, ACCEPTANCE_NOT_RUN,
                        f"{len(cmds)} acceptance command(s) are configured "
                        f"but did not run, because the pipeline failed "
                        f"before reaching them: {shown} - fix the failure "
                        f"above; nothing here says the contract is unmet")

    survivors = surviving_pre_existing_tests(root, snapshot)
    if survivors:
        shown = ", ".join(survivors[:3])
        more = f" (+{len(survivors) - 3} more)" if len(survivors) > 3 else ""
        # A surviving file is only evidence once it has been RUN and
        # passed. `tests_ran` says the pipeline ran tests of its own,
        # which is a different question and was the wrong one to ask.
        if survivors_passed is True:
            weak = shallow_survivors(root, survivors)
            return Evidence(True, INDEPENDENT_PRE_EXISTING,
                            f"{len(survivors)} test file(s) that predate the "
                            f"run and it did not modify passed: {shown}{more}"
                            + (f" — but {weak}" if weak else ""),
                            shallow=bool(weak))
        if survivors_passed is False:
            # Louder than "unverified": the one instrument this run did
            # not author disagrees with it.
            return Evidence(False, PRE_EXISTING_FAILED,
                            f"the pre-existing test file(s) this run did not "
                            f"modify FAILED: {survivors_detail or shown}")
        return Evidence(False, SELF_AUTHORED,
                        f"{len(survivors)} pre-existing test file(s) survived "
                        f"({shown}{more}) but none could be run, so nothing "
                        f"independent has actually agreed"
                        + (f" — {survivors_detail}" if survivors_detail else ""))

    # Everything below here is a run judged only by its own work.
    if snapshot and not survivors:
        return Evidence(
            False, SELF_AUTHORED,
            f"every one of the {len(snapshot)} pre-existing test file(s) was "
            f"modified during the run, so no test survived that the agent "
            f"did not write or rewrite")
    if not tests_ran:
        return Evidence(
            False, NO_TESTS,
            "no test suite ran, so the only evidence is that the planned "
            "steps completed")
    return Evidence(
        False, SELF_AUTHORED,
        "every test that passed was written during this run — the run "
        "marked its own homework. Supply `acceptance_cmds` in "
        ".agentchanti.yaml, or keep a test the agent does not touch, to "
        "earn a verified result")


def run_acceptance_commands(executor, cmds: Iterable[str]
                            ) -> tuple[Optional[bool], list[str]]:
    """Run the user's own acceptance commands. ``(passed, failures)``.

    These are the one instrument in the run that the model neither wrote
    nor can edit, so they are also the only ones allowed to fail the run
    on their own. ``None`` when none were supplied.
    """
    cmds = [c for c in (cmds or ()) if c and c.strip()]
    if not cmds:
        return None, []
    failures: list[str] = []
    for cmd in cmds:
        log.info(f"[Acceptance] {cmd}")
        try:
            success, output = executor.run_command(cmd)
        except Exception as exc:                      # noqa: BLE001
            failures.append(f"{cmd} — raised {exc!r}")
            continue
        if not success:
            failures.append(f"{cmd} — {str(output or '')[-300:]}".rstrip())
    if failures:
        for line in failures:
            log.error(f"[Acceptance] FAILED: {line}")
        return False, failures
    log.info(f"[Acceptance] all {len(cmds)} command(s) passed")
    return True, []


def repair_failed_acceptance(*, executor, cmds, failures, llm_client, memory,
                             task, language=None, max_rounds=3,
                             max_turns=8, escalation_client=None,
                             kb_context_builder=None, project_root=".",
                             run_acceptance=None):
    """Retry a failed acceptance check, repairing the PROJECT between rounds.

    Returns ``(passed, rounds_spent, failures)``.

    `acceptance_cmds` used to run once, at the very end, with nothing
    after it -- so a plan that simply never built what the contract
    requires failed the run outright, after every one of its own gates
    had passed. Measured 2026-08-19 run 29: 765k tokens and 13 minutes to
    discover on the final line that the backend had no protected
    endpoint, a fact fixed at 23:42 when the plan chose `randomUUID()`
    sessions and never installed a JWT library. Nothing between planning
    and exit could see it, because the pipeline verifies what the PLAN
    declares and this plan declared no such route.

    The model may READ the check -- that is the point, and
    `_acceptance_refusal` allows it -- but cannot write it. That refusal
    is what makes this loop safe: the cheapest path to green is to build
    the missing thing, because the contract itself is out of reach. It
    was inert in production until 2026-08-20 (`protect_acceptance_files`
    was called only from a test), which is why wiring it was a
    prerequisite for this function existing at all.

    "Until green" is bounded twice, because an unbounded loop against a
    contract nothing can satisfy would spend the entire budget proving
    it:

    * `max_rounds` caps the attempts outright; and
    * a **progress requirement** stops early when a round leaves the
      failure output byte-identical, which is `observe_gate_verdict`'s
      reasoning -- a measurement that does not move while the code does
      is not measuring the code. Here it is the stronger form: the
      recovery loop reported it changed nothing AND the verdict is
      unchanged, so another round has nothing new to work with.

    Only the FAILING commands are retried; ones already passing are not
    re-run between rounds, so a repair cannot be credited for them. They
    are all re-run once at the end, because a fix for one command can
    break another.
    """
    from .agent_loop import build_step_tools, run_recovery_loop

    if not cmds or max_rounds <= 0 or llm_client is None:
        return False, 0, list(failures or ())

    failing = list(failures or ())
    last_signature = None
    rounds = 0

    for attempt in range(1, max_rounds + 1):
        detail = "\n".join(failing)[-4000:]
        signature = detail.strip()
        if last_signature is not None and signature == last_signature:
            log.warning(
                "[Acceptance] repair round %d produced a byte-identical "
                "failure — stopping: the check is not responding to the "
                "changes being made", attempt)
            break
        last_signature = signature
        rounds = attempt

        log.info("[Acceptance] repair round %d/%d — handing the failure to "
                  "the agent loop", attempt, max_rounds)
        tools = build_step_tools(executor, memory,
                                 kb_context_builder=kb_context_builder,
                                 project_root=project_root)
        # The final round gets the stronger model, the same escalation
        # `_run_diagnosis_loop` makes on its last attempt.
        client = (escalation_client
                  if (attempt == max_rounds and escalation_client is not None)
                  else llm_client)
        run_recovery_loop(
            client, tools,
            step_text=(
                "The run's user-supplied acceptance check FAILED. It is the "
                "one verification in this run that you did not write, and "
                "you may read it but not edit it. Change the PROJECT until "
                "it passes.\nCommand(s): " + "; ".join(cmds)),
            task=task, error_info=detail, display=None, step_idx=0,
            language=language, max_turns=max_turns,
            verify_cmd=None, escalation_client=escalation_client)

        passed, failing = (run_acceptance or run_acceptance_commands)(executor, cmds)
        if passed:
            log.info("[Acceptance] all command(s) passed after %d repair "
                      "round(s)", attempt)
            return True, attempt, []

    return False, rounds, failing
