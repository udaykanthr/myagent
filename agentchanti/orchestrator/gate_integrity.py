"""Is the GATE the defect, rather than the code it judges?

``verify_passed`` already encodes "exit 0 is not proof". This is the
mirror: **exit 1 is not proof either**, because a gate can be an invalid
instrument rather than a failing test.

WHY THIS EXISTS
---------------
Observed on a React/Vite run. The planner declared this acceptance gate::

    node -e "...if(!/@media \\\\(max-width: 48rem\\\\)[\\\\s\\\\S]*.../.test(s))process.exit(1)"

Note the DOUBLE backslashes. Under a POSIX shell the ``"..."`` quoting
collapses ``\\\\`` to ``\\`` before node ever sees it, so node compiles
``[\\s\\S]`` — "any character", the intended meaning. Under Windows
``cmd.exe`` there is no such collapsing: node receives ``[\\\\s\\\\S]``,
which is a character class matching a literal backslash, ``s`` or ``S``.
The regex can then never match ordinary CSS, so the gate was unsatisfiable
on Windows and satisfiable on Linux — from identical plan text.

The cost was not academic. The CSS edit was correct on the very first
turn. The primary loop, the escalation to a stronger model, and the
recovery loop all failed against that one gate — 24 turns across three
attempts, ~182k tokens, and the run reported failure on working code. The
escalated model even PROVED the gate wrong at turn 2 by printing each
sub-condition (all true) — and had nowhere to put that finding, because
the gate was the only thing allowed to decide.

WHY A DIFFERENTIAL RE-RUN, AND NOT A PARSER
-------------------------------------------
The broken payload is *syntactically valid* JavaScript — ``\\\\(`` parses
fine as "literal backslash, then a capture group". So a syntax check
(the ``ast.parse`` approach used by ``unrunnable_gate_reason``) cannot
catch it, in any language. Nor can "does this look like a mis-escape?",
which is a guess.

What IS decidable is behaviour: run the same text under the other shell
dialect's reading and see whether it passes. That needs no parser and no
knowledge of the payload's language — it works identically for
``python -c``, ``node -e``, ``ruby -e`` or anything else — and it yields
proof rather than suspicion.

THE SAFETY BOUNDARY
-------------------
A variant is only ever a **platform-equivalent re-reading of the identical
text**, produced by one whitelisted pure transform. It is never authored
by a model and never semantically different. Widen that and this stops
being a gate-integrity check and becomes a machine for manufacturing
false greens — "mutate the gate until something passes" — which is the
exact failure this project's verification layers exist to prevent.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import List, Tuple

_logger = logging.getLogger(__name__)


def collapse_posix_escapes(cmd: str) -> Tuple[str, bool]:
    """Apply the backslash collapsing a POSIX shell would have applied.

    Returns ``(rewritten, changed)``.

    Only ``\\\\`` -> ``\\`` inside DOUBLE-QUOTED regions is collapsed:

    * That is the transform proven to differ between the platforms, and
      the one that broke a real run. POSIX also unescapes ``\\$`` and
      ``\\```; those are left alone because they are rarer, and every
      extra transform widens the blast radius of a wrong guess.
    * Outside quotes a backslash on Windows is usually a path separator
      (``venv\\Scripts\\activate``). Rewriting there would corrupt working
      commands to fix a bug that only occurs in quoted payloads.
    * ``\\"`` is deliberately preserved: POSIX turns it into a literal
      quote, and so do the Windows argv rules, so the two platforms
      already agree and there is nothing to reconcile.
    """
    out: List[str] = []
    in_double_quotes = False
    changed = False
    i, n = 0, len(cmd)

    while i < n:
        ch = cmd[i]

        if not in_double_quotes:
            if ch == '"':
                in_double_quotes = True
            out.append(ch)
            i += 1
            continue

        if ch == '\\' and i + 1 < n:
            nxt = cmd[i + 1]
            if nxt == '\\':
                # The pair a POSIX shell would have eaten one level of.
                out.append('\\')
                i += 2
                changed = True
                continue
            if nxt == '"':
                # Escaped quote: identical on both platforms, and it must
                # NOT toggle the quote state — treating it as a closing
                # quote would mis-scan the rest of the command.
                out.append('\\')
                out.append('"')
                i += 2
                continue

        if ch == '"':
            in_double_quotes = False
        out.append(ch)
        i += 1

    return ''.join(out), changed


# ---------------------------------------------------------------------------
# POSIX idioms that cmd.exe cannot execute at all
# ---------------------------------------------------------------------------
#
# Distinct from the backslash problem above, which is one text read two
# ways. These are constructs that simply do not exist under cmd.exe, so
# the gate fails on the SHELL before the code under test is ever
# consulted. A syntax check cannot see them — the inline payload is
# perfectly valid Python — and the failure is identical no matter what
# the step writes.
#
# Measured 2026-08-17, a gate carrying four of them at once::
#
#     python main.py > /dev/null 2>&1 & timeout 3 python -c "...psutil..." & wait
#
# `> /dev/null` makes cmd try to create the path `\dev\null` and exit 1
# before anything else runs; `timeout N <cmd>` is a POSIX coreutil and
# Windows' `timeout` takes `/t N` with no command, answering "Invalid
# syntax"; `wait` does not exist; and `&` is a sequential separator here,
# not a background operator, so nothing was ever backgrounded either.
# Twenty turns across two models, 467k tokens, every attempt returning
# the identical error.
_POSIX_ONLY_IDIOMS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(^|\s)\d?>\s*/dev/null"),
     "`> /dev/null` — cmd.exe tries to create the path \\dev\\null and "
     "exits 1 before the command runs (use `> nul`)"),
    (re.compile(r"(^|\s)timeout\s+\d+\s+\S"),
     "`timeout N <command>` — Windows' timeout takes `/t N` and runs no "
     "command, so it answers 'Invalid syntax'"),
    (re.compile(r"(^|&|\||;)\s*wait\s*($|&|\||;)"),
     "`wait` — no such command on Windows"),
    (re.compile(r"(^|\s)2>&1\s*&\s*$"),
     "a trailing `&` — cmd.exe reads `&` as a sequential separator, not "
     "as 'run in the background'"),
]


def posix_only_idiom_reason(cmd: str) -> str | None:
    """Why *cmd* cannot run under this platform's shell, or None.

    Returns None on POSIX, where every one of these is correct. The check
    is about the shell the gate will actually be handed to, so it must
    never fire on a platform where the idiom works.
    """
    if not cmd or os.name != 'nt':
        return None
    for pattern, reason in _POSIX_ONLY_IDIOMS:
        if pattern.search(cmd):
            return (f"the gate uses a POSIX shell idiom this platform "
                    f"cannot run: {reason}")
    return None


def _to_cmd_dialect(cmd: str) -> str:
    """Rewrite POSIX-only idioms into their cmd.exe equivalents.

    A pure text transform, in the same spirit as the backslash collapse:
    it changes dialect, never meaning, and the result is only ever
    *believed if it passes*.
    """
    out = re.sub(r"(^|\s)(\d?)>\s*/dev/null", r"\1\2> nul", cmd)
    out = re.sub(r"(^|\s)timeout\s+\d+\s+", r"\1", out)
    out = re.sub(r"(^|&|\||;)\s*wait\s*($|&|\||;)", r"\1\2", out)
    return re.sub(r"\s*&\s*$", "", out).strip()


def redundant_cd_path_variant(cmd: str) -> str | None:
    r"""*cmd* with the `cd`-ed directory stripped from paths that repeat it.

    A gate that enters a directory and then names paths already prefixed
    with it resolves one level too deep, and no output of the step can
    fix that. Measured 2026-08-19 run 15::

        cd frontend && npm run build
          && findstr /c:"aria-label" frontend\src\components\Navigation.jsx >nul
          && ...

    From inside `frontend/` those paths are `frontend/frontend/src/...`.
    The step failed six times over five rewrites; `observe_gate_verdict`
    correctly called it stalled and suppressed the escalation, but by
    then the agent had already done what an unsatisfiable gate always
    invites and CREATED the paths it named — `frontend/frontend/src/
    pages/{Dashboard,ForgotPassword,Signup}Page.jsx`, all reported as
    unplanned writes.

    Offering the corrected reading as a variant means the loop tries it
    on the first failure, under the same believe-only-if-it-passes rule
    as the other transforms here: a dialect change, never a meaning
    change. The planner's own `cd` is kept — it is usually right, since
    `npm run build` needs it — and only the redundant repetition goes.
    """
    if not cmd:
        return None
    m = re.match(r"^\s*cd\s+([^\s&|;]+)\s*&&\s*(.+)$", cmd, re.DOTALL)
    if not m:
        return None
    target = m.group(1).strip("\"'").replace("\\", "/").strip("./").rstrip("/")
    if not target:
        return None
    rest = m.group(2)
    # Only a path SEGMENT counts: `frontend\src` and `./frontend/src`
    # repeat the cd, while `myfrontend/src` and a bare word do not.
    pattern = re.compile(
        r"(^|[\s\"'=(,])(?:\./|\.\\)?" + re.escape(target) + r"[\\/]")
    rewritten, n = pattern.subn(r"\1", rest)
    if not n or rewritten == rest:
        return None
    return cmd[:m.start(2)] + rewritten


def platform_equivalent_variants(cmd: str) -> List[Tuple[str, str]]:
    """Other readings of *cmd* under a different shell dialect.

    Empty on POSIX: the shell already performed the collapsing there, so
    the command the planner wrote and the command that ran already agree
    and there is no second reading to try.
    """
    variants: List[Tuple[str, str]] = []
    # Not platform-specific: a `cd` repeated in the paths that follow it
    # resolves one level too deep under every shell.
    redundant = redundant_cd_path_variant(cmd)
    if redundant and redundant != cmd:
        variants.append(("redundant-cd-path", redundant))
    if not cmd or os.name != 'nt':
        return variants
    collapsed, changed = collapse_posix_escapes(cmd)
    if changed and collapsed != cmd:
        variants.append(("posix-backslash-collapse", collapsed))
    if posix_only_idiom_reason(cmd):
        translated = _to_cmd_dialect(cmd)
        if translated and translated != cmd:
            variants.append(("posix-shell-idioms", translated))
    return variants


# ---------------------------------------------------------------------------
# Repairs, so a gate proven defective is not re-run in its broken form
# ---------------------------------------------------------------------------

# Keyed by the ORIGINAL command text rather than a step index: the same
# gate is enforced by the main loop, the escalation and the recovery loop,
# and is later re-run by the monotonic GateLedger. Keying by text means a
# repair proven once is known everywhere that command appears, without
# threading a new argument through four call sites.
_repairs: dict[str, str] = {}
_repairs_lock = threading.Lock()


def record_gate_repair(original: str, repaired: str, reason: str) -> None:
    """Remember that *original* is defective and *repaired* is equivalent."""
    if not original or not repaired or original == repaired:
        return
    with _repairs_lock:
        _repairs[original] = repaired
    _logger.warning(
        "[GateIntegrity] gate repaired (%s) — the ORIGINAL form can never "
        "pass on this platform:\n  original: %s\n  repaired: %s",
        reason, original, repaired)


def repaired_gate(cmd: str | None) -> str | None:
    """The proven-equivalent replacement for *cmd*, or None."""
    if not cmd:
        return None
    with _repairs_lock:
        return _repairs.get(cmd)


# ---------------------------------------------------------------------------
# The stall breaker: a gate that returns the same verdict about changed code
# ---------------------------------------------------------------------------
#
# The two mechanisms above catch a gate broken in a way that can be NAMED —
# a mis-escape, an idiom of the wrong shell. This one needs no diagnosis at
# all, and that is the point: it catches the class rather than the spelling.
#
# A gate is a measurement of the artifact. If the artifact changes and the
# measurement does not — byte for byte, repeatedly — then whatever the gate
# is measuring, it is not the artifact. That is decidable from the loop's
# own observations, with no parser and no knowledge of the language.
#
# Measured 2026-08-17: one gate, `Exit code 1, output=246 chars` on all
# twenty attempts, while main.py went through 17 distinct content hashes —
# ten turns of Haiku, then ten of an escalated gpt-5.6-sol, 467k tokens,
# and not one attempt was ever about the code. Three identical verdicts
# were enough to know that at turn three.
#
# Keyed by command text for the same reason as `_repairs`: the primary
# loop, the escalation and the recovery loop all enforce the same gate,
# and the escalation is exactly the expensive repetition worth stopping.
_STALL_THRESHOLD = 3

_verdicts: dict[str, dict] = {}
_verdicts_lock = threading.Lock()


def reset_gate_verdicts() -> None:
    """Forget every observation. Called per run, like the gate ledger."""
    with _verdicts_lock:
        _verdicts.clear()


# Markers that the command got as far as executing project code. Any one
# of them means a failure is ABOUT the artifact, however unchanging its
# text — an interpreter traceback, or a test runner reporting results.
# Their absence is what a shell rejecting the command looks like.
_REACHED_CODE_MARKERS = (
    "Traceback (most recent call last)",
    "AssertionError",
    "AttributeError",
    "NameError",
    "TypeError",
    "ValueError",
    "KeyError",
    "IndexError",
    "FAILED (",
    "= FAILURES =",
    "short test summary",
)


def _reached_the_code(result: str) -> bool:
    """Did this failure come from running the project, or from the shell?"""
    text = result or ""
    if any(m in text for m in _REACHED_CODE_MARKERS):
        return True
    # A runner that collected and ran tests reported on the code even
    # when nothing raised. Matched by shape rather than by regex: a
    # count sitting next to a runner word. split() handles newlines.
    words = text.split()
    for i, w in enumerate(words[:-1]):
        nxt = words[i + 1].rstrip(",.:")
        if w == "Ran" and nxt.isdigit():
            return True
        if w.rstrip(",.:").isdigit() and nxt in (
                "passed", "failed", "test", "tests", "error", "errors"):
            return True
    return False


def observe_gate_verdict(cmd: str | None, result: str,
                         artifact_digest: str) -> str | None:
    """Record one verdict; return a reason once the gate is proven stalled.

    Stalled means all four of:

    * at least ``_STALL_THRESHOLD`` verdicts for this exact command,
    * every one of them byte-identical **and failing** — a gate that ever
      passed is measuring something, and one whose message varies is
      responding to the code,
    * at least two distinct artifact digests, so a model that edited
      nothing does not look like a broken gate,
    * and the failure never reached the project's code at all.

    That last condition was missing, and its absence produced a wrong
    verdict on the check's second live outing (2026-08-18 08:14). A gate
    asserting `game.snake_positions` failed three times with a byte-
    identical ``AttributeError`` while the model edited the file, so the
    first three conditions held and the step was declared unmeasurable —
    escalation suppressed. The recovery loop then satisfied that exact
    gate in three turns, and it passes against the finished artifact.

    The reasoning was simply wrong. An ``AttributeError`` naming a symbol
    the code lacks is byte-identical across every edit that does not add
    that symbol, and it is the most *diagnostic* failure a gate can
    produce — the opposite of one that cannot see the code. What the
    original incident had instead was a shell error (``The system cannot
    find the path specified``) that named nothing about the project,
    because cmd.exe rejected the command before any code ran.

    So the discriminator is whether the command ever reached the code.
    The bias is deliberate: a false negative costs turns the loop had
    anyway, a false positive suppresses real work.

    Returns None while any condition is unmet, so the ordinary case — a
    gate failing differently as the code improves — is untouched.
    """
    if not cmd:
        return None
    with _verdicts_lock:
        entry = _verdicts.setdefault(
            cmd, {"results": [], "digests": set(), "reported": False})
        entry["results"].append((result or "").strip())
        entry["digests"].add(artifact_digest)
        if entry["reported"]:
            return None
        results = entry["results"]
        if len(results) < _STALL_THRESHOLD:
            return None
        if len(set(results[-_STALL_THRESHOLD:])) != 1:
            return None
        if any(r.startswith("exit: success") for r in results):
            return None
        if len(entry["digests"]) < 2:
            return None
        if _reached_the_code(results[-1]):
            return None
        entry["reported"] = True

    _logger.error(
        "[GateIntegrity] gate STALLED — %d identical failing verdicts over "
        "%d different versions of the code. The gate is not measuring the "
        "artifact:\n  gate: %s",
        len(results), len(entry["digests"]), cmd)
    return (
        f"the gate returned the IDENTICAL failure {len(results)} times "
        f"while the code changed {len(entry['digests'])} times, so it is "
        f"not measuring the artifact — the gate is the defect, not the "
        f"code. Gate: {cmd}"
    )


def effective_gate(cmd: str | None) -> str | None:
    """*cmd*, or its repaired form when one was proven."""
    return repaired_gate(cmd) or cmd


def reset_repairs() -> None:
    """Drop every recorded repair (tests, and between runs in-process)."""
    with _repairs_lock:
        _repairs.clear()


# ---------------------------------------------------------------------------
# A gate superseded by the command diagnosis proved equivalent
# ---------------------------------------------------------------------------
#
# The module above catches a gate broken by SHELL ESCAPING. This catches the
# other way a gate is the defect: it names something that does not exist.
#
# Observed twice on hello-world runs, both on working code:
#
#   gate: python -m pytest test_hello.py -q   -> exit 4, no such file
#   ran : python -m pytest tests/test_hello_world.py -> exit 0, 2 passed
#
# The tester had written a conventionally-named file, so the plan's gate
# pointed at a path nobody created. Diagnosis identified this correctly every
# round and proposed the working command; the pipeline RAN that command, saw
# it pass, then re-ran the gate and failed the step. Three rounds, then halt.
#
# `_handle_cmd_step` already has this idea for CMD steps: when a fix command
# is "the same core operation" as the failed one, the step is resolved rather
# than re-running something known to fail. It is restricted to CMD steps and
# compares against the failed command, so a broken CODE/TEST *gate* never
# benefits.
#
# The decision is deliberately behavioural, matching this module's existing
# stance that "what IS decidable is behaviour": re-run both, and only accept
# the substitution when the gate still fails AND the candidate still passes.

_RUNNERS = frozenset({
    "pytest", "unittest", "nose2", "tox",
    "jest", "vitest", "mocha", "jasmine", "ava",
    "rspec", "phpunit",
})
# Tools whose SUBCOMMAND decides what they do: `go test` is a gate,
# `go build` is not.
_SUBCOMMAND_RUNNERS = frozenset({"go", "cargo", "gradle", "mvn", "dotnet"})
_INTERPRETERS = frozenset({"python", "python3", "py", "node", "ruby", "perl"})
# Script runners — `npm test` really is a gate, so these carry an operation.
# pip is deliberately ABSENT: there is no `pip test`, and an installer
# proves nothing about the code. Python's actual gates (pytest, unittest,
# tox) are runners in their own right and are covered above, including the
# `poetry run pytest` / `uv run pytest` forms.
_PKG_MANAGERS = frozenset({"npm", "yarn", "pnpm", "bun"})
# Modules that provision an environment rather than judge code, even when
# invoked as `python -m <module>`.
_INSTALLER_MODULES = frozenset({"pip", "ensurepip", "venv", "virtualenv"})
# Verbs that mutate the environment rather than judge the code.
_INSTALL_VERBS = frozenset({
    "install", "add", "ci", "sync", "uninstall", "remove", "update",
    "upgrade", "restore", "fetch",
})
_SEGMENT_RE = re.compile(r"&&|\|\||;|\|")


def _basename(token: str) -> str:
    return re.split(r"[\\/]", token.strip().strip('"\''))[-1].lower()


def _segment_operation(tokens: List[str]) -> str | None:
    """The instrument one shell segment drives, or None.

    Position matters, and an early version of this ignored it: scanning
    every token for a known runner name made `pip install pytest` report
    "pytest" — the package being INSTALLED read as the instrument. Since
    that command exits 0 whenever pytest is already present, it could have
    been adopted as a stand-in for a test suite, which is exactly the
    "gate quietly replaced by something weaker" this whole mechanism has to
    refuse. An installer verifies nothing; it must have no operation at all.
    """
    if not tokens:
        return None
    head = _basename(tokens[0])
    positional = [t for t in tokens[1:] if not t.startswith("-")]

    # Installing/removing is a side effect, never a verification.
    if positional and positional[0].lower() in _INSTALL_VERBS:
        return None

    # `npm test`, `npm run test` — the runner lives in package.json, so the
    # script name is the identity. Without this a JS gate has no operation
    # and could never be superseded.
    if head in _PKG_MANAGERS:
        if not positional:
            return None
        script = positional[0].lower()
        if script == "run" and len(positional) > 1:
            script = positional[1].lower()
        return f"{head}:{script}"

    # `go test ./...` vs `go build` are different instruments.
    if head in _SUBCOMMAND_RUNNERS:
        return f"{head}:{positional[0].lower()}" if positional else head

    if head in _RUNNERS:
        return head

    if head in _INTERPRETERS:
        # `python -m pytest ...` — the module IS the runner.
        if "-m" in tokens:
            idx = tokens.index("-m")
            if idx + 1 < len(tokens):
                module = _basename(tokens[idx + 1])
                after = [t for t in tokens[idx + 2:] if not t.startswith("-")]
                # `python -m pip install ...` / `python -m venv env`: an
                # installer is still an installer when routed through -m.
                if (module in _INSTALLER_MODULES
                        or (after and after[0].lower() in _INSTALL_VERBS)):
                    return None
                return module if module in _RUNNERS else f"module:{module}"
        # `python hello.py` — the script identifies the operation.
        if positional:
            return "script:" + _basename(positional[0])
        return None

    # `poetry run pytest`, `uv run pytest -q`, `pipenv run pytest`.
    lowered = [t.lower() for t in tokens]
    if "run" in lowered:
        after = [t for t in tokens[lowered.index("run") + 1:]
                 if not t.startswith("-")]
        if after and _basename(after[0]) in _RUNNERS:
            return _basename(after[0])
    return None


def gate_operation(cmd: str) -> set[str]:
    """The tool identities *cmd* invokes — its "core operation".

    `python -m pytest a.py` and `pytest b.py` are both {"pytest"}: same
    instrument, different argument. That is the level at which one command
    can stand in for another. A command that verifies nothing — `echo ok`,
    `pip install pytest`, `cd build` — yields the empty set and can
    therefore never be accepted as a substitute, which is the point.
    """
    ops: set[str] = set()
    for segment in _SEGMENT_RE.split(cmd or ""):
        found = _segment_operation([t for t in segment.split() if t])
        if found:
            ops.add(found)
    return ops


def same_gate_operation(a: str, b: str) -> bool:
    """Do *a* and *b* drive the same instrument?"""
    return bool(gate_operation(a) & gate_operation(b))


def prove_gate_superseded(gate: str, candidate: str, run) -> bool:
    """Is *candidate* a working stand-in for a *gate* that cannot pass?

    *run* is called as ``run(cmd) -> (ok, output)``.

    Both are re-run at decision time rather than trusting the earlier
    observations: the files have changed since, and a stale "it passed once"
    is exactly the kind of evidence that lets a weak gate through. Accepting
    a substitute is only safe while the original genuinely fails, so a gate
    that has started passing is left completely alone.
    """
    if not gate or not candidate or gate.strip() == candidate.strip():
        return False
    if not same_gate_operation(gate, candidate):
        return False
    gate_ok, _ = run(gate)
    if gate_ok:
        return False        # the gate works — there is nothing to supersede
    candidate_ok, _ = run(candidate)
    return bool(candidate_ok)
