"""Per-wave git snapshots of the target project + monotonic gate ledger.

Safety by checkpoint/rollback instead of edit vetoes: after every green
wave the project workdir is committed to a machine-managed git repo, and
a ledger records every per-step acceptance gate that has passed. When a
later fix round leaves a previously-green gate red (a regression), the
project rolls back to the last snapshot instead of shipping the
regression.

The snapshot repo is only ever one agentchanti creates itself (marked
with ``.git/agentchanti-managed``). When the workdir already lives in a
git repository — the user's own — snapshots are disabled: nesting a
machine repo inside a user repo would hide their files from their own
tracking, and their git history is already a better safety net.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from threading import Lock

_logger = logging.getLogger(__name__)

# A gate that runs a whole SUITE, as opposed to one inline assertion.
# Kept deliberately separate from classification._TEST_CMD_RE: that one
# matches step *prose* ("run the tests") and omits `unittest`, which is
# exactly the runner a plain-Python task asks for.
_SUITE_GATE_RE = re.compile(
    r'(?<![-\w])(?:'
    r'python[0-9.]*\s+-m\s+unittest'
    r'|python[0-9.]*\s+-m\s+pytest'
    r'|pytest'
    r'|npm\s+(?:run\s+)?test|yarn\s+test|pnpm\s+test'
    r'|npx\s+(?:vitest|jest)'
    r'|(?:vitest|jest|mocha|rspec|phpunit)'
    r'|go\s+test'
    r'|cargo\s+test'
    r')(?![-\w])',
    re.IGNORECASE,
)


def is_suite_gate(cmd: str) -> bool:
    """True when *cmd* runs a test suite rather than one inline assertion.

    A suite is the task's own statement of correctness — it is where
    requirements like "2000+ frames without the player moving" end up.
    An inline `python -c "assert ..."` gate is a planner's guess at the
    same thing, and the two can contradict each other.
    """
    return bool(cmd) and bool(_SUITE_GATE_RE.search(cmd))

# Identity flags so snapshot commits work on machines with no git
# user configured; --no-verify at the call sites keeps user-level
# hook templates from interfering with machine commits.
_GIT_ID = [
    "-c", "user.name=agentchanti",
    "-c", "user.email=agentchanti@local",
    "-c", "commit.gpgsign=false",
]

_MARKER_NAME = "agentchanti-managed"

_DEFAULT_GITIGNORE = """\
.agentchanti/
venv/
.venv/
node_modules/
__pycache__/
*.pyc
.pytest_cache/
db.sqlite3
"""


# ── Gate ledger ───────────────────────────────────────────────────────


# Signatures of a gate command that could not even launch — the
# interpreter/executable was not found or the cwd was wrong. These mean
# the recorded command is un-runnable in the current environment, NOT
# that the project's code regressed, so they must never trigger a
# rollback of otherwise-green code. Genuine code failures (assertion
# errors, ImportError/ModuleNotFoundError from the project's own modules,
# non-zero test exits) are deliberately excluded.
_HARNESS_ERROR_SIGNATURES = (
    "the system cannot find the path specified",
    "the system cannot find the file specified",
    "is not recognized as an internal or external command",
    "no such file or directory",
    "command not found",
    "can't open file",
    "cannot open file",
)


def _is_harness_error(out: str | None) -> bool:
    """True when *out* shows the gate command failed to launch (env/cwd),
    rather than the project's code failing the check."""
    low = (out or "").lower()
    return any(sig in low for sig in _HARNESS_ERROR_SIGNATURES)


# Windows NTSTATUS failures start at 0xC0000000: 0xC0000409 fast-fail,
# 0xC0000005 access violation, 0xC0000374 heap corruption. POSIX reports a
# fatal signal as a negative return code.
_NTSTATUS_FAILURE_BASE = 0xC0000000

# Most a single gate is sampled before the ledger gives up on getting two
# agreeing verdicts. Three covers the worst honest sequence — a crash that
# costs a sample, then the failure and its confirmation — while keeping a
# genuinely red gate to one extra run per re-check.
_MAX_GATE_SAMPLES = 3


def is_abnormal_exit(code: int | None) -> bool:
    """True when the process DIED rather than reporting a verdict.

    A gate that crashes has produced no verdict at all, so treating it as
    a regression is a category error. Observed: a green `python -m
    unittest -v` gate was re-run by the monotonic check, the interpreter
    fast-failed with 0xC0000409 (3221226505) after printing ordinary test
    output, and the ledger read that as "the tests regressed" and rolled
    back a correct test file. See [[silent-crash-diagnostics]] — the
    crash is intermittent and unrelated to the code under test.
    """
    # Strictly an int: anything else (None, a Mock, an executor that does
    # not track it) is absence of evidence, and guessing "crashed" there
    # would suppress every genuine regression.
    if not isinstance(code, int) or isinstance(code, bool):
        return False
    if code < 0:
        return True                       # POSIX: killed by a signal
    return code >= _NTSTATUS_FAILURE_BASE


class GateLedger:
    """Acceptance commands that have passed at least once this run.

    Thread-safe (steps in the same wave run in parallel). Keyed by the
    exact command string — the same gate recorded by several steps is
    rechecked once.
    """

    def __init__(self) -> None:
        self._gates: dict[str, str] = {}  # cmd -> step label
        self._last_output: dict[str, str] = {}  # cmd -> output of its last run
        self._lock = Lock()

    def record(self, cmd: str, step_label: str = "") -> None:
        if not cmd:
            return
        # The ledger is where a gate's side effects get multiplied: every
        # recorded gate is re-run after every later wave. A destructive
        # one is therefore worst exactly here, and is refused entry
        # rather than being recorded and guarded at run time.
        from .gate_safety import destructive_reason
        unsafe = destructive_reason(cmd)
        if unsafe:
            _logger.error(
                "[GateSafety] not recording a destructive gate (%s) — it "
                "would be re-run after every later wave: %s",
                step_label or "?", unsafe)
            return
        with self._lock:
            if cmd not in self._gates:
                self._gates[cmd] = step_label
                _logger.info("[GateLedger] recorded gate (%s): %s",
                             step_label or "?", cmd)

    def gates(self) -> dict[str, str]:
        with self._lock:
            return dict(self._gates)

    def last_output(self, cmd: str) -> str | None:
        """Output of *cmd*'s most recent run, or None if it never ran."""
        with self._lock:
            return self._last_output.get(cmd)

    def reset(self) -> None:
        with self._lock:
            self._gates.clear()
            self._last_output.clear()

    def _sample_gate(self, executor, cmd: str,
                     timeout: int) -> tuple[str, str, int | None]:
        """Run *cmd* once and classify what came back.

        ``"pass"``    — the gate is green.
        ``"crash"``   — the process died, so it reported no verdict at all.
        ``"harness"`` — the command could not launch (env/cwd), likewise
                        no verdict about the code.
        ``"fail"``    — the code under test genuinely failed the check.
        """
        ok, out = executor.run_command(cmd, timeout=timeout)
        exit_code = getattr(executor, "last_exit_code", None)
        # Kept so a PASSING gate's output stays inspectable. A green
        # verdict is not self-describing: `vitest --passWithNoTests` over
        # zero test files exits 0, and only its output says so.
        with self._lock:
            self._last_output[cmd] = out or ""
        if ok:
            return "pass", out, exit_code
        if is_abnormal_exit(exit_code):
            return "crash", out, exit_code
        if _is_harness_error(out):
            return "harness", out, exit_code
        return "fail", out, exit_code

    def _confirmed_failure(self, executor, cmd: str, label: str,
                           timeout: int) -> str | None:
        """Output of a failure seen **twice**, or ``None`` if unconfirmed.

        Rolling back is enormously more expensive than re-running: it
        discards a whole wave of real work, and it fails the run. So no
        single sample may decide it. A failure must reproduce before it is
        believed, and any sample that PASSES dismisses the earlier one.

        Observed 2026-08-12: the same `python -m unittest -v`, one second
        apart, on bytes that had not changed — exit 0, then exit 1. The
        ledger called the second sample a REGRESSION and rolled the run
        back; by hand afterwards the suite passed 12/12, and that
        project's pygame teardown was measured segfaulting 6 times in 30.
        The abnormal-exit guard below could not save it, because the
        deciding sample came back as an ordinary exit 1.
        """
        crashes = 0
        seen_failure: str | None = None

        for _attempt in range(_MAX_GATE_SAMPLES):
            verdict, out, exit_code = self._sample_gate(
                executor, cmd, timeout)

            if verdict == "pass":
                if seen_failure is not None or crashes:
                    _logger.warning(
                        "[GateLedger] gate PASSES on re-run — the earlier "
                        "result did not reproduce, so it is flaky rather "
                        "than a regression (%s): %s", label or "?", cmd)
                return None

            if verdict == "harness":
                # The command can no longer launch (missing interpreter /
                # wrong cwd) — inconclusive, not a code regression. Rolling
                # back over this would discard good code (observed: a
                # `cd sub && venv\Scripts\python.exe ...` gate whose venv
                # path stopped resolving from the sub-dir).
                _logger.warning(
                    "[GateLedger] gate no longer launches — harness/env "
                    "error, treating as inconclusive rather than a "
                    "regression (%s): %s", label or "?", cmd)
                return None

            if verdict == "crash":
                # A dead process is not a verdict. Retry: the native
                # fast-fail this guards against is intermittent. Two
                # crashes mean the gate cannot be read at all here — more
                # attempts would only cost time.
                crashes += 1
                if crashes >= 2:
                    _logger.warning(
                        "[GateLedger] gate crashed again — treating as "
                        "inconclusive rather than a regression (%s): %s",
                        label or "?", cmd)
                    return None
                _logger.warning(
                    "[GateLedger] gate process terminated abnormally "
                    "(%s) — retrying once before believing it (%s): %s",
                    describe_abnormal_exit(exit_code) or exit_code,
                    label or "?", cmd)
                log_crash_diagnostics(exit_code, cmd)
                continue

            if seen_failure is not None:
                _logger.warning(
                    "[GateLedger] REGRESSION — previously-passing gate "
                    "fails twice in a row (%s): %s", label or "?", cmd)
                return out
            seen_failure = out
            _logger.info(
                "[GateLedger] gate failed (%s): %s — re-running once to "
                "confirm before treating it as a regression",
                label or "?", cmd)

        _logger.warning(
            "[GateLedger] gate never failed twice in %d sample(s) — "
            "inconclusive rather than a regression (%s): %s",
            _MAX_GATE_SAMPLES, label or "?", cmd)
        return None

    def recheck(self, executor, timeout: int = 300) -> list[tuple[str, str, str]]:
        """Re-run every recorded gate; return the ones that now fail.

        Each regression is ``(cmd, step_label, output_tail)``, and each has
        been confirmed by a second failing run. An empty list means
        monotonic progress holds.
        """
        regressions: list[tuple[str, str, str]] = []
        for cmd, label in self.gates().items():
            out = self._confirmed_failure(executor, cmd, label, timeout)
            if out is not None:
                regressions.append((cmd, label, (out or "")[-1500:]))
        return regressions


_ledger = GateLedger()


def get_gate_ledger() -> GateLedger:
    return _ledger


# ── Project snapshots ─────────────────────────────────────────────────


class ProjectSnapshots:
    """Wave-granular git snapshots of the pipeline's working directory."""

    def __init__(self, root: str = ".", enabled: bool = True) -> None:
        self.root = os.path.abspath(root)
        self.enabled = enabled
        self.managed = False           # True only for a repo we created
        self._last_sha: str | None = None
        # Last snapshot whose gate recheck actually passed. Distinct from
        # _last_sha: a wave is committed before its gates are rechecked, so
        # _last_sha may be the very commit that introduced a regression.
        # Rolling back must target the last *verified* state, not HEAD.
        self._last_green_sha: str | None = None

    # -- plumbing ------------------------------------------------------

    def _git(self, *args: str) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["git", *_GIT_ID, *args],
                cwd=self.root, capture_output=True, text=True, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
        out = ((result.stdout or "") + (result.stderr or "")).strip()
        return result.returncode == 0, out

    def _marker_path(self) -> str:
        return os.path.join(self.root, ".git", _MARKER_NAME)

    def _ensure_gitignore(self) -> None:
        """Make sure every default ignore rule is present in .gitignore.

        Appends missing rules instead of overwriting, so a user-authored
        .gitignore is preserved. Idempotent on both fresh and resumed repos.
        """
        path = os.path.join(self.root, ".gitignore")
        existing = ""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = f.read()
            except OSError:
                return
        present = {ln.strip() for ln in existing.splitlines()}
        missing = [ln for ln in _DEFAULT_GITIGNORE.splitlines()
                   if ln.strip() and ln.strip() not in present]
        if not missing:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write("\n".join(missing) + "\n")
        except OSError:
            pass

    def _is_tracked(self, path: str) -> bool:
        ok, out = self._git("ls-files", "--", path)
        return ok and bool(out.strip())

    def _purge_tracked_volatile(self) -> None:
        """Untrack volatile runtime dirs that an earlier run may have committed.

        `.gitignore` never untracks already-committed files, so a stale
        `.agentchanti/` (live logs, cache) stays tracked and every
        `reset --hard` tries to restore/unlink the log file the running
        process holds open — on Windows that fails with "Invalid argument"
        and disables rollback. Dropping them from the index (working tree
        untouched) removes that hazard.
        """
        removed: list[str] = []
        for rel in (".agentchanti", "venv", ".venv", "node_modules",
                    "__pycache__"):
            if self._is_tracked(rel):
                self._git("rm", "-r", "--cached", "--ignore-unmatch", "--", rel)
                removed.append(rel)
        if removed:
            self._commit("agentchanti: stop tracking volatile runtime state")
            _logger.info("[Snapshots] Untracked volatile dir(s) so rollback "
                         "can't be blocked by open handles: %s",
                         ", ".join(removed))

    def _head_sha(self) -> str | None:
        ok, out = self._git("rev-parse", "HEAD")
        return out.strip() if ok else None

    # -- lifecycle -----------------------------------------------------

    def start(self) -> bool:
        """Initialise snapshotting. Returns True when snapshots are live."""
        if not self.enabled:
            return False

        if os.path.isfile(self._marker_path()):
            # Resumed run inside a repo this tool created earlier.
            self.managed = True
            # Older runs may have committed .agentchanti/ (logs, cache) before
            # the ignore rule existed; while those stay tracked, reset --hard
            # tries to unlink the live log Windows holds open and rollback
            # dies. Re-assert the ignore rules and untrack volatile state.
            self._ensure_gitignore()
            self._purge_tracked_volatile()
            self._last_sha = self._head_sha()
            # Nothing has run yet this session, so the resumed HEAD is the
            # baseline we can safely fall back to.
            self._last_green_sha = self._last_sha
            _logger.info("[Snapshots] Resuming managed snapshot repo at %s",
                         self.root)
            return True

        ok, _ = self._git("rev-parse", "--is-inside-work-tree")
        if ok:
            # The workdir belongs to an existing (user) repository.
            self.enabled = False
            _logger.info(
                "[Snapshots] Workdir is inside an existing git repo — "
                "wave snapshots disabled (the user's git is the safety net)")
            return False

        ok, out = self._git("init", "-q")
        if not ok:
            self.enabled = False
            _logger.warning("[Snapshots] git init failed — snapshots "
                            "disabled: %s", out[:200])
            return False

        self._ensure_gitignore()

        try:
            with open(self._marker_path(), "w", encoding="utf-8") as f:
                f.write("snapshot repo created by agentchanti\n")
        except OSError:
            pass

        self.managed = True
        self._commit("agentchanti: baseline (pre-run)")
        self._last_green_sha = self._last_sha
        _logger.info("[Snapshots] Initialised snapshot repo at %s", self.root)
        return True

    def _commit(self, message: str) -> str | None:
        ok, out = self._git("status", "--porcelain")
        if ok and not out.strip():
            return self._last_sha  # nothing new to snapshot
        self._git("add", "-A")
        ok, out = self._git("commit", "-q", "--no-verify", "-m", message)
        if not ok:
            _logger.warning("[Snapshots] commit failed: %s", out[:200])
            return self._last_sha
        self._last_sha = self._head_sha()
        return self._last_sha

    def commit_wave(self, label: str) -> str | None:
        """Snapshot the workdir after a wave. Returns the commit sha.

        The commit is *not* yet a rollback target — call :meth:`mark_green`
        once the wave's gates have been rechecked and still pass.
        """
        if not self.managed:
            return None
        sha = self._commit(f"agentchanti: {label}")
        if sha:
            _logger.info("[Snapshots] %s -> %s", label, sha[:12])
        return sha

    def mark_green(self) -> str | None:
        """Promote the latest snapshot to "last verified green".

        Only call this after the gate ledger has been rechecked and found
        clean. Everything committed since the previous green snapshot then
        becomes permanent as far as rollback is concerned.
        """
        if not self.managed:
            return None
        self._last_green_sha = self._last_sha
        return self._last_green_sha

    def last_green_sha(self) -> str | None:
        return self._last_green_sha

    def rollback_to_last(self) -> tuple[bool, str]:
        """Hard-restore the workdir to the last *verified green* snapshot.

        Deliberately not HEAD: waves are committed before their gates are
        rechecked, so HEAD can be the very commit that introduced the
        regression — resetting to it would "roll back" to the broken state
        and report success while discarding nothing.

        Untracked files are removed too (``clean -fd``) — but ignored
        ones (venv/, node_modules/, .agentchanti/) survive, so the
        environment does not need rebuilding after a rollback.
        """
        if not self.managed or not self._last_green_sha:
            return False, "no snapshot available"
        ok1, out1 = self._git("reset", "--hard", self._last_green_sha)
        ok2, out2 = self._git("clean", "-fd")
        ok = ok1 and ok2
        if ok:
            _logger.warning("[Snapshots] Rolled back workdir to %s",
                            self._last_green_sha[:12])
            self._last_sha = self._last_green_sha
        return ok, f"{out1}\n{out2}".strip()


# ── Native-crash diagnostics ─────────────────────────────────────────
# Windows NTSTATUS codes seen from child interpreters in real runs.
_NTSTATUS_NAMES = {
    0xC0000005: "ACCESS_VIOLATION (bad memory access in a native module)",
    0xC0000409: "STACK_BUFFER_OVERRUN / fast-fail "
                "(__fastfail, often a CRT or SDL abort)",
    0xC000001D: "ILLEGAL_INSTRUCTION",
    0xC00000FD: "STACK_OVERFLOW",
    0xC0000374: "HEAP_CORRUPTION",
}

_crash_hint_shown = False


def describe_abnormal_exit(code: int | None) -> str:
    """Human-readable cause for an abnormal exit, or ``""``.

    The bare number tells nobody anything: a run that dies with "exit code
    3221226505" reads as a mystery, when it is a native crash in a child
    interpreter and the pipeline's own code is not implicated.
    """
    if not isinstance(code, int) or isinstance(code, bool):
        return ""
    if code < 0:
        return f"killed by signal {-code}"
    if code < _NTSTATUS_FAILURE_BASE:
        return ""
    name = _NTSTATUS_NAMES.get(code & 0xFFFFFFFF)
    hexcode = f"0x{code & 0xFFFFFFFF:08X}"
    return f"{hexcode} — {name}" if name else f"{hexcode} — native crash"


def log_crash_diagnostics(code: int | None, command: str = "") -> None:
    """Explain an abnormal exit, and once per run say how to capture a dump.

    The pipeline already retries these (a crash is not a verdict), but the
    underlying fault stays invisible: the child dies before it can report
    anything, so the only way to see WHERE is a crash dump. Printing the
    enabling command once turns "this happens sometimes" into something a
    user can actually investigate.
    """
    global _crash_hint_shown
    detail = describe_abnormal_exit(code)
    if not detail:
        return
    _logger.warning("[Crash] child process terminated abnormally: %s%s",
                detail, f" — while running `{command}`" if command else "")
    if _crash_hint_shown or os.name != "nt":
        return
    _crash_hint_shown = True
    _logger.warning(
        "[Crash] this is a fault inside the child interpreter (commonly a "
        "native extension such as SDL/pygame during teardown), not a "
        "failure of the code under test — the pipeline retries it rather "
        "than trusting the result. To capture a dump for diagnosis, run "
        "once as Administrator:\n"
        '  reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error '
        'Reporting\LocalDumps" /v DumpFolder /t REG_EXPAND_SZ /d '
        '"%%LOCALAPPDATA%%\CrashDumps" /f\n'
        '  reg add "HKLM\SOFTWARE\Microsoft\Windows\Windows Error '
        'Reporting\LocalDumps" /v DumpType /t REG_DWORD /d 2 /f\n'
        "then reproduce and inspect the .dmp in %LOCALAPPDATA%\CrashDumps.")


def reset_crash_hint() -> None:
    """Test helper: re-arm the once-per-run hint."""
    global _crash_hint_shown
    _crash_hint_shown = False
