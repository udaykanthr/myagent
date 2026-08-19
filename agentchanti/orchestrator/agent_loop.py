"""
Bounded agent micro-loop — tool-calling step execution.

When ``agent_loop: true`` is configured and the provider supports native
tool calling, CODE and TEST steps run through this loop instead of the
generate → review → retry pipeline: the model edits files and runs
commands via :class:`~agentchanti.agent_tools.AgentTools`, observes real
execution output, and self-corrects — capped at a fixed number of turns
so cost stays predictable.

The system prompt below is deliberately byte-identical across all steps
of a run so provider prompt caches and local KV caches get a stable
prefix.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from collections import Counter
from threading import Lock

from ..agent_tools import NO_TESTS_MARKER, AgentTools, _truncate
from ..executor import NO_OUTPUT_MARKER
from ..llm.chat_types import Message, ToolCall
from .gate_integrity import (observe_gate_verdict, platform_equivalent_variants,
                             record_gate_repair)

_logger = logging.getLogger(__name__)


def verify_passed(result: str) -> bool:
    """True when a verify command actually proved something.

    ``exit: success`` alone is not proof. A test runner that collected
    NOTHING exits 0 on CPython below 3.12 (unittest only gained a non-zero
    status for a zero-test run in 3.12), so a step whose discovery quietly
    broke would satisfy its own gate having executed no tests at all —
    the exact false-green this project's verification layers exist to
    prevent, arrived at through the gate rather than around it.

    ``AgentTools.run_command`` already labels that case; the gate just has
    to stop ignoring the label.
    """
    return result.startswith("exit: success") and NO_TESTS_MARKER not in result


def truncate_middle(text: str, limit: int) -> str:
    """Length-cap *text* while keeping BOTH ends.

    Error output puts the conclusion at the end — a Python traceback names
    the exception on its last line — so a plain ``text[:limit]`` slice
    hands the model everything EXCEPT the actual error (observed: a
    recovery loop spent its whole budget on read-only turns hunting for a
    ``NoReverseMatch`` that the head-slice had cut off). Keeps ~1/4 head
    for the command/context and ~3/4 tail for the failure itself.
    """
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    return (text[:head]
            + f"\n... [{len(text) - limit} chars truncated] ...\n"
            + text[-tail:])


# ── Telemetry ─────────────────────────────────────────────────────────
# Per-run loop statistics, consumed by the CLI summary and the A/B
# benchmark harness. Thread-safe: steps in the same wave run in parallel.

_stats_lock = Lock()
_loop_runs: list[dict] = []


def _record_loop_run(step_idx: int, turns_used: int,
                     tool_counts: Counter, outcome: str,
                     recovery: bool) -> None:
    with _stats_lock:
        _loop_runs.append({
            "step_idx": step_idx,
            "turns": turns_used,
            "tool_calls": dict(tool_counts),
            "outcome": outcome,
            "recovery": recovery,
        })
    _logger.info(
        "[AgentLoop] stats: step=%d turns=%d outcome=%s recovery=%s tools=%s",
        step_idx + 1, turns_used, outcome, recovery, dict(tool_counts))


def _finish_unstarted(step_idx: int, attempt_label: str, recovery: bool,
                      summary: str) -> tuple[bool, str]:
    """End a loop that never began, recording it like any other run.

    A step refused before turn 1 is still a loop run as far as telemetry
    is concerned — leaving it out would make the session summary claim
    fewer attempts than the pipeline actually made, and the zero turns
    are precisely the number worth seeing.
    """
    _record_loop_run(step_idx, 0, Counter(), "gate-unrunnable", recovery)
    record_attempt(step_idx, attempt_label, "gate-unrunnable", [], [], summary)
    return False, summary


def get_loop_stats() -> list[dict]:
    """All loop runs recorded in this process (copy)."""
    with _stats_lock:
        return [dict(r) for r in _loop_runs]


def reset_loop_stats() -> None:
    with _stats_lock:
        _loop_runs.clear()


# ── Cross-attempt memory ──────────────────────────────────────────────
# The retry ladder (loop → escalation → recovery → recovery escalation)
# gave each attempt a blank conversation carrying only the previous
# attempt's error string. It never learned what the previous attempt
# actually DID, so every attempt re-derived the same hypotheses and
# re-edited the same files. Observed on a Pygame run: four attempts,
# 8 turns each — 54 turns and 497k tokens — churning src/ghost.py,
# src/game.py and src/player.py with conflicting fixes, none of them
# finding the two-line bug. Record each attempt's edits, commands and
# verdict so the next one starts knowing which doors are already shut.

_JOURNAL_MAX_FILES = 8       # per attempt, in the rendered digest
_JOURNAL_MAX_COMMANDS = 4    # per attempt, most recent first
_JOURNAL_MAX_ATTEMPTS = 4    # most recent attempts shown
_JOURNAL_SUMMARY_CHARS = 240

_journal_lock = Lock()
_attempts: dict[int, list[dict]] = {}


# Journal label for a run on the escalation (stronger) model. Checked by
# escalation_already_failed() to keep the ladder monotonic.
ESCALATION_ATTEMPT_LABEL = "escalation (stronger model)"

# Marker prepended to error_info when the gate was proven to be measuring
# something other than the artifact. Read by the escalation wrapper: a
# stronger model cannot satisfy an instrument that ignores the code, and
# sending it in spends a second full turn budget to learn that again.
GATE_STALLED_MARKER = "[gate-stalled]"


def record_attempt(step_idx: int, label: str, outcome: str,
                   edited: list[str], commands: list[tuple[str, bool]],
                   summary: str) -> None:
    """Append one finished attempt to *step_idx*'s journal."""
    with _journal_lock:
        _attempts.setdefault(step_idx, []).append({
            "label": label,
            "outcome": outcome,
            "edited": list(edited),
            "commands": list(commands),
            "summary": (summary or "").strip(),
        })


def get_attempts(step_idx: int) -> list[dict]:
    with _journal_lock:
        return [dict(a) for a in _attempts.get(step_idx, [])]


def reset_attempt_journal() -> None:
    with _journal_lock:
        _attempts.clear()
    # Gate verdicts are per-run for the same reason the journal is: they
    # are keyed by command text, and a second run in the same process
    # (library API, tests) would otherwise inherit the first run's
    # observations and could declare a fresh gate stalled on sight.
    from .gate_integrity import reset_gate_verdicts
    reset_gate_verdicts()


def attempt_digest(step_idx: int) -> str:
    """Compact record of prior attempts, for the next attempt's context.

    Empty string when nothing has been tried yet, so callers can append
    it unconditionally. Deliberately terse: this rides in the opening
    user message of every retry, and the whole point of the ladder is
    that budget is already scarce by the time it is consulted.
    """
    attempts = get_attempts(step_idx)
    if not attempts:
        return ""

    lines = ["Previous attempts at this step ALL FAILED. What was already "
             "tried (do not simply repeat it):"]
    shown = attempts[-_JOURNAL_MAX_ATTEMPTS:]
    offset = len(attempts) - len(shown)
    for i, att in enumerate(shown, start=offset + 1):
        lines.append(f"  attempt {i} — {att['label']} — {att['outcome']}")

        edited = att["edited"]
        if edited:
            counts = Counter(edited)
            shown_files = counts.most_common(_JOURNAL_MAX_FILES)
            rendered = ", ".join(
                f"{path} (x{n})" if n > 1 else path
                for path, n in shown_files)
            if len(counts) > _JOURNAL_MAX_FILES:
                rendered += f", +{len(counts) - _JOURNAL_MAX_FILES} more"
            lines.append(f"    edited: {rendered}")
        else:
            lines.append("    edited: nothing")

        for cmd, ok in att["commands"][-_JOURNAL_MAX_COMMANDS:]:
            lines.append(f"    ran: {cmd[:160]} -> "
                         f"{'ok' if ok else 'FAILED'}")

        if att["summary"]:
            lines.append("    concluded: "
                         + truncate_middle(att["summary"],
                                           _JOURNAL_SUMMARY_CHARS)
                           .replace("\n", " "))

    # The single most useful signal: the same files keep being rewritten
    # and the gate stays red, so the cause is somewhere nobody has looked.
    all_edited = Counter(f for a in attempts for f in a["edited"])
    repeated = [f for f, n in all_edited.items() if n >= 2]
    if len(attempts) >= 2 and repeated:
        lines.append(
            "  NOTE: " + ", ".join(sorted(repeated)[:6])
            + " have been edited across multiple failed attempts. Re-editing "
              "them the same way will not work — look for the cause "
              "somewhere the previous attempts did not read.")
    return "\n".join(lines)


def loop_stats_summary() -> str | None:
    """One-line human summary for the end-of-run log, or None if unused."""
    runs = get_loop_stats()
    if not runs:
        return None
    outcomes = Counter(r["outcome"] for r in runs)
    total_turns = sum(r["turns"] for r in runs)
    recoveries = sum(1 for r in runs if r["recovery"])
    return (f"[AgentLoop] session: {len(runs)} loop run(s), "
            f"{total_turns} total turns "
            f"(avg {total_turns / len(runs):.1f}), "
            f"{recoveries} recovery run(s), outcomes: {dict(outcomes)}")


# Tools that inspect without changing anything — used to detect loops
# stuck in analysis mode.
_READ_ONLY_TOOLS = frozenset({"read_file", "list_files", "search_code"})

# Read-only intervention thresholds (consecutive inspection-only turns).
# Lowered from 3/4 → 2/3: with the step's target files pre-loaded into the
# opening message, the model rarely needs several read_file turns, so nudge
# it to act one turn sooner and stop paying to re-send inspection output on
# every subsequent turn.
_ACT_NOW_NUDGE_AT = 2
_WITHHOLD_READONLY_AT = 3

# Re-running a command that already failed, without having changed anything
# in between, cannot produce a different answer — but models do it anyway,
# usually by fiddling with the working directory. Observed on a Pygame run:
# turns 4-7 of an 8-turn budget spent re-running one identical failing gate
# from four directories (`cd /d %TEMP%\...`, bare, again, `cd .`), ~38k sent
# tokens, while the actual defect — an unreachable pellet its own maze
# validator was reporting — went untouched.
#
# The system prompt already says "do not retry the same command unchanged".
# These make it a mechanism rather than a request, on the same escalation
# ladder as the read-only intervention above.
_REPEAT_CMD_NUDGE_AT = 1
_WITHHOLD_RUN_COMMAND_AT = 2

# Commands whose failure is an environment or argument problem, never a
# defect in the project's source. The repeat nudge below used to tell every
# repeat "the failure is in the code, not in how the command is invoked —
# edit the source that produced it". For `pip install pygame==2.6.0`
# failing because that version has no wheel for this Python, that is simply
# false: no project source existed yet. The model obeyed anyway, and the
# only source it could invent to satisfy the instruction was a local
# `pygame/` stub package that shadowed the real library.
#
# CODE/TEST wording is unchanged — only a repeat of one of these commands
# takes the environment branch.
_ENV_CMD_RE = re.compile(
    r"\b(?:pip3?|uv|npm|pnpm|yarn|apt|apt-get|brew|choco|winget|"
    r"conda|poetry|gem|cargo|go)\s+(?:install|add|get|i)\b"
    r"|\bpython\s+-m\s+(?:pip|venv)\b"
    r"|\bnpx\s+"
    r"|\b(?:python|py)\s+-m\s+venv\b",
    re.IGNORECASE,
)

# The command shape above only catches installs. The more general signal
# is the ERROR: a command that cannot be found, or an import that cannot
# be resolved, is an environment problem whatever the command was.
#
# Observed on the cmd-recovery benchmark: the model fixed messy.py
# correctly (`python -m ruff check messy.py` -> "All checks passed!") but
# the step's gate was the bare console script `ruff`, which is not on the
# child process's PATH. The gate failed on working code, the loop spent
# 22 turns across 3 attempts and 61.9k tokens hunting PATH, and the
# pipeline reported failure on a finished task. `ruff check messy.py` is
# not an install command, so the shape test alone routed it to "edit the
# source" -- advice that was wrong twice over, since the source was
# already correct.
_ENV_ERROR_RE = re.compile(
    r"is not recognized as an internal or external command"
    r"|command not found"
    r"|No module named"
    r"|executable file not found"
    r"|The system cannot find the (?:path|file) specified"
    r"|is not recognized as the name of a cmdlet",
    re.IGNORECASE,
)

# Wrappers that change how a command's output is delivered but not what it
# does. A model that has been told to stop re-running something tends to
# re-run it *dressed differently* instead, so an exact-string comparison
# sees three distinct commands where there is one. Observed verbatim:
#
#   cd /d %CD% && python -m unittest test_pacman -v 2>&1 | head -100
#   cd /d %CD% && python -m unittest test_pacman -v 2>&1 | head -150
#   python -m unittest test_pacman -v
#
_CD_PREFIX_RE = re.compile(r"^\s*cd\s+(?:/d\s+)?(?:\"[^\"]*\"|\S+)\s*&&\s*",
                           re.IGNORECASE)
_OUTPUT_PIPE_RE = re.compile(
    r"\s*(?:2>&1\s*)?\|\s*(?:head|tail|more)\b[^|]*$", re.IGNORECASE)


def normalize_command(cmd: str) -> str:
    """*cmd* reduced to what it actually runs, for repeat detection.

    Strips ``cd <dir> &&`` prefixes and trailing output-limiting pipes, so
    the same work dressed up differently compares equal. Deliberately
    conservative: only wrappers that cannot change the command's effect are
    removed, because a false match would suppress a legitimate re-run.
    """
    out = (cmd or "").strip()
    while True:
        stripped = _CD_PREFIX_RE.sub("", out, count=1)
        if stripped == out:
            break
        out = stripped.strip()
    prev = None
    while prev != out:
        prev = out
        out = _OUTPUT_PIPE_RE.sub("", out).strip()
    out = re.sub(r"\s*2>&1\s*$", "", out).strip()
    return " ".join(out.split())

# Pre-load caps: keep the injected file bundle from dominating the prompt.
_PRELOAD_MAX_FILES = 6
_PRELOAD_MAX_CHARS = 12_000
# Below this much remaining budget a truncated file is more confusing than
# useful — skip it and let the loop read_file if it actually needs it.
_PRELOAD_MIN_USEFUL_CHARS = 1_500
# A file listing longer than this belongs to a tree the model should
# explore with filtered calls, not carry in full in every turn.
_PRELOAD_MAX_LISTING_CHARS = 3_000

# Stable prefix — keep byte-identical across steps (see module docstring).
# Step-specific data (task, context, platform quirks) belongs in the user
# message, never here.
AGENT_LOOP_SYSTEM_PROMPT = """\
You are a coding agent executing one step of a larger implementation plan.
Work autonomously with the provided tools until the step is complete.

Rules:
- Read a file before editing it; base edits on its actual current content.
- Prefer edit_file for changes to existing files; write_file only for new \
files or full rewrites.
- After making changes, verify them: run the relevant command or test and \
check its output. Do not claim success without evidence.
- Stay within the scope of this step. Do not refactor unrelated code.
- If a command fails, read the error and fix the cause; do not retry the \
same command unchanged.
- When the step is complete and verified, reply with a short plain-text \
summary (no tool calls). If you cannot complete it, explain what is blocking.
"""


def _platform_note() -> str:
    """Environment line for the loop's user message.

    Lives in the user message, not the system prompt, so the system
    prompt stays byte-identical across platforms (see module docstring).
    Windows needs it spelled out: an observed loop burned a turn on
    ``sed -n '1,200p' file`` (exit 1 — no sed on Windows) instead of
    calling read_file.
    """
    import os
    if os.name == "nt":
        return ("Environment: Windows (cmd.exe shell). POSIX text tools "
                "(sed, awk, grep, cat, head, tail, ls) are NOT available "
                "in run_command — use read_file / edit_file / search_code "
                "for file inspection and changes.")
    return ""


def _build_user_message(step_text: str, task: str, language: str | None,
                        context: str, preloaded: str = "",
                        listing: str = "") -> str:
    parts = [f"Overall task: {task}", f"Current step: {step_text}"]
    if language:
        parts.append(f"Project language: {language}")
    note = _platform_note()
    if note:
        parts.append(note)
    if context:
        parts.append(f"Project state:\n{context}")
    if listing:
        parts.append(listing)
    if preloaded:
        parts.append(preloaded)
    return "\n\n".join(parts)


def _preload_listing(tools: AgentTools) -> str:
    """The project's file list, up front.

    Orientation is the loop's reflex first move: measured on a 7-step run,
    every single step opened with ``list_files`` — a whole turn out of
    eight spent learning a layout the harness can hand over for free. The
    answer also rides along in every later turn once fetched, so paying a
    round trip for it buys nothing.

    Kept small and skipped when it would be large: a listing long enough
    to need truncating is one the model should explore with its own
    filtered calls rather than read in full.
    """
    try:
        body = tools._tool_list_files()
    except Exception:
        return ""
    if not body or body.startswith("ERROR"):
        return ""
    if len(body) > _PRELOAD_MAX_LISTING_CHARS:
        return ""
    return ("Project files (already listed for you — do NOT call "
            "list_files again unless you have changed the tree):\n" + body)


# A dependency smaller than this is cheaper to send whole than to outline,
# and the values matter: a constants module IS its assignments.
_OUTLINE_MIN_CHARS = 2_000
# Longest literal kept inline in an outline (e.g. a maze layout is elided).
_OUTLINE_MAX_LITERAL = 80


def _py_signature_outline(source: str) -> str | None:
    """API surface of a Python module: declarations, no bodies.

    A step importing `map.py` needs to know that `Map.is_walkable(x, y)`
    exists, not how it is implemented — and bodies are ~95% of the bytes.
    Measured on a generated game module: 21,364 chars of source reduce to
    1,138 chars of signatures.

    Returns ``None`` when the source will not parse, so the caller falls
    back to sending the real text rather than a half-outline.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None

    lines: list[str] = []

    def render_def(node, indent: str) -> None:
        a = node.args
        names = [x.arg for x in getattr(a, "posonlyargs", [])] +                 [x.arg for x in a.args]
        if a.vararg:
            names.append("*" + a.vararg.arg)
        names += [x.arg for x in a.kwonlyargs]
        if a.kwarg:
            names.append("**" + a.kwarg.arg)
        kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        lines.append(f"{indent}{kw} {node.name}({', '.join(names)})")

    def literal(node) -> str:
        try:
            text = ast.unparse(node)
        except Exception:
            return "..."
        text = " ".join(text.split())
        return text if len(text) <= _OUTLINE_MAX_LITERAL else "..."

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            render_def(node, "")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(literal(b) for b in node.bases)
            lines.append(f"class {node.name}" + (f"({bases})" if bases else ""))
            members = [b for b in node.body
                       if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef,
                                         ast.Assign, ast.AnnAssign))]
            if not members:
                lines.append("    ...")
            for b in members:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    render_def(b, "    ")
                elif isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                    lines.append(f"    {b.target.id} = {literal(b.value) if b.value else '...'}")
                elif isinstance(b, ast.Assign):
                    for t in b.targets:
                        if isinstance(t, ast.Name):
                            lines.append(f"    {t.id} = {literal(b.value)}")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            lines.append(f"{node.target.id} = "
                         + (literal(node.value) if node.value else "..."))
        elif isinstance(node, ast.Assign):
            # Module-level constants are the whole point of a constants
            # module, so keep short values verbatim.
            for t in node.targets:
                if isinstance(t, ast.Name):
                    lines.append(f"{t.id} = {literal(node.value)}")

    return "\n".join(lines) if lines else None


def _missing_required(tools: AgentTools,
                      required: set[str] | None) -> list[str]:
    """Declared target files that are still absent from disk.

    A pure file-existence check — no LLM call, no judgement. Paths that
    escape the project root are ignored rather than raising: a malformed
    `target:` line must not be able to hold a step open forever.
    """
    if not required:
        return []
    missing = []
    for path in sorted(required):
        if not path or not path.strip():
            continue
        try:
            full = tools._resolve(path.strip())
        except ValueError:
            continue
        if not os.path.exists(full):
            missing.append(path.strip())
    return missing


def _preload_target_files(tools: AgentTools,
                          paths: list[str] | None,
                          full_paths: set[str] | None = None) -> str:
    """Read the step's context up front so the loop doesn't burn its first
    turns on read_file round-trips — whose output then rides along,
    re-sent, in every later turn.

    Files in *full_paths* (the step's own targets — what it is about to
    edit) are sent verbatim. Everything else is a DEPENDENCY, and a
    dependency is only consulted for its API: what a step importing
    `map.py` needs is that `Map.is_walkable(x, y)` exists, not how it is
    implemented. Those are reduced to a signature outline, which measured
    8% of source size on a generated game module (21,364 -> 1,750 chars)
    while keeping every constant's value.

    Only existing, non-empty files inside the project root are included;
    files the step will *create* are skipped (nothing to read yet). The
    bundle goes into the opening user message so it lands in the cacheable
    prefix instead of growing the conversation mid-loop.
    """
    if not paths:
        return ""
    # ``None`` means "no dependency information available", so every
    # file is sent whole — the previous behaviour. Outlining is opt-in
    # via an explicit set, so a caller that cannot say which file is
    # being edited never gets an outline of the file it must modify.
    outline_ok = full_paths is not None
    full_set = {str(p).replace("\\", "/") for p in (full_paths or ())}
    seen: set[str] = set()
    blocks: list[str] = []
    total = 0
    for path in paths:
        rel = str(path).replace("\\", "/")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        try:
            full = tools._resolve(path)
        except (ValueError, TypeError):
            continue  # outside the project root — skip
        if not os.path.isfile(full) or os.path.getsize(full) == 0:
            continue
        body = tools._tool_read_file(path)
        if body.startswith("ERROR"):
            continue
        if (outline_ok and rel not in full_set and rel.endswith(".py")
                and len(body) >= _OUTLINE_MIN_CHARS):
            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    outline = _py_signature_outline(fh.read())
            except OSError:
                outline = None
            # An unparseable module falls through to the real text: half an
            # outline is worse than none.
            if outline:
                body = (f"{rel} (API outline — signatures only, bodies "
                        f"omitted; read_file it if you need an "
                        f"implementation)\n{outline}")
        room = _PRELOAD_MAX_CHARS - total
        if len(body) > room:
            # This used to `break`, so ONE oversized file emptied the whole
            # bundle — including every smaller file behind it. Generated
            # modules routinely run 20-40 KB, so in practice nothing was
            # ever preloaded: `[PlanStep] Injected 3 plan-context files`
            # would log while the loop's opening message stayed under
            # 1.1k tokens and the model still spent turn 1 on read_file.
            #
            # Truncate to the remaining budget instead. read_file itself
            # truncates at _MAX_READ_CHARS, so a head-of-file is exactly
            # what the model would have got from the round-trip we are
            # replacing — with the same explicit truncation notice, so it
            # knows to read the rest if it needs it.
            if room < _PRELOAD_MIN_USEFUL_CHARS:
                continue        # no room left worth spending; try the next
            body = _truncate(body, room, f"{rel} preload")
        total += len(body)
        blocks.append(body)
        if len(blocks) >= _PRELOAD_MAX_FILES:
            break
    if not blocks:
        return ""
    return ("Relevant existing files (already read for you — do NOT call "
            "read_file on these again):\n\n" + "\n\n".join(blocks))


def _cd_prefix(cmd: str | None) -> str:
    """The ``cd <dir> && `` prefix of *cmd* when it has one, else ``""``.

    Verify commands for sub-project layouts arrive as ``cd app && npm
    test``; an install that heals that verify must run in the same
    directory or it lands in the wrong package/venv.
    """
    if cmd:
        m = re.match(r"^(cd\s+\S+\s*&&\s*)", cmd)
        if m:
            return m.group(1)
    return ""


def _venv_python(project_root: str) -> str | None:
    """Absolute path to the project's own venv interpreter, or None.

    Dependency installs must land in the interpreter the gates run under
    (``venv\\Scripts\\python.exe``), not whatever bare ``python`` resolves
    to on PATH — installing into the system interpreter leaves the venv
    (and therefore the verify gate) still missing the package, while the
    test command run under system Python "passes" in the wrong place.
    """
    for rel in (os.path.join("venv", "Scripts", "python.exe"),
                os.path.join(".venv", "Scripts", "python.exe"),
                os.path.join("venv", "bin", "python"),
                os.path.join(".venv", "bin", "python")):
        cand = os.path.join(project_root, rel)
        if os.path.isfile(cand):
            return cand
    return None


def _flagless_tokens(cmd: str) -> list[str]:
    """Lower-cased non-flag tokens of a shell command, `&&` flattened."""
    toks: list[str] = []
    for seg in (cmd or "").split("&&"):
        for t in seg.split():
            t = t.strip("\"'")
            if t and not t.startswith("-"):
                toks.append(t.lower())
    return toks


def commands_equivalent_modulo_flags(candidate: str | None,
                                     original: str | None) -> bool:
    """True when *candidate* is *original* with only flag tokens changed.

    Same executables, same non-flag arguments, in the same order —
    differing only in ``-``/``--`` tokens (e.g. ``pip install pygame``
    vs ``pip install --yes pygame``). Identical strings return False:
    the gate already ran that exact command itself, so an earlier
    success proves nothing new.
    """
    if not candidate or not original or candidate.strip() == original.strip():
        return False
    a, b = _flagless_tokens(candidate), _flagless_tokens(original)
    return bool(a) and a == b


def _npm_package_dir(project_root: str,
                     planned_files: list[str] | None) -> str:
    """Directory of the ``package.json`` that owns *planned_files*, or ``""``.

    Node resolves dependencies from the manifest's own directory, so a
    step writing ``backend/routes/auth.js`` needs its packages in
    ``backend/``, whatever the repo root happens to contain. Walks up from
    each file to the nearest manifest and requires the answer to be
    unanimous — a step spanning two packages has no single right target,
    and the caller's existing behaviour is the safer answer there.

    Returns ``""`` for the repo root itself, so callers can treat "root"
    and "unknown" identically: both mean "add no --prefix".
    """
    dirs: set[str] = set()
    for f in planned_files or []:
        if not f:
            continue
        parts = f.replace("\\", "/").strip("/").split("/")[:-1]
        while True:
            rel = "/".join(parts)
            manifest = os.path.join(project_root, rel, "package.json")
            if os.path.isfile(manifest):
                dirs.add(rel)
                break
            if not parts:
                break
            parts = parts[:-1]
    if len(dirs) != 1:
        return ""
    return dirs.pop()


def _run_install(tools: AgentTools, install_cmd: str) -> bool:
    _logger.info("[AgentLoop] env self-heal: %s", install_cmd)
    result = tools.execute(ToolCall(name="run_command",
                                    arguments={"command": install_cmd},
                                    id="env-heal"))
    return result.startswith("exit: success")


def attempt_env_self_heal(tools: AgentTools, verify_output: str,
                          language: str | None, healed: set[str],
                          verify_cmd: str | None = None,
                          planned_files: list[str] | None = None,
                          target_files=None) -> bool:
    """Install a missing third-party dependency named in failing output.

    ``No module named X`` / ``Cannot find package 'X'`` are environment
    problems that editing project files can never fix — the loop that hit
    them burned its whole turn budget while the real fix was one pip
    install (mirrors the BulkTest self-heal, which the agent-loop path
    bypasses). *healed* accumulates already-attempted names so a dep that
    keeps failing is only installed once per loop.

    Returns True when an install succeeded and the caller should re-run
    verification.
    """
    lang = (language or "python").lower()
    from .pipeline import _missing_js_packages, _missing_third_party_module
    if lang in ("javascript", "typescript"):
        pkgs = [p for p in _missing_js_packages(verify_output)
                if p not in healed]
        if not pkgs:
            return False
        healed.update(pkgs)
        install_cmd = "npm install -D " + " ".join(pkgs)
        # Install beside the manifest that owns the failing step, not at
        # the repo root. `_cd_prefix` below can only see a `cd` the GATE
        # happens to carry, and a correct root-relative backend gate —
        # `node -e "require('./backend/x')"` — carries none, so the heal
        # landed at the top level. Measured 2026-08-19: a backend step
        # missing `jsonwebtoken` and `supertest` installed both into a
        # repo-root `node_modules`, leaving a `package.json` belonging to
        # no project; the app only worked because Node walks up, and
        # shipping `backend/` alone would break.
        # The step's DECLARED TARGETS, not `planned_files` — the latter is
        # the loop's reading list (`_loop_preload_paths` adds every file
        # the step imports), so a backend step that reads a frontend
        # module spans two packages, the unanimity rule declines, and the
        # install falls back to the repo root. Measured 2026-08-19 run 13:
        # `jsonwebtoken` correctly went to `backend/` for one step and to
        # the root for the next, from the same loop, minutes apart.
        _pkg_dir = _npm_package_dir(
            getattr(tools, "project_root", "."),
            list(target_files or ()) or planned_files)
        if _pkg_dir:
            install_cmd = (f"npm --prefix {_pkg_dir} install -D "
                           + " ".join(pkgs))
            _logger.info("[AgentLoop] env self-heal targets %s/ (the manifest "
                         "owning this step), not the repo root", _pkg_dir)
            return _run_install(tools, install_cmd)
    else:
        memory = getattr(tools, "_memory", None)
        project_files = list(memory.all_files()) if memory is not None else []
        # Files the step is about to create count as local too. A TEST step
        # whose gate is `python -m unittest -v test_main` fails with
        # "No module named test_main" before the file exists, and memory
        # cannot know better — so the heal tried `pip install test_main`,
        # a pointless call and the exact dependency-confusion hazard
        # _missing_third_party_module exists to prevent.
        project_files += [p for p in (planned_files or []) if p]
        mod = _missing_third_party_module(verify_output, project_files)
        if not mod or mod in healed:
            return False
        healed.add(mod)
        # Install into the venv the gates use, not bare `python`. The
        # absolute interpreter path also survives the `cd {sub} &&` prefix
        # added below.
        py = _venv_python(getattr(tools, "project_root", ".")) or "python"
        py_tok = f'"{py}"' if py != "python" else py
        install_cmd = f"{py_tok} -m pip install {mod}"
    install_cmd = _cd_prefix(verify_cmd) + install_cmd
    return _run_install(tools, install_cmd)


def run_agent_loop(
    llm_client,
    tools: AgentTools,
    step_text: str,
    task: str,
    display=None,
    step_idx: int = 0,
    language: str | None = None,
    max_turns: int = 8,
    verify_cmd: str | None = None,
    context: str = "",
    preload_files: list[str] | None = None,
    preload_full_paths: set[str] | None = None,
    required_files: set[str] | None = None,
    _recovery: bool = False,
    attempt_label: str = "first attempt",
) -> tuple[bool, str]:
    """Run one step as a capped tool-calling loop.

    Exit conditions, in order:
    - Model stops calling tools AND ``verify_cmd`` (when given) passes
      → ``(True, summary)``. A failing ``verify_cmd`` is fed back to the
      model as a new user message and the loop continues.
    - Model stops calling tools without having used any tool at all
      → ``(False, ...)`` — a step that changed nothing cannot have
      succeeded.
    - ``max_turns`` exhausted → ``(False, ...)``.

    *required_files* are the step's plan-declared targets. While any of
    them is missing the loop declines to exit EARLY on a green gate —
    only that exit, so the guard can spend turns the step already had but
    can never turn a passing step into a failing one.

    Returns the same ``(success, error_info)`` contract as the step
    handlers in ``step_handlers.py``.
    """
    # A gate the shell cannot execute is decidable before a single token
    # is spent, and it is the same conclusion the stall breaker reaches
    # after three verdicts — with the turns already paid for. Measured
    # 2026-08-18: the plan-time check named this exact gate as
    # unrunnable at 00:38:48, nothing consumed the warning, and the run
    # then spent 180k tokens and 14 turns proving it right. An advisory
    # nobody acts on is indistinguishable from silence.
    #
    # Translating first, refusing second: an unrunnable gate whose other
    # dialect IS runnable is a defective instrument with a working
    # equivalent, and using it is strictly better than failing the step.
    # Only when no runnable reading exists is the step ended here.
    if verify_cmd:
        # Imported here: plan_step is a heavy module and this is the only
        # place in the loop that needs it.
        from .plan_step import unrunnable_gate_reason
        _unrunnable = unrunnable_gate_reason(verify_cmd)
        if _unrunnable:
            _runnable = next(
                (v for _r, v in platform_equivalent_variants(verify_cmd)
                 if not unrunnable_gate_reason(v)), None)
            if _runnable:
                _logger.warning(
                    "[GateIntegrity] step %d: the gate cannot run as "
                    "written (%s) — using the equivalent this platform's "
                    "shell can execute:\n  original: %s\n  using: %s",
                    step_idx + 1, _unrunnable, verify_cmd, _runnable)
                # The ledger must learn it too, or the monotonic recheck
                # re-runs the original, fails exactly as it always did,
                # and reads that as a regression.
                record_gate_repair(verify_cmd, _runnable,
                                   "unrunnable-on-this-platform")
                verify_cmd = _runnable
            else:
                _logger.error(
                    "[GateIntegrity] step %d: refusing to start — the "
                    "gate cannot run on this platform in any reading, so "
                    "no work the step does could satisfy it: %s\n  gate: %s",
                    step_idx + 1, _unrunnable, verify_cmd)
                return _finish_unstarted(
                    step_idx, attempt_label, _recovery,
                    f"{GATE_STALLED_MARKER} the step was never started: its "
                    f"verify command cannot run on this platform, so no "
                    f"edit could change the verdict. {_unrunnable}\n"
                    f"Gate: {verify_cmd}")

    preloaded = _preload_target_files(tools, preload_files,
                                      preload_full_paths)
    listing = _preload_listing(tools)
    messages = [
        Message(role="system", content=AGENT_LOOP_SYSTEM_PROMPT),
        Message(role="user",
                content=_build_user_message(step_text, task, language,
                                            context, preloaded, listing)),
    ]
    definitions = tools.definitions()
    action_definitions = [d for d in definitions
                          if d.name not in _READ_ONLY_TOOLS]
    editing_definitions = [d for d in definitions if d.name != "run_command"]
    any_tool_used = False
    tool_counts: Counter = Counter()
    read_only_streak = 0
    repeat_cmd_streak = 0
    # Commands that have failed and whose cause nobody has touched since.
    # An edit clears it: re-running after a change is verification, not a
    # rut, and must not be penalised.
    #
    # Seeded from earlier attempts at THIS step. The streak used to reset
    # with every attempt, so a command re-run once per attempt across the
    # loop -> escalation -> recovery ladder never tripped the nudge at all
    # — observed on cmd-recovery, where `ruff check messy.py` failed
    # identically in all three attempts and the intervention never fired.
    # An edit inside this attempt still clears the seed, so a genuine
    # fix-then-verify sequence is not penalised.
    failed_since_edit: set[str] = {
        _n for _a in get_attempts(step_idx)
        for _c, _c_ok in _a.get("commands", [])
        if not _c_ok and (_n := normalize_command(_c))
    }
    healed: set[str] = set()

    # Last run_command the model executed that exited 0 — evidence for
    # the unpassable-gate escape below (recovery loops only).
    last_ok_cmd: str | None = None

    # Cross-attempt memory: what this attempt actually touched and ran.
    edited_files: list[str] = []
    commands_run: list[tuple[str, bool]] = []

    # Set when an edit lands, cleared when the gate runs. Without it the
    # gate would re-run after every inspection turn and re-prove the same
    # red result — and the final verification would re-run a suite the
    # early gate just ran against identical files.
    _dirty_since_gate = False
    _gate_cache: str | None = None

    # Set once the gate is proven to be measuring something other than
    # the artifact (see gate_integrity.observe_gate_verdict). Ends the
    # loop and suppresses the escalation, which would otherwise spend a
    # stronger model's whole turn budget on the same broken instrument.
    _stalled_reason: str | None = None

    def _finish(outcome: str, turns: int,
                result: tuple[bool, str]) -> tuple[bool, str]:
        _record_loop_run(step_idx, turns, tool_counts, outcome, _recovery)
        record_attempt(step_idx, attempt_label, outcome,
                       edited_files, commands_run, result[1])
        return result

    def _variant_gate_ok() -> bool:
        """Escape hatch for gates no correct work can pass.

        A failed CMD step's recovery gate is the original command
        verbatim; when that command is malformed (observed:
        `pip install --yes pygame` — pip has no --yes), the gate stays
        red forever even though the loop already ran the corrected
        command successfully. Accept a same-command-modulo-flags success
        the loop itself produced. Recovery loops only — CODE/TEST gates
        keep strict semantics.
        """
        return (_recovery and verify_cmd is not None
                and commands_equivalent_modulo_flags(last_ok_cmd, verify_cmd))

    from .wave_snapshots import (describe_abnormal_exit, is_abnormal_exit,
                                 log_crash_diagnostics)

    def _artifact_digest() -> str:
        """A fingerprint of the files this attempt has written so far.

        Cheap and approximate on purpose: it only has to distinguish
        "the code changed between these two verdicts" from "it did not".
        Unreadable files hash as their name, so a delete still moves the
        digest.
        """
        import hashlib

        root = getattr(tools, "project_root", "") or ""
        h = hashlib.sha1()
        for path in sorted(set(edited_files)):
            h.update(path.encode("utf-8", "replace"))
            try:
                with open(os.path.join(root, path), "rb") as fh:
                    h.update(fh.read())
            except OSError:
                pass
        return h.hexdigest()

    def _verify_once(cmd: str | None = None) -> str:
        # Last line of defence, and the only one covering gates the plan
        # never declared: `verify_cmd_for_language`, a failed CMD step's
        # recovery gate (which is the failed command verbatim), and the
        # platform variants built from either. A gate here runs
        # unattended and repeatedly, so it is refused outright rather
        # than trimmed — trimming belongs to plan time, where the planner
        # can still be asked for a better command. The refusal string
        # does not start with `exit: success`, so `verify_passed` reads
        # it as a failure and the step cannot exit green on it.
        from .gate_safety import destructive_reason
        _cmd = cmd or verify_cmd
        _unsafe = destructive_reason(_cmd or "")
        if _unsafe:
            _logger.error("[GateSafety] refusing to run this gate: %s",
                          _unsafe)
            return (f"gate REFUSED — it runs a destructive command: "
                    f"{_unsafe}. A verify command must only observe; it is "
                    f"re-run after every later wave, so its side effects "
                    f"happen repeatedly.")
        return tools.execute_all([_verify_call(_cmd)])[0].content

    def _try_platform_variants(result: str) -> str:
        """Re-run the gate under the other shell dialect's reading of it.

        A gate is an instrument, and an instrument can be broken. When the
        SAME command text means something different on POSIX than it does
        under cmd.exe, a red verdict may be measuring the platform rather
        than the code — observed on a run where the correct edit landed on
        turn 1 and three separate recovery mechanisms then spent 24 turns
        failing against a regex that could not match anything on Windows.

        Only a variant that PASSES is believed: it proves the original was
        unsatisfiable, because the two forms are the same text and differ
        only in an escaping step one shell performs and the other does not.
        A variant that also fails proves nothing and is discarded, leaving
        the original verdict untouched.
        """
        nonlocal verify_cmd
        for reason, variant in platform_equivalent_variants(verify_cmd):
            variant_result = _verify_once(variant)
            if not verify_passed(variant_result):
                continue
            record_gate_repair(verify_cmd, variant, reason)
            _logger.warning(
                "[AgentLoop] step %d: the gate FAILED as written but PASSES "
                "under the %s reading of the identical command — treating "
                "the gate, not the code, as the defect",
                step_idx + 1, reason)
            # Adopt the repaired form for the rest of this loop so the
            # early gate and exit verification stop re-running a command
            # already proven incapable of passing.
            verify_cmd = variant
            return variant_result
        return result

    def _run_verify() -> str:
        result = _verify_once()
        # A crashed verifier produced no verdict, so failing the step on it
        # is a category error — the same one already guarded in
        # GateLedger.recheck and the BulkTest plan gate, and missing here.
        # Observed: three consecutive attempts where `python -m unittest -v`
        # PASSED seconds earlier and then the exit verification
        # access-violated (0xC0000005), so a run whose tests were green was
        # reported as failed. The model itself spotted the flakiness and
        # ran the suite ten times in a loop to prove it.
        if not result.startswith("exit: success"):
            code = getattr(getattr(tools, "_executor", None),
                           "last_exit_code", None)
            if is_abnormal_exit(code):
                _logger.warning(
                    "[AgentLoop] step %d: verification process terminated "
                    "abnormally (%s) — retrying once before believing it",
                    step_idx + 1, describe_abnormal_exit(code) or code)
                log_crash_diagnostics(code, verify_cmd)
                result = _verify_once()
        while (not result.startswith("exit: success")
               and attempt_env_self_heal(tools, result, language, healed,
                                         verify_cmd,
                                         planned_files=preload_files,
                                         target_files=required_files)):
            result = _verify_once()
        # Last, and only on a still-red verdict: the env self-heal above
        # can turn a red gate green by fixing the environment, and asking
        # "is the gate itself broken?" before that would spend a
        # subprocess on a question already about to be answered.
        if not verify_passed(result):
            result = _try_platform_variants(result)
        # Observed only after every repair path has had its turn, so a
        # gate that self-heals or passes under another dialect never
        # counts as stalled. The digest is what makes the observation
        # evidence: it says the code under the gate really did change.
        nonlocal _stalled_reason
        # A step that has not finished writing its declared targets is
        # EXPECTED to fail its gate, identically, while the digest moves
        # with every file it writes — which is the stall signature exactly.
        # Measured 2026-08-19 run 19: a step declaring five components was
        # cut short at `turns=5, write_file: 3`; its gate reads two of
        # them, so until both existed the failure was a constant ENOENT
        # naming the missing FILE rather than anything about the code. It
        # was declared unmeasurable three times in one run, and a recovery
        # loop then redid the work and passed the same gate. This is the
        # false positive the check's own bias statement warns about — the
        # kind that suppresses real work — so the observation is withheld
        # until the step has produced everything it promised.
        if _stalled_reason is None and not _missing_required(
                tools, required_files):
            _stalled_reason = observe_gate_verdict(
                verify_cmd, result, _artifact_digest())
        return result

    def _gate_result() -> str:
        """The gate's verdict, re-running it only when files have changed.

        The early gate below and the exit verification ask the same
        question; when nothing was edited in between, the answer cannot
        have changed and a second suite run is pure wall-clock.
        """
        nonlocal _dirty_since_gate, _gate_cache
        if _gate_cache is not None and not _dirty_since_gate:
            return _gate_cache
        _gate_cache = _run_verify()
        _dirty_since_gate = False
        return _gate_cache

    for turn in range(1, max_turns + 1):
        # Final turn: withhold tools so the model must produce a text
        # summary instead of burning the last turn on another tool call.
        final_turn = turn == max_turns
        if final_turn and any_tool_used:
            messages.append(Message(role="user", content=(
                "Turn budget exhausted — tools are no longer available. "
                "Reply now with a short summary of what you completed and "
                "whether it was verified.")))
        if final_turn:
            tools_for_turn = None
        elif read_only_streak >= _WITHHOLD_READONLY_AT:
            # The act-now nudge was ignored — withhold inspection tools so
            # the only moves left are the ones that change something.
            tools_for_turn = action_definitions
        elif repeat_cmd_streak >= _WITHHOLD_RUN_COMMAND_AT:
            # Stuck re-running a failing command. Take the command away and
            # the only remaining moves edit the code — which is where the
            # defect is. The harness runs the gate itself after every turn,
            # so nothing is lost by the model not running it.
            tools_for_turn = editing_definitions
        else:
            tools_for_turn = definitions
        response = llm_client.chat(messages, tools=tools_for_turn)

        if response.has_tool_calls:
            any_tool_used = True
            tool_counts.update(tc.name for tc in response.tool_calls)
            names = ", ".join(tc.name for tc in response.tool_calls)
            _logger.info("[AgentLoop] step %d turn %d/%d: %s",
                         step_idx + 1, turn, max_turns, names)
            if display is not None:
                display.step_info(step_idx,
                                  f"Agent loop {turn}/{max_turns}: {names}")
            messages.append(response.to_message())
            # Enforce the narrowed offer. Withholding a tool from the list
            # is only a request; a model that ignores it kept getting the
            # tool executed anyway, so the intervention did nothing at all.
            # Passing the offer through makes the refusal real.
            #
            # The final turn is deliberately NOT enforced. Tools are absent
            # from that offer to prod a text summary, but a model that edits
            # anyway may just have fixed the step — the gate runs after this
            # block, so that late write can still turn the step green.
            # Refusing it would throw away a working fix to enforce a
            # formatting preference.
            _allowed = ({d.name for d in tools_for_turn}
                        if tools_for_turn is not None else None)
            _tool_msgs = tools.execute_all(response.tool_calls,
                                           allowed=_allowed)
            messages.extend(_tool_msgs)
            _repeated_cmd: str | None = None
            _repeated_out: str = ""
            # A green gate the model ran itself, as the last thing it did
            # this turn. Without this the early gate below re-ran the very
            # command the model had just run seconds earlier — observed as
            # back-to-back identical `python -m unittest -v` lines.
            _self_gate: str | None = None
            _last_call = response.tool_calls[-1] if response.tool_calls else None
            for _tc, _tm in zip(response.tool_calls, _tool_msgs):
                _content = _tm.content or ""
                if _tc.name == "run_command":
                    _cmd = _tc.arguments.get("command", "")
                    _ok = _content.startswith("exit: success")
                    if _cmd:
                        if not _ok:
                            _norm = normalize_command(_cmd)
                            if _norm and _norm in failed_since_edit:
                                _repeated_cmd = _cmd
                                _repeated_out = _content
                            failed_since_edit.add(_norm)
                        commands_run.append((_cmd, _ok))
                    if _ok:
                        last_ok_cmd = _cmd
                    # A command changes the world too — it can install a
                    # dependency, generate a file or BE the fix. Treating
                    # only edits as invalidating let a stale red verdict
                    # outlive the command that fixed it.
                    _dirty_since_gate = True
                    # Only a PASS is reusable: _run_verify self-heals the
                    # environment on failure, and a cached red result would
                    # skip that repair entirely.
                    if (verify_cmd and _tc is _last_call and _ok
                            and _cmd.strip() == verify_cmd.strip()):
                        _self_gate = _content
                elif _tc.name in ("write_file", "edit_file"):
                    # Errors come back as strings rather than raising, so
                    # only count edits that actually landed.
                    _path = (_tc.arguments.get("path") or "").replace("\\", "/")
                    if _path and not _content.lower().startswith("error"):
                        edited_files.append(_path)
                        _dirty_since_gate = True
                        # The world changed — every prior failure is now
                        # worth re-testing, so nothing counts as a repeat.
                        failed_since_edit.clear()
            if _self_gate is not None:
                _gate_cache = _self_gate
                _dirty_since_gate = False

            # Read-only intervention: some models settle into inspecting
            # file after file without ever acting (observed: whole 8-turn
            # budgets of read_file, twice in one run, even with the
            # failing output already in context). After a couple of
            # consecutive inspection-only turns, tell the model to act.
            if all(tc.name in _READ_ONLY_TOOLS for tc in response.tool_calls):
                read_only_streak += 1
            else:
                read_only_streak = 0
            if read_only_streak == _ACT_NOW_NUDGE_AT and turn <= max_turns - 2:
                _logger.info("[AgentLoop] step %d: %d read-only turns — "
                             "injecting act-now nudge", step_idx + 1,
                             _ACT_NOW_NUDGE_AT)
                messages.append(Message(role="user", content=(
                    f"You have spent {_ACT_NOW_NUDGE_AT} consecutive turns "
                    "only inspecting files. You have enough context — ACT "
                    "now: apply the fix with edit_file or write_file, or run "
                    "the command that completes this step. "
                    f"{max_turns - turn} turn(s) remain.")))
            elif read_only_streak == _WITHHOLD_READONLY_AT:
                # Nudge ignored (observed twice in one run: the model went
                # straight back to read_file). Escalate: from the next
                # turn only acting tools are offered.
                _logger.info("[AgentLoop] step %d: nudge ignored — "
                             "withholding read-only tools", step_idx + 1)
                messages.append(Message(role="user", content=(
                    "Inspection tools are now disabled. Only write_file, "
                    "edit_file and run_command are available — apply the "
                    "fix or run the completing command now.")))

            # Repeated-command intervention: same ladder, different rut.
            # A command that failed before and failed again with nothing
            # edited in between has told the model everything it is going
            # to tell it.
            if _repeated_cmd:
                repeat_cmd_streak += 1
            else:
                repeat_cmd_streak = 0
            if repeat_cmd_streak == _REPEAT_CMD_NUDGE_AT:
                _is_env_cmd = bool(_ENV_CMD_RE.search(_repeated_cmd or "")
                                   or _ENV_ERROR_RE.search(_repeated_out))
                _is_silent_failure = NO_OUTPUT_MARKER in _repeated_out
                _logger.info("[AgentLoop] step %d: re-ran a failing command "
                             "unchanged — injecting fix-the-cause nudge%s",
                             step_idx + 1,
                             " (environment variant)" if _is_env_cmd
                             else " (silent-failure variant)"
                             if _is_silent_failure else "")
                if _is_env_cmd:
                    messages.append(Message(role="user", content=(
                        f"You already ran `{_repeated_cmd[:160]}` earlier in "
                        "this step and it failed the same way. Re-running it "
                        "cannot change the result. This is an environment or "
                        "argument problem, not a defect in the project's "
                        "source — read the error above and change the "
                        "command itself: drop or change a pinned version "
                        "that has no build for this platform or Python, "
                        "target a different package name, or — if the tool "
                        "is installed but not on PATH — invoke it through "
                        "the interpreter instead (`python -m <tool> ...`) "
                        "rather than by its bare name. Do NOT write a local "
                        "module or package that stands in for the "
                        "dependency: it would shadow the real one, and the "
                        "step would look finished while the functionality "
                        "stayed missing. If the dependency genuinely cannot "
                        "be installed here, say so plainly instead. "
                        f"{max_turns - turn} turn(s) remain.")))
                elif _is_silent_failure:
                    # A silent failure is a different rut. "Read the error
                    # and fix the source" is unactionable when there IS no
                    # error text, and its certainty is sometimes simply
                    # wrong: observed on a `node -e` gate whose regex was
                    # mis-escaped for this platform, where the source was
                    # already correct and every run printed nothing. The
                    # model burned its turns guessing, then eventually
                    # printed the conditions one by one and found them all
                    # true — the right move, reached far too late.
                    #
                    # Ask for that evidence directly. Note this cannot be
                    # used to dodge the step: the acceptance gate is run by
                    # the harness, not by the model, so a model that talks
                    # itself out of the work still does not finish.
                    messages.append(Message(role="user", content=(
                        f"You already ran `{_repeated_cmd[:160]}` earlier in "
                        "this step and it failed the same way, producing NO "
                        "output — so it has not told you what is wrong, and "
                        "running it again cannot. Make the failure "
                        "observable before assuming the source is at fault: "
                        "run a version that prints each condition it checks "
                        "separately, so you can see which one is actually "
                        "false. If every condition it asserts turns out to "
                        "be true, then the check itself is malformed rather "
                        "than the code — say so explicitly and quote the "
                        "output that shows it. Otherwise fix the source. "
                        f"{max_turns - turn} turn(s) remain.")))
                else:
                    messages.append(Message(role="user", content=(
                        f"You already ran `{_repeated_cmd[:160]}` earlier in "
                        "this step and it failed the same way. Re-running it, or "
                        "running it from a different directory, cannot change "
                        "the result — the failure is in the code, not in how the "
                        "command is invoked. Read the error above and edit the "
                        f"source that produced it. {max_turns - turn} turn(s) "
                        "remain.")))
            elif repeat_cmd_streak == _WITHHOLD_RUN_COMMAND_AT:
                _logger.info("[AgentLoop] step %d: still re-running a failing "
                             "command — withholding run_command",
                             step_idx + 1)
                messages.append(Message(role="user", content=(
                    "run_command is now disabled. Fix the cause with "
                    "edit_file or write_file — the step's command is run for "
                    "you after every turn, so you do not need to run it.")))

            # Early gate: the loop already treats "the gate passes" as the
            # definition of a completed step — when the model claims done
            # AND when the turn budget runs out. It just never asked until
            # one of those happened, so a step that was finished at turn 4
            # kept probing to turn 8. Measured on a Pac-Man run: 53 turns
            # across 7 loop runs, avg 7.6 of a max 8, and the late turns
            # are the dear ones (~27k prompt tokens each vs ~2k at the
            # start) because the whole conversation is resent every turn.
            #
            # Asking early costs one subprocess and no tokens at all.
            if (verify_cmd and edited_files and not final_turn
                    and (_dirty_since_gate or _self_gate is not None)):
                _early = _gate_result()
                if not verify_passed(_early) and _stalled_reason:
                    return _finish("gate-stalled", turn, (False, (
                        f"{GATE_STALLED_MARKER} {_stalled_reason}\n\n"
                        f"{truncate_middle(_early, 1000)}")))
                if verify_passed(_early):
                    _missing = _missing_required(tools, required_files)
                    if _missing:
                        # The gate can go green before the step is done. A
                        # plan declaring `target: tests/__init__.py,
                        # tests/test_map.py, tests/test_movement_invariants.py`
                        # had the first two written; `python -m unittest -v`
                        # passed on those alone, the loop exited at turn 3 of
                        # 8, and the run was reported complete with the
                        # adversarial test file — the whole point of the task
                        # — never created. Declining to stop early is safe:
                        # it can only spend turns the step already had, never
                        # turn a passing step into a failure.
                        _logger.info(
                            "[AgentLoop] step %d: gate %r passes on turn "
                            "%d/%d but %d declared target(s) do not exist "
                            "yet (%s) — not exiting early",
                            step_idx + 1, verify_cmd, turn, max_turns,
                            len(_missing), ", ".join(_missing))
                        messages.append(Message(role="user", content=(
                            f"`{verify_cmd}` passes, but the step declares "
                            "target file(s) that do not exist yet: "
                            + ", ".join(_missing)
                            + ". A green gate is not the same as a finished "
                            "step — the gate cannot see a file you have not "
                            "written. Create them now with the content the "
                            "step describes. "
                            f"{max_turns - turn} turn(s) remain.")))
                        continue
                    _logger.info(
                        "[AgentLoop] step %d verified early on turn %d/%d "
                        "— gate %r passes, ending the loop instead of "
                        "spending the remaining turn(s)",
                        step_idx + 1, turn, max_turns, verify_cmd)
                    return _finish("verified-early", turn, (True, (
                        f"Step verified complete on turn {turn}: "
                        f"{verify_cmd} passes.")))
            continue

        # Model stopped calling tools — it believes the step is done.
        summary = response.text.strip()
        if not any_tool_used:
            _logger.warning("[AgentLoop] step %d: model finished without "
                            "using any tool", step_idx + 1)
            return _finish("no-tools", turn, (False, (
                "Agent loop made no tool calls — no files were changed and "
                f"no commands were run. Model said: {summary[:500]}")))

        if verify_cmd:
            if display is not None:
                display.step_info(step_idx, f"Verifying: {verify_cmd}")
            result = _gate_result()
            if verify_passed(result):
                _logger.info("[AgentLoop] step %d verified in %d turn(s)",
                             step_idx + 1, turn)
                return _finish("verified", turn, (True, summary))
            if _variant_gate_ok():
                # Tell the LEDGER too, or the step passes and the run
                # still dies: the monotonic recheck re-runs the declared
                # command, the malformed original fails exactly as it
                # always did, and that reads as a regression. Observed —
                # a gate with `&& npm run build` written INSIDE the
                # `node -e "..."` string (a JS syntax error, unpassable by
                # any code) was correctly recovered via the variant, then
                # rechecked in its original form, declared a REGRESSION,
                # and the whole wave was rolled back over working code.
                record_gate_repair(verify_cmd, last_ok_cmd, "flag-variant")
                _logger.info(
                    "[AgentLoop] step %d: gate command fails but the loop "
                    "ran a flag-variant of it successfully (%r) — "
                    "accepting the variant as the gate", step_idx + 1,
                    last_ok_cmd)
                return _finish("verified-variant", turn, (True, summary))
            # After the variant escape, so a gate that merely needed the
            # other dialect is never called stalled.
            if _stalled_reason:
                return _finish("gate-stalled", turn, (False, (
                    f"{GATE_STALLED_MARKER} {_stalled_reason}\n\n"
                    f"{truncate_middle(result, 1000)}")))
            if final_turn:
                return _finish("verify-failed", turn, (False, (
                    f"Verification still failing after {max_turns} turns:\n"
                    f"{truncate_middle(result, 1000)}")))
            # A zero-test run may have exited 0, so "failed" would be a
            # confusing thing to tell the model — name the real problem.
            _no_tests = NO_TESTS_MARKER in result
            _logger.info(
                "[AgentLoop] step %d: %s on turn %d — feeding back",
                step_idx + 1,
                "verify collected no tests" if _no_tests
                else "verification failed", turn)
            # The model is about to get another go. Its next "done" claim
            # must be re-proved against a fresh run: the cache exists only
            # to stop the early gate and the exit check from running the
            # same suite twice over identical files, not to pin a red
            # verdict across a repair cycle.
            _gate_cache = None
            messages.append(response.to_message())
            messages.append(Message(role="user", content=(
                (f"Verification command COLLECTED NO TESTS:\n{verify_cmd}\n\n"
                 f"{result}\n\nIt may have exited 0, but nothing ran, so it "
                 "proves nothing. Make the tests discoverable and verify "
                 "again."
                 ) if _no_tests else
                (f"Verification command failed:\n{verify_cmd}\n\n{result}\n\n"
                 "The step is not complete. Fix the problem and verify "
                 "again."))))
            continue

        _logger.info("[AgentLoop] step %d finished in %d turn(s)",
                     step_idx + 1, turn)
        return _finish("done", turn, (True, summary))

    # Exhausted without a final text answer (e.g. text-mode model ignored
    # the no-tools instruction). The work may still be done — let the
    # deterministic check have the last word.
    if verify_cmd and any_tool_used:
        result = _gate_result()
        if verify_passed(result):
            _logger.info("[AgentLoop] step %d: turns exhausted but "
                         "verification passes — accepting", step_idx + 1)
            return _finish("exhausted-verified", max_turns, (True, (
                "Step verified complete (turn budget exhausted "
                "before the model summarized).")))
        if _variant_gate_ok():
            _logger.info(
                "[AgentLoop] step %d: turns exhausted, gate command fails "
                "but a successful flag-variant ran (%r) — accepting",
                step_idx + 1, last_ok_cmd)
            return _finish("exhausted-verified-variant", max_turns, (True, (
                "Step complete: the gate command is malformed but the loop "
                f"ran an equivalent command successfully: {last_ok_cmd}")))

    return _finish("exhausted", max_turns, (False, (
        f"Agent loop exhausted {max_turns} turns without completing the "
        f"step: {step_text[:200]}")))


def run_agent_loop_with_escalation(llm_client, tools: AgentTools,
                                   step_text: str, task: str,
                                   escalation_client=None,
                                   **kw) -> tuple[bool, str]:
    """Run the loop; on failure, retry once with a stronger model.

    A weak model exhausting its turns is a capability floor, not a step
    the pipeline must fail on (observed: 8 read-only turns while the
    one-line fix sat in context). When ``models: escalation:`` names a
    stronger model, that client gets one fresh loop with the failed
    attempt's error in context. No escalation configured → identical to
    :func:`run_agent_loop`.
    """
    success, info = run_agent_loop(llm_client, tools, step_text, task, **kw)
    if success or escalation_client is None or escalation_client is llm_client:
        return success, info
    if not getattr(escalation_client, "supports_tools", lambda: False)():
        return success, info
    if GATE_STALLED_MARKER in (info or ""):
        # The gate has been proven not to measure the artifact, so a
        # stronger model has nothing to be stronger AT. Measured: after
        # the weak model's ten turns against an unrunnable gate, the
        # escalation spent ten more and 467k tokens total, every verdict
        # byte-identical to the first.
        _logger.warning(
            "[AgentLoop] step %d: NOT escalating — the gate is the defect, "
            "not the code; a stronger model cannot satisfy it either",
            kw.get("step_idx", 0) + 1)
        return success, info
    _logger.info(
        "[AgentLoop] step %d: loop failed — escalating to stronger model",
        kw.get("step_idx", 0) + 1)
    kw = dict(kw)
    # Digest = what was tried (narrative, heavily truncated). The full
    # error still rides alongside it: a verify failure's stack trace and
    # assertion text are the most actionable thing the next attempt has,
    # and the digest's per-attempt summary is far too short to carry them.
    _digest = attempt_digest(kw.get("step_idx", 0))
    kw["context"] = (
        (kw.get("context") or "")
        + (("\n\n" + _digest) if _digest else "")
        + "\n\nA previous attempt by another model FAILED:\n"
        + truncate_middle(info, 2000))
    kw["attempt_label"] = ESCALATION_ATTEMPT_LABEL
    return run_agent_loop(escalation_client, tools, step_text, task, **kw)


def escalation_already_failed(step_idx: int) -> bool:
    """True when the stronger model has already had a failed run at this step.

    The ladder used to be loop(weak) -> loop(strong) -> recovery(weak) ->
    recovery(strong): after the stronger model failed, the next attempt
    went back to the weaker one. Observed on a Pac-Man run, step 7 spent
    32 turns across those four attempts — 47% of the whole run's turns —
    and only the final strong attempt succeeded. Each attempt re-sends the
    conversation, so turns are the dominant driver of prompt tokens.
    """
    return any(
        a.get("label") == ESCALATION_ATTEMPT_LABEL
        and a.get("outcome") != "verified"
        for a in get_attempts(step_idx)
    )


def _verify_call(verify_cmd: str):
    from ..llm.chat_types import ToolCall
    return ToolCall(name="run_command", arguments={"command": verify_cmd},
                    id="verify")


def verify_cmd_for_language(language: str | None,
                            project_root: str = ".") -> str | None:
    """Deterministic test command for the loop's exit verification.

    Returns None when no trustworthy command exists for the language —
    a wrong verify command is worse than none (the loop would chase
    failures in the verifier instead of the code).
    """
    import json
    import os

    lang = (language or "python").lower()
    if lang == "python":
        # Django projects test through manage.py — pytest is usually not
        # even installed there, so a pytest verifier fails regardless of
        # the app's real state (mirrors BulkTest's Django detection).
        if os.path.isfile(os.path.join(project_root, "manage.py")):
            return "python manage.py test --noinput"
        return "python -m pytest -q"
    if lang in ("javascript", "typescript"):
        # Only trust `npm test` when the project actually defines it;
        # guessing a runner (jest vs vitest) misfires too often.
        pkg_path = os.path.join(project_root, "package.json")
        try:
            with open(pkg_path, "r", encoding="utf-8") as f:
                pkg = json.load(f)
            if (pkg.get("scripts") or {}).get("test"):
                return "npm test --silent"
        except (OSError, json.JSONDecodeError):
            pass
        return None
    if lang == "go":
        return "go test ./..."
    return None


# One-shot scaffolding commands fail on a SECOND invocation precisely when
# the first one (or the recovery) succeeded: mkdir → "already exists",
# django-admin startproject → "conflicts with existing", npm create →
# refuses a non-empty directory, python -m venv → half-usable dir. Any
# compound command containing one of these is excluded whole — planned
# commands routinely chain scaffold + install with `&&`.
# Venv must match CREATION invocations only — a bare `venv` alternative
# also matched the harmless activation path `venv\Scripts\activate`,
# which silently disqualified every plan-declared verify: that carried
# the planner's activation prefix (observed: all 12 CODE gates dropped).
_NON_REVERIFIABLE_RE = re.compile(
    r"\b(?:mkdir|(?:python3?|py)\s+-m\s+venv|virtualenv\s+\S+"
    r"|startproject|startapp"
    r"|git\s+init|npm\s+(?:create|init)|npx\s+create-|yarn\s+create"
    r"|cargo\s+(?:new|init)|rails\s+new|dotnet\s+new)\b",
    re.IGNORECASE,
)


def reverifiable_cmd(cmd: str | None) -> str | None:
    """Return *cmd* when re-running it is a safe deterministic verify gate
    for the recovery loop, else ``None``.

    A failed CMD step's own command is the natural ground truth for "did
    the recovery actually work" — without it the loop accepts the model's
    final summary on faith (observed: `npm run build:css` exited 1 twice,
    the model summarized what it had attempted, and the step was logged
    as recovered while the CSS was never built). Only commands that are
    not one-shot scaffolding qualify; for the rest, ``None`` keeps the
    summary-based exit (with its blocked-admission check).
    """
    if not cmd or _NON_REVERIFIABLE_RE.search(cmd):
        return None
    return cmd


def build_step_tools(executor, memory, kb_context_builder=None,
                     project_root: str = ".") -> AgentTools:
    """Assemble :class:`AgentTools` from the objects a step handler holds."""
    searcher = getattr(kb_context_builder, "_searcher", None) \
        if kb_context_builder is not None else None
    return AgentTools(project_root=project_root, executor=executor,
                      searcher=searcher, memory=memory)


def agent_loop_enabled(cfg, llm_client) -> bool:
    """True when the config opts in AND the provider can do native tools."""
    return (cfg is not None
            and getattr(cfg, "AGENT_LOOP", False)
            and llm_client is not None
            and getattr(llm_client, "supports_tools", lambda: False)())


# Marker prepended to error_info after a failed recovery attempt so the
# pipeline's diagnosis stage doesn't launch a second (redundant) loop.
RECOVERY_FAILED_MARKER = "[agent-loop-recovery-failed]"

# Exact line the recovery prompt asks the model to end with when it could
# NOT recover the step. Checked case-insensitively on the summary.
RECOVERY_BLOCKED_MARKER = "RECOVERY: blocked"


def run_recovery_loop(llm_client, tools: AgentTools, step_text: str,
                      task: str, error_info: str,
                      display=None, step_idx: int = 0,
                      language: str | None = None,
                      max_turns: int = 8,
                      verify_cmd: str | None = None,
                      escalation_client=None) -> tuple[bool, str]:
    """One bounded loop attempt to recover a failed step.

    Replaces the diagnose → fix → re-run machinery: the model gets the
    real error, inspects the actual project state with tools, fixes the
    cause and completes the step in place.

    Without a ``verify_cmd`` the loop's exit rests on the model's final
    summary, so the prompt asks for an explicit verdict line and a
    summary that admits the blocker is treated as a failure — previously
    an honest "the build still fails" summary was logged as a recovery.
    """
    context = (
        "A previous attempt at this step FAILED. Error:\n"
        f"{truncate_middle(error_info, 4000)}\n\n"
        "Investigate the actual state of the project, fix the cause, and "
        "complete the step. If the step is a failed shell command, prefer "
        "correcting and re-running that command (fix the path, drop a bad "
        "`cd`, adjust a flag) over recreating its effects by hand — do NOT "
        "hand-write the files a scaffolder would have generated. "
        "If the failure is an environment limitation "
        "you cannot fix (e.g. a required tool is not installed and cannot "
        "be installed), say so clearly in your summary and end it with "
        f"the exact line: {RECOVERY_BLOCKED_MARKER}")
    if verify_cmd:
        context += (
            f"\n\nThis step is complete ONLY when `{verify_cmd}` exits "
            "successfully — it will be run to verify your work.")

    def _attempt(client, label: str,
                 latest_error: str | None = None) -> tuple[bool, str]:
        # Recovery follows the main loop (and possibly its escalation), so
        # 1-3 failed attempts have already left edits in the working tree.
        # Recompute the digest per attempt so each one sees everything
        # tried so far, including the recovery attempt before it.
        digest = attempt_digest(step_idx)
        ctx = (context + "\n\n" + digest) if digest else context
        if latest_error:
            ctx += ("\n\nA previous recovery attempt by another model "
                    "FAILED:\n" + truncate_middle(latest_error, 2000))
        s, i = run_agent_loop(
            client, tools, step_text, task,
            display=display, step_idx=step_idx, language=language,
            max_turns=max_turns, verify_cmd=verify_cmd, context=ctx,
            _recovery=True, attempt_label=label)
        # Only meaningful when no verify_cmd gated the exit — a passing
        # deterministic check outranks the model's own pessimism.
        if s and not verify_cmd \
                and RECOVERY_BLOCKED_MARKER.lower() in i.lower():
            _logger.warning(
                "[AgentLoop] step %d: recovery summary admits the step is "
                "still blocked — not counting it as recovered",
                step_idx + 1)
            return False, f"Recovery loop reported itself blocked: {i[:800]}"
        return s, i

    # Recovery loops are where turn budgets die (observed repeatedly:
    # "mid-fix at turn 8"). Escalation is decided AFTER the blocked-
    # admission check — a "done" summary that admits the blocker is a
    # failure the stronger model should get a shot at (observed: a
    # blocked npx-tailwind recovery never escalated because the wrapper
    # saw the self-reported success).
    _escalation_usable = (
        escalation_client is not None
        and escalation_client is not llm_client
        and getattr(escalation_client, "supports_tools", lambda: False)())

    # Keep the ladder monotonic. If the stronger model already failed this
    # step in the main loop, a recovery run on the WEAKER one is the least
    # likely rung to succeed and costs a full turn budget to find out —
    # observed step 7 spending 8 turns there between two strong-model
    # attempts, inside a 32-turn step that was 47% of the run's turns.
    if _escalation_usable and escalation_already_failed(step_idx):
        _logger.info(
            "[AgentLoop] step %d: the stronger model already failed this "
            "step — starting recovery there instead of re-trying the "
            "weaker one", step_idx + 1)
        return _attempt(escalation_client, "recovery + escalation")

    success, info = _attempt(llm_client, "recovery")
    if not success and _escalation_usable:
        _logger.info(
            "[AgentLoop] step %d: recovery failed — escalating to "
            "stronger model", step_idx + 1)
        success, info = _attempt(escalation_client, "recovery + escalation",
                                 latest_error=info)
    return success, info
