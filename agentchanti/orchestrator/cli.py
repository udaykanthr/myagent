"""
CLI entry point — argument parsing and main execution flow.
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime

from ..config import Config
from ..llm.ollama import OllamaClient
from ..llm.lm_studio import LMStudioClient
from ..llm.base import LLMError
from ..llm import build_embed_client
from ..llm.cancellation import install_sigint_handler
from ..agents.planner import PlannerAgent
from ..agents.coder import CoderAgent
from ..agents.reviewer import ReviewerAgent
from ..agents.tester import TesterAgent
from ..executor import Executor
from ..embedding_store import EmbeddingStore
from ..cli_display import CLIDisplay, token_tracker, log
from ..language import (
    detect_language, detect_language_from_task, get_test_framework,
    get_language_name, get_code_block_lang,
)
from ..project_scanner import scan_project, format_scan_for_planner, collect_source_files
from ..checkpoint import (
    save_checkpoint, load_checkpoint, clear_checkpoint,
)
from .. import git_utils
from ..knowledge import KnowledgeBase
from ..step_cache import StepCache
from ..report import generate_html_report, StepReport
from ..plugins.registry import PluginRegistry

from .memory import FileMemory
from .crash_diagnostics import install_crash_diagnostics, set_activity
from .pipeline import (
    build_step_waves, _execute_step, _run_diagnosis_loop,
    run_wiring_verification,
)
from .plan_step import build_waves as _build_plan_waves
from ..agents.analyser import build_project_context, AnalyseAgent, parse_briefing_packages


def _rematch_plan_steps(new_steps, old_plan_steps, dependencies):
    """Re-match edited step descriptions to original PlanStep objects.

    Preserves structured metadata (step_type, target_files, exports,
    imports_from, command, inline_code) when the description is similar
    enough. Falls back to UNCLASSIFIED for steps that can't be matched.
    """
    from .plan_step import PlanStep, from_legacy_steps
    from difflib import SequenceMatcher

    result: list[PlanStep] = []
    used: set[int] = set()  # indices into old_plan_steps already matched

    for new_idx, desc in enumerate(new_steps):
        desc_clean = desc.strip().lower()
        best_score = 0.0
        best_old_idx = -1

        for old_idx, old_ps in enumerate(old_plan_steps):
            if old_idx in used:
                continue
            old_clean = old_ps.description.strip().lower()
            score = SequenceMatcher(None, desc_clean, old_clean).ratio()
            if score > best_score:
                best_score = score
                best_old_idx = old_idx

        if best_score >= 0.6 and best_old_idx >= 0:
            # Re-use the old PlanStep with updated description and index
            old = old_plan_steps[best_old_idx]
            ps = PlanStep(
                id=old.id,
                step_type=old.step_type,
                description=desc,
                depends_on=list(old.depends_on),
                command=old.command,
                target_files=list(old.target_files),
                exports=list(old.exports),
                imports_from={k: list(v) for k, v in old.imports_from.items()},
                inline_code=dict(old.inline_code),
                index=new_idx,
            )
            result.append(ps)
            used.add(best_old_idx)
        else:
            # Can't match — create UNCLASSIFIED placeholder
            dep_ids = [str(d + 1) for d in dependencies.get(new_idx, set())]
            result.append(PlanStep(
                id=str(new_idx + 1),
                step_type="UNCLASSIFIED",
                description=desc,
                depends_on=dep_ids,
                index=new_idx,
            ))

    return result


def _blank_project_scaffold_hint(language: str | None) -> str:
    """Return language-appropriate scaffolding examples for blank-project prompt."""
    lang = (language or "").lower()
    if lang == "python":
        return "e.g. `python -m venv venv`, `pip install <packages>`"
    if lang in ("javascript", "typescript"):
        return "e.g. `npm create vite@latest`, `npm install`, framework setup"
    if lang == "go":
        return "e.g. `go mod init`, `go get <packages>`"
    if lang == "rust":
        return "e.g. `cargo init`, `cargo add <crates>`"
    # Generic fallback
    return "e.g. project init command, package install, framework setup"


def _parse_kb_topics(task: str, re_mod) -> list[str]:
    """
    Extract KB topics from a REQUIREMENTS_SPEC embedded in *task*.

    Handles both formats the LLM may output:

    Comma-separated (preferred):
      KB topics: Tailwind CSS, React hooks, Vitest

    Bullet list (common LLM habit):
      KB topics:
      - Tailwind CSS
      - React hooks
      - Vitest

    Returns a list of clean topic strings, empty if 'none' or not found.
    """
    m = re_mod.search(
        r'KB topics[^:\n]*:\s*(.*?)(?=\n[A-Z][^\n]*:|$)',
        task,
        re_mod.IGNORECASE | re_mod.DOTALL,
    )
    if not m:
        return []

    raw = m.group(1).strip()
    if not raw or raw.lower() in ('none', 'n/a'):
        return []

    # Detect bullet list: any line starting with "- "
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if any(l.startswith('- ') for l in lines):
        topics = [
            l.lstrip('- ').strip().rstrip('.')
            for l in lines
            if l.startswith('- ')
        ]
    else:
        # Comma-separated — may span multiple lines
        flat = ' '.join(lines)
        topics = [t.strip().rstrip('.') for t in flat.split(',')]

    return [t for t in topics if t and t.lower() not in ('none', 'n/a')]


def _parse_kb_doc_titles(task: str) -> list[str]:
    """
    Extract explicit KB doc titles from a REQUIREMENTS_SPEC embedded in *task*.

    Parses the `KB docs:` line that IntentAgent emits when it was given a list
    of available global KB doc titles and selected the relevant ones.

    Handles both formats:
      KB docs: Tailwind CSS v4 Setup Guide, React Component Patterns
      KB docs:
      - Tailwind CSS v4 Setup Guide
      - React Component Patterns

    Returns exact title strings ready for GlobalKBStore.get_by_titles().
    """
    import re as _re
    m = _re.search(
        r'KB docs[^:\n]*:\s*(.*?)(?=\n[A-Z][^\n]*:|$)',
        task,
        _re.IGNORECASE | _re.DOTALL,
    )
    if not m:
        return []

    raw = m.group(1).strip()
    if not raw or raw.lower() in ('none', 'n/a'):
        return []

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if any(l.startswith('- ') for l in lines):
        titles = [l.lstrip('- ').strip().rstrip('.') for l in lines if l.startswith('- ')]
    else:
        flat = ' '.join(lines)
        titles = [t.strip().rstrip('.') for t in flat.split(',')]

    return [t for t in titles if t and t.lower() not in ('none', 'n/a')]


# Keep a module-level reference so the crash-log handle stays open for
# the whole process — faulthandler writes to the raw fd at fault time.
_FAULT_LOG = None


def _arm_faulthandler() -> None:
    """Capture native crashes (segfault/access violation) to a file.

    Silent process deaths (no traceback, no exit message) have occurred
    when native extensions were used across threads. Python-level guards
    exist for the known cases; this makes any remaining one identify
    itself: on a fatal signal, faulthandler dumps every thread's stack
    to .agentchanti/crash.log.
    """
    global _FAULT_LOG
    import faulthandler
    try:
        os.makedirs(".agentchanti", exist_ok=True)
        _FAULT_LOG = open(os.path.join(".agentchanti", "crash.log"), "a",
                          encoding="utf-8", errors="replace")
        _FAULT_LOG.write(f"\n=== session start {datetime.now().isoformat()} "
                         f"pid={os.getpid()} ===\n")
        _FAULT_LOG.flush()
        faulthandler.enable(file=_FAULT_LOG, all_threads=True)
    except OSError:
        faulthandler.enable()  # fall back to stderr


def main():
    install_sigint_handler()
    _arm_faulthandler()
    install_crash_diagnostics()
    try:
        # Early returns inside _main_impl (--version, kb subcommand, an
        # aborted prompt) yield None, which stays 0.
        exit_code = _main_impl() or 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception:
        # Last-resort safety net: without this, an unhandled exception
        # anywhere in the pipeline (e.g. a worker-thread error re-raised
        # via future.result(), or an LLM client error) propagates to
        # Python's default excepthook, which prints only to stderr and
        # bypasses the `logging` module entirely — so the run's own log
        # file shows no trace of the crash at all, and the traceback can
        # be lost if a Rich/Textual live display has taken over the
        # terminal. Log it here (with traceback) so the log file always
        # has a record, then re-raise so the process still exits
        # non-zero and the traceback still reaches stderr.
        log.exception("Unhandled exception — pipeline crashed")
        raise
    sys.exit(exit_code)


def _reconcile_plan_graph(plan_graph, plan_steps, pending, step_results,
                          memory, language) -> None:
    """Mark this wave's nodes built and report broken export promises.

    Best-effort and never fatal: a parse failure here must not take down a
    run whose code is fine. Only steps that actually completed are marked,
    and a file we could not read or parse yields no ``actual_exports``,
    which :meth:`PlanGraph.reconcile` treats as "no evidence" rather than
    "export missing".
    """
    if plan_graph is None or not plan_steps:
        return
    try:
        from ..language_backend import get_backend
        backend = get_backend(language)
    except Exception:
        return

    files = {}
    try:
        files = memory.as_dict() or {}
    except Exception:
        pass

    for idx in pending:
        if step_results.get(idx) != "done":
            continue
        step = next((s for s in plan_steps if s.index == idx), None)
        if step is None:
            continue
        actual: list[str] = []
        for target in getattr(step, "target_files", None) or []:
            content = files.get(target) or files.get(
                str(target).replace("\\", "/"))
            if not content:
                continue
            try:
                actual.extend(backend.extract_exports(content) or [])
            except Exception:
                continue
        plan_graph.mark_built(step.id, actual)
        missing = plan_graph.reconcile(step.id)
        if missing:
            # Advisory, not a verdict: export extraction is per-language
            # and heuristic, so an absence here is a hint to look rather
            # than proof. Claiming "these imports WILL fail" was wrong the
            # first time it fired — every named symbol was a module-level
            # constant the extractor could not see, and the gate importing
            # them passed.
            log.warning(
                "[PlanGraph] step %s: declared export(s) not found in the "
                "written file(s): %s — check before downstream steps "
                "import them", step.id, ", ".join(sorted(missing)[:8]))


def _done_step_ids(plan_steps, indices, step_results) -> list[str]:
    """Ids of the steps in *indices* the pipeline itself calls done."""
    out: list[str] = []
    for idx in indices:
        if step_results.get(idx) != "done":
            continue
        step = next((s for s in (plan_steps or []) if s.index == idx), None)
        if step is not None:
            out.append(step.id)
    return out


def _ghost_resolve_wave(plan_steps, pending, step_results, language,
                        stage: str) -> None:
    """Resolve the ghost's expectations for a finished wave.

    Wrapped whole: this is a shadow observer, and an observer that can
    take down the run it observes is worse than no observer.
    """
    try:
        from .ghost import get_ghost
        from .wave_snapshots import get_gate_ledger

        ghost = get_ghost()
        if ghost is None:
            return
        done = _done_step_ids(plan_steps, pending, step_results)
        if not done:
            return
        _gates = get_gate_ledger().gates().keys()
        ghost.resolve(done, language=language, gate_cmds=_gates, stage=stage)

        # Repair what can be repaired now, not after the run: a missing
        # dependency healed at this wave is one the next wave's steps can
        # rely on. Each heal re-checks its own expectation and reverts a
        # source edit that did not close the gap.
        from .ghost_heal import get_healer
        healer = get_healer()
        if healer is not None:
            healer.heal(done, language=language, gate_cmds=_gates,
                        stage=stage)
            # A test directory the run's own command cannot enter is a
            # gap the same wave can close, and closing it early is what
            # gives the later steps a gate that measures their own work.
            healer.heal_uncollected_tests()
    except Exception as exc:
        log.debug("[Ghost] wave resolution skipped: %s", exc)


def _ghost_final_report(plan_steps, step_results, memory, language,
                        pipeline_success: bool) -> None:
    """Re-resolve every step against the final tree and log disagreements.

    A final pass matters even though each wave already resolved: later
    waves, the bulk-test fix round and the smoke-test repair all edit
    files an earlier wave had already certified, so the end-of-run state
    is the only one that describes what actually shipped.
    """
    try:
        from .ghost import get_ghost
        from .wave_snapshots import get_gate_ledger

        ghost = get_ghost()
        if ghost is None:
            return
        all_indices = [s.index for s in (plan_steps or [])]
        done = _done_step_ids(plan_steps, all_indices, step_results)
        _gates = get_gate_ledger().gates().keys()
        ghost.resolve(done, language=language, gate_cmds=_gates,
                      stage="final")

        # A last repair pass over the finished tree: later waves and the
        # fix rounds can reopen a gap an earlier wave closed.
        from .ghost_heal import get_healer
        healer = get_healer()
        if healer is not None:
            healer.heal(done, language=language, gate_cmds=_gates,
                        stage="final")
            healer.heal_uncollected_tests()
            _heal_line = healer.summary()
            if _heal_line:
                log.info(_heal_line)
                for _r in healer.results:
                    log.info("[GhostHeal]   %s", _r.describe())
        try:
            tracked = list((memory.as_dict() or {}).keys())
        except Exception:
            tracked = []
        ghost.report(done, tracked_files=tracked,
                     pipeline_success=pipeline_success)
    except Exception as exc:
        log.debug("[Ghost] final report skipped: %s", exc)


# A suite that collected nothing. Every runner says so in its own words,
# and all of them exit 0 while doing it — `--passWithNoTests` is even an
# explicit request for that. The verdict is therefore not readable from
# the exit code, only from the output.
_EMPTY_SUITE_RE = re.compile(
    r"no test files found"          # vitest
    r"|no tests ran"                # pytest
    r"|collected 0 items"           # pytest
    r"|no tests found"              # jest
    r"|ran 0 tests"                 # unittest
    r"|\[no test files\]"           # go test
    r"|no tests to run",            # misc
    re.IGNORECASE,
)


def _gate_scope(cmd: str) -> str:
    """The directory *cmd* runs in, from a leading ``cd X &&``, else ``""``.

    ``""`` is the repo root, which contains everything.
    """
    m = re.match(r"^\s*cd\s+([^\s&|;]+)\s*&&", cmd or "")
    if not m:
        return ""
    return m.group(1).strip("\"'").replace("\\", "/").strip("./").rstrip("/")


def _suite_covers(suite_cmd: str, gate_cmd: str) -> bool:
    """Whether *suite_cmd* could have exercised what *gate_cmd* checks.

    Decided by working directory, which is the one thing both commands
    always state. A suite rooted at the repo covers everything; a suite
    rooted in ``frontend/`` says nothing whatsoever about a gate that
    runs at the root against ``backend/``.
    """
    suite_scope = _gate_scope(suite_cmd)
    if not suite_scope:
        return True
    gate_scope = _gate_scope(gate_cmd)
    return gate_scope == suite_scope or gate_scope.startswith(suite_scope + "/")


def _green_suites_contradicting(regressions) -> list[tuple[str, str]]:
    """Green suite gates that a rollback would sacrifice, else ``[]``.

    Returns non-empty only when ALL of:
    - a test-suite gate is recorded and is NOT among the regressions, and
    - no regressing gate is itself a suite, and
    - that suite actually **ran tests**, and
    - that suite **could have covered** the gate it is overruling.

    The first two conditions were the original rule. A red suite means the
    stage broke real behaviour and ordinary rollback is right; this is
    strictly the case where inline assertions and the suite disagree and
    the suite wins.

    The last two come from a measured false verdict (2026-08-19). The
    claim being made here is strong — "the suite encodes the task's stated
    invariants, so the gate is the suspect" — and it was made on behalf of
    `cd frontend && npm run test -- --run`, whose script the agent had
    written as `vitest --passWithNoTests --run` and which found **no test
    files at all**. It overruled a real gate on `backend/services/
    authValidation.js`, a file two directories outside anything it could
    have run, and told the reader to go fix a `verify:` line that was
    correct. The gate was red because a dependency-fix round had rewritten
    that CommonJS module into ESM and dropped its exports — a genuine
    regression, correctly detected, and dismissed.

    A suite earns authority over a gate by having run something, and by
    having run something that could have covered it. Returning [] is
    always the safe answer — the caller rolls back as before.
    """
    from .wave_snapshots import get_gate_ledger, is_suite_gate

    failed = {cmd for cmd, _label, _out in regressions}
    if any(is_suite_gate(cmd) for cmd in failed):
        return []

    ledger = get_gate_ledger()
    contradicting: list[tuple[str, str]] = []
    for cmd, label in ledger.gates().items():
        if not is_suite_gate(cmd) or cmd in failed:
            continue
        out = ledger.last_output(cmd) or ""
        if _EMPTY_SUITE_RE.search(out):
            log.warning(
                "[Monotonic] suite gate (%s) collected no tests, so it "
                "cannot overrule a failing gate: %s",
                label or "?", cmd)
            continue
        uncovered = [c for c in failed if not _suite_covers(cmd, c)]
        if uncovered:
            log.warning(
                "[Monotonic] suite gate (%s) runs in '%s/' and cannot have "
                "exercised %d failing gate(s) outside it: %s",
                label or "?", _gate_scope(cmd) or ".", len(uncovered),
                "; ".join(uncovered))
            continue
        contradicting.append((cmd, label))
    return contradicting


def _enforce_monotonic_gates(snapshots, executor, stage: str,
                             repair=None, display=None) -> bool:
    """Snapshot *stage*, then re-run every acceptance gate recorded so far.

    Must be called after **every** stage that can write source files —
    each wave, the bulk-test fix round, and the smoke-test repair loop.
    Skipping any of them lets that stage's edits ship unverified: the
    smoke test repaired a launch crash by changing ``Player.update()``'s
    signature, the already-green ``python -m unittest discover`` gate went
    red, nothing rechecked it, and the run reported ``Finished``.

    *repair* is an optional zero-arg callable invoked once when a
    regression is found, before deciding to roll back. Use it where the
    stage's edits are more likely correct than the failing gate — a
    smoke-test fix that changes an API a test still stubs is a stale test,
    not a bad fix, and rolling it back would re-break a working app to
    satisfy the stub. Gates are re-run afterwards; only a still-red gate
    rolls back.

    Returns True when every gate passes — the stage is then committed and
    becomes the new rollback target. On an unresolved regression the
    workdir is restored to the last green snapshot and False is returned,
    so the caller fails the run instead of reporting success over a red
    gate.
    """
    from ..cli_display import set_status
    from .wave_snapshots import get_gate_ledger

    if not snapshots.managed:
        # No managed repo (the workdir is the user's own git repo), so a
        # rollback is neither possible nor wanted — their history is the
        # safety net. Leave the verdict to the caller's own checks.
        return True

    def _names(regs):
        return ", ".join(f"step {label or '?'}: `{cmd}`"
                         for cmd, label, _out in regs)

    _n_gates = len(get_gate_ledger().gates())
    set_status(display,
               f"Re-checking {_n_gates} acceptance gate(s) after {stage}...")
    regressions = get_gate_ledger().recheck(executor)

    if regressions and repair is not None:
        set_status(display, f"Gate regression after {stage} — repairing...")
        log.warning(
            "[Monotonic] %s broke %d previously-passing gate(s): %s — "
            "attempting one repair round before rolling back.",
            stage, len(regressions), _names(regressions))
        try:
            repair()
        except Exception as exc:
            log.warning("[Monotonic] Repair round raised: %s", exc)
        regressions = get_gate_ledger().recheck(executor)
        if not regressions:
            log.info("[Monotonic] Repair round restored every gate — "
                     "keeping the changes from %s.", stage)

    if not regressions:
        snapshots.commit_wave(stage)
        snapshots.mark_green()
        set_status(display, "")
        return True

    # ── Gate/suite contradiction: report it, do not undo the work ──
    # Some gates cannot be satisfied without violating the task. Observed:
    # a plan gated step 3 on `assert p.can_move()` against a `can_move()`
    # implemented as `direction != STOP` — i.e. "is MOVING" — so it demanded
    # a freshly-built Pac-Man already be in motion, while the brief demanded
    # "2000+ frames without the player moving". Diagnosis duly made the
    # player auto-start ("# Ensure the player starts in a valid moving state
    # for acceptance/tests"), the suite's idle test then failed, its fix
    # removed the auto-start, and THAT regressed the gate. Rollback then
    # discarded the fix and restored the artifact that violates the brief.
    #
    # When the suite is green and only inline gates are red, the suite is
    # the better authority — it is where the task's own invariants live.
    # Keep the work and name both sides. Still returns False: an unresolved
    # red gate must never be reported as success.
    _conflict = _green_suites_contradicting(regressions)
    if _conflict:
        set_status(display, "")
        log.error(
            "[Monotonic] GATE CONFLICT after %s — %d inline gate(s) are red "
            "while the task's own test suite is GREEN. The suite encodes the "
            "task's stated invariants, so the gate is the suspect, not the "
            "code. NOT rolling back — the working tree that satisfies the "
            "suite is preserved for inspection.", stage, len(regressions))
        for _cmd, _label, _out in regressions:
            log.error("[Monotonic]   red gate (step %s): %s",
                      _label or "?", _cmd)
        for _cmd, _label in _conflict:
            log.error("[Monotonic]   green suite (step %s): %s",
                      _label or "?", _cmd)
        log.error(
            "[Monotonic] Fix the plan's verify: line for the step(s) above — "
            "it asserts live behaviour on a freshly-constructed object.")
        return False

    set_status(display, f"Rolling back — {stage} left gate(s) red")
    log.warning("[Monotonic] %s left %d gate(s) red: %s",
                stage, len(regressions), _names(regressions))
    # A rollback is the most destructive thing this pipeline does — it
    # discards a whole wave of work and fails the run — and until now the
    # log recorded only WHICH gate went red, never WHY. `recheck` has
    # captured the failing output all along (`(out or "")[-1500:]`) and
    # `_names` dropped it on the floor, so a reader saw `output=810
    # chars` in the executor line and nothing else. Two separate
    # investigations of a rolled-back run (2026-08-19 runs 18 and 20)
    # could not determine the cause from the log, and the rollback had
    # already deleted the files needed to reproduce it by hand.
    for _cmd, _label, _out in regressions:
        log.warning("[Monotonic]   step %s output:\n%s",
                    _label or "?", (_out or "(no output captured)").strip())
    rb_ok, rb_msg = snapshots.rollback_to_last()
    if rb_ok:
        log.warning(
            "[Monotonic] Workdir rolled back to the last green snapshot "
            "— the changes from %s were discarded.", stage)
    else:
        log.warning(
            "[Monotonic] Rollback unavailable (%s) — regressing state left "
            "in place for inspection.", rb_msg)
    return False


def _main_impl():
    # Dispatch `agentchanti kb ...` to the KB CLI before argparse sees it,
    # so the KB subcommand tree is fully independent of the main task args.
    if len(sys.argv) > 1 and sys.argv[1] == "kb":
        from ..kb.cli import kb_main
        kb_main(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="AgentChanti — Multi-Agent Local Coder")
    from .. import __version__ as _agentchanti_version
    parser.add_argument("--version", action="version",
                        version=f"agentchanti {_agentchanti_version}",
                        help="Print the installed agentchanti version and exit")
    parser.add_argument("task", nargs="?", help="The coding task to perform")
    parser.add_argument("--prompt-from-file", help="Read prompt from a text file")
    parser.add_argument("--provider", choices=["ollama", "lm_studio", "openai", "gemini", "anthropic"],
                        default=None, help="The LLM provider to use (default: from config or lm_studio)")
    parser.add_argument("--model", default=None,
                        help="The model name to use (default: from config)")
    parser.add_argument("--embed-model", default=None,
                        help="Embedding model name (default: from config)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Disable semantic embeddings")
    parser.add_argument("--language", default=None,
                        help="Override detected language (e.g. python, javascript)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming responses")
    parser.add_argument("--no-git", action="store_true",
                        help="Disable git integration")
    parser.add_argument("--resume", action="store_true",
                        help="Force resume from checkpoint")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore checkpoint and start fresh")
    parser.add_argument("--auto", action="store_true",
                        help="Non-interactive mode: auto-approve plan, "
                             "skip all prompts (for backend/service use)")
    parser.add_argument("--config", default=None,
                        help="Path to .agentchanti.yaml config file")
    parser.add_argument("--no-diff", action="store_true",
                         help="Disable diff preview before writing files")
    parser.add_argument("--no-cache", action="store_true",
                         help="Disable step-level caching")
    parser.add_argument("--clear-cache", action="store_true",
                         help="Clear step cache before running")
    parser.add_argument("--no-knowledge", action="store_true",
                         help="Disable project knowledge base")
    parser.add_argument("--report", action="store_true", default=True,
                         help="Generate HTML report after run (default: on)")
    parser.add_argument("--no-report", action="store_true",
                         help="Disable HTML report generation")
    parser.add_argument("--generate-config", "--generate-yaml", action="store_true",
                         help="Generate a .agentchanti.yaml file with current settings and exit")
    parser.add_argument("--no-search", action="store_true",
                         help="Disable web search agent for planning and error diagnosis")
    parser.add_argument("--no-kb", action="store_true",
                         help="Disable KB context injection (debugging)")
    args = parser.parse_args()

    # ── 0. Load config ──
    cfg = Config.load(args.config)

    # CLI overrides
    model = args.model or cfg.DEFAULT_MODEL
    embed_model = args.embed_model or cfg.EMBEDDING_MODEL

    # Update config object with CLI overrides (for --generate-yaml)
    if args.provider is not None:
        cfg.PROVIDER = args.provider
    if args.model:
        cfg.DEFAULT_MODEL = args.model
    if args.embed_model:
        cfg.EMBEDDING_MODEL = args.embed_model
    if args.no_embeddings:
        cfg.NO_EMBEDDINGS = True
    if args.language:
        cfg.LANGUAGE = args.language
    if args.no_stream:
        cfg.STREAM_RESPONSES = False

    # ── 0.5. Generate YAML and exit ──
    if args.generate_config:
        yaml_content = cfg.to_yaml()
        with open(".agentchanti.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print("\n  ✨ Generated .agentchanti.yaml with current settings.\n")
        return

    # Handle prompt-from-file
    if args.prompt_from_file:
        try:
            with open(args.prompt_from_file, "r", encoding="utf-8") as f:
                args.task = f.read().strip()
        except Exception as e:
            print(f"\n  [ERROR] Could not read prompt file: {e}\n")
            return

    if not args.task:
        parser.print_help()
        return

    # ── 1. Detect language ──
    if args.language:
        language = args.language
    else:
        language = detect_language_from_task(args.task) or detect_language()
    log.info(f"Language: {language} ({get_language_name(language)})")

    # Load custom language backends from config
    if cfg.LANGUAGE_BACKENDS:
        from ..language_backend import load_custom_backends
        load_custom_backends(cfg.LANGUAGE_BACKENDS)

    # ── 2. Init LLM client ──
    stream_enabled = cfg.STREAM_RESPONSES and not args.no_stream
    llm_kwargs = dict(
        max_retries=cfg.LLM_MAX_RETRIES,
        retry_delay=cfg.LLM_RETRY_DELAY,
        stream=stream_enabled,
        max_output_tokens=cfg.MAX_OUTPUT_TOKENS,
    )

    provider = args.provider or cfg.PROVIDER
    if provider == "ollama":
        # read_timeout is Ollama-only — it cannot go in llm_kwargs, which
        # every provider shares.
        llm_client = OllamaClient(
            base_url=cfg.OLLAMA_BASE_URL, model=model,
            read_timeout=cfg.LLM_READ_TIMEOUT, **llm_kwargs)
    elif provider == "openai":
        from ..llm.openai_client import OpenAIClient
        api_key = cfg.OPENAI_API_KEY
        if not api_key:
            print("\n  [ERROR] OpenAI provider requires an API key.\n"
                  "  Set OPENAI_API_KEY env var or add it to .agentchanti.yaml.\n")
            return
        llm_client = OpenAIClient(
            base_url=cfg.OPENAI_BASE_URL, model=model,
            api_key=api_key,
            reasoning_effort=cfg.OPENAI_REASONING_EFFORT, **llm_kwargs)
    elif provider == "gemini":
        from ..llm.gemini_client import GeminiClient
        api_key = cfg.GEMINI_API_KEY
        if not api_key:
            print("\n  [ERROR] Gemini provider requires an API key.\n"
                  "  Set GEMINI_API_KEY env var or add it to .agentchanti.yaml.\n")
            return
        llm_client = GeminiClient(
            base_url=cfg.GEMINI_BASE_URL, model=model,
            api_key=api_key, **llm_kwargs)
    elif provider == "anthropic":
        from ..llm.anthropic_client import AnthropicClient
        api_key = cfg.ANTHROPIC_API_KEY
        if not api_key:
            print("\n  [ERROR] Anthropic provider requires an API key.\n"
                  "  Set ANTHROPIC_API_KEY env var or add it to .agentchanti.yaml.\n")
            return
        llm_client = AnthropicClient(
            base_url=cfg.ANTHROPIC_BASE_URL, model=model,
            api_key=api_key, **llm_kwargs)
    else:
        llm_client = LMStudioClient(
            base_url=cfg.LM_STUDIO_BASE_URL, model=model,
            reasoning_effort=cfg.LM_STUDIO_REASONING_EFFORT, **llm_kwargs)

    # ── 3. Scan existing project ──
    scan_result = scan_project(".")
    source_files = collect_source_files(".")
    log.info(f"Project scan: {scan_result['file_count']} files detected, "
             f"{len(source_files)} source files collected")
    project_context = format_scan_for_planner(
        scan_result, max_chars=cfg.PLANNER_CONTEXT_CHARS,
        source_files=source_files)

    # ── 4. Init embedding store (SQLite-backed for persistence) ──
    # Build a dedicated embed client (respects embedding_provider config).
    # Kept as a top-level var so KB components can reuse it instead of llm_client.
    embed_client = None if args.no_embeddings else build_embed_client(cfg)
    embed_store = None
    if args.no_embeddings:
        log.info("Embeddings disabled")
    elif embed_client is None:
        log.info(
            "Embeddings disabled: Anthropic has no embedding API. "
            "Set 'embedding_provider' in .agentchanti.yaml (ollama/openai/gemini)."
        )
    else:
        try:
            from ..embedding_store_sqlite import SQLiteEmbeddingStore
            import os
            db_path = os.path.join(cfg.EMBEDDING_CACHE_DIR, "embeddings.db")
            embed_store = SQLiteEmbeddingStore(
                embed_client, embed_model=embed_model, db_path=db_path)
            log.info(f"Embeddings enabled with SQLite cache (model: {embed_model})")
        except Exception as e:
            log.warning(f"SQLite embedding store failed ({e}), falling back to in-memory")
            embed_store = EmbeddingStore(embed_client, embed_model=embed_model)

    # ── 4b. Init step cache ──
    step_cache = None
    if not args.no_cache:
        import os
        cache_dir = os.path.join(cfg.EMBEDDING_CACHE_DIR, "cache")
        step_cache = StepCache(cache_dir=cache_dir,
                               ttl_hours=cfg.STEP_CACHE_TTL_HOURS)
        if args.clear_cache:
            step_cache.clear()
        log.info(f"Step cache enabled (TTL: {cfg.STEP_CACHE_TTL_HOURS}h)")

    # ── 4c. Init knowledge base ──
    knowledge_base = None
    if not args.no_knowledge:
        import os
        kb_path = os.path.join(cfg.EMBEDDING_CACHE_DIR, "knowledge.json")
        knowledge_base = KnowledgeBase(path=kb_path)
        log.info(f"Knowledge base loaded ({knowledge_base.size} entries)")

    # ── 4c-bis. Import plan optimizer ──
    from .plan_optimizer import optimize_plan, optimize_structured_plan

    # ── 4d. Init plugin registry ──
    plugin_registry = PluginRegistry()
    if cfg.PLUGINS:
        plugin_registry.discover(cfg.PLUGINS)
        log.info(f"Plugins loaded: {plugin_registry.size}")

    # ── 4f. Init search agent ──
    search_agent = None
    if cfg.SEARCH_ENABLED and not args.no_search:
        from ..agents.search import SearchAgent
        search_agent = SearchAgent(
            provider=cfg.SEARCH_PROVIDER,
            api_key=cfg.SEARCH_API_KEY,
            api_url=cfg.SEARCH_API_URL,
            max_results=cfg.SEARCH_MAX_RESULTS,
            max_page_chars=cfg.SEARCH_MAX_PAGE_CHARS,
            llm_client=llm_client,
        )
        log.info(f"Search agent enabled (provider: {cfg.SEARCH_PROVIDER})")
    else:
        log.info("Search agent disabled")

    # ── 4g. Init KB context builder and runtime watcher (Phase 4) ──
    kb_context_builder = None
    kb_runtime_watcher = None
    if cfg.KB_ENABLED and not args.no_kb:
        try:
            import os as _os
            from ..kb.startup import KBStartupManager
            from ..kb.context_builder import ContextBuilder
            from ..kb.runtime_watcher import RuntimeWatcher

            # Use embed_client for KB vector ops; fall back to llm_client if unavailable
            kb_api_client = embed_client or llm_client

            # Smart startup check — handles global KB, local KB
            KBStartupManager().run(project_root=_os.getcwd(), api_client=kb_api_client)

            kb_context_builder = ContextBuilder(project_root=_os.getcwd(), api_client=kb_api_client)
            kb_runtime_watcher = RuntimeWatcher(
                debounce_seconds=cfg.KB_WATCHER_DEBOUNCE_SECONDS,
            )
            kb_runtime_watcher.start(project_root=_os.getcwd(), api_client=kb_api_client)
            log.info("[KB] Context builder and runtime watcher initialised")
        except Exception as kb_exc:
            log.warning(f"[KB] Initialisation failed (non-fatal): {kb_exc}")
            kb_context_builder = None
            kb_runtime_watcher = None
    else:
        log.info("[KB] KB context injection disabled")

    # ── 4e. Step reports (for HTML report) ──
    step_reports: list[StepReport] = []

    # ── 5. Init agents (with per-agent model support) ──
    def _make_llm_for_agent(agent_name: str):
        """Create an LLM client for a specific agent, honouring per-agent
        model and (optionally) per-agent provider overrides.

        A `<agent>_provider` config key routes the override to a different
        backend than the run's provider — e.g. an ollama run can escalate
        to `escalation: gpt-5.4` + `escalation_provider: openai`. Without
        the provider override the model inherits the run provider and a
        cross-provider model 404s against the wrong endpoint.
        """
        agent_model = cfg.get_agent_model(agent_name) or model
        agent_provider = cfg.get_agent_provider(agent_name) or provider
        if agent_model == model and agent_provider == provider:
            return llm_client  # reuse the main client — nothing overridden
        # Create a separate client with the agent-specific provider + model
        if agent_provider == "ollama":
            return OllamaClient(
                base_url=cfg.OLLAMA_BASE_URL, model=agent_model,
                read_timeout=cfg.LLM_READ_TIMEOUT, **llm_kwargs)
        elif agent_provider == "openai":
            from ..llm.openai_client import OpenAIClient
            return OpenAIClient(
                base_url=cfg.OPENAI_BASE_URL, model=agent_model,
                api_key=cfg.OPENAI_API_KEY,
                reasoning_effort=cfg.OPENAI_REASONING_EFFORT, **llm_kwargs)
        elif agent_provider == "gemini":
            from ..llm.gemini_client import GeminiClient
            return GeminiClient(
                base_url=cfg.GEMINI_BASE_URL, model=agent_model,
                api_key=cfg.GEMINI_API_KEY, **llm_kwargs)
        elif agent_provider == "anthropic":
            from ..llm.anthropic_client import AnthropicClient
            return AnthropicClient(
                base_url=cfg.ANTHROPIC_BASE_URL, model=agent_model,
                api_key=cfg.ANTHROPIC_API_KEY, **llm_kwargs)
        else:
            return LMStudioClient(
                base_url=cfg.LM_STUDIO_BASE_URL, model=agent_model,
                reasoning_effort=cfg.LM_STUDIO_REASONING_EFFORT, **llm_kwargs)

    # Custom prompt suffixes from config
    planner_suffix = cfg.PROMPT_SUFFIXES.get("planner_suffix", "")
    coder_suffix = cfg.PROMPT_SUFFIXES.get("coder_suffix", "")
    reviewer_suffix = cfg.PROMPT_SUFFIXES.get("reviewer_suffix", "")
    tester_suffix = cfg.PROMPT_SUFFIXES.get("tester_suffix", "")

    planner = PlannerAgent("Planner", "Senior Software Architect",
                           "Create a step-by-step plan for the coding task and related testcases.",
                           _make_llm_for_agent("planner"),
                           prompt_suffix=planner_suffix)
    from ..agents.intent import IntentAgent, parse_intent_spec
    intent_agent = IntentAgent("IntentAnalyzer", "Requirements Analyst",
                               "Analyze the prompt and search the web if intent is ambiguous to produce a formal REQUIREMENTS_SPEC.",
                               _make_llm_for_agent("intent"))
    coder = CoderAgent("Coder", "Senior Software Developer",
                       f"Write clean {get_language_name(language)} code for a single step.",
                       _make_llm_for_agent("coder"),
                       prompt_suffix=coder_suffix)
    reviewer = ReviewerAgent("Reviewer", "Code Reviewer",
                             "Review code for errors and style issues.",
                             _make_llm_for_agent("reviewer"),
                             prompt_suffix=reviewer_suffix)
    tester = TesterAgent("Tester", "Software Engineer in Test",
                         "Create unit tests for the provided code.",
                         _make_llm_for_agent("tester"),
                         prompt_suffix=tester_suffix)

    # Escalation model for failed steps (config: models.escalation), used by
    # BOTH execution paths: the agent loop retries a failed step with it, and
    # the classic diagnosis loop spends its final attempt on it. The banner
    # said [AgentLoop] while classic silently never escalated, so a classic
    # failure looked like the strong model had been tried and lost.
    if cfg.get_agent_model("escalation"):
        _escalation_client = _make_llm_for_agent("escalation")
        coder.escalation_client = _escalation_client
        tester.escalation_client = _escalation_client
        # CMD-step recoveries only hold the raw client, not an agent —
        # attach there too (observed: a failed npx-tailwind CMD recovery
        # never escalated because this was the one unwired path).
        llm_client.escalation_client = _escalation_client
        log.info("[Escalation] Model configured: %s (provider: %s) — "
                 "agent-loop retries and the final classic diagnosis attempt",
                 cfg.get_agent_model("escalation"),
                 cfg.get_agent_provider("escalation") or provider)

    executor = Executor()

    # ── 6. Init display ──
    display = CLIDisplay(args.task or "Config Generation")
    
    # Inject pricing into tracker
    token_tracker.pricing = cfg.PRICING
    
    log.info(f"Task: {args.task}")
    log.info(f"Provider: {provider}, Model: {model}")

    # Wire streaming progress callback
    if stream_enabled:
        # We'll set per-step callbacks in the execution loop
        pass

    # ── 7. Check for checkpoint ──
    checkpoint_file = cfg.CHECKPOINT_FILE
    resuming = False
    checkpoint_state = None
    step_results: dict[int, str] = {}
    start_from = 0

    if not args.fresh:
        checkpoint_state = load_checkpoint(checkpoint_file)
        if checkpoint_state:
            if args.resume or args.auto:
                resuming = True
                log.info("Auto-resuming from checkpoint" if args.auto else "Resuming (--resume)")
            else:
                display.pause()
                resuming = CLIDisplay.prompt_resume(checkpoint_state)
                display.resume()

    # ── 8. Restore state or create git checkpoint ──
    checkpoint_branch: str | None = None
    use_git = not args.no_git and git_utils.is_git_repo()

    if resuming and checkpoint_state:
        log.info("Resuming from checkpoint...")
        memory = FileMemory(embedding_store=embed_store, top_k=cfg.EMBEDDING_TOP_K)
        if kb_runtime_watcher is not None:
            memory.watcher_created_files = kb_runtime_watcher.created_files
        memory.update(checkpoint_state.get("file_memory", {}))
        steps = checkpoint_state["steps"]
        step_results = checkpoint_state.get("step_results", {})
        start_from = checkpoint_state.get("completed_step", -1) + 1

        # Load dependencies if saved, else parse them out of saved strings as a fallback
        loaded_deps = checkpoint_state.get("dependencies")
        if loaded_deps is not None:
            dependencies = {int(k): set(v) for k, v in loaded_deps.items()}
        else:
            _, dependencies = executor.parse_step_dependencies(steps)

        # Restore structured PlanStep objects if checkpoint has them
        from .plan_step import PlanStep, from_legacy_steps
        saved_plan_steps = checkpoint_state.get("plan_steps")
        if saved_plan_steps:
            plan_steps_parsed = [PlanStep.from_dict(d) for d in saved_plan_steps]
            log.info(f"Restored {len(plan_steps_parsed)} structured PlanSteps from checkpoint")
        else:
            # Legacy checkpoint without plan_steps — create wrappers
            plan_steps_parsed = from_legacy_steps(steps, dependencies)

        # A restored plan never passed through the gate checks: they live
        # in the planning branch, which resume skips entirely. The
        # run-time refusals still hold, but the backstop that makes a
        # destructive gate *impossible* rather than merely refused has to
        # run here too — otherwise a checkpoint is a way to smuggle one
        # past it. Measured: a resume brought back, verbatim, the gate
        # that had already burned the run which wrote the checkpoint.
        from .gate_safety import neutralize_destructive_gates
        from .plan_step import unrunnable_gate_reason
        for _sid, _was, _why in neutralize_destructive_gates(
                plan_steps_parsed):
            log.warning(
                "[GateSafety] restored step %s: dropped the destructive "
                "tail of its verify: %s — was `%s`", _sid, _why, _was)
        # Unrunnable gates are reported, not removed. Dropping one would
        # let the step pass unchecked, and the stall breaker now ends
        # such a step cheaply; a loud line is what a reader needs to know
        # the plan, not the code, is what failed.
        for _ps in plan_steps_parsed:
            _why = unrunnable_gate_reason(getattr(_ps, "verify_cmd", "") or "")
            if _why:
                log.warning(
                    "[Plan] restored gate for step %s can never pass on "
                    "this platform: %s — gate: %s",
                    _ps.id, _why, _ps.verify_cmd)

        language = checkpoint_state.get("language", language)

        # Restore intent_spec — the planning phase that normally produces it
        # (parse_intent_spec on the enriched task) is skipped on resume.
        # Without this, step handlers crash with UnboundLocalError.
        try:
            intent_spec = parse_intent_spec(
                checkpoint_state.get("task") or args.task)
        except Exception:
            intent_spec = None

        # Restore ProjectContext if saved (avoids re-running analysis LLM call)
        saved_project_context = checkpoint_state.get("project_context")
        if saved_project_context:
            from ..agents.analyser import ProjectContext
            _resumed_project_context = ProjectContext.from_dict(saved_project_context)
            log.info("[Resume] Restored ProjectContext from checkpoint (0 LLM tokens)")
        else:
            _resumed_project_context = None

        display.set_steps(steps)
        # Mark completed steps
        for idx in range(start_from):
            display.steps[idx]["status"] = "done"

        if "display_state" in checkpoint_state:
            ds = checkpoint_state["display_state"]
            if "elapsed" in ds:
                display.start_time = time.monotonic() - ds["elapsed"]
            if "steps" in ds:
                for i, saved_step in enumerate(ds["steps"]):
                    if i < len(display.steps):
                        display.steps[i].update(saved_step)

        display.render()
    else:
        # Fresh start
        if use_git:
            log.info("Creating git checkpoint branch...")
            checkpoint_branch = git_utils.create_checkpoint_branch(args.task)
            if checkpoint_branch:
                log.info(f"Git checkpoint: {checkpoint_branch}")
            else:
                log.warning("Failed to create git checkpoint branch")

        # ── 9. Plan ──
        display.show_status("Analyzing task and mapping relevant files...")
        log.info("Planning...")

        # Detect blank projects (no package manager / build config files)
        _has_project_config = bool(scan_result.get("key_files"))
        if _has_project_config:
            planner_context = f"Existing project:\n{project_context}"
        else:
            _scaffold_hint = _blank_project_scaffold_hint(language)
            planner_context = (
                f"PROJECT STATE: BLANK / EMPTY directory — no build config files found.\n"
                f"The plan MUST start with project scaffolding / initialization steps "
                f"({_scaffold_hint}) before writing any source code.\n"
            )
            if project_context:
                planner_context += f"\nCurrent directory contents:\n{project_context}"

        # KB injection is deferred to after pre_analyze so the IntentAgent's
        # REQUIREMENTS_SPEC (which includes a "KB topics:" field) can be used
        # to filter down to only the relevant entries.  Placeholder comment
        # here — actual injection happens below after pre_analyze completes.

        # Baseline test analysis before planning — run existing tests to
        # identify which files pass/fail so the planner only touches broken ones.
        # The task intent determines directive strictness: test-fix tasks get
        # strict "don't touch passing files" rules; feature tasks allow updates.
        from ..agents.planner import _classify_task_intent
        _task_intent = _classify_task_intent(args.task)
        test_analysis = ""
        if _has_project_config:
            try:
                from .test_analyzer import perform_baseline_test_analysis
                from .memory import FileMemory as _PreMemory
                _pre_memory = _PreMemory(embedding_store=embed_store, top_k=cfg.EMBEDDING_TOP_K)
                if source_files:
                    _pre_memory.update(source_files)
                test_analysis = perform_baseline_test_analysis(
                    _pre_memory, executor, language,
                    task_intent=_task_intent,
                )
                if test_analysis:
                    log.info("[Planning] Baseline test analysis (intent=%s):\n%s",
                             _task_intent, test_analysis)
            except Exception as _test_exc:
                log.warning("[Planning] Baseline test analysis failed: %s", _test_exc)

        # Pre-analysis: map relevant files, classify intent, enrich context
        _pre_mem_local = locals().get('_pre_memory')
        # Detect subproject root so IntentAgent can run npm commands from the
        # correct directory (e.g. angular-bootstrap-app/ instead of repo root).
        _intent_subproject: str | None = None
        if _pre_mem_local is not None:
            try:
                from .step_handlers import _detect_subproject_root
                _intent_subproject = _detect_subproject_root(_pre_mem_local)
            except Exception:
                pass
        analysis_context = planner.pre_analyze(
            args.task,
            source_files=source_files,
            kb_context_builder=kb_context_builder,
            knowledge_base=knowledge_base,
            test_analysis=test_analysis,
            language=language,
            baseline_passing_files=getattr(
                _pre_mem_local, '_tester_baseline_passing_files', None),
            baseline_failing_files=getattr(
                _pre_mem_local, '_tester_baseline_failing_files', None),
            search_agent=search_agent,
            intent_agent=intent_agent,
            cli_display=display,
            subproject_cwd=_intent_subproject,
            executor=executor,
        )
        if analysis_context:
            planner_context = analysis_context + "\n\n" + planner_context

        # ── QUESTION short-circuit ────────────────────────────────────────────
        # If IntentAgent classified the task as QUESTION, the answer is already
        # in the REQUIREMENTS_SPEC.  Skip briefing, global KB, and the planner.
        if getattr(planner, '_is_question_task', False):
            _answer = getattr(planner, '_question_answer', '')
            if _answer:
                print(f"\n{'─' * 60}")
                print(_answer)
                print(f"{'─' * 60}\n")
            display.finish()
            return

        # Record whether the RAW task asks for tests before enrichment
        # replaces it — the enriched spec routinely adds testing language,
        # so this must be decided now. Parked on args because memory does
        # not exist yet on the fresh path (it is created after planning);
        # it is copied onto memory at creation time below.
        from ..language import task_requests_tests
        args._raw_task_requests_tests = task_requests_tests(args.task)

        # The user's own words, kept before enrichment overwrites them.
        # The acceptance seed fingerprints the task to decide whether a
        # contract on disk was written for THIS task, and the enriched
        # spec is LLM output — it varies between runs of the identical
        # prompt, so fingerprinting it would re-seed on every run and
        # replace a contract that was perfectly good. Only the raw text
        # is stable enough to answer "is this the same task".
        args._raw_task = args.task

        # Update task if IntentAgent enriched it during pre_analyze
        args.task = getattr(planner, '_enriched_task', args.task)
        intent_spec = parse_intent_spec(args.task)

        # ── Filtered KB injection ─────────────────────────────────────────────
        # Parse "KB topics:" from the REQUIREMENTS_SPEC the IntentAgent just
        # produced.  Use those topics to filter knowledge_base entries so the
        # planner only sees docs relevant to this specific task — not the full
        # 83-entry dump which includes irrelevant framework docs and old fixes.
        if knowledge_base and knowledge_base.size > 0:
            import re as _re_kb
            _kb_topics: list[str] = []
            _kb_topics = _parse_kb_topics(args.task, _re_kb)

            if _kb_topics:
                # Targeted injection: only entries whose text overlaps with the
                # stated topics.  Always include the stack summary (1 entry).
                kb_context = knowledge_base.format_for_task(_kb_topics)
                log.info(
                    "Filtered KB injection: topics=%s", _kb_topics,
                )
            else:
                # "none" or no KB topics field → inject only stack + packages
                # (no patterns/fixes which tend to be task-specific noise).
                kb_context = knowledge_base.format_stack_only()
                log.info("KB topics: none — injecting stack summary only")

            if kb_context:
                planner_context += f"\n\n{kb_context}"

        log.info("[Planning] Pre-analysis context injected")

        # Apply LLM-corrected language (set by pre_analyze when heuristics were wrong)
        _llm_detected = getattr(planner, '_detected_language', None)
        if _llm_detected and _llm_detected != language:
            log.info(
                "Language corrected by LLM during pre-analysis: %s → %s (%s)",
                language, _llm_detected, get_language_name(_llm_detected),
            )
            language = _llm_detected
            # Re-describe coder agent role with the corrected language
            coder.role = f"Write clean {get_language_name(language)} code for a single step."

        MAX_PLAN_RETRIES = 3
        plan = None
        raw_steps = None
        # The previous attempt's steps, kept so a re-plan cannot silently
        # weaken a gate it was never asked to touch. See
        # plan_step.carry_forward_strong_gates.
        _previous_plan_steps: list = []

        for plan_attempt in range(1, MAX_PLAN_RETRIES + 1):
            display.show_status(
                f"Requesting steps from planner...{f' (retry {plan_attempt})' if plan_attempt > 1 else ''}"
            )
            try:
                plan = planner.process(args.task, context=planner_context,
                                       language=language,
                                       plan_mode=getattr(cfg, "PLAN_MODE",
                                                         "content"))
            except LLMError as exc:
                # A model that spends its whole output budget on hidden
                # reasoning returns nothing, every retry, deterministically
                # (observed: minimax-m3:cloud, 3 x 16384 tokens, 7.5
                # minutes, no plan).  That is a configuration problem the
                # user can act on — a raw traceback tells them nothing
                # about which model failed or what to change.
                _model = getattr(llm_client, "model", "?")
                _tries = getattr(llm_client, "max_retries", "?")
                log.error("Planner produced no usable plan with model %s: %s",
                          _model, exc)
                print(f"\nPlanning failed: no usable response from "
                      f"'{_model}' after {_tries} attempts.")
                print(f"  Cause: {exc}")
                print("  If this model reasons before answering, it may be "
                      "spending its whole output budget on hidden thinking.")
                print("  Try a larger max_output_tokens, or set a different "
                      "planner model under `models:` in .agentchanti.yaml.")
                sys.exit(1)
            log.info(f"Plan (attempt {plan_attempt}):\n{plan}")

            # ── Planner no-op signal ──
            # If the planner determined the task is already satisfied it
            # emits ==DONE== instead of steps.  Honour that and exit
            # cleanly — unless the briefing itself demands a change, in
            # which case the DONE is provably wrong: reject it and
            # re-plan with the contradiction quoted back (observed twice:
            # ==DONE== whose reason DESCRIBED the required one-line fix).
            if "==DONE==" in plan:
                _done_reason = ""
                for _line in plan.splitlines():
                    if _line.startswith("reason:"):
                        _done_reason = _line[len("reason:"):].strip()
                        break
                from ..agents.planner import done_contradicted_by_briefing
                # The briefing is only copied onto memory AFTER planning
                # succeeds — at this point it lives on the planner.
                _contradiction = done_contradicted_by_briefing(
                    getattr(planner, "_task_briefing", "")
                    or getattr(memory, "_task_briefing", "") or "")
                if _contradiction and plan_attempt < MAX_PLAN_RETRIES:
                    log.warning(
                        "[Plan] ==DONE== rejected — the briefing requires "
                        "a change: %s", _contradiction[:200])
                    planner_context += (
                        "\n\n[PLANNER CORRECTION] Your previous response "
                        "was ==DONE==, but the change below is REQUIRED "
                        "and has NOT been made:\n"
                        f"{_contradiction}\n"
                        "Emit a plan whose steps make this change. "
                        "Do NOT output ==DONE==.")
                    continue
                _done_msg = _done_reason or "Task already satisfied — no changes needed."
                log.info("[Plan] Planner signalled ==DONE==: %s", _done_msg)
                display.show_status(_done_msg)
                display.finish()
                print(f"\n  ✓ {_done_msg}\n")
                return

            # ── 10. Parse steps + dependencies ──
            from .plan_step import (
                parse_structured_plan, is_structured_plan, validate_plan,
                fix_nested_workspace_collision,
                fix_import_dependencies, project_file_reader, check_gate_quality,
                check_gate_consistency, repair_verify_commands,
                carry_forward_strong_gates,
                steps_as_text_list, steps_dependencies_dict,
                from_legacy_steps, parse_heuristic_plan, PlanStep,
                reclassify_manifest_steps, plan_looks_truncated,
                plan_salvageable, route_blind_edits,
            )
            from .gate_safety import (
                check_gate_safety, neutralize_destructive_gates,
            )
            plan_steps_parsed: list[PlanStep] | None = None

            _is_structured = is_structured_plan(plan)
            log.info(f"[Plan] Structured plan detected: {_is_structured}")
            if _is_structured:
                plan_steps_parsed = parse_structured_plan(plan)
                if plan_steps_parsed:
                    log.info(
                        f"[Plan] Parsed {len(plan_steps_parsed)} structured steps: "
                        f"{[(s.id, s.step_type, s.index) for s in plan_steps_parsed]}"
                    )
                    # A re-plan is triggered by one unusable gate but
                    # regenerates every step, so gates that were never in
                    # question get rewritten too — and the planner has no
                    # reason to preserve the strength of the ones it was
                    # not asked about. Restore those before judging this
                    # attempt, so a carried gate can also end the churn.
                    if _previous_plan_steps:
                        _kept = carry_forward_strong_gates(
                            _previous_plan_steps, plan_steps_parsed)
                        if _kept:
                            log.info(
                                "[Plan] Carried %d strong acceptance gate(s) "
                                "across the re-plan: %s",
                                len(_kept), ", ".join(_kept))
                    # Remember the objects, not a copy: the in-place gate
                    # repairs further down mutate these same steps, so the
                    # next attempt inherits the best version of this one.
                    _previous_plan_steps = plan_steps_parsed
                    errors = validate_plan(plan_steps_parsed)
                    if errors:
                        log.warning(f"[Plan] Validation warnings: {errors}")
                    ws_fixes = fix_nested_workspace_collision(plan_steps_parsed)
                    if ws_fixes:
                        log.info(f"[Plan] Auto-fixed workspace collision: {ws_fixes}")
                    dep_fixes = fix_import_dependencies(plan_steps_parsed, read_file=project_file_reader)
                    if dep_fixes:
                        log.info(f"[Plan] Auto-fixed import dependencies: {dep_fixes}")
                    blind_fixes = route_blind_edits(plan_steps_parsed)
                    if blind_fixes:
                        log.info(f"[Plan] Blind-edit routing: {blind_fixes}")

                    # ── Target reachability ──
                    # A gate can assert seven true things about a file the
                    # application never loads. Observed: `src/App.css` was
                    # imported by nothing (main.jsx loads only index.css),
                    # so repeated "restyle the header" runs wrote a full
                    # dark palette into a file Vite never bundled — twelve
                    # `.site-header` rules in the source, one in the built
                    # CSS, and no visible change across many runs while
                    # every check stayed green.
                    #
                    # This replans rather than merely warning. It was
                    # advisory for exactly one run, on the grounds that
                    # the right repair varies — and that run proved the
                    # point the wrong way round: the warning fired
                    # correctly, nothing consumed it, the step edited the
                    # dead file anyway, every gate went green and the UI
                    # was unchanged for the fourth time. An advisory
                    # nobody acts on is indistinguishable from silence.
                    _reach_gaps: list[tuple[str, str]] = []
                    try:
                        from .reachability import unreachable_stylesheet_reason
                        for _s in plan_steps_parsed:
                            _why = unreachable_stylesheet_reason(
                                _s, plan_steps_parsed, project_file_reader)
                            if _why:
                                _reach_gaps.append((_s.id, _why))
                    except Exception as _reach_exc:      # never fail a run
                        log.debug("[Plan] reachability check skipped: %s",
                                  _reach_exc)

                    if _reach_gaps and plan_attempt < MAX_PLAN_RETRIES:
                        log.warning(
                            "[Plan] %d step(s) target a file the app never "
                            "loads — replanning: %s", len(_reach_gaps),
                            ", ".join(sid for sid, _ in _reach_gaps))
                        display.show_status(
                            "Plan targets an unloaded stylesheet, retrying...")
                        planner_context += (
                            "\n\n[PLANNER CORRECTION] These steps edit a file "
                            "the application never loads, so their work "
                            "cannot reach the browser — the tests, the build "
                            "and the smoke test will all still pass:\n"
                            + "\n".join(f"  - step {sid}: {why}"
                                        for sid, why in _reach_gaps)
                            + "\n\nFix the TARGET, not the verify command. "
                            "Either retarget the stylesheet the entry point "
                            "actually imports, or add an explicit step that "
                            "imports this one and declare that dependency. "
                            "Re-emit the COMPLETE plan.")
                        continue
                    if _reach_gaps:
                        log.warning(
                            "[Plan] Proceeding after %d attempt(s) with %d "
                            "step(s) targeting an unloaded file — their "
                            "changes will not be visible: %s",
                            MAX_PLAN_RETRIES, len(_reach_gaps),
                            ", ".join(sid for sid, _ in _reach_gaps))

                    # ── Acceptance-gate quality ──
                    # A CODE step whose verify: only imports the module
                    # cannot fail on wrong behaviour, which makes the whole
                    # monotonic-gate machinery decorative. Observed: a
                    # Pac-Man run shipped with three of four ghosts spawned
                    # inside wall tiles — every gate green, smoke test
                    # green, pipeline "Finished". Send the plan back with
                    # the specific complaint rather than accepting gates
                    # that can only ever pass.
                    # A gate that damages the machine is judged FIRST and
                    # separately. The other two checks read a gate as a
                    # measurement and ask how good it is; this one asks
                    # what else it does. Measured: a planner gate ending
                    # `taskkill /im python.exe /f` force-killed every
                    # python.exe on the box — the pipeline included — and
                    # every existing check had passed it. See gate_safety.
                    _gate_gaps = (check_gate_safety(plan_steps_parsed)
                                  + check_gate_quality(plan_steps_parsed)
                                  + check_gate_consistency(plan_steps_parsed))

                    # Repair the offending lines before considering a
                    # re-plan. A re-plan regenerates the whole decomposition
                    # to fix one command — expensive, and it churns targets
                    # and dependencies that were never in question.
                    if _gate_gaps:
                        _repaired = repair_verify_commands(
                            plan_steps_parsed, _gate_gaps,
                            getattr(planner, "llm_client", None), args.task)
                        if _repaired:
                            log.info(
                                "[Plan] Repaired %d acceptance gate(s) "
                                "in place (no re-plan): %s",
                                len(_repaired), ", ".join(_repaired))
                            # Re-judge rather than subtract: a replacement
                            # can be substantive yet still assume the wrong
                            # working directory, and that check reads the
                            # whole plan, not one command.
                            _gate_gaps = (
                                check_gate_quality(plan_steps_parsed)
                                + check_gate_consistency(plan_steps_parsed))

                    if _gate_gaps and plan_attempt < MAX_PLAN_RETRIES:
                        log.warning(
                            "[Plan] %d step(s) have a verify: that cannot "
                            "fail on wrong behaviour — replanning: %s",
                            len(_gate_gaps),
                            ", ".join(f"{sid} ({why})"
                                      for sid, why in _gate_gaps))
                        display.show_status(
                            "Plan has import-only acceptance gates, "
                            "retrying...")
                        _by_id = {s.id: s for s in plan_steps_parsed}
                        _detail = "\n".join(
                            f"  - step {sid}: "
                            f"`{getattr(_by_id.get(sid), 'verify_cmd', '')}`"
                            f" — {why}"
                            for sid, why in _gate_gaps)
                        # Spelled out separately when a gate is unsafe
                        # rather than merely weak: "assert a concrete
                        # value" is not the correction for a command that
                        # kills processes, and a planner told only that
                        # will keep the destructive tail while making the
                        # assertion stronger.
                        _unsafe_ids = {sid for sid, _ in
                                       check_gate_safety(plan_steps_parsed)}
                        _safety_note = (
                            "\n\nSteps " + ", ".join(sorted(_unsafe_ids)) +
                            " are a different problem: their verify: RUNS A "
                            "DESTRUCTIVE COMMAND. A gate is re-run after "
                            "every later wave, so its side effects happen "
                            "repeatedly. Never kill processes by name "
                            "(taskkill /im, pkill, killall), delete trees "
                            "(rm -rf, del /s, git clean), or reset the "
                            "working tree. A gate must only observe. If you "
                            "need to check that the app starts, assert it "
                            "from inside a short python -c that imports and "
                            "constructs it — do not launch a blocking "
                            "process and kill it."
                        ) if _unsafe_ids else ""
                        planner_context += (
                            "\n\n[PLANNER CORRECTION] These steps have a "
                            "verify: command that passes as long as the "
                            "file parses, so it can never detect wrong "
                            "behaviour:\n" + _detail + _safety_note +
                            "\n\nRewrite EVERY one of those verify: lines "
                            "so it asserts a concrete value the step's "
                            "description promises (or runs a test suite). "
                            "Keep them single-line and runnable from the "
                            "project root. Re-emit the COMPLETE plan.")
                        continue
                    if _gate_gaps:
                        log.warning(
                            "[Plan] Proceeding after %d attempt(s) with %d "
                            "import-only gate(s) — these steps cannot fail "
                            "on wrong behaviour: %s",
                            MAX_PLAN_RETRIES, len(_gate_gaps),
                            ", ".join(sid for sid, _ in _gate_gaps))

                    # Backstop, on the accepted plan. Repair and re-plan
                    # both get their chance above and both can fail — and
                    # the branch immediately above deliberately proceeds
                    # with gates that are still imperfect. That is right
                    # for a weak gate and wrong for a destructive one:
                    # there is no attempt count after which running
                    # `taskkill /im python.exe /f` becomes acceptable.
                    for _sid, _was, _why in neutralize_destructive_gates(
                            plan_steps_parsed):
                        log.warning(
                            "[GateSafety] step %s: dropped the destructive "
                            "tail of its verify: %s — was `%s`",
                            _sid, _why, _was)

                    raw_steps = steps_as_text_list(plan_steps_parsed)
                else:
                    log.warning("[Plan] Structured parse returned 0 steps, falling back")

            if plan_steps_parsed is None:
                # Heuristic fallback: handles weaker LLMs that output markdown
                # headers with **Key:** value metadata instead of --STEP format.
                heuristic_steps = parse_heuristic_plan(plan)
                if heuristic_steps:
                    log.info(
                        f"[Plan] Heuristic parser extracted {len(heuristic_steps)} "
                        f"steps from non-standard format"
                    )
                    dep_fixes = fix_import_dependencies(heuristic_steps, read_file=project_file_reader)
                    if dep_fixes:
                        log.info(f"[Plan] Auto-fixed import dependencies: {dep_fixes}")
                    plan_steps_parsed = heuristic_steps
                    raw_steps = steps_as_text_list(plan_steps_parsed)

            if plan_steps_parsed is None:
                log.info("[Plan] Using legacy step parser (no structured plan)")
                raw_steps = executor.parse_plan_steps(plan)

            if not raw_steps:
                log.warning(f"Plan attempt {plan_attempt}: no steps parsed")
                if plan_attempt < MAX_PLAN_RETRIES:
                    continue
                log.error("Could not parse any steps from the plan.")
                print("\n  [ERROR] Could not parse any steps. Check the log file.\n")
                return

            # ── Truncation guard ──
            # A plan can parse cleanly yet be a stub: the planner ran out of
            # output tokens mid-plan (observed: a 6-step plan for a ~15-file
            # task, cut mid-step, that ran only because downstream recovery
            # backfilled the missing files). Detect it and re-plan for a
            # complete plan rather than shipping the stub silently. A
            # structured plan that DID close with ==END== is trusted even if
            # the cap flag fired (the cap likely hit trailing whitespace).
            _prov_truncated = getattr(
                getattr(planner, "llm_client", None), "_last_truncated", False)
            _struct_truncated, _trunc_reason = plan_looks_truncated(
                plan, plan_steps_parsed)
            _has_end = "==END==" in (plan or "")
            if _struct_truncated or (_prov_truncated and not _has_end):
                _reason = _trunc_reason or "the planner hit its output-token limit"
                # Salvage: only the ==END== marker is missing, the provider's
                # output cap did NOT fire, and every parsed step is
                # structurally complete — the plan is almost certainly whole.
                # Re-planning here costs a full second generation and churns
                # paths (an observed re-plan renamed the project directory).
                if (not _prov_truncated and "==END==" in _reason
                        and plan_salvageable(plan_steps_parsed)):
                    log.warning(
                        "[Plan] ==END== marker missing but all %d parsed "
                        "steps are structurally complete — salvaging plan "
                        "instead of re-planning",
                        len(plan_steps_parsed or []))
                elif plan_attempt < MAX_PLAN_RETRIES:
                    log.warning(
                        "[Plan] Plan looks truncated (%s) — re-planning for a "
                        "complete plan", _reason)
                    display.show_status(
                        "Plan was cut off — requesting a complete plan...")
                    planner_context += (
                        "\n\n[PLANNER CORRECTION] Your previous plan was CUT "
                        f"OFF ({_reason}) and is INCOMPLETE. Produce the "
                        "COMPLETE plan this time: keep each step description "
                        "terse, include a step for every file the requirements "
                        "name, and finish with the ==END== marker.")
                    continue
                else:
                    log.error(
                        "[Plan] Plan still truncated after %d attempts — "
                        "proceeding with a possibly-incomplete plan (%s)",
                        MAX_PLAN_RETRIES, _reason)
                    display.show_status(
                        "Warning: plan may be incomplete (planner output was "
                        "truncated).")

            # Validate plan quality — skip for structured plans, which are
            # already validated by validate_plan() above and whose step
            # descriptions don't populate the legacy text list reliably.
            if plan_steps_parsed is not None:
                break
            is_valid, reason = Executor.validate_plan_quality(raw_steps)
            if is_valid:
                break

            log.warning(f"Plan attempt {plan_attempt} rejected: {reason}")
            if plan_attempt < MAX_PLAN_RETRIES:
                display.show_status(f"Plan too vague ({reason}), retrying...")
            else:
                log.warning(f"Proceeding with low-quality plan after {MAX_PLAN_RETRIES} attempts")
                print(f"\n  [WARN] Plan quality is low ({reason}). You may want to replan or edit.\n")

        if plan_steps_parsed is not None:
            steps = steps_as_text_list(plan_steps_parsed)
            dependencies = steps_dependencies_dict(plan_steps_parsed)
        else:
            steps, dependencies = executor.parse_step_dependencies(raw_steps)

        # ── 10b. Post-plan optimization ──
        pre_opt_count = len(steps)
        if plan_steps_parsed is not None:
            # Structured path: optimize directly on PlanStep objects
            plan_steps_parsed = optimize_structured_plan(
                plan_steps_parsed, knowledge_base=knowledge_base,
                kb_context_builder=kb_context_builder,
                language=language)
            steps = steps_as_text_list(plan_steps_parsed)
            dependencies = steps_dependencies_dict(plan_steps_parsed)
        else:
            # Legacy path
            steps, dependencies = optimize_plan(
                steps, knowledge_base=knowledge_base,
                kb_context_builder=kb_context_builder,
                dependencies=dependencies,
                language=language)
            plan_steps_parsed = from_legacy_steps(steps, dependencies)
        if len(steps) < pre_opt_count:
            log.info(f"[Planning] Optimized: {pre_opt_count} → {len(steps)} steps")

        # Reclassify CODE steps targeting only protected dependency manifests
        # (package.json, requirements.txt, etc.) as CMD install steps.
        plan_steps_parsed = reclassify_manifest_steps(plan_steps_parsed)
        steps = steps_as_text_list(plan_steps_parsed)
        dependencies = steps_dependencies_dict(plan_steps_parsed)

        # ── 11. Plan approval loop ──
        if args.auto:
            log.info(f"Auto-approved {len(steps)} steps (--auto mode)")
        while not args.auto:
            display.pause()  # stop Rich Live so print()/input() are visible
            # Reattach dependency markers so they are visible and editable in TUI
            display_steps = []
            for i, step in enumerate(steps):
                if dependencies.get(i):
                    deps_str = ", ".join(str(d + 1) for d in sorted(dependencies[i]))
                    display_steps.append(f"{step} (depends: {deps_str})")
                else:
                    display_steps.append(f"{step} (depends: none)")

            # Try TUI editor first, fall back to text-based approval
            action, removed, edited_steps = CLIDisplay.prompt_plan_approval(
                display_steps, use_tui=True)
            if action == "approve":
                break
            elif action == "replan":
                display.resume()  # restart Live for spinner during replan
                display.show_status("Re-planning...")
                plan = planner.process(args.task, context=planner_context,
                                       language=language,
                                       plan_mode=getattr(cfg, "PLAN_MODE",
                                                         "content"))
                log.info(f"Re-plan:\n{plan}")

                if is_structured_plan(plan):
                    plan_steps_parsed = parse_structured_plan(plan)
                    if plan_steps_parsed:
                        dep_fixes = fix_import_dependencies(plan_steps_parsed, read_file=project_file_reader)
                        if dep_fixes:
                            log.info(f"[Plan] Auto-fixed import dependencies: {dep_fixes}")
                        raw_steps = steps_as_text_list(plan_steps_parsed)
                    else:
                        raw_steps = executor.parse_plan_steps(plan)
                        plan_steps_parsed = None
                else:
                    raw_steps = executor.parse_plan_steps(plan)
                    plan_steps_parsed = None

                if not raw_steps:
                    log.error("Could not parse any steps from re-plan.")
                    print("\n  [ERROR] Could not parse re-plan steps.\n")
                    return

                if plan_steps_parsed is not None:
                    steps = steps_as_text_list(plan_steps_parsed)
                    dependencies = steps_dependencies_dict(plan_steps_parsed)
                else:
                    steps, dependencies = executor.parse_step_dependencies(raw_steps)

                if plan_steps_parsed is not None:
                    plan_steps_parsed = optimize_structured_plan(
                        plan_steps_parsed, knowledge_base=knowledge_base,
                        kb_context_builder=kb_context_builder,
                        language=language)
                    steps = steps_as_text_list(plan_steps_parsed)
                    dependencies = steps_dependencies_dict(plan_steps_parsed)
                else:
                    steps, dependencies = optimize_plan(
                        steps, knowledge_base=knowledge_base,
                        kb_context_builder=kb_context_builder,
                        dependencies=dependencies,
                        language=language)
                    plan_steps_parsed = from_legacy_steps(steps, dependencies)
            elif action == "edit" and edited_steps:
                new_steps, new_deps = executor.parse_step_dependencies(edited_steps)
                # Preserve structured PlanStep metadata when possible
                if plan_steps_parsed and len(new_steps) == len(steps):
                    # Same number of steps — check if descriptions match
                    _old = [s.strip() for s in steps]
                    _new = [s.strip() for s in new_steps]
                    if _old == _new:
                        # No actual changes — keep structured metadata intact
                        log.info("[Plan] Edit returned unchanged steps, preserving structured metadata")
                        steps = new_steps
                        dependencies = new_deps
                    else:
                        # Steps changed — try to re-match by description overlap
                        steps = new_steps
                        dependencies = new_deps
                        plan_steps_parsed = _rematch_plan_steps(
                            steps, plan_steps_parsed, dependencies)
                else:
                    # Step count changed — still try to re-match by description
                    # to preserve structured metadata (type, command, target_files)
                    steps = new_steps
                    dependencies = new_deps
                    if plan_steps_parsed:
                        plan_steps_parsed = _rematch_plan_steps(
                            steps, plan_steps_parsed, dependencies)
                    else:
                        plan_steps_parsed = from_legacy_steps(steps, dependencies)

        display.resume()  # restart Live after approval loop exits
        display.set_steps(steps)
        display.render()
        log.info(f"Approved {len(steps)} steps.")

        memory = FileMemory(embedding_store=embed_store, top_k=cfg.EMBEDDING_TOP_K)
        if kb_runtime_watcher is not None:
            memory.watcher_created_files = kb_runtime_watcher.created_files
        # Raw-task test-request flag, computed before enrichment (see above).
        # Gates unsolicited auto-coverage generation in the pipeline.
        memory._task_requests_tests = getattr(
            args, '_raw_task_requests_tests', True)
        # Propagate task briefing to memory so all downstream agents can use it
        _briefing_text = getattr(planner, '_task_briefing', '')
        if _briefing_text:
            memory._task_briefing = _briefing_text

        # Pre-load existing source files into memory so the coder
        # can see and modify them instead of creating new files
        if source_files:
            memory.update(source_files)
            log.info(f"Pre-loaded {len(source_files)} source files into memory")

    # ── 11b. Project analysis phase ──
    # Build structured ProjectContext from static analysis + LLM enrichment.
    # Gives Coder and Tester awareness of end-to-end goal, installed packages,
    # import patterns, and test strategy.
    # On resume, reuse the saved ProjectContext instead of calling LLM again.
    _resumed_pc = locals().get('_resumed_project_context')
    if resuming and _resumed_pc is not None:
        # Checkpoint has full enriched ProjectContext — reuse it
        project_context = _resumed_pc
        log.info(
            "[Analysis] Reusing ProjectContext from checkpoint (0 LLM tokens): "
            "lang=%s, fw=%s, test_fw=%s, %d pkgs, %d testable units",
            project_context.language, project_context.framework,
            project_context.test_framework,
            len(project_context.installed_packages),
            len(project_context.testable_units),
        )
    elif resuming:
        # Old checkpoint without ProjectContext — use static analysis only (0 LLM tokens)
        project_context = build_project_context(
            args.task, steps,
            source_files=source_files or {},
            language=language,
        )
        log.info(
            "[Analysis] Resume: static analysis only, skipping LLM enrichment "
            "(0 LLM tokens): lang=%s",
            project_context.language,
        )
    else:
        project_context = build_project_context(
            args.task, steps,
            source_files=source_files or {},
            language=language,
        )
        if cfg.ANALYSER_ENABLED:
            display.show_status("Analysing project...")
            try:
                analyser = AnalyseAgent(
                    "Analyser", "Senior Technical Analyst",
                    "Analyse the task and project to guide downstream agents.",
                    _make_llm_for_agent("analyser"))
                project_context = analyser.enrich_context(
                    project_context, args.task, steps, source_files or {})
                log.info(
                    "[Analysis] ProjectContext: lang=%s, fw=%s, test_fw=%s, "
                    "%d pkgs, %d testable units",
                    project_context.language, project_context.framework,
                    project_context.test_framework,
                    len(project_context.installed_packages),
                    len(project_context.testable_units),
                )
            except Exception as analyse_exc:
                log.warning("[Analysis] LLM enrichment failed (non-fatal): %s",
                            analyse_exc)
        else:
            log.info("[Analysis] LLM enrichment skipped (analyser_enabled: false)")

    # Inject packages from the task briefing's "New packages:" line so that
    # _ensure_packages_installed installs them before the first CODE step —
    # even when the plan has no explicit CMD install step.
    for _pkg in parse_briefing_packages(getattr(memory, '_task_briefing', '')):
        if _pkg not in project_context.required_packages:
            project_context.required_packages.append(_pkg)
            log.info("[PreAnalysis] Briefing package injected: %s", _pkg)

    # ── 12. Build execution waves ──
    # Use phase-aware wave builder when structured plan steps are available.
    # This ensures all sub-steps of phase N (e.g. 1.1, 1.2) complete before
    # phase N+1 (e.g. 2.1, 2.2) begins, even when explicit depends: is missing.
    if plan_steps_parsed:
        plan_waves = _build_plan_waves(plan_steps_parsed)
        waves = [[s.index for s in w] for w in plan_waves]
    else:
        waves = build_step_waves(steps, dependencies)
    log.info(f"Execution waves: {waves}")

    # Build step reports for HTML output
    step_reports = [StepReport(index=i, text=steps[i]) for i in range(len(steps))]

    # ── 13. Execute waves ──
    # The agent-loop attempt journal is keyed by step index, so a second
    # run in the same process (library API, tests) would otherwise show a
    # step the previous run's attempts as if they were its own.
    from .agent_loop import reset_attempt_journal
    reset_attempt_journal()

    # Graph of what the plan promises to build. Nodes start as `planned`
    # (nothing exists yet on a blank project, which is exactly why import
    # resolution cannot use the KB code graph) and flip to `built` as
    # steps complete, so declared exports can be checked against what was
    # actually produced.
    from .plan_graph import PlanGraph
    plan_graph = PlanGraph(plan_steps_parsed or [])
    _unresolved = plan_graph.unresolved_imports(plan_steps_parsed or [])
    if _unresolved:
        # Usually third-party or pre-existing files — informational only.
        log.debug("[PlanGraph] %d import(s) not produced by any step: %s",
                  len(_unresolved),
                  ", ".join(f"{sid}->{spec}" for sid, spec in _unresolved[:8]))

    # ── Ghost: snapshot the plan's declared postconditions + pre-state ──
    # Read-only shadow (orchestrator/ghost.py). Built HERE because the plan
    # is final by this point — blind-edit routing, dependency fixes and
    # verify repair have all run — and because no step has executed yet, so
    # the file hashes it records are a true pre-run baseline. It never
    # writes, never runs a command and never changes a verdict.
    from .ghost import start_ghost, reset_ghost
    from .ghost_heal import start_healer, reset_healer
    reset_ghost()
    reset_healer()

    # Which test files predate the run? Taken here, alongside the ghost's
    # pre-state and before the first step, because the question at the end
    # is not "is there a suite" but "is there a suite the agent did not
    # write". See orchestrator/evidence.py.
    # A greenfield build has no pre-existing suite, so it is judged by
    # tests it wrote itself — and three measured runs shipped exit 0 over
    # artifacts that failed every external probe while their own tests
    # were green. Seeding one from the TASK, here, before any step has
    # run, is the only moment a check can be written that the code cannot
    # have shaped. It is snapshotted below like any other pre-existing
    # file, so rewriting it forfeits independence and says so.
    if getattr(cfg, "SEED_ACCEPTANCE_TESTS", True):
        try:
            from .acceptance_seed import seed_acceptance_tests
            # The enriched task is what the suite should be WRITTEN from
            # — it is the fuller statement of the requirement — but the
            # raw task is what decides whether an existing contract was
            # written for this same task, because only the raw text is
            # stable across runs. See `_raw_task` above.
            seed_acceptance_tests(args.task, os.getcwd(), llm_client,
                                  language=language,
                                  identity_task=getattr(args, "_raw_task",
                                                        None))
        except Exception as _seed_exc:      # never fail a run over this
            # WARNING, not DEBUG. Losing the run's only independent check
            # is not a detail, and a silent skip is how the first live
            # attempt at this shipped a NameError that nothing noticed.
            log.warning("[AcceptanceSeed] skipped (%s: %s) — this run has "
                        "no seeded independent evidence",
                        type(_seed_exc).__name__, _seed_exc)

    from .evidence import (acceptance_instrument_files as _acc_files,
                           snapshot_test_files as _snap_tests)
    # Files the acceptance commands invoke are read-only to the agent for
    # the rest of the run — see AgentTools._acceptance_refusal.
    _acc_instruments = _acc_files(
        getattr(cfg, "ACCEPTANCE_CMDS", []) or [], os.getcwd())
    if _acc_instruments:
        memory._acceptance_files = _acc_instruments
        log.info("[Evidence] %d acceptance instrument(s) protected from "
                 "agent writes: %s", len(_acc_instruments),
                 ", ".join(sorted(_acc_instruments)))
    _pre_existing_tests = _snap_tests(os.getcwd())
    if _pre_existing_tests:
        log.info(f"[Evidence] {len(_pre_existing_tests)} pre-existing test "
                 f"file(s) recorded as independent evidence candidates")

    # `require_independent_evidence` can be UNSATISFIABLE before the first
    # step runs, and saying so now is the whole point. Independent evidence
    # is exactly three things — user `acceptance_cmds`, a pre-existing test
    # file the run leaves alone, or a contract the seeder wrote — and the
    # seeder is Python-only (`SEED_BASENAME` is a .py, `evidence` filters
    # `.py`, `seed_strength` is an AST analysis). A greenfield JavaScript
    # build therefore has none of the three, no matter how well it goes.
    #
    # Measured 2026-08-19: a run executed all 20 steps, passed every gate,
    # built clean and ran its suite green, then failed on the last line
    # with "nothing outside this run's own output verified it" — 691k
    # tokens to reach a verdict that was already decided at startup. The
    # message reads like the model's fault; it is a configuration that
    # cannot succeed. Warned rather than refused: the run's artifacts are
    # still worth having, and the user may add `acceptance_cmds` and
    # resume from the checkpoint.
    if getattr(cfg, "REQUIRE_INDEPENDENT_EVIDENCE", False):
        _have_acc = bool(getattr(cfg, "ACCEPTANCE_CMDS", []) or [])
        _seedable = not language or language.lower() in ("python", "py")
        if not (_have_acc or _pre_existing_tests or _seedable):
            log.warning(
                "[Evidence] require_independent_evidence is set, but nothing "
                "can satisfy it in this run: no `acceptance_cmds` are "
                "configured, no pre-existing test file was found, and the "
                "acceptance seeder does not support %s (Python only). The "
                "run will do its work and then exit non-zero regardless of "
                "the result. Add `acceptance_cmds:` to .agentchanti.yaml — "
                "a command the agent cannot edit — or unset "
                "`require_independent_evidence`.", language)

    # The top-level directories the final plan names as target roots. A
    # sub-project root is otherwise a claim about the ONE tree the run
    # builds, and everything downstream re-roots into it — CMD cwd, gate
    # `cd` prefixes, unprefixed write paths. This set is what lets a
    # multi-root plan ("React frontend + Express backend") say that
    # `backend/` is a sibling of `frontend/`, not a directory inside it.
    if plan_steps_parsed:
        _declared_roots: set[str] = set()
        for _ps in plan_steps_parsed:
            for _t in getattr(_ps, "target_files", None) or []:
                _norm = (_t or "").replace("\\", "/").lstrip("/")
                while _norm.startswith("./"):
                    _norm = _norm[2:]
                if "/" in _norm:
                    _head = _norm.split("/")[0]
                    if _head and _head not in (".", ".."):
                        _declared_roots.add(_head)
        memory._plan_declared_roots = _declared_roots
        # Every declared target, not just its leading directory. A write
        # the plan asked for is planned however unusual it looks, which
        # is what keeps `phantom_root_manifest_reason` from refusing a
        # legitimate workspaces root.
        memory._plan_declared_files = {
            (_t or "").replace(chr(92), "/").lstrip("/").lstrip("./")
            for _ps in plan_steps_parsed
            for _t in (getattr(_ps, "target_files", None) or [])
        }
        if len(_declared_roots) > 1:
            log.info("[Plan] Declares %d target root(s): %s",
                     len(_declared_roots), ", ".join(sorted(_declared_roots)))

    if getattr(cfg, "GHOST_SHADOW", True) and plan_steps_parsed:
        # On a resume, the steps below `start_from` finished in an earlier
        # run, against a tree the pre-state snapshot below cannot see —
        # so every one of their postconditions would read as "the step
        # changed nothing". They are named here so the ghost declines to
        # judge them rather than reporting ten false violations.
        _carried = [s.id for s in plan_steps_parsed
                    if getattr(s, "index", -1) < start_from]
        _ghost = start_ghost(plan_steps_parsed, os.getcwd(), _carried)
        # Healing is what turns detection into value: the shadow saw a
        # dependency missing from the app's interpreter at wave 2 of a
        # real run and said nothing more, while the smoke test crashed on
        # it six waves later. Repairs are mechanical and verified — see
        # ghost_heal for the state-vs-content rule that bounds them.
        if _ghost is not None and getattr(cfg, "GHOST_HEAL", True):
            start_healer(_ghost, executor,
                         allow_source_edits=getattr(
                             cfg, "GHOST_HEAL_SOURCE_EDITS", True))

    # Clear any lingering planning/analysis status message before execution
    # starts. Without this, "Requesting steps from planner...", "Analysing
    # project...", etc. stay pinned to the STATUS panel for the entire run
    # because nothing inside _execute_step touches show_status. The wiring
    # verification phase sets/clears its own status independently.
    display.show_status("")
    pipeline_success = True

    # Per-wave snapshots + monotonic gate ledger: every green wave is a
    # git commit in the (machine-managed) workdir repo, and every gate
    # that passes is recorded so later fix rounds can be checked for
    # regressions and rolled back instead of shipped.
    from .wave_snapshots import ProjectSnapshots, get_gate_ledger
    get_gate_ledger().reset()
    snapshots = ProjectSnapshots(
        os.getcwd(), enabled=getattr(cfg, "WAVE_SNAPSHOTS", True))
    snapshots.start()

    for wave_idx, wave in enumerate(waves):
        # Filter out already-completed steps (for resume)
        pending = [i for i in wave if i >= start_from]
        if not pending:
            continue

        log.info(f"Wave {wave_idx+1}: executing steps {[i+1 for i in pending]}")
        # `splitlines()[0]` on an empty description raises, and a status
        # line must never be able to kill a run. Observed: a 20B model in
        # content mode emitted 7 of 9 steps as target+content with no
        # prose at all, and the pipeline crashed with IndexError at the
        # start of wave 2 — after wave 1 had already succeeded.
        set_activity(
            f"wave {wave_idx+1}/{len(waves)} steps {[i+1 for i in pending]}: "
            + "; ".join((steps[i].splitlines() or ["(no description)"])[0][:60]
                        for i in pending)
        )

        if len(pending) == 1:
            # Single step — execute directly
            idx = pending[0]
            step_text = steps[idx]
            _ps = next((s for s in plan_steps_parsed if s.index == idx), None) if plan_steps_parsed else None
            if _ps is None and plan_steps_parsed:
                log.warning(
                    "[PlanStep] No PlanStep found for idx=%d. "
                    "Available indices: %s",
                    idx, [s.index for s in plan_steps_parsed],
                )
            idx, success, error_info = _execute_step(
                idx, step_text,
                steps=steps,
                llm_client=llm_client, executor=executor,
                coder=coder, reviewer=reviewer, tester=tester,
                task=args.task, memory=memory, display=display,
                language=language, cfg=cfg, auto=args.auto,
                search_agent=search_agent,
                kb_context_builder=kb_context_builder,
                knowledge_base=knowledge_base,
                project_context=project_context,
                plan_step=_ps,
                all_plan_steps=plan_steps_parsed,
                intent_spec=intent_spec,
            )

            if success:
                step_results[idx] = "done"
                ds = {"elapsed": time.monotonic() - display.start_time, "steps": display.steps}
                save_checkpoint(checkpoint_file, args.task, steps, idx,
                                memory.as_dict(), step_results, language,
                                display_state=ds,
                                plan_steps=plan_steps_parsed,
                                project_context=project_context)

                # Budget check after step
                if display.budget_check(cfg.BUDGET_LIMIT):
                    log.error(f"Budget exceeded (${token_tracker.total_cost:.4f}). Halting.")
                    pipeline_success = False
                    break
            else:
                # Diagnosis loop
                fixed = _run_diagnosis_loop(
                    idx, step_text, error_info,
                    steps=steps,
                    llm_client=llm_client, executor=executor,
                    coder=coder, reviewer=reviewer, tester=tester,
                    task=args.task, memory=memory, display=display,
                    language=language, cfg=cfg, auto=args.auto,
                    search_agent=search_agent,
                    kb_context_builder=kb_context_builder,
                    knowledge_base=knowledge_base,
                    project_context=project_context,
                    plan_step=_ps,
                    all_plan_steps=plan_steps_parsed,
                    intent_spec=intent_spec,
                )
                if fixed:
                    display.complete_step(idx, "done")
                    step_results[idx] = "done"
                    ds = {"elapsed": time.monotonic() - display.start_time, "steps": display.steps}
                    save_checkpoint(checkpoint_file, args.task, steps, idx,
                                    memory.as_dict(), step_results, language,
                                    display_state=ds,
                                    plan_steps=plan_steps_parsed,
                                project_context=project_context)

                    # Budget check after fix
                    if display.budget_check(cfg.BUDGET_LIMIT):
                        log.error(f"Budget exceeded (${token_tracker.total_cost:.4f}). Halting.")
                        pipeline_success = False
                        break
                else:
                    pipeline_success = False
                    break
        else:
            # Multi-step wave — execute in parallel
            failed_steps: list[tuple[int, str]] = []

            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=min(len(pending), 4)) as pool:
                futures = {}
                for idx in pending:
                    _ps = next((s for s in plan_steps_parsed if s.index == idx), None) if plan_steps_parsed else None
                    if _ps is None and plan_steps_parsed:
                        log.warning(
                            "[PlanStep] No PlanStep found for idx=%d. "
                            "Available indices: %s",
                            idx, [s.index for s in plan_steps_parsed],
                        )
                    f = pool.submit(
                        _execute_step, idx, steps[idx],
                        steps=steps,
                        llm_client=llm_client, executor=executor,
                        coder=coder, reviewer=reviewer, tester=tester,
                        task=args.task, memory=memory, display=display,
                        language=language, cfg=cfg, auto=args.auto,
                        search_agent=search_agent,
                        kb_context_builder=kb_context_builder,
                        knowledge_base=knowledge_base,
                        project_context=project_context,
                        plan_step=_ps,
                        all_plan_steps=plan_steps_parsed,
                        intent_spec=intent_spec,
                    )
                    futures[f] = idx

                for future in as_completed(futures):
                    idx, success, error_info = future.result()
                    if success:
                        step_results[idx] = "done"
                    else:
                        failed_steps.append((idx, error_info))

                # Budget check after wave
                if display.budget_check(cfg.BUDGET_LIMIT):
                    log.error(f"Budget exceeded (${token_tracker.total_cost:.4f}) after parallel wave. Halting.")
                    pipeline_success = False
                    break

            # Save checkpoint for completed steps
            max_completed = max(
                (i for i in step_results if step_results[i] == "done"),
                default=start_from - 1)
            ds = {"elapsed": time.monotonic() - display.start_time, "steps": display.steps}
            save_checkpoint(checkpoint_file, args.task, steps, max_completed,
                            memory.as_dict(), step_results, language,
                            display_state=ds,
                            plan_steps=plan_steps_parsed)

            # Handle failures
            for idx, error_info in failed_steps:
                step_text = steps[idx]
                _ps = next((s for s in plan_steps_parsed if s.index == idx), None) if plan_steps_parsed else None
                fixed = _run_diagnosis_loop(
                    idx, step_text, error_info,
                    steps=steps,
                    llm_client=llm_client, executor=executor,
                    coder=coder, reviewer=reviewer, tester=tester,
                    task=args.task, memory=memory, display=display,
                    language=language, cfg=cfg, auto=args.auto,
                    search_agent=search_agent,
                    kb_context_builder=kb_context_builder,
                    knowledge_base=knowledge_base,
                    project_context=project_context,
                    plan_step=_ps,
                    all_plan_steps=plan_steps_parsed,
                    intent_spec=intent_spec,
                )
                if fixed:
                    display.complete_step(idx, "done")
                    step_results[idx] = "done"
                    ds = {"elapsed": time.monotonic() - display.start_time, "steps": display.steps}
                    save_checkpoint(checkpoint_file, args.task, steps, idx,
                                    memory.as_dict(), step_results, language,
                                    display_state=ds,
                                    plan_steps=plan_steps_parsed,
                                project_context=project_context)

                    # Budget check after fix
                    if display.budget_check(cfg.BUDGET_LIMIT):
                        log.error(f"Budget exceeded (${token_tracker.total_cost:.4f}). Halting.")
                        pipeline_success = False
                        break
                else:
                    pipeline_success = False
                    break

            if not pipeline_success:
                break

        # Flip this wave's plan-graph nodes to `built` and check the
        # exports they promised against what the files actually define. A
        # step that silently drops a declared export is a defect every
        # downstream importer will hit, and it is far cheaper to name here
        # than to debug from the resulting ImportError several waves later.
        _reconcile_plan_graph(plan_graph, plan_steps_parsed, pending,
                              step_results, memory, language)

        # Same moment, shadow copy: resolve this wave's declared
        # postconditions against the real tree. Observations accumulate in
        # the ghost's journal; nothing is reported or acted on until the
        # end of the run.
        _ghost_resolve_wave(plan_steps_parsed, pending, step_results,
                            language, f"wave {wave_idx + 1}")

        # Snapshot the wave, then re-run every gate recorded so far. A step
        # can break a sibling's already-green gate — including one recorded
        # in this same wave, since wave steps run in parallel — so the
        # recheck covers all gates, not just those from earlier waves.
        # Only a wave that leaves every gate green becomes a rollback target;
        # without this, HEAD (and so the rollback target) silently advances
        # onto the commit that introduced the regression.
        if not _enforce_monotonic_gates(
                snapshots, executor, f"wave {wave_idx + 1}", display=display):
            pipeline_success = False
            break

    # ── 13.5. Bulk test execution + per-file fix ──
    # All TEST steps with inline code deferred their runs until now so that:
    #   • parallel wave steps don't race to run the full suite simultaneously
    #   • source fixes for one test can't break another before it's verified
    # Run all test files once; fix failing ones one at a time; final run-all.
    from ..cli_display import set_status
    verif_ok = False
    if pipeline_success:
        from .pipeline import run_bulk_test_execution_and_fix
        set_status(display, "Running the full test suite...")
        verif_ok, verif_err = run_bulk_test_execution_and_fix(
            memory=memory,
            executor=executor,
            coder=coder,
            display=display,
            language=language,
            task=args.task,
            cfg=cfg,
            project_context=project_context,
            kb_context_builder=kb_context_builder,
            all_plan_steps=plan_steps_parsed,
            search_agent=search_agent,
        )
        if not verif_ok:
            pipeline_success = False
            log.warning(f"[BulkTest] Pipeline marked failed: {verif_err[:200]}")

        # ── Monotonic-progress check ──
        # Fix rounds may touch source files; a fix that turns a
        # previously-green per-step gate red is a regression. Re-run the
        # recorded gates and roll the workdir back to the last green
        # snapshot rather than shipping the regression.
        if not _enforce_monotonic_gates(snapshots, executor, "bulk-test fixes",
                                        display=display):
            pipeline_success = False
            verif_ok = False

    # ── 13.6. Wiring verification ──
    # One LLM call that checks all fix-scope files together for cross-file
    # integration issues (entry-point mounts, import/export mismatches, etc.).
    # Skipped when the bulk test run just executed real tests and they all
    # passed — see should_run_wiring_verification() for the full rationale.
    from .pipeline import should_run_wiring_verification
    _run_wiring = should_run_wiring_verification(
        memory,
        pipeline_success=pipeline_success,
        bulk_test_verif_ok=verif_ok,
        wiring_enabled=cfg.WIRING_VERIFICATION_ENABLED,
    )
    if not _run_wiring and pipeline_success and cfg.WIRING_VERIFICATION_ENABLED:
        log.info(
            "[WiringVerification] Skipped — bulk tests just ran and "
            "passed; wiring is implicitly verified."
        )
    if _run_wiring:
        import os as _os
        wv_ok, wv_err = run_wiring_verification(
            memory=memory,
            executor=executor,
            coder=coder,
            display=display,
            task=args.task,
            language=language,
            cfg=cfg,
            kb_context_builder=kb_context_builder,
            project_root=_os.getcwd(),
        )
        if not wv_ok:
            log.warning(f"[WiringVerification] Fix failed: {wv_err[:200]}")

    # ── 13.7. Runtime smoke verification ──
    # Tests can pass while the app crashes at launch (GUI apps especially —
    # tests mock the graphics library and never render a frame).  Launch the
    # entry point briefly and feed any crash traceback into a bounded fix
    # loop.  Skips silently when there is no runnable entry point.
    if pipeline_success:
        from .smoke_test import run_smoke_verification
        set_status(display, "Launching the app to check it starts...")
        smoke_ok, smoke_err = run_smoke_verification(
            memory=memory,
            executor=executor,
            coder=coder,
            display=display,
            task=args.task,
            language=language,
            cfg=cfg,
        )
        if not smoke_ok:
            pipeline_success = False
            log.warning(f"[SmokeTest] Pipeline marked failed: {smoke_err[:300]}")

        # An adversarial property check used to run here — one bounded loop
        # authoring a randomised-dt invariant test, aimed at the class of
        # defect a fixed timestep hides (a Pac-Man run shipped with ghosts
        # walking through walls while every gate was green). It was removed:
        # over three runs it never once produced its test file, cost ~33%
        # of a run's tokens (147k sent / 10k received on the last one),
        # falsely failed a pipeline whose gates were all green, and
        # overwrote a verified test file from an earlier wave. The idea is
        # sound; a loop that authors the test is not the way to get it.
        # See git history for the implementation.
        else:
            # The smoke-test repair loop rewrites source files to fix a
            # launch crash, and it is the LAST thing that can do so.
            # Those edits were
            # previously never gate-checked: a repair that changed
            # `Player.update()`'s signature turned the green
            # `python -m unittest discover` gate red, and the run still
            # printed `Finished`. Re-check here so success always means
            # every gate is green — this is the final word on the run.
            #
            # A red gate here usually means the repair was RIGHT and a
            # test still encodes the old API (observed: a test stubbing
            # `update(self, game_map)` after the real signature changed),
            # so give the test-fix machinery one round to catch the tests
            # up rather than reflexively discarding a fix that made the
            # app run.
            from .pipeline import run_bulk_test_execution_and_fix as _btf

            def _repair_tests_after_smoke():
                _btf(
                    memory=memory,
                    executor=executor,
                    coder=coder,
                    display=display,
                    language=language,
                    task=args.task,
                    cfg=cfg,
                    project_context=project_context,
                    kb_context_builder=kb_context_builder,
                    all_plan_steps=plan_steps_parsed,
                    search_agent=search_agent,
                )

            if not _enforce_monotonic_gates(
                    snapshots, executor, "smoke-test fixes",
                    repair=_repair_tests_after_smoke, display=display):
                pipeline_success = False
                log.warning(
                    "[SmokeTest] The launch fix left a previously-passing "
                    "gate red — reporting failure rather than shipping it.")

    # ── 14. Populate step reports from display state ──
    for i, sr in enumerate(step_reports):
        if i < len(display.steps):
            ds = display.steps[i]
            sr.status = ds.get("status", sr.status)
            sr.step_type = ds.get("type", sr.step_type)
            tokens = ds.get("tokens", {})
            sr.tokens_sent = tokens.get("sent", 0)
            sr.tokens_recv = tokens.get("recv", 0)
            sr.duration = ds.get("duration", 0.0)

    # ── 15. Extract knowledge (runs on both success and failure) ──
    # Patterns/fixes from completed steps are valuable regardless of
    # overall pipeline outcome — especially fixes learned from failures.
    if knowledge_base:
        set_status(display, "Extracting learnings from this run...")
        try:
            knowledge_base.extract_from_run(
                args.task, steps, memory.as_dict(), llm_client)
        except Exception as e:
            log.warning(f"Knowledge extraction failed: {e}")
    # Every post-wave stage is done; clear the footer so the finish screen
    # is not printed under a stale "still working" message.
    set_status(display, "")

    # ── 16. Finish ──
    _cached = token_tracker.total_cached_tokens
    _sent_breakdown = f"sent={token_tracker.total_prompt_tokens}"
    if _cached > 0:
        # Show gross vs. cached-net so the prompt-cache discount is
        # visible: of the tokens sent, how many were cache hits (billed
        # at a discount) vs. full-price.
        _pct = _cached * 100 // max(1, token_tracker.total_prompt_tokens)
        _sent_breakdown += (
            f" [cached={_cached} ({_pct}%), "
            f"full-price={token_tracker.full_price_prompt_tokens}]")
    # Built before the branch: a FAILED run is the expensive one, and it
    # used to report a bare total with no send/cache breakdown — exactly
    # the number needed to tell a cheap failure from a runaway one.
    _token_line = (f"Total tokens: {token_tracker.total_tokens} "
                   f"({_sent_breakdown}, "
                   f"recv={token_tracker.total_completion_tokens})")

    # Shadow reconciliation, on both the success and failure paths — the
    # failed run is the one whose disagreements are most worth reading.
    _ghost_final_report(plan_steps_parsed, step_results, memory, language,
                        pipeline_success)

    # ── Whose evidence is this? ──────────────────────────────────────
    # The user's acceptance commands are the only instrument here the
    # model neither wrote nor can edit, so they are the only ones allowed
    # to fail the run outright. Everything else feeds a second verdict —
    # verified vs merely completed — which changes what the run may
    # CLAIM without inventing a failure it cannot prove.
    from .evidence import classify as _classify_evidence
    from .evidence import run_acceptance_commands as _run_acceptance
    _acceptance_cmds = list(getattr(cfg, "ACCEPTANCE_CMDS", []) or [])
    _acceptance_passed = None
    if pipeline_success and _acceptance_cmds:
        _acceptance_passed, _acc_failures = _run_acceptance(
            executor, _acceptance_cmds)
        if _acceptance_passed is False:
            pipeline_success = False
            log.error("Pipeline failed: user acceptance command(s) did not "
                      "pass — " + "; ".join(_acc_failures[:3]))

    # Run the surviving pre-existing tests before claiming they passed.
    # They are the run's only independent instrument, and until this was
    # added the claim rested on `tests_ran` — a flag about the pipeline's
    # OWN tests. Two measured runs reported them as passing while every
    # test in them errored.
    from .evidence import (run_pre_existing_tests as _run_survivors,
                           surviving_pre_existing_tests as _survivors)
    _surv = _survivors(os.getcwd(), _pre_existing_tests)
    _surv_passed, _surv_detail = (None, "")
    if _surv:
        _surv_passed, _surv_detail = _run_survivors(executor, os.getcwd(),
                                                    _surv)
        log.info("[Evidence] pre-existing suite(s) %s — %s",
                 {True: "PASSED", False: "FAILED"}.get(_surv_passed,
                                                       "could not be run"),
                 _surv_detail)

    _evidence = _classify_evidence(
        os.getcwd(), _pre_existing_tests,
        tests_ran=bool(verif_ok) or any(
            getattr(s, "step_type", "") == "TEST" for s in plan_steps_parsed),
        acceptance_passed=_acceptance_passed,
        acceptance_cmds=_acceptance_cmds,
        survivors_passed=_surv_passed,
        survivors_detail=_surv_detail)
    log.info(_evidence.log_line())

    if (pipeline_success and not _evidence.independent
            and getattr(cfg, "REQUIRE_INDEPENDENT_EVIDENCE", False)):
        log.error("Pipeline failed: require_independent_evidence is set and "
                  "nothing outside this run's own output verified it")
        pipeline_success = False

    if pipeline_success:
        display.finish(success=True, evidence=_evidence)
        clear_checkpoint(checkpoint_file)
        from .agent_loop import loop_stats_summary as _als_fn
        _als = _als_fn()
        if _als:
            log.info(_als)
        log.info(f"Finished. {_token_line}")

        # Generate HTML report
        if args.report and not args.no_report:
            try:
                token_usage = {
                    "sent": token_tracker.total_prompt_tokens,
                    "recv": token_tracker.total_completion_tokens,
                    "total": token_tracker.total_tokens,
                    "cost": token_tracker.total_cost,
                    "total_time": time.monotonic() - display.start_time,
                }
                report_path = generate_html_report(
                    args.task, step_reports, token_usage,
                    pipeline_success=True, output_dir=cfg.REPORT_DIR)
                log.info(f"Report generated: {report_path}")
                print(f"\n  📄 Report: {report_path}")
            except Exception as e:
                log.warning(f"Report generation failed: {e}")

        # Git: offer commit
        if use_git and git_utils.has_changes():
            if args.auto:
                git_choice = "commit"
                log.info("Auto-committing changes (--auto mode)")
            else:
                display.stop_spinner()
                git_choice = CLIDisplay.prompt_git_action("complete")
            if git_choice == "commit":
                ok, msg = git_utils.commit_changes(
                    f"AgentChanti: {args.task[:60]}")
                print(f"  {'Committed!' if ok else 'Commit failed: ' + msg}")
            if checkpoint_branch:
                git_utils.delete_checkpoint_branch(checkpoint_branch)
    else:
        display.finish(success=False)
        from .agent_loop import loop_stats_summary as _als_fail_fn
        _als_fail = _als_fail_fn()
        if _als_fail:
            log.info(_als_fail)
        log.info(f"Pipeline failed. {_token_line}")

        # Generate HTML report even on failure
        if args.report and not args.no_report:
            try:
                token_usage = {
                    "sent": token_tracker.total_prompt_tokens,
                    "recv": token_tracker.total_completion_tokens,
                    "total": token_tracker.total_tokens,
                    "cost": token_tracker.total_cost,
                    "total_time": time.monotonic() - display.start_time,
                }
                report_path = generate_html_report(
                    args.task, step_reports, token_usage,
                    pipeline_success=False, output_dir=cfg.REPORT_DIR)
                log.info(f"Report generated: {report_path}")
                print(f"\n  📄 Report: {report_path}")
            except Exception as e:
                log.warning(f"Report generation failed: {e}")

        # Git: offer rollback
        if use_git and checkpoint_branch:
            if args.auto:
                git_choice = "skip"
                log.info("Auto-skipping git rollback (--auto mode)")
            else:
                display.stop_spinner()
                git_choice = CLIDisplay.prompt_git_action("failed")
            if git_choice == "rollback":
                ok, msg = git_utils.rollback_to_branch(checkpoint_branch)
                print(f"  {'Rolled back!' if ok else 'Rollback failed: ' + msg}")
            elif git_choice == "commit":
                ok, msg = git_utils.commit_changes(
                    f"AgentChanti (partial): {args.task[:50]}")
                print(f"  {'Committed!' if ok else 'Commit failed: ' + msg}")

    # ── 15. Cleanup ──
    if kb_runtime_watcher is not None:
        try:
            kb_runtime_watcher.stop()
        except Exception:
            pass
    executor.cleanup()

    # A halted pipeline must not look like a success to the shell. This
    # branch logged "Pipeline failed", wrote the report, and fell through
    # returning None — so the process exited 0 and anything reading $? (CI,
    # a `&&` chain, a benchmark harness) recorded a run that stopped at step
    # 11 of 12, having never written its tests, as a pass. Returned rather
    # than raised so the cleanup above always runs first.
    return 0 if pipeline_success else 1


if __name__ == "__main__":
    main()
