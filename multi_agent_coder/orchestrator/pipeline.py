"""
Pipeline execution — wave-based parallel/sequential step execution.
"""

import logging
import re

from concurrent.futures import ThreadPoolExecutor, as_completed

from ..cli_display import CLIDisplay, log

from .memory import FileMemory
from .classification import _classify_step
from .step_handlers import (
    _handle_cmd_step, _handle_code_step, _handle_test_step,
    MAX_STEP_RETRIES,
)
from .diagnosis import _diagnose_failure, _apply_fix

_logger = logging.getLogger(__name__)


MAX_DIAGNOSIS_RETRIES = 2   # outer retries: diagnose failure → fix → re-run step

# ── External service dependency detection ─────────────────────
# Patterns that indicate the command failed because an external
# service (database, cache, message broker, etc.) is unavailable.
# These failures cannot be fixed by the agent — the user must
# ensure the service is running.

_EXTERNAL_SERVICE_PATTERNS: list[tuple[str, str]] = [
    # MongoDB
    (r'MongoServerSelectionError|MongoNetworkError|ECONNREFUSED.*27017',
     'MongoDB (default port 27017)'),
    # PostgreSQL
    (r'ECONNREFUSED.*5432|could not connect to server.*5432|pg_hba\.conf|'
     r'SequelizeConnectionRefusedError.*5432',
     'PostgreSQL (default port 5432)'),
    # MySQL / MariaDB
    (r'ECONNREFUSED.*3306|ER_ACCESS_DENIED_ERROR|PROTOCOL_CONNECTION_LOST.*3306',
     'MySQL/MariaDB (default port 3306)'),
    # Redis
    (r'ECONNREFUSED.*6379|Redis connection.*failed|NOAUTH',
     'Redis (default port 6379)'),
    # RabbitMQ
    (r'ECONNREFUSED.*5672|amqp.*connection.*refused',
     'RabbitMQ (default port 5672)'),
    # Elasticsearch
    (r'ECONNREFUSED.*9200|ConnectionError.*9200',
     'Elasticsearch (default port 9200)'),
    # Generic connection refused (with port)
    (r'ECONNREFUSED\s+\d+\.\d+\.\d+\.\d+:\d+',
     'an external service'),
    # Generic connection timeout to localhost
    (r'connect ETIMEDOUT\s+127\.0\.0\.1:\d+|'
     r'connection timed out.*localhost',
     'an external service on localhost'),
]


def _detect_external_service_failure(error_info: str) -> str | None:
    """Check if an error is caused by an unavailable external service.

    Returns a human-readable service name if detected, ``None`` otherwise.
    """
    for pattern, service_name in _EXTERNAL_SERVICE_PATTERNS:
        if re.search(pattern, error_info, re.IGNORECASE):
            return service_name
    return None


# ── System-level / environment issue detection ────────────────
# Patterns that indicate the failure is due to missing system tools,
# runtimes, or project setup files — NOT a code bug.  The agent
# cannot fix these by editing source files.

_SYSTEM_LEVEL_PATTERNS: list[tuple[str, str]] = [
    # Ruby / Bundler
    (r'Could not locate Gemfile', 'Bundler (no Gemfile found — run `bundle init` or create a Gemfile)'),
    (r'bundler:?\s+command not found|bundle:?\s+command not found',
     'Bundler (install with `gem install bundler`)'),
    (r"ruby:?\s+command not found|ruby:?\s+is not recognized",
     'Ruby runtime (install Ruby from https://www.ruby-lang.org)'),
    # Python
    (r'python3?:?\s+command not found|python3?:?\s+is not recognized',
     'Python runtime'),
    (r'pip3?:?\s+command not found|pip3?:?\s+is not recognized',
     'pip (Python package manager)'),
    # Node.js / npm
    (r'node:?\s+command not found|node:?\s+is not recognized',
     'Node.js runtime (install from https://nodejs.org)'),
    (r'npm:?\s+command not found|npm:?\s+is not recognized',
     'npm (install Node.js from https://nodejs.org)'),
    # Java
    (r'javac?:?\s+command not found|javac?:?\s+is not recognized',
     'Java SDK (install JDK)'),
    (r'mvn:?\s+command not found', 'Maven (install Apache Maven)'),
    (r'gradle:?\s+command not found', 'Gradle (install Gradle)'),
    # .NET
    (r'dotnet:?\s+command not found|dotnet:?\s+is not recognized',
     '.NET SDK (install from https://dotnet.microsoft.com)'),
    # Docker
    (r'docker:?\s+command not found|docker:?\s+is not recognized',
     'Docker (install Docker Desktop)'),
    # Generic: "X is not recognized as an internal or external command" (Windows)
    (r"'[^']+' is not recognized as an internal or external command",
     'a required system tool (see error message above)'),
]


def _detect_system_level_failure(error_info: str) -> str | None:
    """Check if an error is caused by a missing system tool or environment setup.

    Returns a human-readable description if detected, ``None`` otherwise.
    """
    for pattern, description in _SYSTEM_LEVEL_PATTERNS:
        if re.search(pattern, error_info, re.IGNORECASE):
            return description
    return None


def build_step_waves(steps: list[str], dependencies: dict[int, set[int]]) -> list[list[int]]:
    """Group step indices into execution waves using topological ordering.

    Each wave is a list of step indices that can execute in parallel.
    Waves execute sequentially.
    """
    n = len(steps)
    remaining: set[int] = set(range(n))
    completed: set[int] = set()
    waves: list[list[int]] = []

    while remaining:
        # Find all steps whose dependencies are satisfied
        wave = [i for i in sorted(remaining)
                if dependencies.get(i, set()).issubset(completed)]
        if not wave:
            # Circular dependency or missing deps — execute remaining sequentially
            wave = [min(remaining)]
        waves.append(wave)
        for i in wave:
            remaining.discard(i)
            completed.add(i)

    return waves


def _execute_step(step_idx: int, step_text: str, *,
                  llm_client, executor, coder, reviewer, tester,
                  task: str, memory: FileMemory, display: CLIDisplay,
                  language: str | None, cfg=None,
                  auto: bool = False,
                  search_agent=None,
                  kb_context_builder=None,
                  code_graph=None,
                  project_profile=None) -> tuple[int, bool, str]:
    """Execute a single step. Returns ``(step_idx, success, error_info)``.

    Catches all exceptions so that a crash inside any handler never
    kills the whole pipeline — the step is marked as failed instead.
    """
    try:
        # --- Project Orientation + KB Context Injection (Phase 4+) ---
        #
        # Project grounding ALWAYS comes first — before KB symbols,
        # before task description, before everything.  It is the LLM's
        # "north star" for the entire session.

        context_parts: list[str] = []

        # 1. Project orientation grounding (always first)
        if project_profile is not None:
            try:
                context_parts.append(project_profile.format_for_prompt())
            except Exception as orient_exc:
                _logger.warning(
                    "[KB] Project orientation formatting failed: %s",
                    orient_exc,
                )

        # 2. KB context (Phase 4 — symbols, error fixes, patterns)
        if kb_context_builder is not None:
            try:
                from ..kb.context_builder import ContextBuilder
                kb_ctx = kb_context_builder.build_context(
                    task_description=step_text,
                    current_file=None,
                    max_tokens=getattr(cfg, "KB_MAX_CONTEXT_TOKENS", 4000) if cfg else 4000,
                )
                if kb_ctx.kb_available or kb_ctx.behavioral_instructions:
                    kb_text = kb_context_builder.format_context_for_prompt(kb_ctx)
                    if kb_text:
                        context_parts.append(kb_text)
                _logger.debug(
                    "[KB] Injected context: %d tokens, sources: %s, "
                    "symbols: %d, errors: %d",
                    kb_ctx.token_count, kb_ctx.sources_used,
                    len(kb_ctx.local_symbols), len(kb_ctx.error_fixes),
                )
            except Exception as kb_exc:
                _logger.warning("[KB] Context injection failed: %s", kb_exc)

        # Combine and store in memory for downstream handlers
        if context_parts:
            memory._kb_context = "\n\n".join(context_parts)

        log.info(f"\n{'='*60}\nTask {step_idx+1}: {step_text}\n"
                 f"Memory: {memory.summary()}\n{'='*60}")

        display.start_step(step_idx)
        step_type = _classify_step(step_text, llm_client, display, step_idx)
        display.steps[step_idx]["type"] = step_type
        display.render()
        log.info(f"Task {step_idx+1}: Classified as [{step_type}]")

        success, error_info = True, ""

        if step_type == "IGNORE":
            display.step_info(step_idx, "Not actionable, skipping.")
            display.complete_step(step_idx, "skipped")

        elif step_type == "CMD":
            success, error_info = _handle_cmd_step(
                step_text, executor, llm_client, memory, display, step_idx,
                language=language)
            display.complete_step(step_idx, "done" if success else "failed")

        elif step_type == "CODE":
            # Extract code graph from kb_context_builder if available
            _graph = code_graph
            if _graph is None and kb_context_builder is not None:
                _graph = getattr(kb_context_builder, "_graph", None)
            success, error_info = _handle_code_step(
                step_text, coder, reviewer, executor,
                task, memory, display, step_idx, language=language, cfg=cfg,
                auto=auto, code_graph=_graph,
                project_profile=project_profile)
            display.complete_step(step_idx, "done" if success else "failed")

        elif step_type == "TEST":
            success, error_info = _handle_test_step(
                step_text, tester, coder, reviewer, executor,
                task, memory, display, step_idx, language=language,
                auto=auto, search_agent=search_agent)
            display.complete_step(step_idx, "done" if success else "failed")

        else:
            display.step_info(step_idx, f"Unknown type '{step_type}', skipping.")
            display.complete_step(step_idx, "skipped")

        return step_idx, success, error_info

    except Exception as exc:
        log.error(f"Task {step_idx+1}: Unhandled exception: {exc}")
        display.step_info(step_idx, f"Error: {type(exc).__name__}: {exc}")
        display.complete_step(step_idx, "failed")
        return step_idx, False, f"Unhandled exception: {type(exc).__name__}: {exc}"


def _run_diagnosis_loop(step_idx: int, step_text: str, error_info: str, *,
                        llm_client, executor, coder, reviewer, tester,
                        task: str, memory: FileMemory, display: CLIDisplay,
                        language: str | None, cfg=None,
                        auto: bool = False,
                        search_agent=None,
                        kb_context_builder=None,
                        project_profile=None) -> bool:
    """Run diagnose → fix → retry loop. Returns ``True`` if the step was fixed.

    All exceptions are caught so that a crash during diagnosis (e.g. an
    embedding error) never kills the whole pipeline — the step is simply
    marked as failed and the pipeline halts gracefully.
    """
    # ── Early exit: external service dependency ──────────────────
    # If the failure is due to an unavailable external service (DB,
    # cache, etc.), diagnosis cannot help — inform the user instead.
    service = _detect_external_service_failure(error_info)
    if service:
        msg = (f"Step requires {service} which is not reachable. "
               f"Please ensure the service is running and accessible, "
               f"then re-run the pipeline.")
        display.step_info(step_idx, msg)
        log.warning(f"Task {step_idx+1}: External service unavailable: {service}")
        log.warning(f"Task {step_idx+1}: Skipping diagnosis — "
                    f"this is not a code issue.")
        display.complete_step(step_idx, "skipped")
        return False

    # ── Early exit: missing system tool / environment setup ─────
    # If the failure is because a runtime, package manager, or project
    # config file is missing, editing code won't help.
    sys_issue = _detect_system_level_failure(error_info)
    if sys_issue:
        msg = (f"System dependency missing: {sys_issue}. "
               f"Please install the required tool and re-run the pipeline.")
        display.step_info(step_idx, msg)
        log.warning(f"Task {step_idx+1}: System-level issue: {sys_issue}")
        log.warning(f"Task {step_idx+1}: Skipping diagnosis — "
                    f"this is an environment issue, not a code bug.")
        display.complete_step(step_idx, "failed")
        return False

    for diag_attempt in range(1, MAX_DIAGNOSIS_RETRIES + 1):
        try:
            display.step_info(
                step_idx, f"Diagnosing failure ({diag_attempt}/{MAX_DIAGNOSIS_RETRIES})...")
            log.info(f"Task {step_idx+1}: Diagnosis attempt "
                     f"{diag_attempt}/{MAX_DIAGNOSIS_RETRIES}")

            step_type = display.steps[step_idx].get("type", "CODE")
            diagnosis = _diagnose_failure(
                step_text, step_type, error_info,
                memory, llm_client, display, step_idx,
                search_agent=search_agent, language=language)

            fix_applied, cmds_succeeded = _apply_fix(
                diagnosis, executor, memory, display, step_idx,
                step_type=step_type)

            if not fix_applied:
                display.step_info(step_idx, "No actionable fix found in diagnosis.")
                log.warning(f"Task {step_idx+1}: Diagnosis produced no actionable fix.")
                continue

            # For CMD steps, the fix commands ARE the corrected step.
            # If they all succeeded, the step is done — no need to re-run
            # the original (which would likely fail again).
            if step_type == "CMD" and cmds_succeeded:
                display.step_info(step_idx, "Fix commands succeeded — step resolved.")
                log.info(f"Task {step_idx+1}: CMD fix commands succeeded, "
                         f"treating step as resolved.")
                display.complete_step(step_idx, "done")
                return True

            # Re-run the step (for CODE/TEST: re-run with fixed files)
            display.step_info(step_idx, "Fix applied — retrying step...")
            _, success, error_info = _execute_step(
                step_idx, step_text,
                llm_client=llm_client, executor=executor,
                coder=coder, reviewer=reviewer, tester=tester,
                task=task, memory=memory, display=display,
                language=language, cfg=cfg, auto=auto,
                search_agent=search_agent,
                kb_context_builder=kb_context_builder,
                project_profile=project_profile,
            )

            if success:
                return True
            else:
                log.warning(f"Task {step_idx+1}: Still failing after "
                            f"diagnosis attempt {diag_attempt}")

        except Exception as exc:
            log.error(f"Task {step_idx+1}: Exception during diagnosis "
                      f"attempt {diag_attempt}: {exc}")
            display.step_info(step_idx, f"Diagnosis error: {type(exc).__name__}")
            continue

    display.step_info(
        step_idx, "Step failed after all fix attempts. Halting pipeline.")
    log.error(f"Task {step_idx+1}: Failed after {MAX_DIAGNOSIS_RETRIES} "
              f"diagnosis attempts. Halting pipeline.")
    return False
