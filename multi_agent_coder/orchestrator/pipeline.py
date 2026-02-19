"""
Pipeline execution — wave-based parallel/sequential step execution.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from ..cli_display import CLIDisplay, log

from .memory import FileMemory
from .classification import _classify_step
from .step_handlers import (
    _handle_cmd_step, _handle_code_step, _handle_test_step,
    MAX_STEP_RETRIES,
)
from .diagnosis import _diagnose_failure, _apply_fix


MAX_DIAGNOSIS_RETRIES = 2   # outer retries: diagnose failure → fix → re-run step


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
                  auto: bool = False) -> tuple[int, bool, str]:
    """Execute a single step. Returns ``(step_idx, success, error_info)``.

    Catches all exceptions so that a crash inside any handler never
    kills the whole pipeline — the step is marked as failed instead.
    """
    try:
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
            success, error_info = _handle_code_step(
                step_text, coder, reviewer, executor,
                task, memory, display, step_idx, language=language, cfg=cfg,
                auto=auto)
            display.complete_step(step_idx, "done" if success else "failed")

        elif step_type == "TEST":
            success, error_info = _handle_test_step(
                step_text, tester, coder, reviewer, executor,
                task, memory, display, step_idx, language=language,
                auto=auto)
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
                        auto: bool = False) -> bool:
    """Run diagnose → fix → retry loop. Returns ``True`` if the step was fixed.

    All exceptions are caught so that a crash during diagnosis (e.g. an
    embedding error) never kills the whole pipeline — the step is simply
    marked as failed and the pipeline halts gracefully.
    """
    for diag_attempt in range(1, MAX_DIAGNOSIS_RETRIES + 1):
        try:
            display.step_info(
                step_idx, f"Diagnosing failure ({diag_attempt}/{MAX_DIAGNOSIS_RETRIES})...")
            log.info(f"Task {step_idx+1}: Diagnosis attempt "
                     f"{diag_attempt}/{MAX_DIAGNOSIS_RETRIES}")

            step_type = display.steps[step_idx].get("type", "CODE")
            diagnosis = _diagnose_failure(
                step_text, step_type, error_info,
                memory, llm_client, display, step_idx)

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
