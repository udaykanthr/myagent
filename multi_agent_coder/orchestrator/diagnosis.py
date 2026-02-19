"""
Diagnosis and fix helpers — analyze step failures and apply fixes.
"""

from ..executor import Executor
from ..cli_display import CLIDisplay, token_tracker, log
from ..diff_display import show_diffs

from .memory import FileMemory
from .step_handlers import _shell_instructions
from .classification import _extract_commands_from_text


def _diagnose_failure(step_text: str, step_type: str, error_info: str,
                      memory: FileMemory, llm_client, display: CLIDisplay,
                      step_idx: int) -> str:
    display.step_info(step_idx, "Analyzing failure root cause...")

    context_files = memory.related_context(step_text)

    prior_context = ""
    all_files = memory.all_files()
    for i in range(step_idx):
        key = f"_cmd_output/step_{i+1}.txt"
        if key in all_files:
            prior_context += f"Step {i+1} output:\n{all_files[key]}\n\n"
    if prior_context:
        prior_context = "Previously executed steps:\n" + prior_context

    prompt = (
        "A step in our automated coding pipeline has FAILED after multiple retries.\n"
        "Analyze the failure and provide a concrete fix.\n\n"
        f"Step {step_idx+1}: {step_text}\n"
        f"Step type: {step_type}\n\n"
        f"Error details:\n{error_info}\n\n"
    )
    if prior_context:
        prompt += f"{prior_context}\n"
    if context_files:
        prompt += f"Relevant project files:\n{context_files}\n\n"
    prompt += f"All project files: {memory.summary()}\n\n"

    # Step-type-specific fix instructions
    if step_type == "CMD":
        prompt += (
            "This is a COMMAND step. Do NOT generate code files.\n"
            "Use the exact names, paths, and values from previous steps.\n"
            "Respond with:\n"
            "1. ROOT CAUSE: one-line explanation of what went wrong\n"
            "2. FIX: provide the corrected shell command in backticks.\n"
            f"{_shell_instructions()}"
        )
    else:
        prompt += (
            "Respond with:\n"
            "1. ROOT CAUSE: one-line explanation of what went wrong\n"
            "2. FIX: provide the COMPLETE corrected file(s). Do NOT use diffs or patches.\n"
            "   Write the ENTIRE file content, not just the changed parts.\n\n"
            "CRITICAL FORMAT — the #### [FILE]: marker must be OUTSIDE and BEFORE the code block:\n\n"
            "#### [FILE]: path/to/file.py\n"
            "```python\n"
            "# entire file contents here\n"
            "```\n\n"
            "WRONG (do NOT do this):\n"
            "```python\n"
            "#### [FILE]: path/to/file.py   <-- WRONG! marker inside code block\n"
            "```\n\n"
            "WRONG (do NOT do this):\n"
            "```diff\n"
            "-old line   <-- WRONG! do not use diff format\n"
            "+new line\n"
            "```\n"
        )

    sent_before = token_tracker.total_prompt_tokens
    recv_before = token_tracker.total_completion_tokens

    diagnosis = llm_client.generate_response(prompt)

    sent_delta = token_tracker.total_prompt_tokens - sent_before
    recv_delta = token_tracker.total_completion_tokens - recv_before
    display.step_tokens(step_idx, sent_delta, recv_delta)

    log.info(f"Step {step_idx+1}: Diagnosis:\n{diagnosis}")
    return diagnosis


def _apply_fix(diagnosis: str, executor: Executor, memory: FileMemory,
               display: CLIDisplay, step_idx: int,
               step_type: str = "CODE") -> tuple[bool, bool]:
    """Apply fixes from a diagnosis response.

    Returns ``(applied, cmds_succeeded)`` where *applied* is True if any
    fix action was taken and *cmds_succeeded* is True if all fix commands
    ran successfully (relevant for CMD steps where the fix command itself
    is the corrected step).
    """
    applied = False
    cmds_succeeded = True

    # Only write code files for CODE / TEST steps, never for CMD
    if step_type != "CMD":
        files = executor.parse_code_blocks(diagnosis)

        # Fallback: try fuzzy parsing (handles diff blocks, inline file
        # comments, and other common LLM diagnosis formats)
        if not files:
            files = executor.parse_code_blocks_fuzzy(diagnosis)
            if files:
                log.info(f"Step {step_idx+1}: Standard parser found nothing, "
                         f"fuzzy parser extracted: {list(files.keys())}")

        if files:
            show_diffs(files, log_only=True)
            written = executor.write_files(files)
            memory.update(files)
            display.step_info(step_idx, f"Fixed files: {', '.join(written)}")
            log.info(f"Step {step_idx+1}: Applied code fixes to: {', '.join(written)}")
            applied = True

    # Extract and run fix commands (from triple-backtick blocks + inline backticks)
    fix_commands = _extract_commands_from_text(diagnosis)
    for cmd in fix_commands:
        display.step_info(step_idx, f"Running fix: {cmd}")
        log.info(f"Step {step_idx+1}: Running fix command: {cmd}")
        success, output = executor.run_command(cmd)
        if output:
            truncated = output[:4000] if len(output) > 4000 else output
            memory.update({
                f"_fix_output/step_{step_idx+1}.txt": f"$ {cmd}\n\n{truncated}"
            })
        if not success:
            cmds_succeeded = False
        applied = True

    return applied, cmds_succeeded
