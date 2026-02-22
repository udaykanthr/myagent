"""
Diagnosis and fix helpers — analyze step failures and apply fixes.
"""

import os

from ..executor import Executor
from ..cli_display import CLIDisplay, token_tracker, log
from ..diff_display import show_diffs, _detect_hazards

from .memory import FileMemory
from .step_handlers import _shell_instructions, _strip_protected_files
from .classification import _extract_commands_from_text, _looks_like_command


def _diagnose_failure(step_text: str, step_type: str, error_info: str,
                      memory: FileMemory, llm_client, display: CLIDisplay,
                      step_idx: int,
                      search_agent=None,
                      language: str | None = None) -> str:
    display.step_info(step_idx, "Analyzing failure root cause...")

    # ── Optional: search the web for error documentation ────
    search_context = ""
    if search_agent is not None:
        display.step_info(step_idx, "Searching web for error documentation...")
        try:
            search_context = search_agent.search_for_error(
                error_info, step_text, language=language)
            if search_context:
                log.info(f"Step {step_idx+1}: Search agent found documentation")
        except Exception as exc:
            log.warning(f"Step {step_idx+1}: Search agent error: {exc}")

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
    if search_context:
        prompt += (
            "The following web search results may contain relevant documentation,\n"
            "error explanations, or solutions. Use them to inform your fix:\n\n"
            f"{search_context}\n\n"
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
            "2. FIX: provide the corrected shell command inside a code block:\n"
            "```bash\n"
            "command here\n"
            "```\n"
            "3. SPECIAL CASE: If the command failed because the directory is not empty (e.g. create-react-app .), "
            "the FIX is to create the app in a new subdirectory (e.g. `npx create-react-app my-app ...`) "
            "instead of the current directory.\n"
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

    explanation = CLIDisplay.extract_explanation(diagnosis)
    if explanation:
        display.add_llm_log(explanation, source="Diagnosis")

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
            # Strip protected manifest files before any further processing
            files = _strip_protected_files(files)

        if files:
            # Filter out files with hazardous diffs (e.g. truncation,
            # dependency removal) — these would corrupt the project.
            safe_files: dict[str, str] = {}
            for filepath, content in files.items():
                full_path = os.path.join(".", filepath)
                if os.path.isfile(full_path):
                    try:
                        with open(full_path, "r", encoding="utf-8",
                                  errors="replace") as f:
                            old_content = f.read()
                        hazards = _detect_hazards(filepath, old_content, content)
                        if hazards:
                            msgs = "; ".join(m for _, m in hazards)
                            log.warning(f"Step {step_idx+1}: Skipping hazardous "
                                        f"fix for {filepath}: {msgs}")
                            display.step_info(step_idx,
                                              f"Skipped unsafe fix for {filepath}")
                            continue
                    except OSError:
                        pass
                safe_files[filepath] = content

            if safe_files:
                show_diffs(safe_files, log_only=True)
                written = executor.write_files(safe_files)
                memory.update(safe_files)
                display.step_info(step_idx, f"Fixed files: {', '.join(written)}")
                log.info(f"Step {step_idx+1}: Applied code fixes to: "
                         f"{', '.join(written)}")
                applied = True

    # Extract and run fix commands (from triple-backtick blocks + inline backticks)
    fix_commands = _extract_commands_from_text(diagnosis)

    # Fallback: if no commands found, look for raw lines that look like commands
    # (e.g. "npx create-react-app ..." sitting on its own line)
    if not fix_commands and step_type == "CMD":
        for line in diagnosis.splitlines():
            line = line.strip()
            # Heuristic: line must start with a known command, contain spaces (args),
            # and not be a numbered list item (e.g. "1. npx ...")
            if not line or len(line.split()) < 2:
                continue
            # Remove leading bullets/numbers if present
            clean_line = line.lstrip('1234567890.-* ').strip()
            if _looks_like_command(clean_line) and clean_line not in fix_commands:
                fix_commands.append(clean_line)
        
        if fix_commands:
            log.info(f"Step {step_idx+1}: Fuzzy command parser found: {fix_commands}")

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
