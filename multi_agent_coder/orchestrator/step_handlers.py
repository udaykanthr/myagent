"""
Step handlers — CMD, CODE, and TEST step execution logic.
"""

import os
import shutil

from ..config import Config
from ..agents.coder import CoderAgent
from ..agents.reviewer import ReviewerAgent
from ..agents.tester import TesterAgent
from ..executor import Executor
from ..cli_display import CLIDisplay, token_tracker, log
from ..language import get_code_block_lang, get_test_framework

from .memory import FileMemory
from .classification import _extract_command_from_step


MAX_STEP_RETRIES = 3

# Map test runner binary → install command
_RUNNER_INSTALL = {
    "pytest": "pip install pytest",
    "jest": "npm install --save-dev jest",
    "npx": "npm install --save-dev jest",
    "mocha": "npm install --save-dev mocha",
    "vitest": "npm install --save-dev vitest",
    "go": None,  # built-in, no install needed
    "cargo": None,
    "rspec": "gem install rspec",
    "phpunit": "composer require --dev phpunit/phpunit",
}


def _get_runner_install_cmd(runner: str) -> str:
    """Return the install command for a test runner binary."""
    return _RUNNER_INSTALL.get(runner, f"pip install {runner}")


import platform


def _shell_instructions() -> str:
    """Return OS-aware shell command guidance for LLM prompts."""
    if os.name == 'nt':
        return (
            "Use plain CMD commands that work in Windows cmd.exe.\n"
            "For listing files use: dir /s /b\n"
            "For reading a file use: type <path>\n"
            "For creating a directory use: mkdir <path>\n"
            "For installing Python packages use: pip install <package>\n"
            "Do NOT use PowerShell cmdlets like Get-ChildItem, Select-Object, etc.\n"
        )
    else:
        return (
            f"Use standard shell commands for {platform.system()}.\n"
            "For listing files use: find . -type f\n"
            "For reading a file use: cat <path>\n"
            "For creating a directory use: mkdir -p <path>\n"
            "For installing Python packages use: pip install <package>\n"
        )


def _shell_examples() -> str:
    """Return OS-aware example commands for the planner prompt."""
    if os.name == 'nt':
        return "  1. List all project files with `dir /s /b`"
    else:
        return "  1. List all project files with `find . -type f`"


# File extensions and names that don't need code review
_NON_CODE_EXTENSIONS = {
    '.md', '.txt', '.rst', '.log', '.csv',
    '.yml', '.yaml', '.toml', '.ini', '.cfg',
    '.json', '.xml',
    '.html', '.css',
    '.env', '.env.example', '.gitignore', '.dockerignore',
    '.editorconfig',
}
_NON_CODE_FILENAMES = {
    'README', 'README.md', 'README.rst', 'README.txt',
    'LICENSE', 'LICENSE.md', 'LICENSE.txt',
    'CHANGELOG', 'CHANGELOG.md',
    'CONTRIBUTING', 'CONTRIBUTING.md',
    'Makefile', 'Dockerfile', 'Procfile',
    '.gitignore', '.dockerignore', '.editorconfig',
    'requirements.txt', 'setup.cfg',
}


def _all_non_code_files(filenames: list[str]) -> bool:
    """Return True if every file in the list is non-functional (docs, config, etc.)."""
    if not filenames:
        return False
    for f in filenames:
        basename = f.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        _, ext = os.path.splitext(basename)
        if basename not in _NON_CODE_FILENAMES and ext.lower() not in _NON_CODE_EXTENSIONS:
            return False
    return True


def _build_prior_steps_context(memory: FileMemory, step_idx: int) -> str:
    """Collect outputs of prior steps from memory for context."""
    parts: list[str] = []
    all_files = memory.all_files()
    for i in range(step_idx):
        key = f"_cmd_output/step_{i+1}.txt"
        if key in all_files:
            parts.append(f"Step {i+1} output:\n{all_files[key]}")
    if not parts:
        return ""
    return "Previously executed steps:\n" + "\n\n".join(parts) + "\n\n"


def _handle_cmd_step(step_text: str, executor: Executor,
                     llm_client, memory: FileMemory,
                     display: CLIDisplay, step_idx: int,
                     language: str | None = None) -> tuple[bool, str]:
    cmd = _extract_command_from_step(step_text)

    if cmd:
        pass  # use extracted command
    else:
        display.step_info(step_idx, "Generating command...")

        prior_context = _build_prior_steps_context(memory, step_idx)
        file_summary = memory.summary()

        gen_prompt = (
            "You are a shell command generator. Given a task step, output "
            "ONLY the shell command to accomplish it. No explanations, no "
            "markdown, no backticks — just the raw command.\n"
            f"{_shell_instructions()}\n"
        )
        if prior_context:
            gen_prompt += (
                f"{prior_context}"
                "IMPORTANT: Use the exact names, paths, and values from the "
                "previous steps above. Do NOT guess or use defaults.\n\n"
            )
        if file_summary != "(no files yet)":
            gen_prompt += f"Project files: {file_summary}\n\n"
        gen_prompt += f"Step: {step_text}\n\nCommand:"
        sent_before = token_tracker.total_prompt_tokens
        recv_before = token_tracker.total_completion_tokens

        cmd = llm_client.generate_response(gen_prompt).strip()

        sent_delta = token_tracker.total_prompt_tokens - sent_before
        recv_delta = token_tracker.total_completion_tokens - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        cmd = cmd.strip('`').strip()
        if cmd.startswith('```'):
            cmd = cmd.split('\n', 1)[-1].rsplit('```', 1)[0].strip()

        if not cmd:
            display.step_info(step_idx, "Could not generate command, skipping.")
            log.warning(f"Step {step_idx+1}: LLM returned empty command.")
            return True, ""

    display.step_info(step_idx, f"Running: {cmd}")
    log.info(f"Step {step_idx+1}: Running command: {cmd}")

    success, output = executor.run_command(cmd)
    log.info(f"Step {step_idx+1}: Command output:\n{output}")

    if output:
        truncated = output[:4000] if len(output) > 4000 else output
        memory.update({
            f"_cmd_output/step_{step_idx+1}.txt": f"$ {cmd}\n\n{truncated}"
        })

    if success:
        display.step_info(step_idx, "Command succeeded.")
        return True, ""
    else:
        display.step_info(step_idx, "Command failed. See log.")
        log.warning(f"Step {step_idx+1}: Command failed.")
        return False, f"Command `{cmd}` failed.\nOutput:\n{output}"


def _handle_code_step(step_text: str, coder: CoderAgent, reviewer: ReviewerAgent,
                      executor: Executor, task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int,
                      language: str | None = None,
                      cfg: Config | None = None) -> tuple[bool, str]:
    feedback = ""
    context_window = cfg.CONTEXT_WINDOW if cfg else 8192
    ctx_budget = int(context_window * 0.8)

    for attempt in range(1, MAX_STEP_RETRIES + 1):
        context = f"Task: {task}"
        related = memory.related_context(step_text, max_tokens=ctx_budget)
        if related:
            context += f"\nExisting files (overwrite as needed):\n{related}"
        if memory.summary() != "(no files yet)":
            context += f"\nAll project files: {memory.summary()}"
        if feedback:
            context += f"\nFeedback: {feedback}"

        display.step_info(step_idx, f"Coding (attempt {attempt}/{MAX_STEP_RETRIES})...")
        sent_before = token_tracker.total_prompt_tokens
        recv_before = token_tracker.total_completion_tokens

        response = coder.process(step_text, context=context, language=language)

        sent_delta = token_tracker.total_prompt_tokens - sent_before
        recv_delta = token_tracker.total_completion_tokens - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        files = executor.parse_code_blocks(response)
        if not files:
            feedback = "No file markers found. Use #### [FILE]: path/to/file.py format."
            display.step_info(step_idx, "No files parsed, retrying...")
            log.warning(f"Step {step_idx+1}: No files parsed from coder response.")
            continue

        written = executor.write_files(files)
        memory.update(files)
        display.step_info(step_idx, f"Written: {', '.join(written)}")

        # Skip review for non-code files (README, LICENSE, configs, etc.)
        if _all_non_code_files(list(files.keys())):
            display.step_info(step_idx, "Non-code files, skipping review ✔")
            log.info(f"Step {step_idx+1}: Skipped review (non-code files: {list(files.keys())})")
            return True, ""

        # Review
        display.step_info(step_idx, "Reviewing code...")
        sent_before = token_tracker.total_prompt_tokens
        recv_before = token_tracker.total_completion_tokens

        review = reviewer.process(
            f"Review this code:\n{response}",
            context=f"Step: {step_text}",
            language=language,
        )

        sent_delta = token_tracker.total_prompt_tokens - sent_before
        recv_delta = token_tracker.total_completion_tokens - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        log.info(f"Step {step_idx+1}: Review:\n{review}")

        review_lower = review.lower()
        # Accept if the reviewer explicitly approves
        approved = any(phrase in review_lower for phrase in (
            "code looks good",
            "looks good",
            "no issues",
            "no critical issues",
            "no bugs found",
            "code is correct",
            "functionally correct",
            "lgtm",
        ))

        if approved:
            display.step_info(step_idx, "Review passed ✔")
            return True, ""

        # On the last attempt, accept the code if the review only has
        # minor/style suggestions (no keywords indicating actual bugs)
        if attempt == MAX_STEP_RETRIES:
            has_critical = any(kw in review_lower for kw in (
                "error", "bug", "crash", "undefined", "missing import",
                "will fail", "won't work", "does not work", "broken",
                "incorrect", "wrong", "typeerror", "nameerror",
                "syntaxerror", "attributeerror", "keyerror",
            ))
            if not has_critical:
                display.step_info(step_idx, "Review has only minor suggestions, accepting ✔")
                log.info(f"Step {step_idx+1}: Accepted on last attempt "
                         f"(review had no critical keywords)")
                return True, ""

        feedback = review
        display.step_info(step_idx, "Review found issues, retrying...")
        log.warning(f"Step {step_idx+1}: Review issues: {review[:200]}")

    log.error(f"Step {step_idx+1}: Failed after {MAX_STEP_RETRIES} attempts.")
    return False, f"Code step failed after {MAX_STEP_RETRIES} attempts.\nLast review feedback:\n{feedback}"


def _handle_test_step(step_text: str, tester: TesterAgent, coder: CoderAgent,
                      reviewer: ReviewerAgent, executor: Executor,
                      task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int,
                      language: str | None = None) -> tuple[bool, str]:
    lang_tag = get_code_block_lang(language) if language else "python"
    test_cmd = get_test_framework(language)["command"] if language else "pytest"

    # Ensure the test runner binary is installed before attempting to run tests
    parts = test_cmd.split()
    runner = parts[0]
    # For "npx <tool>", the binary to check is "npx" itself
    if not shutil.which(runner):
        actual_tool = parts[1] if runner == "npx" and len(parts) > 1 else runner
        install_cmd = _get_runner_install_cmd(actual_tool)
        display.step_info(step_idx, f"`{runner}` not found, installing...")
        log.info(f"Step {step_idx+1}: Auto-installing: {install_cmd}")
        ok, out = executor.run_command(install_cmd)
        if ok:
            display.step_info(step_idx, f"Installed `{actual_tool}`")
        else:
            log.warning(f"Step {step_idx+1}: Failed to install "
                        f"{actual_tool}: {out[:200]}")

    code_summary = ""
    for fname, content in memory.all_files().items():
        code_summary += f"#### [FILE]: {fname}\n```{lang_tag}\n{content}\n```\n\n"

    feedback = ""
    last_test_output = ""

    for gen_attempt in range(1, MAX_STEP_RETRIES + 1):
        display.step_info(step_idx, f"Generating tests (attempt {gen_attempt})...")
        gen_context = f"Code:\n{code_summary}"
        if feedback:
            gen_context += f"\nFeedback: {feedback}"

        sent_before = token_tracker.total_prompt_tokens
        recv_before = token_tracker.total_completion_tokens

        test_response = tester.process(step_text, context=gen_context, language=language)

        sent_delta = token_tracker.total_prompt_tokens - sent_before
        recv_delta = token_tracker.total_completion_tokens - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        test_files = executor.parse_code_blocks(test_response)
        if not test_files:
            feedback = "No test files found. Use #### [FILE]: format."
            display.step_info(step_idx, "No test files parsed, retrying...")
            continue

        # Review tests
        display.step_info(step_idx, "Reviewing tests...")
        sent_before = token_tracker.total_prompt_tokens
        recv_before = token_tracker.total_completion_tokens

        review = reviewer.process(
            f"Review these tests for correctness, especially import paths:\n{test_response}",
            context=f"Project files: {memory.summary()}\n{code_summary}",
            language=language,
        )

        sent_delta = token_tracker.total_prompt_tokens - sent_before
        recv_delta = token_tracker.total_completion_tokens - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        log.info(f"Step {step_idx+1}: Test review:\n{review}")

        review_lower = review.lower()
        test_approved = any(phrase in review_lower for phrase in (
            "code looks good", "looks good", "no issues",
            "no critical issues", "no bugs found", "code is correct",
            "functionally correct", "lgtm", "tests look good",
        ))

        # On last attempt, accept if no critical issues found
        if not test_approved and gen_attempt == MAX_STEP_RETRIES:
            has_critical = any(kw in review_lower for kw in (
                "error", "bug", "crash", "undefined", "missing import",
                "will fail", "won't work", "incorrect", "wrong import",
            ))
            if not has_critical:
                test_approved = True
                log.info(f"Step {step_idx+1}: Test accepted on last attempt (minor issues only)")

        if not test_approved:
            feedback = review
            display.step_info(step_idx, "Test review found issues, regenerating...")
            continue

        # Write and run
        written = executor.write_files(test_files)
        memory.update(test_files)
        display.step_info(step_idx, f"Tests written: {', '.join(written)}")

        prev_output = None
        for run_attempt in range(1, MAX_STEP_RETRIES + 1):
            display.step_info(step_idx, f"Running: {test_cmd} (attempt {run_attempt})...")
            log.info(f"Step {step_idx+1}: Running test command: {test_cmd}")
            success, output = executor.run_tests(test_cmd)
            log.info(f"Step {step_idx+1}: Test run output:\n{output or '(no output)'}")

            last_test_output = output

            if success:
                display.step_info(step_idx, "Tests passed ✔")
                return True, ""

            # Detect stuck loop: same error output repeating means code
            # fixes aren't helping (likely an infra/tool issue, not code)
            if prev_output and output == prev_output and run_attempt > 1:
                display.step_info(step_idx,
                                  "Same error repeating — not a code issue, stopping retry loop.")
                log.warning(f"Step {step_idx+1}: Identical test output on attempt "
                            f"{run_attempt}, breaking retry loop.")
                break
            prev_output = output

            # If test runner itself is not installed, try to install it
            if "not installed" in output or "not on PATH" in output:
                runner_parts = test_cmd.split()
                actual_tool = runner_parts[1] if runner_parts[0] == "npx" and len(runner_parts) > 1 else runner_parts[0]
                install_cmd = _get_runner_install_cmd(actual_tool)
                display.step_info(step_idx, f"Installing `{actual_tool}`...")
                log.info(f"Step {step_idx+1}: Installing test runner: {install_cmd}")
                ok, out = executor.run_command(install_cmd)
                if ok:
                    display.step_info(step_idx, f"Installed `{actual_tool}`, re-running...")
                    success, output = executor.run_tests(test_cmd)
                    last_test_output = output
                    if success:
                        display.step_info(step_idx, "Tests passed after runner install ✔")
                        return True, ""
                continue  # retry with coder fix if runner install + rerun still failed

            # Auto-install missing packages before asking coder to fix
            missing_pkgs = executor.detect_missing_packages(output)
            if missing_pkgs:
                display.step_info(step_idx, f"Installing missing packages: {', '.join(missing_pkgs)}")
                log.info(f"Step {step_idx+1}: Auto-installing: {missing_pkgs}")
                install_ok, install_out = executor.install_packages(missing_pkgs)
                if install_ok:
                    display.step_info(step_idx, "Packages installed, re-running tests...")
                    log.info(f"Step {step_idx+1}: Re-running test command: {test_cmd}")
                    success, output = executor.run_tests(test_cmd)
                    log.info(f"Step {step_idx+1}: Test re-run after install:\n{output or '(no output)'}")
                    last_test_output = output
                    if success:
                        display.step_info(step_idx, "Tests passed after package install ✔")
                        return True, ""
                else:
                    log.warning(f"Step {step_idx+1}: Package install failed: {install_out}")

            display.step_info(step_idx, "Tests failed, asking coder to fix...")
            error_detail = output[:500] if output else f"(command `{test_cmd}` produced no output — it may have crashed or the test framework may not be installed)"
            fix_context = (
                f"Test command: `{test_cmd}`\n"
                f"Test errors:\n{error_detail}\n"
                f"Project files:\n{code_summary}"
            )

            sent_before = token_tracker.total_prompt_tokens
            recv_before = token_tracker.total_completion_tokens

            fix_response = coder.process(
                "Fix the code so tests pass.", context=fix_context, language=language)

            sent_delta = token_tracker.total_prompt_tokens - sent_before
            recv_delta = token_tracker.total_completion_tokens - recv_before
            display.step_tokens(step_idx, sent_delta, recv_delta)

            fix_files = executor.parse_code_blocks(fix_response)
            if fix_files:
                executor.write_files(fix_files)
                memory.update(fix_files)
                code_summary = ""
                for fname, content in memory.all_files().items():
                    code_summary += f"#### [FILE]: {fname}\n```{lang_tag}\n{content}\n```\n\n"

        log.error(f"Step {step_idx+1}: Tests still failing after {MAX_STEP_RETRIES} fixes.")
        return False, f"Tests still failing after {MAX_STEP_RETRIES} fix attempts.\nLast test output:\n{last_test_output}"

    log.error(f"Step {step_idx+1}: Could not generate valid tests after {MAX_STEP_RETRIES} attempts.")
    return False, f"Could not generate valid tests after {MAX_STEP_RETRIES} attempts.\nLast feedback:\n{feedback}"
