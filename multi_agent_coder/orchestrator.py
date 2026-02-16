import argparse
import os
import platform
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Config
from .llm.ollama import OllamaClient
from .llm.lm_studio import LMStudioClient
from .llm.base import LLMError
from .agents.planner import PlannerAgent
from .agents.coder import CoderAgent
from .agents.reviewer import ReviewerAgent
from .agents.tester import TesterAgent
from .executor import Executor
from .embedding_store import EmbeddingStore
from .cli_display import CLIDisplay, token_tracker, log
from .language import (
    detect_language, detect_language_from_task, get_test_framework,
    get_language_name, get_code_block_lang,
)
from .project_scanner import scan_project, format_scan_for_planner
from .checkpoint import (
    save_checkpoint, load_checkpoint, clear_checkpoint,
)
from . import git_utils

MAX_STEP_RETRIES = 3
MAX_DIAGNOSIS_RETRIES = 2   # outer retries: diagnose failure → fix → re-run step


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


# ───────────────────────────────────────────────────────────────
# FileMemory — thread-safe, with context-window budget
# ───────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token)."""
    return len(text) // 4


class FileMemory:
    """Tracks every file's path and current contents across all steps.

    When an EmbeddingStore is provided, context retrieval uses semantic
    similarity instead of simple filename matching.
    """

    def __init__(self, embedding_store: EmbeddingStore | None = None,
                 top_k: int = 5):
        self._files: dict[str, str] = {}   # filepath -> contents
        self._store = embedding_store
        self._top_k = top_k
        self._lock = threading.Lock()

    def update(self, files: dict[str, str]):
        """Store or overwrite file contents and update embeddings."""
        with self._lock:
            self._files.update(files)
            if self._store:
                for fpath, content in files.items():
                    self._store.add(fpath, content)

    def get(self, filepath: str) -> str | None:
        with self._lock:
            return self._files.get(filepath)

    def all_files(self) -> dict[str, str]:
        with self._lock:
            return dict(self._files)

    def as_dict(self) -> dict[str, str]:
        """Snapshot for checkpoint serialization."""
        with self._lock:
            return dict(self._files)

    def related_context(self, step_text: str, max_tokens: int | None = None) -> str:
        """Build a compact context string with the most relevant files.

        Uses semantic search when embeddings are available, otherwise
        falls back to filename substring matching.

        When *max_tokens* is given, files are accumulated until the
        budget is reached.
        """
        with self._lock:
            if self._store and self._store.size > 0:
                return self._semantic_context(step_text, max_tokens)
            return self._substring_context(step_text, max_tokens)

    def _semantic_context(self, step_text: str, max_tokens: int | None) -> str:
        results = self._store.search(step_text, top_k=self._top_k)
        parts: list[str] = []
        budget = max_tokens or float("inf")
        used = 0
        for fpath, score in results:
            content = self._files.get(fpath, "")
            if not content:
                continue
            entry = (
                f"#### [FILE]: {fpath} (relevance: {score:.2f})\n"
                f"```\n{content}\n```"
            )
            entry_tokens = _estimate_tokens(entry)
            if used + entry_tokens > budget:
                break
            parts.append(entry)
            used += entry_tokens
        log.debug(f"[FileMemory] Semantic search returned {len(parts)} files "
                  f"({used} est. tokens)")
        return "\n\n".join(parts)

    def _substring_context(self, step_text: str, max_tokens: int | None) -> str:
        parts: list[str] = []
        budget = max_tokens or float("inf")
        used = 0
        for fpath, content in self._files.items():
            basename = fpath.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
            if basename in step_text or fpath in step_text:
                entry = f"#### [FILE]: {fpath}\n```\n{content}\n```"
                entry_tokens = _estimate_tokens(entry)
                if used + entry_tokens > budget:
                    break
                parts.append(entry)
                used += entry_tokens
        return "\n\n".join(parts)

    def summary(self) -> str:
        with self._lock:
            if not self._files:
                return "(no files yet)"
            return ", ".join(self._files.keys())


# ───────────────────────────────────────────────────────────────
# Step classification
# ───────────────────────────────────────────────────────────────

def _classify_step(step_text: str, llm_client, display: CLIDisplay, step_idx: int) -> str:
    display.step_info(step_idx, "Classifying step...")
    prompt = (
        "Classify the following task step into exactly one category.\n"
        "Reply with ONLY one word: CMD, CODE, TEST, or IGNORE\n\n"
        "  CMD    = anything that can be done by running shell commands, including:\n"
        "           - scanning or listing files/directories (ls, tree, find, dir)\n"
        "           - reading or inspecting file contents (cat, type, head)\n"
        "           - checking project structure or dependencies\n"
        "           - installing packages (pip install, npm install)\n"
        "           - running scripts, builds, or any CLI tool\n"
        "           - navigating or exploring a codebase\n\n"
        "  CODE   = create or modify source code files (writing new code or editing existing files)\n\n"
        "  TEST   = write or run unit/integration tests\n\n"
        "  IGNORE = not actionable by a program (e.g. open an IDE, review code visually,\n"
        "           think about architecture, make a decision)\n\n"
        f"Step: {step_text}\n\n"
        "Category:"
    )
    sent_before = token_tracker.total_prompt_tokens
    recv_before = token_tracker.total_completion_tokens

    response = llm_client.generate_response(prompt).strip().upper()

    sent_delta = token_tracker.total_prompt_tokens - sent_before
    recv_delta = token_tracker.total_completion_tokens - recv_before
    display.step_tokens(step_idx, sent_delta, recv_delta)

    for keyword in ("IGNORE", "CMD", "CODE", "TEST"):
        if keyword in response:
            return keyword
    return "CODE"


# ───────────────────────────────────────────────────────────────
# Step handlers — now accept ``language``
# ───────────────────────────────────────────────────────────────

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


def _handle_code_step(step_text: str, coder: CoderAgent, reviewer: ReviewerAgent,
                      executor: Executor, task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int,
                      language: str | None = None) -> tuple[bool, str]:
    feedback = ""
    ctx_budget = int(Config.CONTEXT_WINDOW * 0.8)

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

        for run_attempt in range(1, MAX_STEP_RETRIES + 1):
            display.step_info(step_idx, f"Running tests (attempt {run_attempt})...")
            success, output = executor.run_tests(test_cmd)
            log.info(f"Step {step_idx+1}: Test run output:\n{output}")

            last_test_output = output

            if success:
                display.step_info(step_idx, "Tests passed ✔")
                return True, ""

            # Auto-install missing packages before asking coder to fix
            missing_pkgs = executor.detect_missing_packages(output)
            if missing_pkgs:
                display.step_info(step_idx, f"Installing missing packages: {', '.join(missing_pkgs)}")
                log.info(f"Step {step_idx+1}: Auto-installing: {missing_pkgs}")
                install_ok, install_out = executor.install_packages(missing_pkgs)
                if install_ok:
                    display.step_info(step_idx, "Packages installed, re-running tests...")
                    success, output = executor.run_tests(test_cmd)
                    log.info(f"Step {step_idx+1}: Test re-run after install:\n{output}")
                    last_test_output = output
                    if success:
                        display.step_info(step_idx, "Tests passed after package install ✔")
                        return True, ""
                else:
                    log.warning(f"Step {step_idx+1}: Package install failed: {install_out}")

            display.step_info(step_idx, "Tests failed, asking coder to fix...")
            fix_context = (
                f"Test errors:\n{output[:500]}\n"
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


# ───────────────────────────────────────────────────────────────
# Diagnosis / fix helpers (unchanged logic, pass language)
# ───────────────────────────────────────────────────────────────

def _diagnose_failure(step_text: str, step_type: str, error_info: str,
                      memory: FileMemory, llm_client, display: CLIDisplay,
                      step_idx: int) -> str:
    display.step_info(step_idx, "Analyzing failure root cause...")

    context_files = memory.related_context(step_text)

    prior_context = _build_prior_steps_context(memory, step_idx)

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
               step_type: str = "CODE") -> bool:
    applied = False

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
        applied = True

    return applied


def _is_file_path(text: str) -> bool:
    """Return True if *text* looks like a bare file/directory path, not a command."""
    text = text.strip()
    # No spaces usually means it's a path, not a command with arguments
    # Exception: single-word commands like "pytest" are handled by _looks_like_command
    if ' ' in text:
        return False
    # Looks like a file path: contains slashes and/or has a file extension
    has_sep = '/' in text or '\\' in text
    has_ext = bool(re.search(r'\.\w{1,5}$', text))
    return has_sep or has_ext


def _looks_like_command(text: str) -> bool:
    """Return True if *text* looks like an executable shell command."""
    text = text.strip()
    if not text:
        return False
    # Reject bare file paths
    if _is_file_path(text):
        return False
    # Extract the first token, splitting on whitespace AND shell operators
    # so that "echo.>file" splits to "echo." and "type nul > file" splits to "type"
    first_token = re.split(r'[\s>|&;<]', text)[0].lower()
    # Strip trailing .exe suffix (not rstrip which eats individual chars)
    if first_token.endswith('.exe'):
        first_token = first_token[:-4]
    # Strip trailing dots (CMD echo. syntax)
    first_token = first_token.rstrip('.')

    known_commands = {
        'pip', 'pip3', 'python', 'python3', 'py',
        'npm', 'npx', 'node', 'yarn', 'pnpm',
        'go', 'cargo', 'rustc', 'mvn', 'gradle', 'javac', 'java',
        'ruby', 'bundle', 'gem', 'rspec',
        'git', 'docker', 'make', 'cmake',
        'mkdir', 'rmdir', 'del', 'copy', 'move', 'ren', 'type', 'dir',
        'ls', 'cat', 'cp', 'mv', 'rm', 'find', 'grep', 'chmod', 'chown',
        'cd', 'echo', 'set', 'export', 'source', 'touch',
        'curl', 'wget', 'ssh', 'scp',
        'apt', 'apt-get', 'brew', 'choco', 'yum', 'dnf', 'pacman',
        'powershell', 'pwsh', 'cmd',
        'pytest', 'jest', 'tox', 'mypy', 'flake8', 'black', 'ruff',
    }
    return first_token in known_commands


def _extract_commands_from_text(text: str) -> list[str]:
    """Extract shell commands from *text*, handling both triple- and single-backtick blocks.

    Prefers triple-backtick code blocks (```cmd, ```bash, ```shell, ```)
    over single-backtick inline code.  Filters out file paths and non-commands.
    """
    commands: list[str] = []
    seen: set[str] = set()

    # 1. Triple-backtick code blocks (```lang\n...\n```)
    for m in re.finditer(r"```(?:\w*)\n(.*?)```", text, re.DOTALL):
        block = m.group(1).strip()
        for line in block.splitlines():
            line = line.strip()
            if line and _looks_like_command(line) and line not in seen:
                commands.append(line)
                seen.add(line)

    # 2. Single-backtick inline commands (`...`)
    for m in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", text):
        cmd = m.group(1).strip()
        if cmd and _looks_like_command(cmd) and cmd not in seen:
            commands.append(cmd)
            seen.add(cmd)

    return commands


def _extract_command_from_step(step_text: str) -> str | None:
    """Extract an inline command from a step description.

    Only matches backtick content that looks like a real command,
    skipping bare file paths like ``tests/test_main.py``.
    """
    for m in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", step_text):
        candidate = m.group(1).strip()
        if _looks_like_command(candidate):
            return candidate
    return None


# ───────────────────────────────────────────────────────────────
# Parallel wave execution
# ───────────────────────────────────────────────────────────────

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
                  language: str | None) -> tuple[int, bool, str]:
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
                task, memory, display, step_idx, language=language)
            display.complete_step(step_idx, "done" if success else "failed")

        elif step_type == "TEST":
            success, error_info = _handle_test_step(
                step_text, tester, coder, reviewer, executor,
                task, memory, display, step_idx, language=language)
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


# ───────────────────────────────────────────────────────────────
# Main entry point
# ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AgentChanti — Multi-Agent Local Coder")
    parser.add_argument("task", help="The coding task to perform")
    parser.add_argument("--provider", choices=["ollama", "lm_studio"],
                        default="lm_studio", help="The LLM provider to use")
    parser.add_argument("--model", default=Config.DEFAULT_MODEL,
                        help="The model name to use")
    parser.add_argument("--embed-model", default=Config.EMBEDDING_MODEL,
                        help="Embedding model name (default: %(default)s)")
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
    args = parser.parse_args()

    # ── 1. Detect language ──
    if args.language:
        language = args.language
    else:
        language = detect_language_from_task(args.task) or detect_language()
    log.info(f"Language: {language} ({get_language_name(language)})")

    # ── 2. Init LLM client ──
    stream_enabled = Config.STREAM_RESPONSES and not args.no_stream
    llm_kwargs = dict(
        max_retries=Config.LLM_MAX_RETRIES,
        retry_delay=Config.LLM_RETRY_DELAY,
        stream=stream_enabled,
    )
    if args.provider == "ollama":
        llm_client = OllamaClient(
            base_url=Config.OLLAMA_BASE_URL, model=args.model, **llm_kwargs)
    else:
        llm_client = LMStudioClient(
            base_url=Config.LM_STUDIO_BASE_URL, model=args.model, **llm_kwargs)

    # ── 3. Scan existing project ──
    scan_result = scan_project(".")
    project_context = format_scan_for_planner(scan_result)
    log.info(f"Project scan: {scan_result['file_count']} files detected")

    # ── 4. Init embedding store ──
    embed_store = None
    if not args.no_embeddings:
        embed_store = EmbeddingStore(llm_client, embed_model=args.embed_model)
        log.info(f"Embeddings enabled (model: {args.embed_model})")
    else:
        log.info("Embeddings disabled")

    # ── 5. Init agents ──
    planner = PlannerAgent("Planner", "Senior Software Architect",
                           "Create a step-by-step plan for the coding task and related testcases.",
                           llm_client)
    coder = CoderAgent("Coder", "Senior Software Developer",
                       f"Write clean {get_language_name(language)} code for a single step.",
                       llm_client)
    reviewer = ReviewerAgent("Reviewer", "Code Reviewer",
                             "Review code for errors and style issues.", llm_client)
    tester = TesterAgent("Tester", "Software Engineer in Test",
                         "Create unit tests for the provided code.", llm_client)
    executor = Executor()

    # ── 6. Init display ──
    display = CLIDisplay(args.task)
    log.info(f"Task: {args.task}")
    log.info(f"Provider: {args.provider}, Model: {args.model}")

    # Wire streaming progress callback
    if stream_enabled:
        # We'll set per-step callbacks in the execution loop
        pass

    # ── 7. Check for checkpoint ──
    checkpoint_file = Config.CHECKPOINT_FILE
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
                resuming = CLIDisplay.prompt_resume(checkpoint_state)

    # ── 8. Restore state or create git checkpoint ──
    checkpoint_branch: str | None = None
    use_git = not args.no_git and git_utils.is_git_repo()

    if resuming and checkpoint_state:
        log.info("Resuming from checkpoint...")
        memory = FileMemory(embedding_store=embed_store, top_k=Config.EMBEDDING_TOP_K)
        memory.update(checkpoint_state.get("file_memory", {}))
        steps = checkpoint_state["steps"]
        step_results = checkpoint_state.get("step_results", {})
        start_from = checkpoint_state.get("completed_step", -1) + 1
        language = checkpoint_state.get("language", language)
        display.set_steps(steps)
        # Mark completed steps
        for idx in range(start_from):
            display.steps[idx]["status"] = "done"
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
        display.show_status("Requesting steps from planner...")
        log.info("Planning...")

        planner_context = ""
        if project_context:
            planner_context = f"Existing project:\n{project_context}"

        plan = planner.process(args.task, context=planner_context)
        log.info(f"Plan:\n{plan}")

        # ── 10. Parse steps + dependencies ──
        display.show_status("Parsing steps...")
        raw_steps = executor.parse_plan_steps(plan)
        if not raw_steps:
            log.error("Could not parse any steps from the plan.")
            print("\n  [ERROR] Could not parse any steps. Check the log file.\n")
            return

        steps, dependencies = executor.parse_step_dependencies(raw_steps)

        # ── 11. Plan approval loop ──
        if args.auto:
            log.info(f"Auto-approved {len(steps)} steps (--auto mode)")
        while not args.auto:
            action, removed = CLIDisplay.prompt_plan_approval(steps)
            if action == "approve":
                break
            elif action == "replan":
                display.show_status("Re-planning...")
                plan = planner.process(args.task, context=planner_context)
                log.info(f"Re-plan:\n{plan}")
                raw_steps = executor.parse_plan_steps(plan)
                if not raw_steps:
                    log.error("Could not parse any steps from re-plan.")
                    print("\n  [ERROR] Could not parse re-plan steps.\n")
                    return
                steps, dependencies = executor.parse_step_dependencies(raw_steps)
            elif action == "edit" and removed:
                # Remove steps by index (descending to preserve indices)
                for idx in sorted(removed, reverse=True):
                    if 0 <= idx < len(steps):
                        steps.pop(idx)
                # Rebuild dependencies after removal
                _, dependencies = executor.parse_step_dependencies(steps)

        display.set_steps(steps)
        display.render()
        log.info(f"Approved {len(steps)} steps.")

        memory = FileMemory(embedding_store=embed_store, top_k=Config.EMBEDDING_TOP_K)

    # ── 12. Build execution waves ──
    # Re-parse dependencies from current steps (they may have been cleaned)
    _, dependencies = executor.parse_step_dependencies(steps)
    waves = build_step_waves(steps, dependencies)
    log.info(f"Execution waves: {waves}")

    # ── 13. Execute waves ──
    pipeline_success = True

    for wave_idx, wave in enumerate(waves):
        # Filter out already-completed steps (for resume)
        pending = [i for i in wave if i >= start_from]
        if not pending:
            continue

        log.info(f"Wave {wave_idx+1}: executing steps {[i+1 for i in pending]}")

        if len(pending) == 1:
            # Single step — execute directly
            idx = pending[0]
            step_text = steps[idx]
            idx, success, error_info = _execute_step(
                idx, step_text,
                llm_client=llm_client, executor=executor,
                coder=coder, reviewer=reviewer, tester=tester,
                task=args.task, memory=memory, display=display,
                language=language,
            )

            if success:
                step_results[idx] = "done"
                save_checkpoint(checkpoint_file, args.task, steps, idx,
                                memory.as_dict(), step_results, language)
            else:
                # Diagnosis loop
                fixed = _run_diagnosis_loop(
                    idx, step_text, error_info,
                    llm_client=llm_client, executor=executor,
                    coder=coder, reviewer=reviewer, tester=tester,
                    task=args.task, memory=memory, display=display,
                    language=language,
                )
                if fixed:
                    step_results[idx] = "done"
                    save_checkpoint(checkpoint_file, args.task, steps, idx,
                                    memory.as_dict(), step_results, language)
                else:
                    pipeline_success = False
                    break
        else:
            # Multi-step wave — execute in parallel
            failed_steps: list[tuple[int, str]] = []

            with ThreadPoolExecutor(max_workers=min(len(pending), 4)) as pool:
                futures = {}
                for idx in pending:
                    f = pool.submit(
                        _execute_step, idx, steps[idx],
                        llm_client=llm_client, executor=executor,
                        coder=coder, reviewer=reviewer, tester=tester,
                        task=args.task, memory=memory, display=display,
                        language=language,
                    )
                    futures[f] = idx

                for future in as_completed(futures):
                    idx, success, error_info = future.result()
                    if success:
                        step_results[idx] = "done"
                    else:
                        failed_steps.append((idx, error_info))

            # Save checkpoint for completed steps
            max_completed = max(
                (i for i in step_results if step_results[i] == "done"),
                default=start_from - 1)
            save_checkpoint(checkpoint_file, args.task, steps, max_completed,
                            memory.as_dict(), step_results, language)

            # Handle failures
            for idx, error_info in failed_steps:
                step_text = steps[idx]
                fixed = _run_diagnosis_loop(
                    idx, step_text, error_info,
                    llm_client=llm_client, executor=executor,
                    coder=coder, reviewer=reviewer, tester=tester,
                    task=args.task, memory=memory, display=display,
                    language=language,
                )
                if fixed:
                    step_results[idx] = "done"
                    save_checkpoint(checkpoint_file, args.task, steps, idx,
                                    memory.as_dict(), step_results, language)
                else:
                    pipeline_success = False
                    break

            if not pipeline_success:
                break

    # ── 14. Finish ──
    if pipeline_success:
        display.finish(success=True)
        clear_checkpoint(checkpoint_file)
        log.info(f"Finished. Total tokens: {token_tracker.total_tokens} "
                 f"(sent={token_tracker.total_prompt_tokens}, "
                 f"recv={token_tracker.total_completion_tokens})")

        # Git: offer commit
        if use_git and git_utils.has_changes():
            if args.auto:
                git_choice = "commit"
                log.info("Auto-committing changes (--auto mode)")
            else:
                git_choice = CLIDisplay.prompt_git_action("complete")
            if git_choice == "commit":
                ok, msg = git_utils.commit_changes(
                    f"AgentChanti: {args.task[:60]}")
                print(f"  {'Committed!' if ok else 'Commit failed: ' + msg}")
            if checkpoint_branch:
                git_utils.delete_checkpoint_branch(checkpoint_branch)
    else:
        display.finish(success=False)
        log.info(f"Pipeline failed. Total tokens: {token_tracker.total_tokens}")

        # Git: offer rollback
        if use_git and checkpoint_branch:
            if args.auto:
                git_choice = "skip"
                log.info("Auto-skipping git rollback (--auto mode)")
            else:
                git_choice = CLIDisplay.prompt_git_action("failed")
            if git_choice == "rollback":
                ok, msg = git_utils.rollback_to_branch(checkpoint_branch)
                print(f"  {'Rolled back!' if ok else 'Rollback failed: ' + msg}")
            elif git_choice == "commit":
                ok, msg = git_utils.commit_changes(
                    f"AgentChanti (partial): {args.task[:50]}")
                print(f"  {'Committed!' if ok else 'Commit failed: ' + msg}")


def _run_diagnosis_loop(step_idx: int, step_text: str, error_info: str, *,
                        llm_client, executor, coder, reviewer, tester,
                        task: str, memory: FileMemory, display: CLIDisplay,
                        language: str | None) -> bool:
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

            fix_applied = _apply_fix(diagnosis, executor, memory, display, step_idx,
                                     step_type=step_type)

            if not fix_applied:
                display.step_info(step_idx, "No actionable fix found in diagnosis.")
                log.warning(f"Task {step_idx+1}: Diagnosis produced no actionable fix.")
                continue

            # Re-run the step
            display.step_info(step_idx, "Fix applied — retrying step...")
            _, success, error_info = _execute_step(
                step_idx, step_text,
                llm_client=llm_client, executor=executor,
                coder=coder, reviewer=reviewer, tester=tester,
                task=task, memory=memory, display=display,
                language=language,
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


if __name__ == "__main__":
    main()
