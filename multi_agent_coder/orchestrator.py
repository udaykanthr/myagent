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

def _handle_cmd_step(step_text: str, executor: Executor,
                     llm_client, memory: FileMemory,
                     display: CLIDisplay, step_idx: int,
                     language: str | None = None) -> tuple[bool, str]:
    match = re.search(r"`([^`]+)`", step_text)

    if match:
        cmd = match.group(1)
    else:
        display.step_info(step_idx, "Generating command...")
        gen_prompt = (
            "You are a shell command generator. Given a task step, output "
            "ONLY the shell command to accomplish it. No explanations, no "
            "markdown, no backticks — just the raw command.\n"
            f"{_shell_instructions()}\n"
            f"Step: {step_text}\n\n"
            "Command:"
        )
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

        if "code looks good" in review.lower():
            display.step_info(step_idx, "Review passed ✔")
            return True, ""
        else:
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

        if "code looks good" not in review.lower():
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

    prompt = (
        "A step in our automated coding pipeline has FAILED after multiple retries.\n"
        "Analyze the failure and provide a concrete fix.\n\n"
        f"Step {step_idx+1}: {step_text}\n"
        f"Step type: {step_type}\n\n"
        f"Error details:\n{error_info}\n\n"
    )
    if context_files:
        prompt += f"Relevant project files:\n{context_files}\n\n"
    prompt += (
        f"All project files: {memory.summary()}\n\n"
        "Respond with:\n"
        "1. ROOT CAUSE: one-line explanation of what went wrong\n"
        "2. FIX: provide corrected code using #### [FILE]: path/file.py markers "
        "with fenced code blocks, OR a shell command in backticks to fix the issue.\n"
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
               display: CLIDisplay, step_idx: int) -> bool:
    applied = False

    files = executor.parse_code_blocks(diagnosis)
    if files:
        written = executor.write_files(files)
        memory.update(files)
        display.step_info(step_idx, f"Fixed files: {', '.join(written)}")
        log.info(f"Step {step_idx+1}: Applied code fixes to: {', '.join(written)}")
        applied = True

    cmd_matches = re.findall(r"`([^`]+)`", diagnosis)
    cmd_indicators = (
        'pip ', 'npm ', 'mkdir ', 'python ', 'install',
        'apt ', 'brew ', 'choco ', 'set ', 'export ',
        'curl ', 'wget ', 'git ', 'New-Item', 'Set-', 'Get-',
    )
    for cmd in cmd_matches:
        cmd = cmd.strip()
        if not cmd or '\n' in cmd or cmd.startswith('#'):
            continue
        if any(ind in cmd for ind in cmd_indicators):
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
    """Execute a single step. Returns ``(step_idx, success, error_info)``."""
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
            if args.resume:
                resuming = True
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
        while True:
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
    """Run diagnose → fix → retry loop. Returns ``True`` if the step was fixed."""
    for diag_attempt in range(1, MAX_DIAGNOSIS_RETRIES + 1):
        display.step_info(
            step_idx, f"Diagnosing failure ({diag_attempt}/{MAX_DIAGNOSIS_RETRIES})...")
        log.info(f"Task {step_idx+1}: Diagnosis attempt "
                 f"{diag_attempt}/{MAX_DIAGNOSIS_RETRIES}")

        step_type = display.steps[step_idx].get("type", "CODE")
        diagnosis = _diagnose_failure(
            step_text, step_type, error_info,
            memory, llm_client, display, step_idx)

        fix_applied = _apply_fix(diagnosis, executor, memory, display, step_idx)

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

    display.step_info(
        step_idx, "Step failed after all fix attempts. Halting pipeline.")
    log.error(f"Task {step_idx+1}: Failed after {MAX_DIAGNOSIS_RETRIES} "
              f"diagnosis attempts. Halting pipeline.")
    return False


if __name__ == "__main__":
    main()
