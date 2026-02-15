import argparse
import re
from .config import Config
from .llm.ollama import OllamaClient
from .llm.lm_studio import LMStudioClient
from .agents.planner import PlannerAgent
from .agents.coder import CoderAgent
from .agents.reviewer import ReviewerAgent
from .agents.tester import TesterAgent
from .executor import Executor
from .cli_display import CLIDisplay, token_tracker, log

MAX_STEP_RETRIES = 3


class FileMemory:
    """Tracks every file's path and current contents across all steps."""

    def __init__(self):
        self._files: dict[str, str] = {}   # filepath -> contents

    def update(self, files: dict[str, str]):
        """Store or overwrite file contents."""
        self._files.update(files)

    def get(self, filepath: str) -> str | None:
        return self._files.get(filepath)

    def all_files(self) -> dict[str, str]:
        return dict(self._files)

    def related_context(self, step_text: str) -> str:
        """Build a compact context string with contents of files
        whose name appears in the step description."""
        parts = []
        for fpath, content in self._files.items():
            basename = fpath.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
            if basename in step_text or fpath in step_text:
                parts.append(f"#### [FILE]: {fpath}\n```\n{content}\n```")
        return "\n\n".join(parts)

    def summary(self) -> str:
        """One-line-per-file overview (name only, no contents)."""
        if not self._files:
            return "(no files yet)"
        return ", ".join(self._files.keys())


def _classify_step(step_text: str, llm_client, display: CLIDisplay, step_idx: int) -> str:
    """
    Send a single step description to the LLM to classify it.
    Returns one of: CMD, CODE, TEST, IGNORE
    """
    display.step_info(step_idx, "Classifying step...")
    prompt = (
        "Classify the following task step into exactly one category.\n"
        "Reply with ONLY one word: CMD, CODE, TEST, or IGNORE\n"
        "  CMD    = run a specific shell command (must contain an actual command)\n"
        "  CODE   = create or modify source code files, make sure path and filename are correct and meaningful\n"
        "  TEST   = write or run unit tests, make sure path and filename are correct and meaningful\n"
        "  IGNORE = not actionable by a program (e.g. open a text editor,\n"
        "           open an IDE, save a file, review code visually,\n"
        "           set up environment , navigate directories)\n\n"
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


def _handle_cmd_step(step_text: str, executor: Executor,
                     display: CLIDisplay, step_idx: int) -> bool:
    """Extract and run a shell command from the step description."""
    match = re.search(r"`([^`]+)`", step_text)
    if not match:
        display.step_info(step_idx, "No command in backticks, skipping.")
        log.info(f"Step {step_idx+1}: No backtick command found, skipping.")
        return True

    cmd = match.group(1)
    display.step_info(step_idx, f"Running: {cmd}")
    log.info(f"Step {step_idx+1}: Running command: {cmd}")

    success, output = executor.run_command(cmd)
    log.info(f"Step {step_idx+1}: Command output:\n{output}")

    if success:
        display.step_info(step_idx, f"Command succeeded.")
    else:
        display.step_info(step_idx, f"Command failed. See log.")
        log.warning(f"Step {step_idx+1}: Command failed.")
    return success


def _handle_code_step(step_text: str, coder: CoderAgent, reviewer: ReviewerAgent,
                      executor: Executor, task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int) -> bool:
    """Send a single step to the coder, write files, and review."""
    feedback = ""

    for attempt in range(1, MAX_STEP_RETRIES + 1):
        context = f"Task: {task}"
        related = memory.related_context(step_text)
        if related:
            context += f"\nExisting files (overwrite as needed):\n{related}"
        if memory.summary() != "(no files yet)":
            context += f"\nAll project files: {memory.summary()}"
        if feedback:
            context += f"\nFeedback: {feedback}"

        display.step_info(step_idx, f"Coding (attempt {attempt}/{MAX_STEP_RETRIES})...")
        sent_before = token_tracker.total_prompt_tokens
        recv_before = token_tracker.total_completion_tokens

        response = coder.process(step_text, context=context)

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
            context=f"Step: {step_text}"
        )

        sent_delta = token_tracker.total_prompt_tokens - sent_before
        recv_delta = token_tracker.total_completion_tokens - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        log.info(f"Step {step_idx+1}: Review:\n{review}")

        if "code looks good" in review.lower():
            display.step_info(step_idx, "Review passed ✔")
            return True
        else:
            feedback = review
            display.step_info(step_idx, "Review found issues, retrying...")
            log.warning(f"Step {step_idx+1}: Review issues: {review[:200]}")

    log.error(f"Step {step_idx+1}: Failed after {MAX_STEP_RETRIES} attempts.")
    return False


def _handle_test_step(step_text: str, tester: TesterAgent, coder: CoderAgent,
                      reviewer: ReviewerAgent, executor: Executor,
                      task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int) -> bool:
    """Generate tests, review them, run them, retry on failure."""
    code_summary = ""
    for fname, content in memory.all_files().items():
        code_summary += f"#### [FILE]: {fname}\n```python\n{content}\n```\n\n"

    feedback = ""

    for gen_attempt in range(1, MAX_STEP_RETRIES + 1):
        display.step_info(step_idx, f"Generating tests (attempt {gen_attempt})...")
        gen_context = f"Code:\n{code_summary}"
        if feedback:
            gen_context += f"\nFeedback: {feedback}"

        sent_before = token_tracker.total_prompt_tokens
        recv_before = token_tracker.total_completion_tokens

        test_response = tester.process(step_text, context=gen_context)

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
            context=f"Project files: {memory.summary()}\n{code_summary}"
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
            success, output = executor.run_tests()
            log.info(f"Step {step_idx+1}: Test run output:\n{output}")

            if success:
                display.step_info(step_idx, "Tests passed ✔")
                return True

            display.step_info(step_idx, "Tests failed, asking coder to fix...")
            fix_context = (
                f"Test errors:\n{output[:500]}\n"
                f"Project files:\n{code_summary}"
            )

            sent_before = token_tracker.total_prompt_tokens
            recv_before = token_tracker.total_completion_tokens

            fix_response = coder.process("Fix the code so tests pass.", context=fix_context)

            sent_delta = token_tracker.total_prompt_tokens - sent_before
            recv_delta = token_tracker.total_completion_tokens - recv_before
            display.step_tokens(step_idx, sent_delta, recv_delta)

            fix_files = executor.parse_code_blocks(fix_response)
            if fix_files:
                executor.write_files(fix_files)
                memory.update(fix_files)
                code_summary = ""
                for fname, content in memory.all_files().items():
                    code_summary += f"#### [FILE]: {fname}\n```python\n{content}\n```\n\n"

        log.error(f"Step {step_idx+1}: Tests still failing after {MAX_STEP_RETRIES} fixes.")
        return False

    log.error(f"Step {step_idx+1}: Could not generate valid tests after {MAX_STEP_RETRIES} attempts.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Coder")
    parser.add_argument("task", help="The coding task to perform")
    parser.add_argument("--provider", choices=["ollama", "lm_studio"],
                        default="lm_studio", help="The LLM provider to use")
    parser.add_argument("--model", default=Config.DEFAULT_MODEL,
                        help="The model name to use")
    args = parser.parse_args()

    # Init LLM
    if args.provider == "ollama":
        llm_client = OllamaClient(base_url=Config.OLLAMA_BASE_URL, model=args.model)
    else:
        llm_client = LMStudioClient(base_url=Config.LM_STUDIO_BASE_URL, model=args.model)

    # Init agents
    planner = PlannerAgent("Planner", "Senior Software Architect",
                           "Create a step-by-step plan for the coding task and related testcases.", llm_client)
    coder = CoderAgent("Coder", "Senior Software Developer",
                       "Write clean Python code for a single step.", llm_client)
    reviewer = ReviewerAgent("Reviewer", "Code Reviewer",
                             "Review code for errors and style issues.", llm_client)
    tester = TesterAgent("Tester", "Software Engineer in Test",
                         "Create unit tests for the provided code.", llm_client)
    executor = Executor()

    # Init display
    display = CLIDisplay(args.task)
    log.info(f"Task: {args.task}")
    log.info(f"Provider: {args.provider}, Model: {args.model}")

    # --- Step 1: Planning ---
    display.render()
    log.info("Planning...")

    sent_before = token_tracker.total_prompt_tokens
    recv_before = token_tracker.total_completion_tokens
    plan = planner.process(args.task)
    log.info(f"Plan:\n{plan}")

    # --- Step 2: Parse steps ---
    steps = executor.parse_plan_steps(plan)
    if not steps:
        log.error("Could not parse any steps from the plan.")
        print("\n  [ERROR] Could not parse any steps. Check the log file.\n")
        return

    display.set_steps(steps)
    display.render()
    log.info(f"Parsed {len(steps)} steps.")

    # --- Step 3: Process each step ---
    memory = FileMemory()
    any_failed = False

    for i, step_text in enumerate(steps):
        log.info(f"\n{'='*60}\nStep {i+1}: {step_text}\nMemory: {memory.summary()}\n{'='*60}")

        # Classify
        display.start_step(i)
        step_type = _classify_step(step_text, llm_client, display, i)
        display.steps[i]["type"] = step_type
        display.render()
        log.info(f"Step {i+1}: Classified as [{step_type}]")

        success = True

        if step_type == "IGNORE":
            display.step_info(i, "Not actionable, skipping.")
            display.complete_step(i, "skipped")

        elif step_type == "CMD":
            success = _handle_cmd_step(step_text, executor, display, i)
            display.complete_step(i, "done" if success else "failed")

        elif step_type == "CODE":
            success = _handle_code_step(step_text, coder, reviewer, executor,
                                        args.task, memory, display, i)
            display.complete_step(i, "done" if success else "failed")

        elif step_type == "TEST":
            success = _handle_test_step(step_text, tester, coder, reviewer, executor,
                                        args.task, memory, display, i)
            display.complete_step(i, "done" if success else "failed")

        else:
            display.step_info(i, f"Unknown type '{step_type}', skipping.")
            display.complete_step(i, "skipped")

        if not success:
            any_failed = True

    display.finish(success=not any_failed)
    log.info(f"Finished. Total tokens: {token_tracker.total_tokens} "
             f"(sent={token_tracker.total_prompt_tokens}, "
             f"recv={token_tracker.total_completion_tokens})")


if __name__ == "__main__":
    main()
