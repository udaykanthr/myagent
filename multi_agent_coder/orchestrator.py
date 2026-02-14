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

MAX_STEP_RETRIES = 3


def _classify_step(step_text: str, llm_client) -> str:
    """
    Send a single step description to the LLM to classify it.
    Returns one of: CMD, CODE, TEST, IGNORE
    """
    prompt = (
        "Classify the following task step into exactly one category.\n"
        "Reply with ONLY one word: CMD, CODE, TEST, or IGNORE\n"
        "  CMD    = run a specific shell command (must contain an actual command)\n"
        "  CODE   = create or modify source code files\n"
        "  TEST   = write or run unit tests\n"
        "  IGNORE = not actionable by a program (e.g. open a text editor,\n"
        "           open an IDE, save a file , review code visually,\n"
        "           set up environment , navigate directories)\n\n"
        f"Step: {step_text}\n\n"
        "Category:"
    )
    response = llm_client.generate_response(prompt).strip().upper()
    # Extract just the keyword from the response
    for keyword in ("IGNORE", "CMD", "CODE", "TEST"):
        if keyword in response:
            return keyword
    return "CODE"  # default fallback


def _handle_cmd_step(step_text: str, executor: Executor) -> bool:
    """Extract and run a shell command from the step description.
    Only runs if a command is specified in backticks, otherwise skips."""
    match = re.search(r"`([^`]+)`", step_text)
    if not match:
        print(f"  [SKIP] No executable command found in backticks.")
        return True

    cmd = match.group(1)
    print(f"  Running: {cmd}")
    success, output = executor.run_command(cmd)
    if output:
        print(f"  Output: {output}")
    if not success:
        print(f"  [WARN] Command failed.")
    return success


def _handle_code_step(step_text: str, coder: CoderAgent, reviewer: ReviewerAgent,
                      executor: Executor, task: str, accumulated_code: dict) -> bool:
    """Send a single step to the coder, write files, and review."""
    feedback = ""

    for attempt in range(1, MAX_STEP_RETRIES + 1):
        context = f"Task: {task}"
        if feedback:
            context += f"\nFeedback: {feedback}"

        print(f"  Coder attempt {attempt}...")
        response = coder.process(step_text, context=context)

        files = executor.parse_code_blocks(response)
        if not files:
            feedback = "No file markers found. Use #### [FILE]: path/to/file.py format."
            print(f"  No files parsed. Retrying...")
            continue

        executor.write_files(files)
        accumulated_code.update(files)

        # Review
        print(f"  Reviewing...")
        review = reviewer.process(
            f"Review this code:\n{response}",
            context=f"Step: {step_text}"
        )
        print(f"  Review: {review[:200]}...")

        if "code looks good" in review.lower():
            return True
        else:
            feedback = review
            print(f"  Review found issues, retrying...")

    print(f"  [FAIL] Could not complete step after {MAX_STEP_RETRIES} attempts.")
    return False


def _handle_test_step(step_text: str, tester: TesterAgent, coder: CoderAgent,
                      executor: Executor, task: str, accumulated_code: dict) -> bool:
    """Generate tests, run them, retry code on failure."""
    code_summary = ""
    for fname, content in accumulated_code.items():
        code_summary += f"#### [FILE]: {fname}\n```python\n{content}\n```\n\n"

    print(f"  Generating tests...")
    test_response = tester.process(step_text, context=f"Code:\n{code_summary}")
    test_files = executor.parse_code_blocks(test_response)

    if not test_files:
        print(f"  No test files generated.")
        return False

    executor.write_files(test_files)

    for attempt in range(1, MAX_STEP_RETRIES + 1):
        print(f"  Running tests (attempt {attempt})...")
        success, output = executor.run_tests()
        if output:
            print(f"  {output[:300]}")

        if success:
            print(f"  Tests passed!")
            return True

        print(f"  Tests failed. Asking coder to fix...")
        fix_response = coder.process(
            "Fix the code so tests pass.",
            context=f"Test errors:\n{output[:500]}"
        )
        fix_files = executor.parse_code_blocks(fix_response)
        if fix_files:
            executor.write_files(fix_files)
            accumulated_code.update(fix_files)

    print(f"  [FAIL] Tests still failing after {MAX_STEP_RETRIES} fix attempts.")
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
                           "Create a step-by-step plan for the coding task.", llm_client)
    coder = CoderAgent("Coder", "Senior Software Developer",
                       "Write clean Python code for a single step.", llm_client)
    reviewer = ReviewerAgent("Reviewer", "Code Reviewer",
                             "Review code for errors and style issues.", llm_client)
    tester = TesterAgent("Tester", "Software Engineer in Test",
                         "Create unit tests for the provided code.", llm_client)
    executor = Executor()

    print(f"\n{'='*60}")
    print(f"  Task: {args.task}")
    print(f"{'='*60}\n")

    # --- Step 1: Get plan ---
    print("--- Planning ---\n")
    plan = planner.process(args.task)
    print(plan)
    print()

    # --- Step 2: Parse steps by splitting numbered list ---
    steps = executor.parse_plan_steps(plan)
    if not steps:
        print("[ERROR] Could not parse any steps from the plan. Exiting.")
        return

    print(f"Parsed {len(steps)} steps.\n")

    # --- Step 3: Process each step one by one ---
    accumulated_code: dict = {}

    for i, step_text in enumerate(steps, 1):
        print(f"\n--- Step {i}: {step_text} ---\n")

        # Classify this step by sending it to the LLM
        step_type = _classify_step(step_text, llm_client)
        print(f"  Classified as: [{step_type}]\n")

        if step_type == "IGNORE":
            print(f"  [SKIP] Not actionable, skipping.")

        elif step_type == "CMD":
            _handle_cmd_step(step_text, executor)

        elif step_type == "CODE":
            _handle_code_step(step_text, coder, reviewer, executor,
                              args.task, accumulated_code)

        elif step_type == "TEST":
            _handle_test_step(step_text, tester, coder, executor,
                              args.task, accumulated_code)
        else:
            print(f"  Unknown type '{step_type}', skipping.")

    print(f"\n{'='*60}")
    print(f"  All steps processed.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
