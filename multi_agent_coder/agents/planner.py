import os

from .base import Agent


def _shell_example() -> str:
    """Return an OS-appropriate file-listing command example."""
    if os.name == 'nt':
        return "  1. List all project files with `dir /s /b`"
    return "  1. List all project files with `find . -type f`"


class PlannerAgent(Agent):
    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += f"""

Provide a step-by-step plan as a numbered list.
Keep each step short and actionable. Do NOT include code in this plan.

IMPORTANT RULES:
- Base your plan ONLY on the project context provided above.
- Do NOT add meta-steps like "Review code", "Identify issues", "Analyze project structure", or "List files".
- The project context is already provided to you; go straight to implementation or modification steps.
- Combine "Identify" and "Fix" into a single actionable step (e.g., "Fix the logic in X" instead of "Identify bug in X" followed by "Fix bug in X").
- Each step should produce a concrete, verifiable result.
- For steps that involve running shell commands (installing packages, running scripts, etc.), include the exact command in backticks.

Example:
  1. Create a new utility function in `utils.py` for input validation
  2. Update the API endpoint in `app.py` to use the new validation (depends: 1)
  3. Create unit tests for the validation logic in `tests/test_utils.py` (depends: 1)
  4. Run tests with `pytest` (depends: 3)

For each step, if it depends on a previous step being completed first, add (depends: N) or (depends: N, M) at the end of the step.
Steps with no dependencies can run in parallel.
"""
        return self.llm_client.generate_response(prompt)
