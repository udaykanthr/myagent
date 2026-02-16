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
- Base your plan ONLY on what actually exists in the project (see context above).
- Do NOT assume files exist unless they are listed in the project context.
- Do NOT include steps to install from requirements.txt or package.json unless those files are confirmed to exist.
- Do NOT add unnecessary setup steps. Focus on what the task actually needs.
- Each step should produce a concrete, verifiable result.

For steps that involve running shell commands (scanning files, listing directories,
installing packages, etc.), include the exact command in backticks, e.g.:
{_shell_example()}
  2. Install a specific package with `pip install pytest`

For each step, if it depends on a previous step being completed first, add (depends: N) or (depends: N, M) at the end of the step.
Steps with no dependencies can run in parallel.
Example:
  1. Create the data model
  2. Create the API endpoints (depends: 1)
  3. Write unit tests (depends: 1, 2)
"""
        return self.llm_client.generate_response(prompt)
