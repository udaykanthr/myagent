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
- All commands run non-interactively (no terminal input). Always include --yes, -y, or --defaults flags for tools that prompt for input (e.g. `npx create-next-app . --yes`, `npm init -y`).
- Do NOT add test steps (writing tests or running tests) unless the user's task EXPLICITLY asks for tests. Never auto-generate test steps on your own.
- CRITICAL — Existing files: When existing source files are shown in the "Existing Source Files" section above, you MUST plan to MODIFY those files rather than creating new ones. Reference the specific file paths (e.g., "Update `src/index.html` to add a navbar"). Only plan to create NEW files when the task genuinely requires new functionality that does not belong in any existing file.

Example:
  1. Create a new utility function in `utils.py` for input validation
  2. Update the API endpoint in `app.py` to use the new validation (depends: 1)

For each step, if it depends on a previous step being completed first, add (depends: N) or (depends: N, M) at the end of the step.
Steps with no dependencies can run in parallel.
"""
        return self.llm_client.generate_response(prompt)
