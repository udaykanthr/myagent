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
        prompt += """

You are a SENIOR SOFTWARE ARCHITECT creating an execution plan that will be
carried out by an automated pipeline. Each step is executed by one of three
agents: a CODER (writes files), a CMD runner (executes shell commands), or a
TESTER (generates and runs unit tests). Your plan MUST be precise enough for
these agents to succeed on the first attempt.

═══════ STEP FORMAT ═══════
Write a numbered list. Each step MUST be a single, concrete action:

  1. Install dependencies with `npm install express cors` (depends: none)
  2. Create the Express server in `src/server.js` with GET /api/health endpoint (depends: 1)
  3. Add input validation utility in `src/utils/validate.js` (depends: 1)
  4. Update `src/server.js` to use validation from `src/utils/validate.js` (depends: 2, 3)

═══════ STEP RULES (CRITICAL) ═══════

1. **Reference EXACT file paths**: Every CODE step must name the specific
   file(s) to create or modify. Use the paths from the project context above.
   Say "Update `src/index.js`" NOT "Update the main file".

2. **One action per step**: Each step should do ONE thing. Don't combine
   "create file AND install package" in one step. Split them.

3. **CMD steps for shell commands**: Installing packages, running scripts,
   creating directories — put the exact command in backticks.
   Examples: `npm install express`, `pip install flask`, `mkdir -p src/utils`

4. **CODE steps for file changes**: Creating or modifying source files.
   Always specify the file path. For modifications, say "Update `path/to/file`"
   and describe WHAT to change.

5. **Existing files = MODIFY, not recreate**: When files already exist in the
   project (shown in context above), plan to UPDATE them. Reference their
   exact paths. Do NOT plan to create a file that already exists.

6. **Dependencies between steps**: Add `(depends: N)` or `(depends: N, M)` at
   the end of steps that need prior steps completed first. Steps without
   dependencies can run in parallel.

7. **Logical ordering**: Dependencies first, then dependents:
   - Install packages → Write code that uses them
   - Create utility files → Write code that imports them
   - Write source code → Write tests for it

8. **NO meta-steps**: Do NOT include steps like "Analyze the project",
   "Review the code", "Identify the bug", or "Plan the implementation".
   Jump straight to actionable steps.

9. **NO test steps unless asked**: Do NOT add test steps (writing or running
   tests) UNLESS the user's task EXPLICITLY asks for tests.

10. **Shell commands are non-interactive**: Always include --yes, -y, or
    --defaults flags for tools that prompt for input:
    - `npx create-next-app . --yes`
    - `npm init -y`
    - `ng new myapp --defaults`

═══════ QUALITY CHECKLIST (verify before outputting) ═══════
- [ ] Every file path in the plan matches an existing project file OR is
      clearly marked as a new file to create
- [ ] No two steps create/modify the same file (consolidate into one step)
- [ ] Import dependencies: if step N creates a module that step M imports,
      M depends on N
- [ ] Package dependencies: if step N installs a package that step M uses,
      M depends on N
- [ ] No vague steps ("improve the code", "fix issues", "update as needed")
- [ ] Each step is specific enough that a developer could execute it without
      asking questions
- [ ] Total steps are between 2-15 (break large tasks down, but don't over-split)
"""
        return self.llm_client.generate_response(prompt)
