from .base import Agent


class ReviewerAgent(Agent):
    def process(self, task: str, context: str = "", language: str | None = None) -> str:
        prompt = self._build_prompt(task, context, language=language)
        prompt += """

Review the provided code for CRITICAL issues ONLY:
- Bugs, logic errors, or incorrect behavior
- Missing imports or undefined variables
- Wrong function signatures or return types
- Security vulnerabilities (SQL injection, XSS, etc.)
- Code that will crash at runtime

Do NOT flag:
- Style suggestions (naming, comments, docstrings)
- Minor improvements or refactoring ideas
- Missing error handling for unlikely edge cases
- Performance optimizations unless there is a clear bug

If the code is functionally correct and will work as intended, respond with EXACTLY:
"Code looks good."

If there are critical issues, list ONLY the critical bugs (not style suggestions)."""
        return self.llm_client.generate_response(prompt)
