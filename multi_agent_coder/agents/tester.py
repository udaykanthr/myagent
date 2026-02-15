from .base import Agent


class TesterAgent(Agent):
    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += """
Generate unit tests using `pytest`.
IMPORTANT: Use the exact file paths from the context to build correct import statements.
For example if the source file is at `src/bubble_sort.py`, import it as `from src.bubble_sort import ...`.

For EACH test file, use EXACTLY this marker format (no extra text after the path):

#### [FILE]: tests/test_example.py
```python
# test code here
```

Rules:
- The path after [FILE]: must be ONLY the file path, nothing else.
- Use forward slashes in paths.
- Do NOT add descriptions, comments, or parentheses after the file path.
"""
        return self.llm_client.generate_response(prompt)
