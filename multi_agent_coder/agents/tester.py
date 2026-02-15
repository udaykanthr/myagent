from .base import Agent


class TesterAgent(Agent):
    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += """
Generate unit tests using `pytest`.
IMPORTANT: Use the exact file paths from the context to build correct import statements.
For example if the source file is at `src/bubble_sort.py`, import it as `from src.bubble_sort import ...`.
Format your response using:
#### [FILE]: tests/test_[file_name].[ext]
```[language]
# test code here
```
"""
        return self.llm_client.generate_response(prompt)
