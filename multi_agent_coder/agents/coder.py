from .base import Agent

class CoderAgent(Agent):
    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += """
Please provide the implementation.
For EACH file, use EXACTLY this marker format (no extra text after the path):

#### [FILE]: src/my_module.py
```python
# code here
```

Rules:
- The path after [FILE]: must be ONLY the file path, nothing else.
- Use forward slashes in paths.
- Do NOT add descriptions, comments, or parentheses after the file path.
"""
        return self.llm_client.generate_response(prompt)
