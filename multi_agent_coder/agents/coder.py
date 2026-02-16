from .base import Agent
from ..language import get_code_block_lang, get_language_name


class CoderAgent(Agent):
    def process(self, task: str, context: str = "", language: str | None = None) -> str:
        prompt = self._build_prompt(task, context, language=language)
        lang_tag = get_code_block_lang(language) if language else "python"
        lang_name = get_language_name(language) if language else "Python"
        prompt += f"""
Please provide the implementation.
Write clean, idiomatic {lang_name} code.
For EACH file, use EXACTLY this marker format (no extra text after the path):

#### [FILE]: src/my_module.py
```{lang_tag}
# code here
```

Rules:
- The path after [FILE]: must be ONLY the file path, nothing else.
- Use forward slashes in paths.
- Do NOT add descriptions, comments, or parentheses after the file path.
"""
        return self.llm_client.generate_response(prompt)
