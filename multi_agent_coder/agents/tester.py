from .base import Agent
from ..language import get_code_block_lang, get_test_framework, get_language_name


class TesterAgent(Agent):
    def process(self, task: str, context: str = "", language: str | None = None) -> str:
        prompt = self._build_prompt(task, context, language=language)
        fw = get_test_framework(language) if language else get_test_framework("python")
        lang_tag = get_code_block_lang(language) if language else "python"
        lang_name = get_language_name(language) if language else "Python"
        test_dir = fw.get("dir", "tests")
        test_ext = fw.get("ext", ".py")

        prompt += f"""
Generate unit tests for the provided code using `{fw['command']}`.
Language: {lang_name}

CRITICAL IMPORT RULE — read carefully:
Convert the source file path to an import by replacing '/' with '.' and removing the file extension.
NEVER drop directory prefixes. If the file is in a subdirectory, the import MUST include that directory.

Examples:
  File at `src/calculator.py`       → `from src.calculator import ...`
  File at `src/utils/helpers.py`    → `from src.utils.helpers import ...`
  File at `app/models/user.py`      → `from app.models.user import ...`
  File at `calculator.py`           → `from calculator import ...`

WRONG (do NOT do this):
  File at `src/calculator.py`       → `from calculator import ...`   ← WRONG! Missing 'src.'

For EACH test file, use EXACTLY this marker format (no extra text after the path):

#### [FILE]: {test_dir}/test_example{test_ext}
```{lang_tag}
# test code here
```

Rules:
- The path after [FILE]: must be ONLY the file path, nothing else.
- Use forward slashes in paths.
- Do NOT add descriptions, comments, or parentheses after the file path.
"""
        return self.llm_client.generate_response(prompt)
