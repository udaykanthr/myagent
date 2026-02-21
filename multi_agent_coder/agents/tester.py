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

        # Extract file paths from context so LLM knows the real project layout
        file_listing = self._extract_file_listing(context)

        prompt += f"""
Generate unit tests for the provided code using `{fw['command']}`.
Language: {lang_name}

{'PROJECT FILE LISTING (use these EXACT paths for imports):' + chr(10) + file_listing + chr(10) if file_listing else ''}
CRITICAL PATH RULES — violations cause import failures:
1. Use the EXACT file paths shown in the code context above for imports.
   Do NOT shorten, rename, or strip directory prefixes from paths.
2. Convert the source file path to an import by replacing '/' with '.' and removing the file extension.
3. NEVER drop directory prefixes. If the file is at `my-app/src/calc.py`, the import MUST include `my_app.src.calc`.

Examples:
  File at `src/calculator.py`       → `from src.calculator import ...`
  File at `src/utils/helpers.py`    → `from src.utils.helpers import ...`
  File at `app/models/user.py`      → `from app.models.user import ...`
  File at `calculator.py`           → `from calculator import ...`

WRONG (do NOT do this):
  File at `src/calculator.py`       → `from calculator import ...`   ← WRONG! Missing 'src.'
"""

        # Add language-specific test guidance
        if language in ("javascript", "typescript"):
            prompt += self._js_test_rules(language, fw)
        elif language == "python" or language is None:
            prompt += self._python_test_rules()

        prompt += f"""
OUTPUT FORMAT — for EACH test file, use EXACTLY this marker:

#### [FILE]: {test_dir}/test_example{test_ext}
```{lang_tag}
// test code here
```

STRICT RULES:
- The path after [FILE]: must be ONLY the file path, nothing else.
- Use forward slashes in paths.
- Do NOT add descriptions, comments, or parentheses after the file path.
- ONLY output test files. Do NOT output source files, package.json, or config files.
- Do NOT modify, recreate, or output package.json, tsconfig.json, or any config file.
- Keep tests focused: test actual function behavior, not framework boilerplate.
- If a function has no `module.exports` or `export`, use the actual function/class names from the source code.
"""
        return self.llm_client.generate_response(prompt)

    @staticmethod
    def _extract_file_listing(context: str) -> str:
        """Pull file paths from #### [FILE]: markers in the context."""
        import re
        paths = re.findall(r"####\s*\[FILE\]:\s*(\S+)", context)
        if not paths:
            return ""
        return "\n".join(f"  - {p}" for p in paths)

    @staticmethod
    def _js_test_rules(language: str | None, fw: dict) -> str:
        """Return JavaScript/TypeScript-specific test guidance."""
        ext = ".ts" if language == "typescript" else ".js"
        import_style = "import" if language == "typescript" else "require"

        return f"""
JAVASCRIPT/TYPESCRIPT TEST RULES (critical for Jest):
1. Import source modules using `require()` for .js or `import` for .ts:
   - JS: `const {{ funcName }} = require('../src/module');`
   - TS: `import {{ funcName }} from '../src/module';`
2. The import path must be RELATIVE from the test file to the source file.
   Example: test at `__tests__/calc.test{ext}`, source at `src/calculator{ext}`
   → Use `require('../src/calculator')` or `import ... from '../src/calculator'`
3. Do NOT use absolute paths or module aliases unless the project defines them.
4. If the source file uses `module.exports`, use `require()` in tests.
   If the source file uses `export`, use `import` in tests.
5. Wrap each test in `describe` and `it`/`test` blocks.
6. Do NOT install packages, modify package.json, or create jest.config files.
7. Do NOT create or modify source files — ONLY write test files.
8. Use `expect(...).toBe(...)`, `toEqual(...)`, `toThrow(...)` etc. for assertions.
9. For async functions, use `async/await` with `expect(...).resolves` or `await expect(...)`.
"""

    @staticmethod
    def _python_test_rules() -> str:
        """Return Python-specific test guidance."""
        return """
PYTHON TEST RULES (critical for pytest):
1. Use `from <module_path> import <name>` matching the EXACT source file path.
2. Do NOT create or modify source files — ONLY write test files.
3. Use `pytest` conventions: functions named `test_*`, use `assert` statements.
4. For exceptions, use `pytest.raises(ExceptionType)`.
5. For fixtures, use `@pytest.fixture` decorator.
6. Do NOT install packages or modify requirements.txt.
"""
