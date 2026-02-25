from .base import Agent
from ..language import (
    get_code_block_lang, get_test_framework, get_language_name,
    detect_language_from_files, detect_test_runner,
)


class TesterAgent(Agent):
    def process(self, task: str, context: str = "",
                language: str | None = None,
                env_info: dict | None = None) -> str:
        # Infer language from context file paths when not explicitly provided
        if language is None:
            file_paths = self._extract_file_paths(context)
            if file_paths:
                language = detect_language_from_files(file_paths)
            # Final fallback to Python if still unknown
            if language is None:
                language = "python"

        # Detect test runner (vitest vs jest) for JS/TS projects
        test_runner: str | None = None
        if language in ("javascript", "typescript"):
            test_runner = detect_test_runner()
            if env_info and env_info.get("test_runner"):
                test_runner = env_info["test_runner"]

        prompt = self._build_prompt(task, context, language=language)
        fw = get_test_framework(language, test_runner=test_runner)
        lang_tag = get_code_block_lang(language)
        lang_name = get_language_name(language)
        test_dir = fw.get("dir", "tests")
        test_ext = fw.get("ext", ".py")

        # Extract file paths from context so LLM knows the real project layout
        file_listing = self._extract_file_listing(context)

        prompt += f"""
Generate unit tests for the provided code using `{fw['command']}`.
Language: {lang_name}

{'PROJECT FILE LISTING (use these EXACT paths for imports):' + chr(10) + file_listing + chr(10) if file_listing else ''}
"""
        # Language-aware import path rules
        if language in ("javascript", "typescript"):
            prompt += self._js_import_path_rules()
        else:
            prompt += self._python_import_path_rules()

        # Add language-specific test guidance
        if language in ("javascript", "typescript"):
            prompt += self._js_test_rules(language, fw, env_info=env_info,
                                          test_runner=test_runner)
        else:
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
    def _extract_file_paths(context: str) -> list[str]:
        """Extract file paths from #### [FILE]: markers in the context."""
        import re
        return re.findall(r"####\s*\[FILE\]:\s*(\S+)", context)

    @staticmethod
    def _extract_file_listing(context: str) -> str:
        """Pull file paths from #### [FILE]: markers in the context."""
        import re
        paths = re.findall(r"####\s*\[FILE\]:\s*(\S+)", context)
        if not paths:
            return ""
        return "\n".join(f"  - {p}" for p in paths)

    @staticmethod
    def _python_import_path_rules() -> str:
        """Import path rules specific to Python projects."""
        return """CRITICAL PATH RULES — violations cause import failures:
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

    @staticmethod
    def _js_import_path_rules() -> str:
        """Import path rules specific to JavaScript/TypeScript projects."""
        return """CRITICAL PATH RULES — violations cause import failures:
1. Use RELATIVE paths from the test file to the source file.
   Do NOT use absolute paths or module aliases unless the project defines them.
2. Keep the original file extension (.ts, .tsx, .js, .jsx) when required by the bundler/runner.
3. NEVER drop directory prefixes.

Examples:
  Source at `src/components/Button.tsx`, test at `__tests__/Button.test.tsx`
    → `import { Button } from '../src/components/Button';`
  Source at `src/utils/helpers.ts`, test at `__tests__/helpers.test.ts`
    → `import { helperFn } from '../src/utils/helpers';`

WRONG (do NOT do this):
  Source at `src/components/Button.tsx` → `import { Button } from './Button';`  ← WRONG! Missing relative path to src/
"""

    @staticmethod
    def _js_test_rules(language: str | None, fw: dict,
                       env_info: dict | None = None,
                       test_runner: str | None = None) -> str:
        """Return JavaScript/TypeScript-specific test guidance."""
        ext = ".ts" if language == "typescript" else ".js"
        env = env_info or {}

        # Vitest path
        if test_runner == "vitest":
            return TesterAgent._vitest_test_rules(language, ext, env)

        # Jest path
        is_esm = env.get("is_esm", False)

        # ESM projects MUST explicitly import Jest globals
        if is_esm:
            import_rule = (
                "CRITICAL — This project uses ES Modules (`\"type\": \"module\"` in package.json).\n"
                "You MUST add this import at the TOP of every test file:\n"
                "   `import { expect, describe, it, test, beforeEach, afterEach } from '@jest/globals';`\n"
                "Without this import, you will get `ReferenceError: expect is not defined`.\n"
                "Use `import` syntax for ALL imports (both source and test globals).\n"
                "Do NOT use `require()` — it is not available in ESM.\n"
            )
            source_import = (
                f"Import source modules using ES `import` syntax:\n"
                f"   `import {{ funcName }} from '../src/module.js';`\n"
                f"IMPORTANT: Include the file extension (.js) in import paths for ESM.\n"
            )
        elif language == "typescript":
            import_rule = (
                "TypeScript project: Jest globals (expect, describe, it, test) are available globally.\n"
                "If you get `ReferenceError: expect is not defined`, add:\n"
                "   `import {{ expect, describe, it, test }} from '@jest/globals';`\n"
            )
            source_import = (
                f"Import source modules using `import` syntax:\n"
                f"   `import {{ funcName }} from '../src/module';`\n"
            )
        else:
            import_rule = (
                "CommonJS project: Jest globals (expect, describe, it, test) are available globally.\n"
                "No need to import them.\n"
            )
            source_import = (
                f"Import source modules using `require()`:\n"
                f"   `const {{ funcName }} = require('../src/module');`\n"
            )

        return f"""
JAVASCRIPT/TYPESCRIPT TEST RULES (critical for Jest):
1. {import_rule}
2. {source_import}
3. The import path must be RELATIVE from the test file to the source file.
   Example: test at `__tests__/calc.test{ext}`, source at `src/calculator{ext}`
4. Do NOT use absolute paths or module aliases unless the project defines them.
5. If the source file uses `module.exports`, use `require()` in tests.
   If the source file uses `export`, use `import` in tests.
6. Wrap each test in `describe` and `it`/`test` blocks.
7. Do NOT install packages, modify package.json, or create jest.config files.
8. Do NOT create or modify source files — ONLY write test files.
9. Use `expect(...).toBe(...)`, `toEqual(...)`, `toThrow(...)` etc. for assertions.
10. For async functions, use `async/await` with `expect(...).resolves` or `await expect(...)`.
"""

    @staticmethod
    def _vitest_test_rules(language: str | None, ext: str,
                           env: dict) -> str:
        """Return Vitest-specific test guidance."""
        tsx_rules = ""
        if language == "typescript":
            ext = ".tsx" if env.get("has_tsx") else ".ts"
            tsx_rules = (
                "For React component tests (.tsx files):\n"
                "   - Use `@testing-library/react` for rendering: "
                "`import { render, screen } from '@testing-library/react';`\n"
                "   - Use `@testing-library/user-event` for interactions: "
                "`import userEvent from '@testing-library/user-event';`\n"
                "   - Test file extension should be `.test.tsx` for components.\n"
            )

        return f"""
JAVASCRIPT/TYPESCRIPT TEST RULES (critical for Vitest):
1. You MUST import test utilities from 'vitest' at the TOP of every test file:
   `import {{ describe, it, expect, beforeEach, afterEach, vi }} from 'vitest';`
   Do NOT use Jest globals — this project uses Vitest, not Jest.
2. Import source modules using ES `import` syntax:
   `import {{ funcName }} from '../src/module';`
3. {tsx_rules if tsx_rules else 'Use `import` syntax for ALL imports.'}
4. The import path must be RELATIVE from the test file to the source file.
   Example: test at `__tests__/calc.test{ext}`, source at `src/calculator{ext}`
5. Do NOT use absolute paths or module aliases unless the project defines them.
6. Wrap each test in `describe` and `it`/`test` blocks.
7. Do NOT install packages, modify package.json, or create vitest/jest config files.
8. Do NOT create or modify source files — ONLY write test files.
9. Use `expect(...).toBe(...)`, `toEqual(...)`, `toThrow(...)` etc. for assertions.
10. For mocking, use `vi.fn()`, `vi.spyOn()`, `vi.mock()` (NOT `jest.fn()`).
11. For async functions, use `async/await` with `expect(...).resolves` or `await expect(...)`.
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
