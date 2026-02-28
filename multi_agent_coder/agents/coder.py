from .base import Agent
from ..language import get_code_block_lang, get_language_name

# Mapping from language key → real file extension for prompt examples.
# This prevents the LLM from creating files like "src/my_module.python"
# when the correct extension is ".py".
_LANG_TO_EXT = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "go": ".go",
    "rust": ".rs",
    "java": ".java",
    "ruby": ".rb",
    "csharp": ".cs",
    "cpp": ".cpp",
    "c": ".c",
    "swift": ".swift",
    "kotlin": ".kt",
    "php": ".php",
    "scala": ".scala",
    "lua": ".lua",
}


class CoderAgent(Agent):
    def process(self, task: str, context: str = "", language: str | None = None) -> str:
        prompt = self._build_prompt(task, context, language=language)
        lang_tag = get_code_block_lang(language) if language else "python"
        lang_name = get_language_name(language) if language else "Python"
        # Use the real file extension for the example, not the lang_tag
        example_ext = _LANG_TO_EXT.get(language or "python", ".py")

        # Extract known file paths from context so the LLM knows the real layout
        import re
        known_files = re.findall(r"####\s*\[FILE\]:\s*(\S+)", context)
        file_listing = "\n".join(f"  - {f}" for f in known_files) if known_files else ""

        prompt += f"""
Write clean, production-quality {lang_name} code that works correctly on
the FIRST attempt with no fixes needed.

{('EXISTING PROJECT FILES (use these EXACT paths — do NOT rename or shorten):'
  + chr(10) + file_listing + chr(10)) if file_listing else ''}
═══════ OUTPUT FORMAT ═══════
For EACH file, use EXACTLY this marker (no extra text after the path):

#### [FILE]: src/my_module{example_ext}
```{lang_tag}
# complete code here
```

═══════ PATH RULES (violations cause import failures) ═══════
1. Use the EXACT file paths from the project listing above — do NOT
   shorten, rename, or strip directory prefixes.
2. The path after [FILE]: must be ONLY the file path, nothing else.
3. Use forward slashes in paths, never backslashes.
4. When modifying an existing file, use the SAME path as shown above.
   Only create a new [FILE] when the task explicitly requires a new file.

═══════ CODE QUALITY CHECKLIST (verify before outputting) ═══════
1. **Imports resolve**: Every `import` / `require()` / `from ... import`
   points to a real file that exists. Match directory structure exactly.
   - Python: `from src.calculator import add` requires `src/calculator.py`
   - JS: `require('./utils')` requires `./utils.js` to exist
2. **Exports match imports**: Every name you import must be defined and
   exported in the source module. Don't import non-existent functions.
3. **Complete files**: Output the ENTIRE file content, not just the changed
   part. Include all existing functions/classes that should remain.
4. **Type correctness**: Function arguments, return types, and variable
   types must be consistent. No implicit type coercions that break.
5. **Error handling**: Handle likely failure cases (file not found, null
   values, network errors) where they'd cause crashes.
6. **Edge cases**: Handle empty input, zero values, boundary conditions.
"""
        # Add language-specific guidance
        if language in ("javascript", "typescript"):
            prompt += self._js_rules(language)
        elif language == "python" or language is None:
            prompt += self._python_rules()

        prompt += f"""
═══════ STRICT PROHIBITIONS ═══════
- FORBIDDEN: Editing `package.json`, `pyproject.toml`, `go.mod`, or
  `requirements.txt` to add dependencies. ALWAYS use shell commands
  (`npm install`, `pip install`, `go get`) instead.
  You may only edit these files to add/modify `scripts` or metadata.
- FORBIDDEN: Leaving TODO comments, placeholder code, or stub
  implementations. Every function must have a complete, working body.
- FORBIDDEN: Using deprecated APIs or functions.
- FORBIDDEN: Hardcoding environment-specific paths or credentials.
- FORBIDDEN: Creating files that duplicate existing files at different paths.

Think step-by-step before writing: What files exist? What are their exports?
What imports will the new/modified code need? Will the code pass tests?
"""
        return self.llm_client.generate_response(prompt)

    @staticmethod
    def _js_rules(language: str | None) -> str:
        """Return JavaScript/TypeScript coding rules."""
        return """
═══════ JAVASCRIPT/TYPESCRIPT RULES ═══════
1. **Module system**: Check `package.json` `"type"` field:
   - `"type": "module"` → use `import/export` (ES Modules)
   - No type field or `"type": "commonjs"` → use `require()/module.exports`
   - NEVER mix `require()` and `import` in the same file.
2. **Exports**: Always export functions/classes that other files need:
   - CommonJS: `module.exports = { funcA, funcB }` or `module.exports = ClassName`
   - ESM: `export function funcA()` or `export default class ClassName`
3. **Async/await**: If a function returns a Promise, callers MUST use
   `await`. Mark caller functions as `async`.
4. **Error handling**: Use `try/catch` around I/O, network calls, and
   JSON parsing. Don't let unhandled rejections crash the process.
5. **Dependencies**: Use `npm install <pkg>` via a CMD step, never edit
   package.json directly to add deps.
"""

    @staticmethod
    def _python_rules() -> str:
        """Return Python coding rules."""
        return """
═══════ PYTHON RULES ═══════
1. **Imports**: Use absolute imports matching the directory structure:
   - File at `src/utils/helpers.py` → `from src.utils.helpers import func`
   - NEVER drop directory prefixes from imports.
2. **`__init__.py`**: If your module is in a package directory, ensure
   `__init__.py` exists (it will be auto-created, but imports must
   assume package structure).
3. **Type hints**: Use type hints for function signatures.
4. **Error handling**: Use specific exception types, not bare `except:`.
5. **Dependencies**: Use `pip install <pkg>` via a CMD step, never edit
   requirements.txt or pyproject.toml directly to add deps.
6. **String formatting**: Use f-strings, not `%` or `.format()`.
"""
