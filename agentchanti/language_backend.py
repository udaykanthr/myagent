"""
Pluggable language backend system for AgentChanti.

Each LanguageBackend encapsulates language-specific behaviour:
test framework, coder/tester prompt rules, import/export extraction,
and CONFIG_BUG handling.  Built-in backends are registered automatically;
custom backends can be loaded via config or setuptools entry points.
"""

from __future__ import annotations

import importlib
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from .cli_display import log


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LanguageBackend(ABC):
    """Base class for language-specific behaviour."""

    #: Canonical language name (e.g. "python", "javascript")
    language: str = ""

    #: Human-readable name
    display_name: str = ""

    # ── Test framework ──

    @abstractmethod
    def get_test_framework(self, test_runner: str | None = None) -> dict[str, Any]:
        """Return test framework config dict.

        Must include at minimum: ``command``, ``dir``, ``ext``.
        Optional: ``prefix``, ``suffix``, ``setup_cmd``, ``config_note``.
        """

    # ── Prompt rules ──

    def get_coder_rules(self) -> str:
        """Language-specific rules appended to Coder agent prompts."""
        return ""

    def get_test_rules(self, test_runner: str | None = None,
                       env_info: dict | None = None) -> str:
        """Language-specific rules appended to Tester agent prompts."""
        return ""

    # ── Import / export extraction ──

    def extract_imports(self, content: str) -> list[str]:
        """Extract import sources from file content."""
        return []

    def extract_exports(self, content: str) -> list[str]:
        """Extract exported symbol names from file content."""
        return []

    # ── CONFIG_BUG handling ──

    def get_config_candidates(self) -> list[str]:
        """Return config file names to check when fixing CONFIG_BUG errors."""
        return []

    def get_config_fix_prompt(self, test_cmd: str) -> str:
        """Return the LLM prompt body for fixing test config issues."""
        return (
            f"Fix the test configuration so the test runner ({test_cmd}) "
            f"can discover and run tests correctly.\n"
            f"Return the corrected config file(s) using #### [FILE]: format.\n"
            f"Do NOT modify test files or source files."
        )

    def get_config_fix_detail(self) -> str:
        """Return description of what's misconfigured (for LLM context)."""
        return (
            "This is caused by a missing or misconfigured test environment.\n\n"
        )

    def get_config_lang_tag(self) -> str:
        """Return the code block language tag for config files."""
        return self.language

    def get_config_no_found_msg(self) -> str:
        """Return message when no config file is found."""
        return "(no test config found)\n\n"


# ---------------------------------------------------------------------------
# Built-in backends
# ---------------------------------------------------------------------------

class PythonBackend(LanguageBackend):
    language = "python"
    display_name = "Python"

    def get_test_framework(self, test_runner: str | None = None) -> dict:
        return {
            "command": "python -m pytest",
            "dir": "tests",
            "ext": ".py",
            "prefix": "test_",
        }

    def get_coder_rules(self) -> str:
        return (
            "- Use type hints for function signatures.\n"
            "- Prefer f-strings over .format() or %.\n"
            "- Use pathlib.Path over os.path for file operations.\n"
        )

    def get_test_rules(self, test_runner=None, env_info=None) -> str:
        return (
            "- Use pytest conventions: function names start with `test_`.\n"
            "- Use `assert` statements, not unittest methods.\n"
            "- Use `pytest.raises()` for expected exceptions.\n"
            "- Import from the package using dotted module paths.\n"
        )

    def extract_imports(self, content: str) -> list[str]:
        imports = []
        for m in re.finditer(r'^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))',
                             content, re.MULTILINE):
            imports.append(m.group(1) or m.group(2))
        return imports

    def extract_exports(self, content: str) -> list[str]:
        exports = []
        # __all__ list
        all_match = re.search(r'__all__\s*=\s*\[([^\]]+)\]', content)
        if all_match:
            exports.extend(re.findall(r'["\'](\w+)["\']', all_match.group(1)))
        # class/def at module level
        for m in re.finditer(r'^(?:class|def)\s+(\w+)', content, re.MULTILINE):
            name = m.group(1)
            if not name.startswith('_'):
                exports.append(name)
        # Module-level assignments — constants are exports too, and missing
        # them made every `TILE_SIZE = 24` / `START = "start"` look absent.
        # Observed: a step declaring TILE_SIZE, TILE_WALL, START, WIN … was
        # reported as defining none of them, while the acceptance gate that
        # imported exactly those symbols passed.
        # Column 0 only, so locals inside a class or function body are not
        # mistaken for module attributes.
        for m in re.finditer(r'^([A-Za-z]\w*)\s*(?::[^=\n]+)?=[^=]',
                             content, re.MULTILINE):
            name = m.group(1)
            if name not in exports:
                exports.append(name)
        return exports

    def get_config_candidates(self) -> list[str]:
        return ["pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "conftest.py"]

    def get_config_fix_prompt(self, test_cmd: str) -> str:
        return (
            "Fix the Python test configuration so pytest can "
            "discover and run tests correctly.\n"
            "Common fixes: add conftest.py, set pythonpath in "
            "pyproject.toml [tool.pytest.ini_options], ensure "
            "__init__.py exists in test directories, or install "
            "missing test dependencies.\n"
            "Return the corrected config file(s) using "
            "#### [FILE]: format.\n"
            "Do NOT modify test files or source files."
        )

    def get_config_fix_detail(self) -> str:
        return (
            "This is caused by a missing or misconfigured Python "
            "test environment (e.g. wrong PYTHONPATH, missing "
            "conftest.py, incorrect pytest options, or missing "
            "test dependencies).\n\n"
        )

    def get_config_lang_tag(self) -> str:
        return "python"

    def get_config_no_found_msg(self) -> str:
        return (
            "(no pytest config found — you may need to create "
            "pytest.ini or add [tool.pytest.ini_options] to "
            "pyproject.toml)\n\n"
        )


# A JavaScript identifier — the shape any real export name has.
_JS_IDENT_RE = re.compile(r'^[A-Za-z_$][\w$]*$')


class JavaScriptBackend(LanguageBackend):
    language = "javascript"
    display_name = "JavaScript"

    def get_test_framework(self, test_runner: str | None = None) -> dict:
        if test_runner == "vitest":
            return {
                "command": "npx vitest run",
                "dir": "__tests__",
                "ext": ".jsx",
                "prefix": "",
                "suffix": ".test",
                "setup_cmd": "npm install --save-dev vitest",
                "config_note": (
                    "Vitest is the test runner. Use "
                    "`import { describe, it, expect } from 'vitest';` "
                    "in every test file. Use .jsx extension for JSX."
                ),
            }
        if test_runner == "ng":
            return {
                "command": "npx ng test --watch=false",
                "dir": "src/app",
                "ext": ".ts",
                "prefix": "",
                "suffix": ".spec",
                "config_note": (
                    "Angular CLI test runner. "
                    "Tests use `describe`/`it`/`expect`. "
                    "Use TestBed for component testing."
                ),
            }
        if test_runner == "mocha":
            return {
                "command": "npx mocha",
                "dir": "test",
                "ext": ".js",
                "prefix": "",
                "suffix": ".test",
                "setup_cmd": "npm install --save-dev mocha",
                "config_note": "Mocha test runner. Use assert or chai for assertions.",
            }
        return {
            "command": "npx jest --forceExit --watchAll=false",
            "dir": "__tests__",
            "ext": ".js",
            "prefix": "",
            "suffix": ".test",
            "setup_cmd": "npm install --save-dev jest",
            "config_note": (
                "Jest must be listed in devDependencies. "
                "If package.json has `\"type\": \"module\"`, use "
                "import/export syntax. Otherwise use require()."
            ),
        }

    def get_coder_rules(self) -> str:
        return (
            "- Use const/let, never var.\n"
            "- Prefer arrow functions for callbacks.\n"
            "- Use template literals over string concatenation.\n"
        )

    def get_test_rules(self, test_runner=None, env_info=None) -> str:
        if test_runner == "vitest":
            return (
                "- Import test utilities: "
                "`import { describe, it, expect } from 'vitest';`\n"
                "- Use .jsx extension for files containing JSX.\n"
                "- Use ES import syntax for all imports.\n"
            )
        rules = "- Use describe/it/expect pattern.\n"
        if env_info and env_info.get("module_type") == "esm":
            rules += (
                "- Import from '@jest/globals': "
                "`import { describe, it, expect } from '@jest/globals';`\n"
                "- Use ES import syntax. Include .js extension in imports.\n"
            )
        else:
            rules += "- Jest globals are available. Use require() for imports.\n"
        return rules

    def extract_imports(self, content: str) -> list[str]:
        imports = []
        for m in re.finditer(
            r'''(?:import\s+.*?\s+from\s+['"](.+?)['"]|'''
            r'''require\s*\(\s*['"](.+?)['"]\s*\))''',
            content,
        ):
            imports.append(m.group(1) or m.group(2))
        return imports

    def extract_exports(self, content: str) -> list[str]:
        """Exported names, across the four spellings JS actually uses.

        Every omission here reads downstream as a MISSING export, and the
        ghost reports that as `violated-exports` — a finding that sends a
        reader after code which is already correct. Measured 2026-08-19
        run 16: `export async function apiRequest(...)` was reported
        missing from a file that plainly exports it, because the pattern
        had no `async` branch. `ghost.plan_anchors` has carried the
        `async` case for some time; the two parsers simply disagreed, and
        the verdict was decided by the weaker one.

        The list form (`export { a, b as c }`) and the object form
        (`module.exports = {a, b}`) were missing for the same reason —
        the CommonJS pattern accepted only `= identifier`, so every
        backend module in this project's own benchmark exported nothing
        as far as this extractor was concerned.
        """
        exports: list[str] = []

        def _add(name: str) -> None:
            # Every branch below slices text out of a pattern match, so
            # the one invariant worth enforcing centrally is that what
            # survives is actually a name. Without it, CSS Modules'
            # `:export { globalStyles: globalStyles; }` yielded the
            # "export" `globalStyles: globalStyles;` — reported by the
            # ghost as a file's exports, verbatim (2026-08-19 run 22).
            if not name or not _JS_IDENT_RE.match(name):
                return
            if name not in exports:
                exports.append(name)

        for m in re.finditer(
            r'export\s+(?:default\s+)?(?:async\s+)?'
            r'(?:function\s*\*?|class|const|let|var)\s+(\w+)',
            content,
        ):
            _add(m.group(1))
        # `export { a, b as c }` — the exported name is the one after `as`.
        for m in re.finditer(r'(?<![:\w$])export\s*\{([^}]*)\}', content):
            for part in m.group(1).split(","):
                _add(part.split(" as ")[-1].strip().strip("\"'"))
        if re.search(r'export\s+default\s+', content):
            _add("default")
        for m in re.finditer(r'module\.exports\s*=\s*(\w+)\s*(?:;|$)',
                             content, re.MULTILINE):
            _add(m.group(1))
        # `module.exports = { a, b: c }` / `exports.name = ...`
        for m in re.finditer(r'module\.exports\s*=\s*\{([^}]*)\}',
                             content, re.DOTALL):
            for part in m.group(1).split(","):
                _add(part.split(":")[0].strip())
        for m in re.finditer(r'(?:module\.)?exports\.(\w+)\s*=', content):
            _add(m.group(1))
        return exports

    def get_config_candidates(self) -> list[str]:
        return [
            "vitest.config.js", "vitest.config.ts",
            "vite.config.js", "vite.config.ts",
            "jest.config.js", "jest.config.ts", "jest.config.mjs",
        ]

    def get_config_fix_prompt(self, test_cmd: str) -> str:
        return (
            "Fix the vitest/jest configuration so tests run correctly.\n"
            "If using vitest with React/JSX, add "
            "`test: { environment: 'jsdom', globals: true }` to the config.\n"
            "Return the corrected config file using #### [FILE]: format.\n"
            "Do NOT modify test files or source files."
        )

    def get_config_fix_detail(self) -> str:
        return (
            "This is caused by a missing or misconfigured test environment "
            "(jsdom not enabled in vitest/vite config, or Jest not configured).\n\n"
        )

    def get_config_lang_tag(self) -> str:
        return "js"

    def get_config_no_found_msg(self) -> str:
        return "(no vitest/jest config found — create one)\n\n"


class TypeScriptBackend(JavaScriptBackend):
    language = "typescript"
    display_name = "TypeScript"

    def get_test_framework(self, test_runner: str | None = None) -> dict:
        fw = super().get_test_framework(test_runner)
        fw["ext"] = ".ts"
        # Only add ts-jest setup for plain Jest — vitest, ng, mocha handle TS natively
        if test_runner not in ("vitest", "ng", "mocha"):
            fw["setup_cmd"] = "npm install --save-dev jest ts-jest @types/jest"
            fw["config_note"] = (
                "TypeScript projects need ts-jest configured. "
                "Ensure jest.config has `transform` set for .ts files."
            )
        return fw


class GoBackend(LanguageBackend):
    language = "go"
    display_name = "Go"

    def get_test_framework(self, test_runner=None) -> dict:
        return {
            "command": "go test ./...",
            "dir": "",
            "ext": ".go",
            "prefix": "",
            "suffix": "_test",
        }

    def get_coder_rules(self) -> str:
        return (
            "- Handle errors explicitly, don't ignore returned errors.\n"
            "- Use short variable declarations (:=) inside functions.\n"
            "- Follow Go naming conventions (exported = PascalCase).\n"
        )

    def get_test_rules(self, test_runner=None, env_info=None) -> str:
        return (
            "- Test functions must start with `Test` and take `*testing.T`.\n"
            "- Use `t.Errorf()` or `t.Fatalf()` for assertions.\n"
            "- Test files must end with `_test.go` in the same package.\n"
        )

    def extract_imports(self, content: str) -> list[str]:
        imports = []
        for m in re.finditer(r'"([\w./]+)"', content):
            imports.append(m.group(1))
        return imports

    def extract_exports(self, content: str) -> list[str]:
        exports = []
        for m in re.finditer(r'^(?:func|type|var|const)\s+([A-Z]\w*)', content, re.MULTILINE):
            exports.append(m.group(1))
        return exports

    def get_config_candidates(self) -> list[str]:
        return ["go.mod"]

    def get_config_fix_prompt(self, test_cmd: str) -> str:
        return (
            "Fix the Go test configuration. Ensure go.mod exists "
            "and test files are in the correct package.\n"
            "Return corrected files using #### [FILE]: format."
        )


class RustBackend(LanguageBackend):
    language = "rust"
    display_name = "Rust"

    def get_test_framework(self, test_runner=None) -> dict:
        return {
            "command": "cargo test",
            "dir": "tests",
            "ext": ".rs",
            "prefix": "test_",
        }

    def get_coder_rules(self) -> str:
        return (
            "- Use Result<T, E> for fallible operations.\n"
            "- Prefer &str over String for function parameters.\n"
            "- Use derive macros for common traits.\n"
        )

    def get_test_rules(self, test_runner=None, env_info=None) -> str:
        return (
            "- Use #[test] attribute on test functions.\n"
            "- Use assert!, assert_eq!, assert_ne! macros.\n"
            "- Integration tests go in tests/ directory.\n"
        )

    def extract_imports(self, content: str) -> list[str]:
        imports = []
        for m in re.finditer(r'use\s+([\w:]+)', content):
            imports.append(m.group(1))
        return imports

    def extract_exports(self, content: str) -> list[str]:
        exports = []
        for m in re.finditer(r'pub\s+(?:fn|struct|enum|trait|type|const|static|mod)\s+(\w+)',
                             content):
            exports.append(m.group(1))
        return exports

    def get_config_candidates(self) -> list[str]:
        return ["Cargo.toml"]


class JavaBackend(LanguageBackend):
    language = "java"
    display_name = "Java"

    def get_test_framework(self, test_runner=None) -> dict:
        return {
            "command": "mvn test",
            "dir": "src/test/java",
            "ext": ".java",
            "prefix": "Test",
        }

    def get_coder_rules(self) -> str:
        return (
            "- Follow Java naming conventions (classes PascalCase, methods camelCase).\n"
            "- Use try-with-resources for AutoCloseable objects.\n"
        )

    def get_test_rules(self, test_runner=None, env_info=None) -> str:
        return (
            "- Use JUnit 5 annotations: @Test, @BeforeEach, @AfterEach.\n"
            "- Use Assertions.assertEquals, assertTrue, assertThrows.\n"
            "- Test class names should match: FooTest for class Foo.\n"
        )

    def extract_imports(self, content: str) -> list[str]:
        imports = []
        for m in re.finditer(r'^import\s+([\w.]+);', content, re.MULTILINE):
            imports.append(m.group(1))
        return imports

    def extract_exports(self, content: str) -> list[str]:
        exports = []
        for m in re.finditer(r'public\s+(?:class|interface|enum)\s+(\w+)', content):
            exports.append(m.group(1))
        return exports

    def get_config_candidates(self) -> list[str]:
        return ["pom.xml", "build.gradle", "build.gradle.kts"]


class RubyBackend(LanguageBackend):
    language = "ruby"
    display_name = "Ruby"

    def get_test_framework(self, test_runner=None) -> dict:
        return {
            "command": "bundle exec rspec",
            "dir": "spec",
            "ext": ".rb",
            "prefix": "",
            "suffix": "_spec",
        }

    def get_coder_rules(self) -> str:
        return (
            "- Follow Ruby style guide (snake_case methods, PascalCase classes).\n"
            "- Use blocks with do/end for multi-line, braces for single-line.\n"
        )

    def get_test_rules(self, test_runner=None, env_info=None) -> str:
        return (
            "- Use RSpec describe/context/it pattern.\n"
            "- Use expect().to / expect().not_to for assertions.\n"
            "- Spec files end with _spec.rb in spec/ directory.\n"
        )

    def extract_imports(self, content: str) -> list[str]:
        imports = []
        for m in re.finditer(r"require\s+['\"](.+?)['\"]", content):
            imports.append(m.group(1))
        return imports

    def extract_exports(self, content: str) -> list[str]:
        exports = []
        for m in re.finditer(r'^(?:class|module)\s+(\w+)', content, re.MULTILINE):
            exports.append(m.group(1))
        return exports

    def get_config_candidates(self) -> list[str]:
        return ["Gemfile", ".rspec"]


class CSharpBackend(LanguageBackend):
    language = "csharp"
    display_name = "C#"

    def get_test_framework(self, test_runner=None) -> dict:
        return {
            "command": "dotnet test",
            "dir": "Tests",
            "ext": ".cs",
            "prefix": "",
            "suffix": "Tests",
            "setup_cmd": "dotnet add package NUnit NUnit3TestAdapter",
        }

    def get_test_rules(self, test_runner=None, env_info=None) -> str:
        return (
            "- Use NUnit or xUnit test attributes: [Test], [Fact].\n"
            "- Use Assert.AreEqual, Assert.That, Assert.Throws.\n"
        )

    def extract_imports(self, content: str) -> list[str]:
        imports = []
        for m in re.finditer(r'^using\s+([\w.]+);', content, re.MULTILINE):
            imports.append(m.group(1))
        return imports

    def extract_exports(self, content: str) -> list[str]:
        exports = []
        for m in re.finditer(r'public\s+(?:class|interface|struct|enum)\s+(\w+)', content):
            exports.append(m.group(1))
        return exports

    def get_config_candidates(self) -> list[str]:
        return ["*.csproj", "*.sln"]


class PHPBackend(LanguageBackend):
    language = "php"
    display_name = "PHP"

    def get_test_framework(self, test_runner=None) -> dict:
        return {
            "command": "vendor/bin/phpunit",
            "dir": "tests",
            "ext": ".php",
            "prefix": "",
            "suffix": "Test",
            "setup_cmd": "composer require --dev phpunit/phpunit",
        }

    def get_test_rules(self, test_runner=None, env_info=None) -> str:
        return (
            "- Test classes extend PHPUnit\\Framework\\TestCase.\n"
            "- Test methods start with `test` prefix.\n"
            "- Use $this->assertEquals(), assertTrue(), assertThrows().\n"
        )

    def extract_imports(self, content: str) -> list[str]:
        imports = []
        for m in re.finditer(r'^use\s+([\w\\]+)', content, re.MULTILINE):
            imports.append(m.group(1))
        return imports

    def extract_exports(self, content: str) -> list[str]:
        exports = []
        for m in re.finditer(r'class\s+(\w+)', content):
            exports.append(m.group(1))
        return exports

    def get_config_candidates(self) -> list[str]:
        return ["phpunit.xml", "phpunit.xml.dist", "composer.json"]


class SwiftBackend(LanguageBackend):
    language = "swift"
    display_name = "Swift"

    def get_test_framework(self, test_runner=None) -> dict:
        return {
            "command": "swift test",
            "dir": "Tests",
            "ext": ".swift",
            "prefix": "",
            "suffix": "Tests",
        }

    def get_test_rules(self, test_runner=None, env_info=None) -> str:
        return (
            "- Test classes extend XCTestCase.\n"
            "- Test methods start with `test` prefix.\n"
            "- Use XCTAssertEqual, XCTAssertTrue, XCTAssertThrowsError.\n"
        )

    def extract_imports(self, content: str) -> list[str]:
        imports = []
        for m in re.finditer(r'^import\s+(\w+)', content, re.MULTILINE):
            imports.append(m.group(1))
        return imports

    def extract_exports(self, content: str) -> list[str]:
        exports = []
        for m in re.finditer(r'(?:public|open)\s+(?:class|struct|enum|protocol)\s+(\w+)',
                             content):
            exports.append(m.group(1))
        return exports

    def get_config_candidates(self) -> list[str]:
        return ["Package.swift"]


class KotlinBackend(LanguageBackend):
    language = "kotlin"
    display_name = "Kotlin"

    def get_test_framework(self, test_runner=None) -> dict:
        return {
            "command": "gradle test",
            "dir": "src/test/kotlin",
            "ext": ".kt",
            "prefix": "",
            "suffix": "Test",
        }

    def get_test_rules(self, test_runner=None, env_info=None) -> str:
        return (
            "- Use JUnit 5 or kotlin.test annotations.\n"
            "- Use assertEquals, assertTrue, assertThrows.\n"
        )

    def extract_imports(self, content: str) -> list[str]:
        imports = []
        for m in re.finditer(r'^import\s+([\w.]+)', content, re.MULTILINE):
            imports.append(m.group(1))
        return imports

    def extract_exports(self, content: str) -> list[str]:
        exports = []
        for m in re.finditer(r'^(?:class|object|interface|enum class)\s+(\w+)',
                             content, re.MULTILINE):
            exports.append(m.group(1))
        return exports

    def get_config_candidates(self) -> list[str]:
        return ["build.gradle.kts", "build.gradle"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BUILTIN_BACKENDS: list[type[LanguageBackend]] = [
    PythonBackend, JavaScriptBackend, TypeScriptBackend,
    GoBackend, RustBackend, JavaBackend, RubyBackend,
    CSharpBackend, PHPBackend, SwiftBackend, KotlinBackend,
]

_registry: dict[str, LanguageBackend] = {}


def _init_registry() -> None:
    """Populate the registry with built-in backends."""
    global _registry
    if _registry:
        return
    for cls in _BUILTIN_BACKENDS:
        backend = cls()
        _registry[backend.language] = backend


def register_backend(backend: LanguageBackend) -> None:
    """Register a custom language backend (overrides built-in if same language)."""
    _init_registry()
    _registry[backend.language] = backend
    log.info("[LanguageBackend] Registered custom backend: %s", backend.language)


def get_backend(language: str | None) -> LanguageBackend:
    """Get the backend for a language. Falls back to Python for unknown languages."""
    _init_registry()
    if language is None:
        return _registry["python"]
    # Handle vitest variant
    base_lang = language.split(":")[0]
    return _registry.get(base_lang, _registry["python"])


def get_all_backends() -> dict[str, LanguageBackend]:
    """Return all registered backends."""
    _init_registry()
    return dict(_registry)


def load_custom_backends(backend_paths: list[str]) -> None:
    """Load custom backends from dotted import paths.

    Each path should point to a LanguageBackend subclass, e.g.
    ``my_pkg.backends.CSharpBackend``.
    """
    _init_registry()
    for path in backend_paths:
        try:
            module_path, class_name = path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            if isinstance(cls, type) and issubclass(cls, LanguageBackend):
                backend = cls()
                register_backend(backend)
            else:
                log.warning("[LanguageBackend] %s is not a LanguageBackend subclass", path)
        except Exception as exc:
            log.warning("[LanguageBackend] Failed to load %s: %s", path, exc)
