"""
Language detection and test framework mapping for multi-language support.
"""

import json
import os
from collections import Counter


# ── Extension → Language mapping ──

EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".php": "php",
    ".scala": "scala",
    ".r": "r",
    ".R": "r",
    ".lua": "lua",
    ".sh": "bash",
    ".bat": "batch",
    ".ps1": "powershell",
}


# ── Test framework configs per language ──

TEST_FRAMEWORKS = {
    "python": {
        "command": "pytest",
        "dir": "tests",
        "ext": ".py",
        "prefix": "test_",
    },
    "javascript": {
        "command": "npx jest --forceExit --watchAll=false",
        "dir": "__tests__",
        "ext": ".js",
        "prefix": "",
        "suffix": ".test",
        "setup_cmd": "npm install --save-dev jest",
        "config_note": (
            "Jest must be listed in devDependencies. "
            "If package.json has `\"type\": \"module\"`, tests must use "
            "`import/export` syntax. Otherwise, use `require()/module.exports`. "
            "Use relative paths from the test file to the source file."
        ),
    },
    "javascript:vitest": {
        "command": "npx vitest run",
        "dir": "__tests__",
        "ext": ".js",
        "prefix": "",
        "suffix": ".test",
        "setup_cmd": "npm install --save-dev vitest",
        "config_note": (
            "Vitest is the test runner. Use `import { describe, it, expect } from 'vitest';` "
            "in every test file. Use ES `import` syntax for all imports."
        ),
    },
    "typescript": {
        "command": "npx jest --forceExit --watchAll=false",
        "dir": "__tests__",
        "ext": ".ts",
        "prefix": "",
        "suffix": ".test",
        "setup_cmd": "npm install --save-dev jest ts-jest @types/jest",
        "config_note": (
            "TypeScript projects need ts-jest or @swc/jest configured. "
            "Ensure jest.config has `transform` set for .ts files. "
            "Use `import/export` syntax in test files."
        ),
    },
    "typescript:vitest": {
        "command": "npx vitest run",
        "dir": "__tests__",
        "ext": ".ts",
        "prefix": "",
        "suffix": ".test",
        "setup_cmd": "npm install --save-dev vitest",
        "config_note": (
            "Vitest is the test runner. Use `import { describe, it, expect } from 'vitest';` "
            "in every test file. Use ES `import` syntax for all imports. "
            "For React components (.tsx), use @testing-library/react."
        ),
    },
    "go": {
        "command": "go test ./...",
        "dir": "",
        "ext": ".go",
        "prefix": "",
        "suffix": "_test",
    },
    "rust": {
        "command": "cargo test",
        "dir": "tests",
        "ext": ".rs",
        "prefix": "test_",
    },
    "java": {
        "command": "mvn test",
        "dir": "src/test/java",
        "ext": ".java",
        "prefix": "Test",
    },
    "ruby": {
        "command": "bundle exec rspec",
        "dir": "spec",
        "ext": ".rb",
        "prefix": "",
        "suffix": "_spec",
    },
}


# ── Keyword → Language mapping for task string detection ──

_TASK_KEYWORDS = {
    "python":     ["python", "flask", "django", "fastapi", "pip", "pytest", "pandas", "numpy"],
    "javascript": ["javascript", "node", "express", "react", "vue", "npm", "webpack", "jest"],
    "typescript": ["typescript", "angular", "nest", "tsx", "tsc"],
    "go":         ["golang", "go module", "gin", "echo framework"],
    "rust":       ["rust", "cargo", "tokio", "actix"],
    "java":       ["java", "spring", "maven", "gradle", "junit"],
    "ruby":       ["ruby", "rails", "sinatra", "rspec", "bundler"],
    "csharp":     ["c#", "csharp", "dotnet", ".net", "asp.net"],
    "cpp":        ["c++", "cpp", "cmake"],
}

_LANGUAGE_NAMES = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "go": "Go",
    "rust": "Rust",
    "java": "Java",
    "ruby": "Ruby",
    "csharp": "C#",
    "cpp": "C++",
    "c": "C",
    "swift": "Swift",
    "kotlin": "Kotlin",
    "php": "PHP",
    "scala": "Scala",
    "r": "R",
    "lua": "Lua",
    "bash": "Bash",
    "batch": "Batch",
    "powershell": "PowerShell",
}

_CODE_BLOCK_LANGS = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "ruby": "ruby",
    "csharp": "csharp",
    "cpp": "cpp",
    "c": "c",
    "swift": "swift",
    "kotlin": "kotlin",
    "php": "php",
    "scala": "scala",
}

# Directories to skip when scanning for language detection
_SKIP_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv",
              "env", "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
              "target", "bin", "obj", ".idea", ".vscode"}


def detect_language(directory: str = ".") -> str:
    """Scan file extensions in *directory* and return the most common language.

    Returns ``"python"`` as default when no recognized files are found.
    """
    ext_counts: Counter = Counter()
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext in EXTENSION_MAP:
                ext_counts[EXTENSION_MAP[ext]] += 1

    if not ext_counts:
        return "python"
    return ext_counts.most_common(1)[0][0]


def detect_language_from_task(task: str) -> str | None:
    """Try to infer the language from keywords in the task string.

    Returns ``None`` if no match is found (caller should fall back to
    ``detect_language``).
    """
    task_lower = task.lower()
    for lang, keywords in _TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in task_lower:
                return lang
    return None


def get_test_framework(language: str, test_runner: str | None = None) -> dict:
    """Return test framework config for *language*, defaulting to pytest.

    When *test_runner* is ``"vitest"`` and the language is JS or TS, returns
    the Vitest-specific framework config instead of Jest.
    """
    if test_runner == "vitest" and language in ("javascript", "typescript"):
        return TEST_FRAMEWORKS[f"{language}:vitest"]
    return TEST_FRAMEWORKS.get(language, TEST_FRAMEWORKS["python"])


def get_language_name(language: str) -> str:
    """Human-readable name for a language key."""
    return _LANGUAGE_NAMES.get(language, language.capitalize())


def get_code_block_lang(language: str) -> str:
    """Markdown fence language tag for a language key."""
    return _CODE_BLOCK_LANGS.get(language, language)


def detect_language_from_files(file_paths: list[str]) -> str | None:
    """Infer the project language from a list of file paths.

    Typically called with paths extracted from ``#### [FILE]:`` markers in
    context strings.  Returns ``None`` when no recognised extensions are found.
    """
    ext_counts: Counter = Counter()
    for path in file_paths:
        _, ext = os.path.splitext(path)
        lang = EXTENSION_MAP.get(ext)
        if lang:
            ext_counts[lang] += 1
    if not ext_counts:
        return None
    return ext_counts.most_common(1)[0][0]


def detect_test_runner(directory: str | None = None) -> str | None:
    """Detect whether a JS/TS project uses Vitest or Jest.

    Checks for ``vitest.config.*`` files first, then ``jest.config.*`` files,
    then falls back to inspecting ``package.json`` devDependencies.

    Returns ``"vitest"``, ``"jest"``, or ``None``.
    """
    cwd = directory or "."

    # Check for vitest config files
    for name in ("vitest.config.ts", "vitest.config.js", "vitest.config.mts",
                 "vitest.config.mjs"):
        if os.path.isfile(os.path.join(cwd, name)):
            return "vitest"

    # Check for jest config files
    for name in ("jest.config.js", "jest.config.ts", "jest.config.mjs",
                 "jest.config.cjs", "jest.config.json"):
        if os.path.isfile(os.path.join(cwd, name)):
            return "jest"

    # Fall back to package.json devDependencies
    pkg_path = os.path.join(cwd, "package.json")
    if os.path.isfile(pkg_path):
        try:
            with open(pkg_path, "r", encoding="utf-8") as f:
                pkg = json.load(f)
            dev_deps = pkg.get("devDependencies", {})
            if "vitest" in dev_deps:
                return "vitest"
            if "jest" in dev_deps:
                return "jest"
        except (json.JSONDecodeError, OSError):
            pass

    return None
