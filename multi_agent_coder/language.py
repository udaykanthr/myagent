"""
Language detection and test framework mapping for multi-language support.
"""

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
        "command": "npx jest",
        "dir": "__tests__",
        "ext": ".js",
        "prefix": "",
        "suffix": ".test",
    },
    "typescript": {
        "command": "npx jest",
        "dir": "__tests__",
        "ext": ".ts",
        "prefix": "",
        "suffix": ".test",
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


def get_test_framework(language: str) -> dict:
    """Return test framework config for *language*, defaulting to pytest."""
    return TEST_FRAMEWORKS.get(language, TEST_FRAMEWORKS["python"])


def get_language_name(language: str) -> str:
    """Human-readable name for a language key."""
    return _LANGUAGE_NAMES.get(language, language.capitalize())


def get_code_block_lang(language: str) -> str:
    """Markdown fence language tag for a language key."""
    return _CODE_BLOCK_LANGS.get(language, language)
