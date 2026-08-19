"""
Post-step dependency validation — detects orphaned exports, broken imports,
and missing connections between files after each pipeline step.

Uses fast regex-based import/export extraction (no tree-sitter dependency)
and a single LLM call to fix all detected gaps.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)


def clean_diff_markers(content: str) -> str:
    """Strip pseudo-diff +/- markers from LLM output that should be clean code.

    Local LLMs sometimes emit a diff-style response (lines prefixed with ``+``
    for additions and ``-`` for removals) instead of complete file content.
    This function detects that pattern and reconstructs clean source code:
    - Lines starting with ``+`` (single, not ``++`` or ``+++``) → strip the ``+``
    - Lines starting with ``-`` (single, not ``--`` or ``---``) → removed, skip
    - Diff headers (``+++``, ``---``, ``@@``) → skip
    - All other lines → kept as-is (context lines)

    Only activates when >5% of non-empty lines carry diff markers to avoid
    false positives on legitimate code that starts a line with ``+`` or ``-``.
    """
    lines = content.splitlines(keepends=True)
    if not lines:
        return content

    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return content

    plus_count = sum(
        1 for l in non_empty
        if l.startswith("+") and not l.startswith("+++")
    )
    minus_count = sum(
        1 for l in non_empty
        if l.startswith("-") and not l.startswith("---")
    )
    diff_count = plus_count + minus_count
    if diff_count == 0 or diff_count / len(non_empty) < 0.05:
        return content

    result: list[str] = []
    for line in lines:
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue  # diff headers
        elif line.startswith("+") and not line.startswith("++"):
            result.append(line[1:])  # strip leading +
        elif line.startswith("-") and not line.startswith("--"):
            pass  # skip removed lines
        else:
            result.append(line)  # context line — keep as-is
    cleaned = "".join(result)
    _logger.debug("[clean_diff_markers] Stripped diff markers from content (%d→%d chars)",
                  len(content), len(cleaned))
    return cleaned


# ── Extension → language family mapping ──────────────────────────

_EXT_TO_LANG_FAMILY: dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
}

# ── Test-file detection patterns ─────────────────────────────────

_TEST_FILE_SUFFIXES = (
    ".test.js", ".test.ts", ".test.jsx", ".test.tsx",
    ".spec.js", ".spec.ts", ".spec.jsx", ".spec.tsx",
    "_test.go", "_test.py",
)

# Config files consumed by tools by convention — never imported by other
# project files.  Flagging them as "orphaned exports" is always a false
# positive and can cause DepCheck to regenerate them without KB context,
# silently downgrading version-specific syntax (e.g. Tailwind v4 → v3).
_TOOL_CONFIG_STEMS = frozenset({
    "postcss.config", "tailwind.config", "vite.config", "vitest.config",
    "jest.config", "babel.config", "webpack.config", "rollup.config",
    "eslint.config", ".eslintrc", "prettier.config", ".prettierrc",
    "stylelint.config", "svelte.config", "next.config", "nuxt.config",
    "astro.config", "remix.config", "wrangler.config",
})


def _is_test_file(file_path: str) -> bool:
    """Return ``True`` if *file_path* looks like a test file."""
    basename = os.path.basename(file_path).lower()
    if basename.startswith("test_"):
        return True
    for suffix in _TEST_FILE_SUFFIXES:
        if basename.endswith(suffix):
            return True
    return False


def _is_tool_config_file(file_path: str) -> bool:
    """Return True if *file_path* is a tool-convention config file.

    These files are read by build tools (PostCSS, Vite, Tailwind, …) via
    filesystem convention, never imported by other source files.  They must
    NOT be flagged as orphaned exports.
    """
    name = os.path.basename(file_path).lower()
    # Strip known JS/TS/JSON/CJS/MJS extensions to get the stem
    for ext in (".js", ".ts", ".cjs", ".mjs", ".json", ".yaml", ".yml"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name in _TOOL_CONFIG_STEMS


# Django app modules are discovered by the framework — via INSTALLED_APPS,
# the URLconf (`include('app.urls')`), admin autodiscover, and app configs —
# not by sibling Python files importing them.  Flagging them as "orphaned
# exports" is a false positive, and the auto-fix (re-exporting them from the
# package `__init__.py`) is actively harmful: `forms.py` / `models.py` import
# the auth User model, so importing them at package-init time runs *during*
# `apps.populate()` and raises "RuntimeError: populate() isn't reentrant",
# breaking `manage.py check` for the whole project.
_DJANGO_APP_MODULES = frozenset({
    "models", "views", "forms", "admin", "apps", "urls", "serializers",
    "signals", "middleware", "tasks", "consumers", "routing", "managers",
    "permissions", "filters", "viewsets", "api", "context_processors",
})


def _django_markers_present(known_files) -> bool:
    """True if the file set looks like a Django project (manage.py/settings.py)."""
    for f in known_files:
        if os.path.basename(f.replace("\\", "/")) in ("manage.py", "settings.py"):
            return True
    return False


def _has_django_app_sibling(file_path: str, known_files) -> bool:
    """True if *file_path*'s directory contains an ``apps.py`` — the definitive
    marker of a Django app package."""
    directory = os.path.dirname(file_path.replace("\\", "/"))
    for f in known_files:
        norm = f.replace("\\", "/")
        if os.path.dirname(norm) == directory and os.path.basename(norm) == "apps.py":
            return True
    return False


def _is_django_app_module(file_path: str, known_files) -> bool:
    """Return True if *file_path* is a Django app module wired by the framework
    rather than imported by sibling source files (so never an orphaned export).
    """
    if not file_path.endswith(".py"):
        return False
    stem = os.path.splitext(os.path.basename(file_path.replace("\\", "/")))[0].lower()
    if stem not in _DJANGO_APP_MODULES:
        return False
    return _django_markers_present(known_files) or _has_django_app_sibling(
        file_path, known_files)


# ── Data structures ──────────────────────────────────────────────

@dataclass
class FileDeps:
    """Import/export information for a single file."""
    file_path: str
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    has_default_export: bool = False
    default_imports: list[str] = field(default_factory=list)


@dataclass
class IntegrationGap:
    """A detected integration problem between files."""
    gap_type: str               # "orphaned_export" | "broken_import" | "missing_connection" | "missing_default_export" | "stale_caller"
    source_file: str            # The file where the problem originates
    target_file: str | None     # The file that should reference/be referenced
    symbol: str                 # The symbol name(s) involved
    description: str            # Human-readable explanation


@dataclass
class DependencySnapshot:
    """Snapshot of all import/export relationships at a point in time."""
    file_deps: dict[str, FileDeps] = field(default_factory=dict)


# ── Import / export regex patterns ───────────────────────────────

# Python
_PY_IMPORT_PATTERNS = [
    re.compile(r"^\s*from\s+([\w.]+)\s+import\s+", re.MULTILINE),
    re.compile(r"^\s*import\s+([\w.]+)", re.MULTILINE),
]
_PY_EXPORT_PATTERNS = [
    re.compile(r"^(?:class|def)\s+(\w+)", re.MULTILINE),
    re.compile(r"__all__\s*=\s*\[([^\]]+)\]"),
]

# JavaScript / TypeScript
_JS_IMPORT_PATTERNS = [
    re.compile(r"""^\s*import\s+.*?\s+from\s+['"](.*?)['"]""", re.MULTILINE),
    re.compile(r"""^\s*import\s+['"](.*?)['"]""", re.MULTILINE),
    re.compile(r"""(?:require|import)\s*\(\s*['"](.*?)['"]\s*\)"""),
]
_JS_EXPORT_PATTERNS = [
    re.compile(r"^\s*export\s+default\s+(?:function|class)\s+(\w+)", re.MULTILINE),
    re.compile(r"^\s*export\s+(?:function|class|const|let|var|type|interface|enum)\s+(\w+)", re.MULTILINE),
    re.compile(r"^\s*export\s*\{([^}]+)\}", re.MULTILINE),
    re.compile(r"module\.exports\s*=\s*(\w+|\{[^}]+\})"),
    re.compile(r"^\s*export\s+default\s+(\w+)\s*;?\s*$", re.MULTILINE),
]

# Go
_GO_IMPORT_PATTERNS = [
    re.compile(r'^\s*"(.*?)"$', re.MULTILINE),
    re.compile(r'^\s*import\s+"(.*?)"$', re.MULTILINE),
]
_GO_EXPORT_PATTERNS = [
    re.compile(r"^(?:func|type|var|const)\s+([A-Z]\w*)", re.MULTILINE),
    re.compile(r"^func\s+\(\w+\s+\*?\w+\)\s+([A-Z]\w*)", re.MULTILINE),
]

# Java
_JAVA_IMPORT_PATTERNS = [
    re.compile(r"^\s*import\s+([\w.]+);", re.MULTILINE),
]
_JAVA_EXPORT_PATTERNS = [
    re.compile(
        r"^\s*(?:public|protected)\s+(?:static\s+)?(?:abstract\s+)?"
        r"(?:class|interface|enum)\s+(\w+)",
        re.MULTILINE,
    ),
]

# Rust
_RUST_IMPORT_PATTERNS = [
    re.compile(r"^\s*use\s+([\w:]+)", re.MULTILINE),
]
_RUST_EXPORT_PATTERNS = [
    re.compile(r"^\s*pub\s+(?:fn|struct|enum|trait|type|const|static|mod)\s+(\w+)", re.MULTILINE),
]

_LANG_PATTERNS: dict[str, tuple[list, list]] = {
    "python":     (_PY_IMPORT_PATTERNS, _PY_EXPORT_PATTERNS),
    "javascript": (_JS_IMPORT_PATTERNS, _JS_EXPORT_PATTERNS),
    "typescript": (_JS_IMPORT_PATTERNS, _JS_EXPORT_PATTERNS),
    "go":         (_GO_IMPORT_PATTERNS, _GO_EXPORT_PATTERNS),
    "java":       (_JAVA_IMPORT_PATTERNS, _JAVA_EXPORT_PATTERNS),
    "rust":       (_RUST_IMPORT_PATTERNS, _RUST_EXPORT_PATTERNS),
}

# ── JS/TS default export detection ────────────────────────────────
# Matches any form of default export: export default function X, export default X,
# export default class X, module.exports = X
_JS_DEFAULT_EXPORT_PATTERNS = [
    re.compile(r"^\s*export\s+default\s+", re.MULTILINE),
    re.compile(r"^\s*module\.exports\s*=", re.MULTILINE),
]

# ── JS/TS default import detection ────────────────────────────────
# Matches: import Foo from '...'  (default import — no braces)
# Does NOT match: import { Foo } from '...'  (named import)
# Does NOT match: import '...'  (side-effect import)
_JS_DEFAULT_IMPORT_PATTERN = re.compile(
    r"""^\s*import\s+([A-Z]\w*)\s+from\s+['"](\..*?)['"]""",
    re.MULTILINE,
)

# ── Function / callable signature patterns (per language) ─────────

# JS/TS: components with destructured-object props.
# This regex matches only the *head* of a component definition:
#   function ComponentName
#   const ComponentName =
#   const ComponentName = function
# The destructure body is parsed separately by `_scan_balanced_destructure`
# so that nested object defaults and string literals containing braces or
# commas are handled correctly (a naive `[^}]*` capture truncates inside
# default object literals and exposes their contents as fake top-level props).
_JSX_COMP_HEAD_RE = re.compile(
    r'(?:export\s+(?:default\s+)?)?'
    r'(?:'
    r'function\s+([A-Z]\w*)'
    r'|(?:const|let|var)\s+([A-Z]\w*)\s*=\s*(?:function\s*)?'
    r')',
    re.MULTILINE,
)

# Python: def func_name(param1, param2: Type, param3=default):
_PY_FUNC_DEF_RE = re.compile(
    r'^def\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)',
    re.MULTILINE,
)

# Go: func FuncName(param1 type1, param2 type2)
_GO_FUNC_DEF_RE = re.compile(
    r'^func\s+([A-Z]\w*)\s*\(([^)]*)\)',
    re.MULTILINE,
)

# Java / C# / Kotlin: public ReturnType methodName(Type1 p1, Type2 p2)
_JAVA_FUNC_DEF_RE = re.compile(
    r'(?:public|protected|private)\s+(?:static\s+)?(?:\w[\w<>\[\],\s]*\s+)'
    r'([a-zA-Z_]\w*)\s*\(([^)]*)\)',
    re.MULTILINE,
)

# Rust: pub fn func_name(param1: type1, param2: type2)
_RUST_FUNC_DEF_RE = re.compile(
    r'^pub\s+fn\s+([a-z_]\w*)\s*\(([^)]*)\)',
    re.MULTILINE,
)

# Ruby: def method_name(param1, param2 = default)
_RUBY_FUNC_DEF_RE = re.compile(
    r'^def\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)',
    re.MULTILINE,
)

# PHP: public function methodName(Type $param1, $param2 = default)
_PHP_FUNC_DEF_RE = re.compile(
    r'(?:public|protected|private)?\s*function\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)',
    re.MULTILINE,
)

# Maps language family → (func_def_pattern, param_style)
# param_style: "destructured" | "positional" | "typed_prefix" | "typed_suffix"
_LANG_FUNC_INFO: dict[str, tuple[re.Pattern, str]] = {
    "python": (_PY_FUNC_DEF_RE,   "python"),
    "go":     (_GO_FUNC_DEF_RE,   "go"),
    "java":   (_JAVA_FUNC_DEF_RE, "typed_suffix"),
    "csharp": (_JAVA_FUNC_DEF_RE, "typed_suffix"),
    "rust":   (_RUST_FUNC_DEF_RE, "rust"),
    "ruby":   (_RUBY_FUNC_DEF_RE, "ruby"),
    "php":    (_PHP_FUNC_DEF_RE,  "php"),
}


# ── Python stdlib modules (for external import detection) ────────

_PYTHON_STDLIB = frozenset({
    "abc", "argparse", "ast", "asyncio", "base64", "bisect", "calendar",
    "collections", "contextlib", "copy", "csv", "ctypes", "dataclasses",
    "datetime", "decimal", "difflib", "email", "enum", "errno",
    "fileinput", "fnmatch", "fractions", "ftplib", "functools", "gc",
    "getpass", "glob", "gzip", "hashlib", "heapq", "hmac", "html",
    "http", "imaplib", "importlib", "inspect", "io", "itertools",
    "json", "keyword", "linecache", "locale", "logging", "lzma",
    "math", "mimetypes", "multiprocessing", "numbers", "operator",
    "os", "pathlib", "pickle", "pkgutil", "platform", "pprint",
    "profile", "queue", "random", "re", "readline", "reprlib",
    "secrets", "select", "shelve", "shlex", "shutil", "signal",
    "site", "smtplib", "socket", "sqlite3", "ssl", "stat",
    "statistics", "string", "struct", "subprocess", "sys", "sysconfig",
    "tempfile", "textwrap", "threading", "time", "timeit", "token",
    "tokenize", "trace", "traceback", "tracemalloc", "types", "typing",
    "unicodedata", "unittest", "urllib", "uuid", "venv", "warnings",
    "wave", "weakref", "webbrowser", "xml", "xmlrpc", "zipfile",
    "zipimport", "zlib", "_thread",
})


# ── Core functions ───────────────────────────────────────────────

def extract_file_deps(
    file_path: str, content: str, language: str | None = None,
) -> FileDeps:
    """Extract import and export information from a single file.

    Uses language-specific regex patterns. If *language* is ``None`` it is
    inferred from the file extension.
    """
    if not language:
        ext = os.path.splitext(file_path)[1].lower()
        language = _EXT_TO_LANG_FAMILY.get(ext)

    if not language or language not in _LANG_PATTERNS:
        return FileDeps(file_path=file_path)

    import_patterns, export_patterns = _LANG_PATTERNS[language]

    imports: list[str] = []
    for pat in import_patterns:
        for m in pat.finditer(content):
            source = m.group(1).strip().strip("'\"")
            if source and source not in imports:
                imports.append(source)

    exports: list[str] = []
    for pat in export_patterns:
        for m in pat.finditer(content):
            raw = m.group(1).strip()
            if "," in raw or raw.startswith("{"):
                # Comma-separated: export { X, Y } or module.exports = { X, Y }
                inner = raw.strip("{}")
                for name in inner.split(","):
                    name = name.strip().split(" as ")[0].split(":")[0].strip()
                    name = name.strip("'\"")  # strip quotes from __all__ entries
                    if name and name not in exports:
                        exports.append(name)
            else:
                if raw and raw not in exports:
                    exports.append(raw)

    # Filter private Python symbols (leading underscore)
    if language == "python":
        exports = [e for e in exports if not e.startswith("_")]

    # Detect JS/TS default exports and default imports
    has_default_export = False
    default_imports: list[str] = []
    if language in ("javascript", "typescript"):
        has_default_export = any(
            pat.search(content) for pat in _JS_DEFAULT_EXPORT_PATTERNS
        )
        for m in _JS_DEFAULT_IMPORT_PATTERN.finditer(content):
            imp_path = m.group(2).strip()
            if imp_path and imp_path not in default_imports:
                default_imports.append(imp_path)

    return FileDeps(
        file_path=file_path, imports=imports, exports=exports,
        has_default_export=has_default_export,
        default_imports=default_imports,
    )


# ── Function signature contract helpers ──────────────────────────

def _scan_balanced_destructure(
    content: str, start_idx: int,
) -> tuple[str, int] | None:
    """Find the matching ``}`` for the ``{`` at *start_idx*.

    Walks the source character by character, counting only ``{``/``}`` braces
    while skipping over string literals (``'…'``, ``"…"``, ``\u0060…\u0060``)
    and ``//``/``/* … */`` comments so that braces inside them do not affect
    nesting depth. Returns ``(inner_text, index_after_closing_brace)`` or
    ``None`` if no matching brace is found.
    """
    n = len(content)
    if start_idx >= n or content[start_idx] != "{":
        return None

    depth = 1
    i = start_idx + 1
    while i < n:
        ch = content[i]
        # Line comment
        if ch == "/" and i + 1 < n and content[i + 1] == "/":
            nl = content.find("\n", i + 2)
            i = n if nl == -1 else nl + 1
            continue
        # Block comment
        if ch == "/" and i + 1 < n and content[i + 1] == "*":
            end = content.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        # String / template literal
        if ch in ("'", '"', "`"):
            quote = ch
            j = i + 1
            while j < n:
                cj = content[j]
                if cj == "\\" and j + 1 < n:
                    j += 2
                    continue
                if cj == quote:
                    j += 1
                    break
                j += 1
            i = j
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return content[start_idx + 1:i], i + 1
        i += 1

    return None


def _split_top_level_params(s: str) -> list[str]:
    """Split *s* on commas that are not inside strings or balanced brackets.

    Used to break a destructure body into individual parameter declarations
    while preserving object/array defaults and string literals (which may
    contain commas) as a single token. Mirrors the lexer behaviour of
    ``_scan_balanced_destructure`` for the inner contents.
    """
    parts: list[str] = []
    n = len(s)
    i = 0
    start = 0
    depth = 0  # combined {}/[]/() depth — well-formed JS keeps these balanced

    while i < n:
        ch = s[i]
        # Line comment
        if ch == "/" and i + 1 < n and s[i + 1] == "/":
            nl = s.find("\n", i + 2)
            i = n if nl == -1 else nl + 1
            continue
        # Block comment
        if ch == "/" and i + 1 < n and s[i + 1] == "*":
            end = s.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        # String / template literal
        if ch in ("'", '"', "`"):
            quote = ch
            j = i + 1
            while j < n:
                cj = s[j]
                if cj == "\\" and j + 1 < n:
                    j += 2
                    continue
                if cj == quote:
                    j += 1
                    break
                j += 1
            i = j
            continue
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            if depth > 0:
                depth -= 1
        elif ch == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
        i += 1

    if start < n:
        parts.append(s[start:n])
    return parts


def _extract_component_props(content: str) -> dict[str, set[str]]:
    """JS/TS only: return {ComponentName: required_props} for destructured-prop components.

    Only props *without* a default value are considered required. Default
    values may contain commas and braces (e.g. ``foo = { a: 1, b: 2 }`` or
    ``foo = 'a, b, c'``); both are handled by the brace-/string-aware
    scanners ``_scan_balanced_destructure`` and ``_split_top_level_params``.
    """
    result: dict[str, set[str]] = {}
    n = len(content)

    for m in _JSX_COMP_HEAD_RE.finditer(content):
        name = m.group(1) or m.group(2)
        if not name:
            continue

        # Locate the opening `(` of the parameter list, then the destructure `{`.
        i = m.end()
        while i < n and content[i].isspace():
            i += 1
        if i >= n or content[i] != "(":
            continue
        i += 1
        while i < n and content[i].isspace():
            i += 1
        if i >= n or content[i] != "{":
            continue

        scan = _scan_balanced_destructure(content, i)
        if scan is None:
            continue
        props_str, _end = scan

        required: set[str] = set()
        for raw in _split_top_level_params(props_str):
            raw = raw.strip()
            if not raw or raw.startswith("..."):
                continue
            if "=" in raw:
                continue  # has default → optional
            m2 = re.match(r"^(\w+)", raw)
            if m2:
                required.add(m2.group(1))
        if required:
            result[name] = required
    return result


def _parse_required_params(params_str: str, style: str) -> set[str]:
    """Extract required (no-default) param names from a parameter string.

    *style* is one of the values from ``_LANG_FUNC_INFO``.
    """
    required: set[str] = set()
    for raw in params_str.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if style == "python":
            if raw in ("/", "self", "cls") or raw.startswith("*") or raw.startswith("**"):
                continue
            if "=" in raw:
                continue
            raw = raw.split(":")[0].strip()
            m = re.match(r"^(\w+)", raw)
            if m:
                required.add(m.group(1))
        elif style == "go":
            # "name type" — first token is name if lowercase
            parts = raw.split()
            if len(parts) >= 2:
                name = parts[0]
                if name and name != "_" and name[0].islower():
                    required.add(name)
        elif style == "typed_suffix":
            # Java/C#: "Type name" or "Type name = default"
            parts = raw.split()
            if len(parts) >= 2:
                name = parts[-1].split("=")[0].strip()
                m = re.match(r"^(\w+)", name)
                if m:
                    if "=" not in raw:
                        required.add(m.group(1))
        elif style == "rust":
            if raw in ("self", "&self", "&mut self"):
                continue
            if ":" in raw:
                name = raw.split(":")[0].strip()
                if name and name != "_":
                    m = re.match(r"^(\w+)", name)
                    if m:
                        required.add(m.group(1))
        elif style in ("ruby", "php"):
            # Ruby: param or param = default; PHP: Type $param or $param = default
            if "=" in raw:
                continue
            # PHP: strip type hint and $ sigil
            raw = re.sub(r"^\w[\w\[\]|?]*\s+", "", raw).lstrip("$")
            m = re.match(r"^(\w+)", raw)
            if m:
                required.add(m.group(1))
    return required


def _extract_func_sigs(content: str, language: str) -> dict[str, set[str]]:
    """Return {callable_name: required_params} for *content* in *language*.

    JS/TS uses the detailed destructured-prop pattern (named params).
    All other languages use their language-specific function def pattern.
    """
    if language in ("javascript", "typescript"):
        return _extract_component_props(content)

    info = _LANG_FUNC_INFO.get(language)
    if not info:
        return {}
    pattern, style = info
    result: dict[str, set[str]] = {}
    for m in pattern.finditer(content):
        name = m.group(1)
        if language == "python" and name.startswith("_"):
            continue  # skip private/dunder functions
        required = _parse_required_params(m.group(2), style)
        if len(required) >= 2:
            result[name] = required
    return result


def _extract_jsx_call_props(content: str, comp_name: str) -> list[set[str]]:
    """Return one set of passed prop names per ``<CompName ...>`` call site.

    Character-level scan with balanced-brace tracking handles multi-line JSX
    tags and ``{expression}`` values that contain ``>`` characters.
    Call sites with a spread (``{...x}``) are skipped.
    """
    all_sites: list[set[str]] = []
    search_str = f"<{comp_name}"
    pos = 0
    while True:
        idx = content.find(search_str, pos)
        if idx == -1:
            break
        pos = idx + 1
        end_of_name = idx + len(search_str)
        if end_of_name < len(content) and (
            content[end_of_name].isalnum() or content[end_of_name] == "_"
        ):
            continue

        props_found: set[str] = set()
        has_spread = False
        i = end_of_name
        brace_depth = 0
        limit = min(len(content), end_of_name + 2000)
        while i < limit:
            ch = content[i]
            if brace_depth > 0:
                if ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    brace_depth -= 1
                i += 1
                continue
            if ch == "{":
                if i + 1 < limit and content[i + 1] == ".":
                    has_spread = True
                brace_depth = 1
                i += 1
                continue
            if ch == "/" and i + 1 < limit and content[i + 1] == ">":
                break
            if ch == ">":
                break
            if ch.isalpha() or ch == "_":
                attr_m = re.match(r"([a-zA-Z_]\w*)\s*=", content[i:])
                if attr_m:
                    props_found.add(attr_m.group(1))
                    i += len(attr_m.group(0))
                    continue
            i += 1

        if not has_spread:
            all_sites.append(props_found)
    return all_sites


def _check_zero_arg_call(content: str, func_name: str) -> bool:
    """Return True if *func_name* is called with empty parentheses somewhere."""
    pat = re.compile(rf"\b{re.escape(func_name)}\s*\(\s*\)", re.MULTILINE)
    return bool(pat.search(content))


def _find_signature_gaps(
    new_files: list[str],
    memory_files: dict[str, str],
) -> list[IntegrationGap]:
    """Detect call sites missing required params after a signature change.

    Works for all supported languages:
    - **JS/TS**: checks named props at JSX call sites (detailed).
    - **Python / Go / Java / Rust / Ruby / PHP**: checks for zero-arg calls
      (``func_name()``) to functions that require 2+ params — the most
      reliable static check for positional-arg languages.
    """
    gaps: list[IntegrationGap] = []
    _code_exts = frozenset({
        ".js", ".jsx", ".ts", ".tsx",
        ".py", ".go", ".java", ".rs", ".cs", ".rb", ".php",
    })

    for nf in new_files:
        ext = os.path.splitext(nf)[1].lower()
        if ext not in _code_exts:
            continue
        language = _EXT_TO_LANG_FAMILY.get(ext)
        if not language:
            continue
        content = memory_files.get(nf, "")
        if not content:
            continue

        callables = _extract_func_sigs(content, language)
        if not callables:
            continue

        is_js_ts = language in ("javascript", "typescript")

        for func_name, required_params in callables.items():
            if len(required_params) < 2:
                continue

            for other_path, other_content in memory_files.items():
                if other_path == nf:
                    continue
                other_ext = os.path.splitext(other_path)[1].lower()
                if other_ext not in _code_exts:
                    continue

                if is_js_ts:
                    # Named-prop check via JSX attribute scan
                    if f"<{func_name}" not in other_content:
                        continue
                    call_sites = _extract_jsx_call_props(other_content, func_name)
                    for passed_props in call_sites:
                        missing = required_params - passed_props
                        if len(missing) >= 2:
                            gaps.append(IntegrationGap(
                                gap_type="stale_caller",
                                source_file=nf,
                                target_file=other_path,
                                symbol=func_name,
                                description=(
                                    f"File '{other_path}' calls <{func_name}> but is "
                                    f"missing required props: {', '.join(sorted(missing))}. "
                                    f"Component defined in '{nf}' expects: "
                                    f"{', '.join(sorted(required_params))}."
                                ),
                            ))
                else:
                    # Zero-arg call check for positional-arg languages
                    if f"{func_name}(" not in other_content:
                        continue
                    if _check_zero_arg_call(other_content, func_name):
                        gaps.append(IntegrationGap(
                            gap_type="stale_caller",
                            source_file=nf,
                            target_file=other_path,
                            symbol=func_name,
                            description=(
                                f"File '{other_path}' calls {func_name}() with no arguments "
                                f"but '{nf}' defines it with {len(required_params)} required "
                                f"parameter(s): {', '.join(sorted(required_params))}."
                            ),
                        ))
    return gaps


def build_snapshot(
    memory_files: dict[str, str], language: str | None = None,
) -> DependencySnapshot:
    """Build a dependency snapshot from all files currently in memory.

    Includes test files so their imports are tracked, allowing us to see
    if an export is only used by tests.
    """
    snapshot = DependencySnapshot()
    for fpath, content in memory_files.items():
        ext = os.path.splitext(fpath)[1].lower()
        if ext not in _EXT_TO_LANG_FAMILY:
            continue
        snapshot.file_deps[fpath] = extract_file_deps(fpath, content, language)
    return snapshot


# ── Import path resolution helpers ───────────────────────────────

def _normalize_import_path(import_source: str, importer_file: str) -> str:
    """Resolve relative import paths to a comparable form.

    JS/TS: ``./components/Header`` from ``src/App.tsx`` → ``src/components/Header``
    Python: ``.models`` from ``app/views.py`` → ``app.models``
    """
    if import_source.startswith("./") or import_source.startswith("../"):
        importer_dir = os.path.dirname(importer_file)
        resolved = os.path.normpath(os.path.join(importer_dir, import_source))
        return resolved.replace("\\", "/")
    if import_source.startswith("."):
        # Python relative import
        importer_dir = os.path.dirname(importer_file).replace("/", ".").replace("\\", ".")
        dots = len(import_source) - len(import_source.lstrip("."))
        relative_module = import_source[dots:]
        parts = importer_dir.split(".")
        base = ".".join(parts[:max(0, len(parts) - dots + 1)])
        if relative_module:
            return f"{base}.{relative_module}" if base else relative_module
        return base
    return import_source


def _file_matches_import(file_path: str, import_source: str) -> bool:
    """Check if *file_path* could be the target of *import_source*.

    Handles extension-less imports, Python dotted paths, and index files.
    """
    fp = file_path.replace("\\", "/")
    imp = import_source.replace("\\", "/")

    fp_noext = os.path.splitext(fp)[0]
    imp_noext = os.path.splitext(imp)[0] if "." in os.path.basename(imp) else imp

    # Direct match
    if fp_noext == imp_noext or fp_noext.endswith("/" + imp_noext):
        return True

    # Python dotted module → path (app.models → app/models)
    imp_as_path = imp.replace(".", "/")
    if fp_noext == imp_as_path or fp_noext.endswith("/" + imp_as_path):
        return True

    # Index file (./components → ./components/index)
    if fp_noext.endswith("/index") and fp_noext.rsplit("/index", 1)[0] == imp_noext:
        return True

    return False


_VENV_DIR_NAMES = ("venv", ".venv", "env", ".env")


def _project_has_installed_package(top_module: str) -> bool:
    """Check whether *top_module* is installed in the TARGET project's own
    virtualenv (searched relative to the current working directory).

    ``importlib.util.find_spec`` in ``_is_external_import`` only sees
    packages installed in *agentchanti's own* Python environment — it has
    no visibility into the venv a pipeline step just created for the
    project being built (e.g. ``arcade``, ``pygame``, ``django``, none of
    which agentchanti itself depends on). Without this check, any
    third-party import that isn't coincidentally also an agentchanti
    dependency gets misclassified as a "broken" local import, and DepCheck
    then has the LLM generate a bogus local stub file with the same name
    (e.g. ``src/arcade.py``) to "fix" it — corrupting the project by
    shadowing the real installed package.
    """
    cwd = os.getcwd()
    for venv_name in _VENV_DIR_NAMES:
        venv_dir = os.path.join(cwd, venv_name)
        if not os.path.isdir(venv_dir):
            continue
        # Windows layout: <venv>/Lib/site-packages/<module>
        candidates = [os.path.join(venv_dir, "Lib", "site-packages")]
        # POSIX layout: <venv>/lib/python3.X/site-packages/<module>
        posix_lib = os.path.join(venv_dir, "lib")
        if os.path.isdir(posix_lib):
            for entry in os.listdir(posix_lib):
                candidates.append(os.path.join(posix_lib, entry, "site-packages"))
        for site_packages in candidates:
            if not os.path.isdir(site_packages):
                continue
            if (os.path.isdir(os.path.join(site_packages, top_module))
                    or os.path.isfile(os.path.join(site_packages, top_module + ".py"))):
                return True
            # Distribution names don't always match the import name
            # (e.g. PyYAML -> yaml), so also check for a dist-info/egg-info
            # directory whose name starts with the module name.
            try:
                for entry in os.listdir(site_packages):
                    if entry.lower().startswith(top_module.lower()) and (
                            entry.endswith(".dist-info") or entry.endswith(".egg-info")):
                        return True
            except OSError:
                pass
    return False


def _is_external_import(import_source: str, importer_file: str) -> bool:
    """Return ``True`` if *import_source* refers to an external package.

    JS/TS: anything not starting with ``./`` or ``../`` (node_modules).
    Python: stdlib modules.
    Go: anything that looks like a stdlib or remote module.
    """
    ext = os.path.splitext(importer_file)[1].lower()

    if ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
        if import_source.startswith("./") or import_source.startswith("../"):
            return False
        if import_source.startswith("@/") or import_source.startswith("~/"):
            return False
        return True

    if ext == ".py":
        if import_source.startswith("."):
            return False
        top_module = import_source.split(".")[0]
        if top_module in _PYTHON_STDLIB:
            return True
        # Also treat installed third-party packages (pytest, pygame, requests, …)
        # as external so they never trigger a "broken import" false positive.
        # Check the TARGET project's own venv first — it's what actually
        # matters, and covers packages agentchanti itself doesn't depend on.
        if _project_has_installed_package(top_module):
            return True
        try:
            import importlib.util as _ilu
            return _ilu.find_spec(top_module) is not None
        except Exception:
            return False

    if ext == ".go":
        # Local relative imports start with "."; stdlib has no "."
        if import_source.startswith("."):
            return False
        return True

    if ext in (".java",):
        # java.* and javax.* are stdlib
        if import_source.startswith("java.") or import_source.startswith("javax."):
            return True
        return False

    if ext == ".rs":
        # std::, core::, alloc:: are stdlib
        if import_source.startswith(("std::", "core::", "alloc::")):
            return True
        return False

    return True


# ── Parent file guessing heuristic ───────────────────────────────

def _guess_parent_file(
    new_file: str, step_text: str, memory_files: dict[str, str],
) -> str | None:
    """Heuristic to guess which file should import *new_file*.

    Strategies (tried in order):
    1. Both file stems are mentioned in the step text.
    2. An index file exists in the same or parent directory.
    3. Common root files (App.tsx, main.py, etc.).
    """
    stem = os.path.splitext(os.path.basename(new_file))[0]
    ext = os.path.splitext(new_file)[1].lower()

    # Strategy 1: step text mentions both the new file and another file
    step_lower = step_text.lower()
    if stem.lower() in step_lower:
        for fpath in memory_files:
            if fpath == new_file:
                continue
            other_stem = os.path.splitext(os.path.basename(fpath))[0]
            if other_stem.lower() in step_lower:
                return fpath

    # Strategy 2: index file in the same or parent directory
    parent_dir = os.path.dirname(new_file)
    for index_name in ("index.tsx", "index.ts", "index.jsx", "index.js", "__init__.py"):
        index_path = os.path.join(parent_dir, index_name).replace("\\", "/")
        if index_path in memory_files:
            return index_path

    # Strategy 2b: Angular/NgModule — component files are declared in
    # the nearest *.module.ts, not imported by main.ts directly.
    if ext in (".ts", ".js") and stem.endswith(".component"):
        for fpath in memory_files:
            if fpath.endswith(".module.ts") and fpath != new_file:
                # Prefer a module in the same project subtree
                if os.path.dirname(fpath).startswith(
                    parent_dir.rsplit("/", 1)[0] if "/" in parent_dir else ""
                ):
                    return fpath

    # Strategy 3: common root files
    if ext in (".tsx", ".jsx", ".ts", ".js"):
        for root_name in ("App.tsx", "App.jsx", "App.ts", "App.js",
                          "main.tsx", "main.ts", "main.jsx", "main.js"):
            for fpath in memory_files:
                if fpath.endswith(root_name) and fpath != new_file:
                    return fpath

    if ext == ".py":
        for fpath in memory_files:
            if fpath.endswith("__init__.py") and fpath != new_file:
                fdir = os.path.dirname(fpath)
                if new_file.startswith(fdir):
                    return fpath

    return None


def _llm_guess_parent_file(
    new_file: str,
    exported_symbols: list[str],
    memory_files: dict[str, str],
    llm_client,
) -> str | None:
    """Use a tiny LLM call to identify which file should import *new_file*.

    Only called when both the plan map and the heuristic return ``None``.
    Keeps the prompt minimal so even small models can answer reliably.
    Returns a validated file path from *memory_files*, or ``None``.
    """
    # Exclude test files and the file itself from candidates
    candidates = [
        f for f in sorted(memory_files.keys())
        if f != new_file and not _is_test_file(f)
    ]
    if not candidates:
        return None

    file_list = "\n".join(f"- {f}" for f in candidates)
    sym_str = ", ".join(exported_symbols) if exported_symbols else os.path.basename(new_file)
    prompt = (
        f"Task: identify which file should import a new module.\n\n"
        f"New file: {new_file}\n"
        f"Exports: {sym_str}\n\n"
        f"Existing project files:\n{file_list}\n\n"
        f"Which ONE file from the list above should import '{os.path.basename(new_file)}'?\n"
        f"Reply with ONLY the exact file path. No explanation."
    )
    try:
        response = llm_client.generate_response(prompt).strip().strip("'\"` \n")
        # Exact match
        if response in memory_files:
            return response
        # Case-insensitive match
        response_lower = response.lower()
        for fpath in memory_files:
            if fpath.lower() == response_lower:
                return fpath
        # Basename match as last resort
        resp_base = os.path.basename(response).lower()
        for fpath in memory_files:
            if os.path.basename(fpath).lower() == resp_base:
                return fpath
        _logger.debug("[DepCheck] LLM parent guess '%s' not in project files", response)
        return None
    except Exception as exc:
        _logger.debug("[DepCheck] LLM parent guess failed: %s", exc)
        return None


# Python entry-point guard: a module with `if __name__ == "__main__":` is
# executed, not imported — "no importer" is expected, not a wiring gap.
_MAIN_GUARD_RE = re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']')


def _is_package_initializer(path: str) -> bool:
    """True for files a package exposes implicitly rather than by name."""
    return os.path.basename((path or "").replace("\\", "/")).lower() in (
        "__init__.py", "index.js", "index.ts", "index.jsx", "index.tsx")


def _plan_declares_import(nf: str, declared_imports: set[str]) -> bool:
    """True if a plan step declares it will import file *nf*.

    Declarations come from ``imports:`` plan lines and may use file paths
    ('src/snake.py'), Python module notation ('src.snake'), or bare module
    names ('snake').
    """
    nf_norm = nf.replace("\\", "/")
    nf_noext = os.path.splitext(nf_norm)[0]
    for decl in declared_imports:
        d = decl.replace("\\", "/")
        dpath = d if "/" in d else d.replace(".", "/")
        if dpath in (nf_norm, nf_noext) or nf_noext.endswith("/" + dpath):
            return True
    return False


def _find_file_by_name(name: str, memory_files: dict[str, str]) -> str | None:
    """Find a file in memory whose stem matches *name* (case-insensitive)."""
    name_lower = name.lower()
    for fpath in memory_files:
        stem = os.path.splitext(os.path.basename(fpath))[0].lower()
        if stem == name_lower:
            return fpath
    return None


# ── Gap detection ────────────────────────────────────────────────

def find_gaps(
    before: DependencySnapshot,
    after: DependencySnapshot,
    new_files: list[str],
    step_text: str,
    memory_files: dict[str, str],
    plan_imported_by: dict[str, str] | None = None,
    pending_target_files: set[str] | None = None,
    plan_declared_imports: set[str] | None = None,
) -> list[IntegrationGap]:
    """Compare before/after snapshots to detect integration gaps.

    Detects:
    1. **Orphaned exports** — newly created file exports symbols but nothing
       imports it.
    2. **Broken imports** — a file imports a path that doesn't resolve to
       any known file (only NEW imports are flagged).
    3. **Missing connections** — step text mentions connecting two entities
       but no import edge exists between them.
    """
    gaps: list[IntegrationGap] = []
    all_known_files = set(memory_files.keys())
    # Include files that pending plan steps will create — these are not
    # broken imports, just files that haven't been written yet.
    if pending_target_files:
        all_known_files |= pending_target_files

    # ── 1. Orphaned exports ──
    for nf in new_files:
        if _is_test_file(nf):
            continue
        if _is_tool_config_file(nf):
            continue  # config files are read by tools, never imported
        if _is_django_app_module(nf, all_known_files):
            _logger.info(
                "[DepCheck] Skipping orphaned_export for '%s': Django app "
                "module (framework-wired via settings/URLconf, not "
                "sibling-imported)", nf)
            continue
        nf_deps = after.file_deps.get(nf)
        if not nf_deps or not nf_deps.exports:
            continue

        # Entry-point modules are executed, not imported — skip.
        nf_content = (memory_files.get(nf)
                      or memory_files.get(nf.replace("\\", "/")) or "")
        if _MAIN_GUARD_RE.search(nf_content):
            _logger.debug(
                "[DepCheck] Skipping orphaned_export for '%s': "
                "entry-point module (__main__ guard)", nf)
            continue

        # A package initializer is never imported BY NAME — Python imports
        # it implicitly when the package is imported, and JS resolves
        # `index.js` from the directory. "No other file imports it" is
        # therefore always true and always meaningless here.
        #
        # Observed: a run where this fired on `pacman/__init__.py`,
        # reported "Likely parent: 'pacman/__init__.py'" (the file as its
        # own parent — a nonsense diagnosis), spent 1,496 tokens
        # generating a fix, wrote the file, and broke the already-green
        # `from pacman.map import Map` gate. The monotonic guard rolled
        # the wave back and the run reported failure.
        if _is_package_initializer(nf):
            _logger.debug(
                "[DepCheck] Skipping orphaned_export for '%s': package "
                "initializer — nothing imports it by name", nf)
            continue

        # A pending plan step already declares it will import this file —
        # the wiring belongs to that future step.  Pre-wiring it here (e.g.
        # re-exporting from __init__.py) couples the package to the file's
        # dependencies and can break unrelated imports.
        if plan_declared_imports and _plan_declares_import(nf, plan_declared_imports):
            _logger.info(
                "[DepCheck] Skipping orphaned_export for '%s': "
                "a pending plan step declares this import", nf)
            continue

        nf_basename = os.path.splitext(os.path.basename(nf))[0].lower()

        is_imported = False
        for other_path, other_deps in after.file_deps.items():
            if other_path == nf:
                continue
            for imp_src in other_deps.imports:
                resolved = _normalize_import_path(imp_src, other_path)
                if _file_matches_import(nf, resolved) or _file_matches_import(nf, imp_src):
                    is_imported = True
                    break
                # Also check if the import's basename matches the new file's
                # basename.  This handles the common pattern where the same
                # symbol is imported from a sibling directory (e.g. App.jsx
                # imports './components/Homepage' while the new file lives at
                # 'pages/Homepage.jsx').  A basename match means the symbol
                # is already wired — the path just differs.
                imp_basename = os.path.splitext(
                    os.path.basename(imp_src)
                )[0].lower()
                if imp_basename == nf_basename and not _is_external_import(imp_src, other_path):
                    is_imported = True
                    break
            if is_imported:
                break

        if not is_imported and len(after.file_deps) > 1:
            # Priority order: plan declaration > heuristic (LLM fallback in caller)
            nf_norm = nf.replace("\\", "/")
            plan_parent = (plan_imported_by or {}).get(nf_norm)
            if plan_parent is None:
                # Also try basename match for plan map (sub-project path differences)
                nf_base = os.path.basename(nf_norm)
                for k, v in (plan_imported_by or {}).items():
                    if os.path.basename(k) == nf_base:
                        plan_parent = v
                        break
            likely_parent = plan_parent or _guess_parent_file(nf, step_text, memory_files)
            if plan_parent:
                _logger.debug("[DepCheck] Plan-declared parent for '%s': %s", nf, plan_parent)

            # ── Circular import guard (bidirectional) ──
            # If the new file (nf) already imports from the likely parent,
            # OR the parent already imports from the new file, wiring the
            # parent to import nf back would create a circular dependency.
            # Skip this gap entirely.
            if likely_parent:
                would_be_circular = False
                parent_norm = likely_parent.replace("\\", "/")
                # Direction 1: nf imports from parent
                for imp_src in (nf_deps.imports or []):
                    resolved = _normalize_import_path(imp_src, nf)
                    if (_file_matches_import(parent_norm, resolved)
                            or _file_matches_import(parent_norm, imp_src)):
                        would_be_circular = True
                        break
                # Direction 2: parent already imports from nf
                if not would_be_circular:
                    parent_deps = after.file_deps.get(likely_parent)
                    if parent_deps:
                        for imp_src in (parent_deps.imports or []):
                            resolved = _normalize_import_path(
                                imp_src, likely_parent)
                            if (_file_matches_import(nf_norm, resolved)
                                    or _file_matches_import(nf_norm, imp_src)):
                                would_be_circular = True
                                break
                if would_be_circular:
                    _logger.info(
                        "[DepCheck] Skipping orphaned_export for '%s' → '%s': "
                        "would create circular import", nf, likely_parent)
                    continue

            gaps.append(IntegrationGap(
                gap_type="orphaned_export",
                source_file=nf,
                target_file=likely_parent,
                symbol=", ".join(nf_deps.exports[:5]),
                description=(
                    f"File '{nf}' exports [{', '.join(nf_deps.exports[:5])}] "
                    f"but no other file imports it."
                    + (f" Likely parent: '{likely_parent}'." if likely_parent else "")
                ),
            ))

    # ── 2. Broken imports ──
    for fpath, deps in after.file_deps.items():
        for imp_src in deps.imports:
            if _is_external_import(imp_src, fpath):
                continue

            # `from . import x` / `from .. import x` — the referenced
            # package is the importer's own parent directory, which
            # exists by construction (scaffolders don't register their
            # __init__.py in memory, so file matching can't prove it).
            # Observed as a false broken_import on every Django app's
            # urls.py, each costing an LLM fix call that got discarded.
            if fpath.endswith(".py") and imp_src.strip(".") == "":
                continue

            resolved = _normalize_import_path(imp_src, fpath)
            found = any(
                _file_matches_import(known, resolved) or _file_matches_import(known, imp_src)
                for known in all_known_files
            )

            if not found:
                # Only flag imports that were NOT present before this step
                before_deps = before.file_deps.get(fpath)
                was_present_before = (
                    before_deps is not None and imp_src in before_deps.imports
                )
                if not was_present_before:
                    gaps.append(IntegrationGap(
                        gap_type="broken_import",
                        source_file=fpath,
                        target_file=None,
                        symbol=imp_src,
                        description=f"File '{fpath}' imports '{imp_src}' but no matching file exists.",
                    ))

    # ── 3. Missing connections ──
    connection_patterns = [
        re.compile(
            r"(?:add|integrate|include|use|import|wire|connect)\s+(\w+)"
            r"\s+(?:to|into|in|within)\s+(\w+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(\w+)\s+(?:component|module|service|class)\s+.*?"
            r"(?:to|into|in)\s+(\w+)",
            re.IGNORECASE,
        ),
    ]
    for pat in connection_patterns:
        m = pat.search(step_text)
        if not m:
            continue
        source_name, target_name = m.group(1), m.group(2)
        source_file = _find_file_by_name(source_name, memory_files)
        target_file = _find_file_by_name(target_name, memory_files)
        if source_file and target_file and source_file != target_file:
            target_deps = after.file_deps.get(target_file)
            if target_deps:
                has_import = any(
                    _file_matches_import(source_file, _normalize_import_path(imp, target_file))
                    or _file_matches_import(source_file, imp)
                    for imp in target_deps.imports
                )
                if not has_import:
                    already = any(
                        g.source_file == source_file and g.target_file == target_file
                        for g in gaps
                    )
                    if not already:
                        gaps.append(IntegrationGap(
                            gap_type="missing_connection",
                            source_file=source_file,
                            target_file=target_file,
                            symbol=source_name,
                            description=(
                                f"Step mentions connecting '{source_name}' to "
                                f"'{target_name}' but '{target_file}' does not "
                                f"import '{source_file}'."
                            ),
                        ))

    # ── 4. Missing default export ──
    # A JS/TS file that was created/modified in this step lacks `export default`
    # but another file imports it with a default import (import Foo from './Foo').
    # This causes a runtime error: the imported value is undefined.
    _js_exts = (".js", ".jsx", ".ts", ".tsx", ".mjs")
    for nf in new_files:
        if _is_test_file(nf):
            continue
        ext = os.path.splitext(nf)[1].lower()
        if ext not in _js_exts:
            continue
        nf_deps = after.file_deps.get(nf)
        if nf_deps is None:
            continue
        if nf_deps.has_default_export:
            continue
        # Check if the file had a default export before (LLM removed it)
        before_deps = before.file_deps.get(nf)
        lost_default = before_deps is not None and before_deps.has_default_export
        # Check if any other file default-imports this file
        is_default_imported = False
        importer_file = None
        for other_path, other_deps in after.file_deps.items():
            if other_path == nf:
                continue
            for di in other_deps.default_imports:
                resolved = _normalize_import_path(di, other_path)
                if _file_matches_import(nf, resolved) or _file_matches_import(nf, di):
                    is_default_imported = True
                    importer_file = other_path
                    break
            if is_default_imported:
                break
        if lost_default or is_default_imported:
            component_name = os.path.splitext(os.path.basename(nf))[0]
            reason = (
                f"was removed during editing (previously had export default)"
                if lost_default else
                f"is default-imported by '{importer_file}'"
            )
            gaps.append(IntegrationGap(
                gap_type="missing_default_export",
                source_file=nf,
                target_file=importer_file,
                symbol=component_name,
                description=(
                    f"File '{nf}' is a JSX/TSX component but has no "
                    f"`export default` statement. It {reason}. "
                    f"Add `export default {component_name};` at the end of the file."
                ),
            ))

    # ── 5. Stale callers — signature contract check ──
    # When a newly written callable gains required params, ensure call sites
    # in other files are updated.  Works across all supported languages.
    try:
        sig_gaps = _find_signature_gaps(new_files, memory_files)
        gaps.extend(sig_gaps)
    except Exception as exc:
        _logger.warning("[DepCheck] Signature contract check failed: %s", exc)

    return gaps


# ── LLM fix prompt ───────────────────────────────────────────────

_FIX_PROMPT_TEMPLATE = """\
You are a dependency integration specialist. Files were just created or modified \
but have integration gaps that must be fixed. Fix ALL gaps with minimal changes.

## Step Context
{step_text}
{kb_context_section}
## Integration Gaps Found
{gaps_formatted}

## Current File Contents
{files_formatted}

## Module System
{module_system_note}

## Rules
- Output ONLY files that need changes (not unchanged files).
- Use #### [FILE]: path format followed by a code block.
- Include COMPLETE file contents for changed files.
- Match existing import style (ESM/CJS, relative/absolute, single/double quotes).
- For component imports, use the correct relative path from importer to importee.
- Do NOT modify package.json, go.mod, or other config/manifest files.
- Do NOT add unnecessary imports — only fix the gaps listed above.
- Do NOT create circular imports (A imports B and B imports A). If wiring \
an import would create a cycle, skip that gap entirely.
- Preserve ALL existing code, comments, and formatting in modified files.
- For MISSING DEFAULT EXPORT gaps: add `export default ComponentName;` at the end \
of the file. The component name should match the filename in PascalCase. \
NEVER remove the existing component function/class — only add the missing export.
- For STALE CALLER gaps: update the call site to pass all listed required \
arguments/props. Use sensible placeholder values or variables already in scope. \
Do NOT change the callee's definition — only fix the call site(s).
"""


def _module_system_of(fpath: str, memory_files: dict) -> str | None:
    """``"esm"``/``"cjs"`` for one JS file, or None when nothing decides it.

    Node decides this per file, from the nearest ``package.json`` — not
    once per repo. A monorepo with a Vite frontend (``"type": "module"``)
    beside an Express backend (``"type": "commonjs"``) is two answers,
    and the file's own directory is the one that knows which.

    Order: an explicit ``.mjs``/``.cjs`` extension, then the nearest
    manifest (a manifest with no ``type`` still bounds the package, and
    Node defaults it to CommonJS), then the file's own syntax.
    """
    norm = (fpath or "").replace("\\", "/").strip("/")
    if norm.endswith(".mjs"):
        return "esm"
    if norm.endswith(".cjs"):
        return "cjs"

    parts = norm.split("/")[:-1]
    while True:
        manifest = "/".join(parts + ["package.json"]) if parts else "package.json"
        raw = memory_files.get(manifest)
        if raw is None and os.path.isfile(manifest):
            try:
                with open(manifest, "r", encoding="utf-8", errors="replace") as fh:
                    raw = fh.read()
            except OSError:
                raw = None
        if raw:
            try:
                declared = json.loads(raw).get("type")
            except (ValueError, AttributeError):
                declared = None
            if declared == "module":
                return "esm"
            return "cjs"
        if not parts:
            break
        parts = parts[:-1]

    content = memory_files.get(fpath, "") or ""
    if "require(" in content:
        return "cjs"
    if "import " in content and " from " in content:
        return "esm"
    return None


_MODULE_SYSTEM_RULE = {
    "esm": "ES Modules (import/export) — use ESM syntax",
    "cjs": "CommonJS (require/module.exports) — use CJS syntax",
}


def _module_system_note(relevant_files, memory_files: dict) -> str:
    """The module-system instruction for a dependency-fix prompt.

    Measured 2026-08-19: this was one run-wide ``any("import " in c ...)``
    over every file in memory, so a single React component anywhere made
    the answer ESM for the whole repo — and the CommonJS branch sat in an
    ``elif`` it could never reach. The fix prompt for a CommonJS Express
    module was told "Project uses ES Modules", rewrote its
    ``module.exports = {...}`` into ``export function``, and dropped every
    declared export. Its previously green gate went red twice, and the
    monotonic check read that as the GATE being wrong and told the reader
    to go fix the plan's verify line.

    Files that disagree are named individually rather than averaged: in a
    two-root repo the average is wrong for one of the roots, and picking
    either silently corrupts the other.
    """
    # With no gaps there are no relevant files; fall back to everything
    # in memory so a caller asking about the repo as a whole still gets
    # an answer (the per-file listing below then covers a mixed repo).
    targets = sorted(relevant_files) or [
        f for f in sorted(memory_files)
        if f.rsplit(".", 1)[-1] in ("js", "jsx", "ts", "tsx", "mjs", "cjs")
    ]
    systems = {f: _module_system_of(f, memory_files) for f in targets}
    known = {v for v in systems.values() if v}
    if not known:
        return "Unknown module system."
    if len(known) == 1:
        return f"Project uses {_MODULE_SYSTEM_RULE[known.pop()]}."
    lines = ["This repo mixes module systems — each file below keeps the one "
             "its own package.json declares. Do NOT convert a file from one "
             "to the other:"]
    lines += [f"  - {f}: {_MODULE_SYSTEM_RULE[v]}"
              for f, v in systems.items() if v]
    return "\n".join(lines)


def build_fix_prompt(
    gaps: list[IntegrationGap],
    memory_files: dict[str, str],
    step_text: str,
    language: str | None,
    kb_context: str = "",
) -> str:
    """Build a single LLM prompt to fix all integration gaps."""
    gap_lines: list[str] = []
    for i, gap in enumerate(gaps, 1):
        tag = gap.gap_type.upper().replace("_", " ")
        gap_lines.append(f"{i}. [{tag}] {gap.description}")
    gaps_formatted = "\n".join(gap_lines)

    # Collect only relevant files
    relevant_files: set[str] = set()
    for gap in gaps:
        relevant_files.add(gap.source_file)
        if gap.target_file:
            relevant_files.add(gap.target_file)

    file_parts: list[str] = []
    for fpath in sorted(relevant_files):
        content = memory_files.get(fpath, "")
        if content:
            file_parts.append(f"#### [FILE]: {fpath}\n```\n{content}\n```")
    files_formatted = "\n\n".join(file_parts)

    # Detect module system — per file, from the nearest manifest.
    module_system_note = "Unknown module system."
    if language in ("javascript", "typescript"):
        module_system_note = _module_system_note(relevant_files, memory_files)
    elif language == "python":
        module_system_note = "Python project. Use standard import syntax."
    elif language == "go":
        module_system_note = "Go project. Use standard import syntax."

    kb_context_section = (
        f"\n## Project Knowledge Base\n{kb_context.strip()}\n"
        if kb_context and kb_context.strip()
        else ""
    )

    return _FIX_PROMPT_TEMPLATE.format(
        step_text=step_text,
        kb_context_section=kb_context_section,
        gaps_formatted=gaps_formatted,
        files_formatted=files_formatted,
        module_system_note=module_system_note,
    )


# ── Main entry point ─────────────────────────────────────────────

def run_dependency_check(
    step_idx: int,
    step_text: str,
    new_files: list[str],
    dep_before: DependencySnapshot,
    dep_after: DependencySnapshot,
    memory,
    llm_client,
    executor,
    display,
    language: str | None,
    cfg=None,
    all_plan_steps=None,
    kb_context: str = "",
) -> dict[str, str]:
    """Run post-step dependency validation and return fixes.

    Returns ``{}`` immediately if the feature is disabled, there are too few
    files in memory, or no integration gaps are detected (zero LLM overhead).
    When gaps *are* found, makes a **single** LLM call to fix all of them.
    """
    # Feature toggle
    if cfg and not getattr(cfg, "DEPENDENCY_CHECK_ENABLED", True):
        return {}

    memory_files = memory.all_files()

    # Need at least 2 files to have a dependency relationship
    if len(memory_files) <= 1:
        return {}

    if not new_files:
        return {}

    # Build the set of all files this session plans to create (from plan steps).
    # These are treated as "session files" even if not yet written to memory.
    session_files: set[str] = set(memory_files.keys())
    # Build plan_imported_by map: source_file → consumer_file (first declared wins)
    # Uses PlanStep.imported_by which is auto-derived from imports_from relationships
    # plus any explicit imported_by: lines the planner wrote.
    # Also collect pending target files so find_gaps() can suppress false
    # broken_import gaps for files that a future step will create.
    plan_imported_by: dict[str, str] = {}
    _pending_targets: set[str] = set()
    _pending_declared_imports: set[str] = set()
    if all_plan_steps:
        for ps in all_plan_steps:
            for tf in (ps.target_files or []):
                _ntf = tf.replace("\\", "/")
                session_files.add(_ntf)
                if ps.status in ("pending", "in_progress"):
                    _pending_targets.add(_ntf)
            if ps.status in ("pending", "in_progress"):
                # Files a future step declares it will import — those steps
                # do the wiring themselves, so their sources aren't orphans.
                _pending_declared_imports.update(ps.imports_from or {})
            for consumer_file in (ps.imported_by or []):
                for tf in (ps.target_files or []):
                    norm_tf = tf.replace("\\", "/")
                    if norm_tf not in plan_imported_by:
                        plan_imported_by[norm_tf] = consumer_file

    # Detect gaps
    display.step_info(step_idx, "[DepCheck] Scanning dependencies...")
    try:
        gaps = find_gaps(
            dep_before, dep_after, new_files, step_text, memory_files,
            plan_imported_by=plan_imported_by or None,
            pending_target_files=_pending_targets or None,
            plan_declared_imports=_pending_declared_imports or None,
        )
    except Exception as exc:
        _logger.warning("[DepCheck] Gap detection failed: %s", exc)
        return {}

    if not gaps:
        _logger.debug("[DepCheck] No integration gaps for step %d", step_idx + 1)
        display.step_info(step_idx, "[DepCheck] All dependencies connected.")
        return {}

    # For orphaned exports where both plan and heuristic returned None,
    # let the main fix prompt identify the correct parent (0 extra LLM
    # calls — parent guessing is folded into the single fix call).
    # Enrich the gap description so the fix LLM knows it must choose.
    for gap in gaps:
        if gap.gap_type == "orphaned_export" and gap.target_file is None:
            _logger.debug(
                "[DepCheck] No parent identified for '%s' — "
                "fix prompt will determine parent", gap.source_file)
            gap.description += (
                " No parent file identified — choose the most appropriate "
                "existing file to add the import to."
            )

    # Report gaps
    display.step_info(
        step_idx,
        f"[DepCheck] Found {len(gaps)} integration gap(s), generating fixes...",
    )
    for gap in gaps:
        _logger.info("[DepCheck] %s: %s", gap.gap_type, gap.description)

    # Single LLM call
    try:
        prompt = build_fix_prompt(gaps, memory_files, step_text, language, kb_context=kb_context)
        display.step_info(step_idx, "[DepCheck] Fixing dependency gaps (LLM)...")
        response = llm_client.generate_response(prompt)
    except Exception as exc:
        _logger.warning("[DepCheck] LLM fix call failed: %s", exc)
        display.step_info(step_idx, "[DepCheck] Fix generation failed, continuing without fixes")
        return {}

    # Parse response
    display.step_info(step_idx, "[DepCheck] Applying fixes...")
    fix_files = executor.parse_code_blocks(response)
    if not fix_files:
        fix_files = executor.parse_code_blocks_fuzzy(response)

    if not fix_files:
        _logger.warning("[DepCheck] Could not parse fix response")
        display.step_info(step_idx, "[DepCheck] No parseable fixes in response")
        return {}

    # Validate: only accept files relevant to the gaps.
    # Three tiers:
    #   1. Gap source/target files — always accepted
    #   2. Session files (plan target_files + memory) — always accepted
    #   3. Wiring files for watcher-created sources — if the runtime watcher
    #      saw a gap source file being CREATED (not just modified) this session,
    #      allow the dep-check to propose a new wiring file (e.g. App.jsx) in
    #      the same project root, as long as the agent hasn't written it yet
    relevant_files: set[str] = set()
    project_roots: set[str] = set()
    for gap in gaps:
        relevant_files.add(gap.source_file)
        if gap.target_file:
            relevant_files.add(gap.target_file)
        root = gap.source_file.replace("\\", "/").split("/")[0]
        if root:
            project_roots.add(root)

    # Files the runtime watcher saw being CREATED on disk this session
    watcher_created: set[str] = getattr(memory, "watcher_created_files", set()) or set()
    # Allow wiring files only when at least one orphaned-export source was
    # freshly created by the agent (not just modified).
    has_watcher_created_sources = any(
        gap.gap_type == "orphaned_export" and any(
            gap.source_file.replace("\\", "/").endswith(wf)
            or wf.endswith(os.path.basename(gap.source_file))
            for wf in watcher_created
        )
        for gap in gaps
    )

    # Build set of files protected by pending plan steps.
    # DepCheck must not overwrite a file that a later planned step will write
    # (via inline_edits, inline_code, or as a target_file) — doing so clobbers
    # content the planner already decided on, causing race-condition corruption
    # when multiple CODE steps execute in the same parallel wave.
    # A common case: main.jsx imports ./index.css which doesn't exist yet, but
    # step 2.2 will write the correct Tailwind v4 content.  Without this guard
    # DepCheck's LLM generates old Tailwind v3 directives that overwrite the
    # planner's correct version.
    _pending_inline_targets: set[str] = set()
    if all_plan_steps:
        for _ps in all_plan_steps:
            if _ps.status in ("pending", "in_progress"):
                for _ef in (_ps.inline_edits or {}).keys():
                    _pending_inline_targets.add(_ef.replace("\\", "/"))
                for _tf in (_ps.target_files or []):
                    _pending_inline_targets.add(_tf.replace("\\", "/"))
                for _ic in (_ps.inline_code or {}).keys():
                    _pending_inline_targets.add(_ic.replace("\\", "/"))

    validated: dict[str, str] = {}
    for fpath, content in fix_files.items():
        matched = fpath
        norm_matched = fpath.replace("\\", "/")
        if fpath not in memory_files:
            for known in memory_files:
                if known.endswith(fpath) or fpath.endswith(os.path.basename(known)):
                    matched = known
                    norm_matched = matched.replace("\\", "/")
                    break
        # Skip files that a future plan step will edit via inline_edits — let
        # the planned step handle the content rather than clobbering it here.
        if _pending_inline_targets and (
            norm_matched in _pending_inline_targets
            or any(
                pit.endswith("/" + norm_matched) or norm_matched.endswith("/" + pit)
                for pit in _pending_inline_targets
            )
        ):
            _logger.debug(
                "[DepCheck] Skipping fix for '%s' — protected by pending plan step", fpath
            )
            continue
        # Never regenerate an existing tool-config file — DepCheck has no KB
        # context to reproduce version-specific syntax (e.g. Tailwind v4 vs v3,
        # Vite vs Vitest config). If the LLM erroneously emits one, discard it.
        if _is_tool_config_file(norm_matched) and matched in memory_files:
            _logger.debug(
                "[DepCheck] Skipping fix for existing tool config '%s'", fpath
            )
            continue
        # Tier 1: gap source/target
        if matched in relevant_files:
            validated[matched] = content
            continue
        # Tier 2: planned session file — but ONLY if it's not already in
        # memory with correct content.  If a file was already written by a
        # completed step (e.g. index.css with Tailwind v4 syntax), the
        # DepCheck LLM may "fix" it using outdated training data (e.g.
        # Tailwind v3 directives).  Only allow overwrites for files that
        # have an actual gap or are not yet written.
        is_session_file = norm_matched in session_files or any(
            sf.endswith("/" + norm_matched) or norm_matched.endswith("/" + sf)
            for sf in session_files
        )
        if is_session_file:
            if matched not in memory_files:
                validated[matched] = content
                _logger.debug("[DepCheck] Accepted planned session file from fix: %s", fpath)
            else:
                _logger.debug(
                    "[DepCheck] Skipping fix for '%s' — already in memory "
                    "with no detected gap (preventing LLM overwrite of "
                    "correct content)", fpath)
            continue
        # Tier 3: wiring file for watcher-created sources
        in_project = any(norm_matched.startswith(root + "/") for root in project_roots)
        not_yet_written = matched not in memory_files
        if has_watcher_created_sources and in_project and not_yet_written:
            validated[matched] = content
            _logger.debug("[DepCheck] Accepted new wiring file from fix: %s", fpath)
        else:
            _logger.warning("[DepCheck] Ignoring unexpected file in fix: %s", fpath)

    # Clean any pseudo-diff markers the LLM may have emitted
    validated = {path: clean_diff_markers(content) for path, content in validated.items()}

    if validated:
        _logger.info("[DepCheck] Generated fixes for %d file(s)", len(validated))
    return validated
