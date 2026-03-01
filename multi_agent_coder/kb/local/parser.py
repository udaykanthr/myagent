"""
Tree-sitter AST parser for extracting structural information from source files.

Supports: Python, JavaScript, TypeScript, Java, C, C++, Go, Rust, Ruby, PHP, C#

Uses tree-sitter >= 0.22 API with individual language packages.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language mapping
# ---------------------------------------------------------------------------

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "c_sharp",
}

SUPPORTED_LANGUAGES: set[str] = set(EXTENSION_TO_LANGUAGE.values())


def detect_language(file_path: str) -> Optional[str]:
    """
    Return the tree-sitter language name for *file_path*, or None if unsupported.

    Parameters
    ----------
    file_path:
        Any file path; only the extension is examined.
    """
    ext = os.path.splitext(file_path)[1].lower()
    return EXTENSION_TO_LANGUAGE.get(ext)


# ---------------------------------------------------------------------------
# Data classes returned by the parser
# ---------------------------------------------------------------------------

@dataclass
class ParsedFunction:
    """A function or method extracted from source code."""
    name: str
    file_path: str
    line_start: int
    line_end: int
    docstring: str = ""
    params: list[str] = field(default_factory=list)
    return_type: str = ""
    parent_class: Optional[str] = None   # set if this is a method


@dataclass
class ParsedClass:
    """A class definition extracted from source code."""
    name: str
    file_path: str
    line_start: int
    line_end: int
    docstring: str = ""
    bases: list[str] = field(default_factory=list)


@dataclass
class ParsedVariable:
    """A module-level or class-level variable."""
    name: str
    file_path: str
    scope: str          # "module" | "class:<ClassName>" | "function:<name>"
    type_hint: str = ""


@dataclass
class ParsedImport:
    """A module import statement."""
    source_file: str
    imported_name: str      # dotted module name or path being imported
    alias: Optional[str] = None


@dataclass
class ParsedCall:
    """A function call site found in a function body."""
    caller_function: str
    callee_name: str
    file_path: str
    line: int


@dataclass
class ParsedFile:
    """All structural information extracted from a single source file."""
    path: str
    language: str
    hash: str
    functions: list[ParsedFunction] = field(default_factory=list)
    classes: list[ParsedClass] = field(default_factory=list)
    variables: list[ParsedVariable] = field(default_factory=list)
    imports: list[ParsedImport] = field(default_factory=list)
    calls: list[ParsedCall] = field(default_factory=list)
    parse_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Language → (tree-sitter Language object) lookup
# ---------------------------------------------------------------------------

# Map language name → callable that returns the raw language pointer
def _get_lang_func(language: str):
    """Return the tree-sitter language() function for *language*, or None."""
    try:
        if language == "python":
            import tree_sitter_python as m  # type: ignore
            return m.language
        elif language == "javascript":
            import tree_sitter_javascript as m  # type: ignore
            return m.language
        elif language == "typescript":
            import tree_sitter_typescript as m  # type: ignore
            return m.language_typescript
        elif language == "java":
            import tree_sitter_java as m  # type: ignore
            return m.language
        elif language == "c":
            import tree_sitter_c as m  # type: ignore
            return m.language
        elif language == "cpp":
            import tree_sitter_cpp as m  # type: ignore
            return m.language
        elif language == "go":
            import tree_sitter_go as m  # type: ignore
            return m.language
        elif language == "rust":
            import tree_sitter_rust as m  # type: ignore
            return m.language
        elif language == "ruby":
            import tree_sitter_ruby as m  # type: ignore
            return m.language
        elif language == "php":
            import tree_sitter_php as m  # type: ignore
            return m.language_php
        elif language == "c_sharp":
            import tree_sitter_c_sharp as m  # type: ignore
            return m.language
    except ImportError:
        pass
    return None


# Cache Language objects to avoid repeated construction
_LANG_CACHE: dict[str, object] = {}
_PARSER_CACHE: dict[str, object] = {}


def _get_ts_language(language: str):
    """
    Return the tree_sitter.Language object for *language*, or None.

    Caches results for performance.
    """
    if language in _LANG_CACHE:
        return _LANG_CACHE[language]
    try:
        import tree_sitter as ts  # type: ignore
        func = _get_lang_func(language)
        if func is None:
            return None
        lang_obj = ts.Language(func())
        _LANG_CACHE[language] = lang_obj
        return lang_obj
    except Exception as exc:
        logger.debug("Cannot load tree-sitter language %s: %s", language, exc)
        return None


def _get_ts_parser(language: str):
    """
    Return a tree-sitter Parser configured for *language*, or None.

    Caches parsers for performance.
    """
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]
    try:
        import tree_sitter as ts  # type: ignore
        lang_obj = _get_ts_language(language)
        if lang_obj is None:
            return None
        parser = ts.Parser(lang_obj)
        _PARSER_CACHE[language] = parser
        return parser
    except Exception as exc:
        logger.warning("Cannot create tree-sitter parser for %s: %s", language, exc)
        return None


# ---------------------------------------------------------------------------
# Query helper — new tree-sitter 0.22+ API
# ---------------------------------------------------------------------------

def _query_matches(lang_obj, query_src: str, node) -> list[tuple[int, dict]]:
    """
    Compile *query_src* against *lang_obj* and run it on *node*.

    Returns a list of (pattern_index, {capture_name: [Node]}) tuples,
    or an empty list if the query fails to compile or no matches are found.
    """
    if lang_obj is None or node is None:
        return []
    try:
        import tree_sitter as ts  # type: ignore
        q = ts.Query(lang_obj, query_src)
        qc = ts.QueryCursor(q)
        return list(qc.matches(node))
    except Exception:
        return []


def _safe_query_matches(lang_obj, query_src: str, node) -> list[dict]:
    """
    Execute query safely, splitting on blank lines to try each sub-pattern.

    Returns a flat list of capture dicts: [{capture_name: [Node]}].
    """
    if lang_obj is None or node is None:
        return []
    sub_patterns = [p.strip() for p in query_src.strip().split("\n\n") if p.strip()]
    all_results: list[dict] = []
    for pattern in sub_patterns:
        try:
            import tree_sitter as ts  # type: ignore
            q = ts.Query(lang_obj, pattern)
            qc = ts.QueryCursor(q)
            for _pat_idx, caps in qc.matches(node):
                all_results.append(caps)
        except Exception:
            pass  # silently skip invalid queries for this language version
    return all_results


# ---------------------------------------------------------------------------
# Language-specific tree-sitter queries (new single-pattern style)
# ---------------------------------------------------------------------------

# Each entry is a dict with query strings per extraction type.
# Queries use optional captures (?).  We'll split multi-pattern strings on
# blank lines so each sub-pattern is tried independently.

_QUERIES: dict[str, dict[str, str]] = {
    "python": {
        "functions": """\
(function_definition
  name: (identifier) @func.name
  parameters: (parameters) @func.params
  return_type: (type)? @func.ret) @func.def
""",
        "classes": """\
(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list)? @class.bases) @class.def
""",
        "imports": """\
(import_statement (dotted_name) @import.mod)

(import_from_statement
  module_name: (dotted_name) @import.mod)

(import_from_statement
  module_name: (relative_import) @import.mod)
""",
        "calls": """\
(call function: (identifier) @call.name)

(call function: (attribute attribute: (identifier) @call.method))
""",
    },
    "javascript": {
        "functions": """\
(function_declaration
  name: (identifier) @func.name
  parameters: (formal_parameters) @func.params) @func.def

(method_definition
  name: (property_identifier) @func.name
  parameters: (formal_parameters) @func.params) @func.def

(variable_declarator
  name: (identifier) @func.name
  value: (arrow_function
    parameters: (formal_parameters) @func.params)) @func.def
""",
        "classes": """\
(class_declaration
  name: (identifier) @class.name) @class.def
""",
        "imports": """\
(import_statement source: (string) @import.mod)

(call_expression
  function: (identifier) @req.keyword
  arguments: (arguments (string) @import.mod))
""",
        "calls": """\
(call_expression function: (identifier) @call.name)

(call_expression function: (member_expression
  property: (property_identifier) @call.method))
""",
    },
    "typescript": {
        "functions": """\
(function_declaration
  name: (identifier) @func.name
  parameters: (formal_parameters) @func.params
  return_type: (type_annotation)? @func.ret) @func.def

(method_definition
  name: (property_identifier) @func.name
  parameters: (formal_parameters) @func.params
  return_type: (type_annotation)? @func.ret) @func.def
""",
        "classes": """\
(class_declaration
  name: (type_identifier) @class.name) @class.def
""",
        "imports": """\
(import_statement source: (string) @import.mod)
""",
        "calls": """\
(call_expression function: (identifier) @call.name)

(call_expression function: (member_expression
  property: (property_identifier) @call.method))
""",
    },
    "java": {
        "functions": """\
(method_declaration
  name: (identifier) @func.name
  parameters: (formal_parameters) @func.params) @func.def
""",
        "classes": """\
(class_declaration
  name: (identifier) @class.name
  superclass: (superclass (type_identifier) @class.bases)?) @class.def

(interface_declaration name: (identifier) @class.name) @class.def
""",
        "imports": """\
(import_declaration (scoped_identifier) @import.mod)
""",
        "calls": """\
(method_invocation name: (identifier) @call.name)
""",
    },
    "go": {
        "functions": """\
(function_declaration
  name: (identifier) @func.name
  parameters: (parameter_list) @func.params) @func.def

(method_declaration
  name: (field_identifier) @func.name
  parameters: (parameter_list) @func.params) @func.def
""",
        "classes": """\
(type_spec
  name: (type_identifier) @class.name
  type: (struct_type)) @class.def
""",
        "imports": """\
(import_spec path: (interpreted_string_literal) @import.mod)
""",
        "calls": """\
(call_expression function: (identifier) @call.name)

(call_expression function: (selector_expression
  field: (field_identifier) @call.method))
""",
    },
    "rust": {
        "functions": """\
(function_item
  name: (identifier) @func.name
  parameters: (parameters) @func.params
  return_type: (_)? @func.ret) @func.def
""",
        "classes": """\
(struct_item name: (type_identifier) @class.name) @class.def

(trait_item name: (type_identifier) @class.name) @class.def
""",
        "imports": """\
(use_declaration argument: (_) @import.mod)
""",
        "calls": """\
(call_expression function: (identifier) @call.name)
""",
    },
    "c": {
        "functions": """\
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @func.name
    parameters: (parameter_list) @func.params)) @func.def
""",
        "classes": """\
(struct_specifier name: (type_identifier) @class.name) @class.def
""",
        "imports": """\
(preproc_include path: (string_literal) @import.mod)

(preproc_include path: (system_lib_string) @import.mod)
""",
        "calls": """\
(call_expression function: (identifier) @call.name)
""",
    },
    "cpp": {
        "functions": """\
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @func.name
    parameters: (parameter_list) @func.params)) @func.def
""",
        "classes": """\
(class_specifier name: (type_identifier) @class.name) @class.def

(struct_specifier name: (type_identifier) @class.name) @class.def
""",
        "imports": """\
(preproc_include path: (string_literal) @import.mod)

(preproc_include path: (system_lib_string) @import.mod)
""",
        "calls": """\
(call_expression function: (identifier) @call.name)
""",
    },
    "ruby": {
        "functions": """\
(method name: (identifier) @func.name
  parameters: (method_parameters)? @func.params) @func.def
""",
        "classes": """\
(class name: (constant) @class.name
  superclass: (superclass (constant) @class.bases)?) @class.def

(module name: (constant) @class.name) @class.def
""",
        "imports": """\
(call method: (identifier) @import.keyword
  arguments: (argument_list (string (string_content) @import.mod)))
""",
        "calls": """\
(call method: (identifier) @call.name)
""",
    },
    "php": {
        "functions": """\
(function_definition
  name: (name) @func.name
  parameters: (formal_parameters) @func.params) @func.def

(method_declaration
  name: (name) @func.name
  parameters: (formal_parameters) @func.params) @func.def
""",
        "classes": """\
(class_declaration name: (name) @class.name) @class.def

(interface_declaration name: (name) @class.name) @class.def
""",
        "imports": """\
(require_expression (string (string_value) @import.mod))

(include_expression (string (string_value) @import.mod))
""",
        "calls": """\
(function_call_expression function: (name) @call.name)
""",
    },
    "c_sharp": {
        "functions": """\
(method_declaration
  name: (identifier) @func.name
  parameters: (parameter_list) @func.params) @func.def
""",
        "classes": """\
(class_declaration name: (identifier) @class.name) @class.def

(interface_declaration name: (identifier) @class.name) @class.def
""",
        "imports": """\
(using_directive (identifier) @import.mod)

(using_directive (qualified_name) @import.mod)
""",
        "calls": """\
(invocation_expression function: (identifier) @call.name)

(invocation_expression function: (member_access_expression
  name: (identifier) @call.method))
""",
    },
}


# ---------------------------------------------------------------------------
# Node text helpers
# ---------------------------------------------------------------------------

def _text(node) -> str:
    """Decode a tree-sitter Node's text as UTF-8."""
    if node is None:
        return ""
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _first_node(caps: dict, *keys: str):
    """Return the first Node found under any of *keys* in a capture dict."""
    for k in keys:
        nodes = caps.get(k)
        if nodes:
            return nodes[0]
    return None


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------

def _extract_docstring(def_node) -> str:
    """
    Try to extract the first docstring from the body of a function/class node.
    Works for Python (expression_statement wrapping a string literal).
    Returns empty string if not found.
    """
    if def_node is None:
        return ""
    for child in def_node.children:
        if child.type in ("block", "class_body", "declaration_list"):
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for sub in stmt.children:
                        if sub.type in ("string", "concatenated_string"):
                            raw = _text(sub)
                            for q in ('"""', "'''", '"', "'"):
                                if (
                                    raw.startswith(q)
                                    and raw.endswith(q)
                                    and len(raw) > 2 * len(q)
                                ):
                                    return raw[len(q):-len(q)].strip()
                            return raw.strip()
                    break  # only check first stmt
    return ""


def _extract_params(params_node) -> list[str]:
    """Extract parameter names from a parameters/formal_parameters node."""
    if params_node is None:
        return []
    params: list[str] = []
    skip_names = {"self", "cls"}
    for child in params_node.children:
        if child.type == "identifier":
            name = _text(child)
            if name and name not in skip_names:
                params.append(name)
        elif child.type in (
            "typed_parameter", "typed_default_parameter",
            "default_parameter", "formal_parameter",
            "rest_pattern", "optional_parameter",
        ):
            # Get first identifier child
            for sub in child.children:
                if sub.type == "identifier":
                    name = _text(sub)
                    if name and name not in skip_names:
                        params.append(name)
                    break
    return params


def _node_inside(inner, outer) -> bool:
    """Return True if *inner* falls within *outer*'s source range."""
    return (
        inner.start_point[0] >= outer.start_point[0]
        and inner.end_point[0] <= outer.end_point[0]
    )


def _find_parent_class(
    fn_start_row: int,
    fn_end_row: int,
    class_ranges: list[tuple[int, int, str]],
) -> Optional[str]:
    """Return name of the tightest enclosing class, or None."""
    best: Optional[tuple[int, int, str]] = None
    for c_start, c_end, c_name in class_ranges:
        if c_start <= fn_start_row and c_end >= fn_end_row:
            if best is None or (c_end - c_start) < (best[1] - best[0]):
                best = (c_start, c_end, c_name)
    return best[2] if best else None


def _build_func_line_map(functions: list[ParsedFunction]) -> dict[int, str]:
    """Map each source line to the name of the innermost enclosing function."""
    line_map: dict[int, str] = {}
    # Process shortest-span first so larger spans don't overwrite inner ones
    for fn in sorted(functions, key=lambda f: f.line_end - f.line_start):
        for ln in range(fn.line_start, fn.line_end + 1):
            line_map[ln] = fn.name
    return line_map


def _find_scope(
    line: int,
    functions: list[ParsedFunction],
    classes: list[ParsedClass],
) -> str:
    """Return scope string for a variable at *line*."""
    best_fn: Optional[ParsedFunction] = None
    for fn in functions:
        if fn.line_start <= line <= fn.line_end:
            if best_fn is None or (fn.line_end - fn.line_start) < (best_fn.line_end - best_fn.line_start):
                best_fn = fn
    if best_fn:
        return f"function:{best_fn.name}"
    best_cls: Optional[ParsedClass] = None
    for cls in classes:
        if cls.line_start <= line <= cls.line_end:
            if best_cls is None or (cls.line_end - cls.line_start) < (best_cls.line_end - best_cls.line_start):
                best_cls = cls
    if best_cls:
        return f"class:{best_cls.name}"
    return "module"


# ---------------------------------------------------------------------------
# Main public parse function
# ---------------------------------------------------------------------------

def parse_code(source_bytes: bytes, language: str, file_path: str = "") -> ParsedFile:
    """
    Parse raw code bytes and return structural information.
    
    Parameters
    ----------
    source_bytes:
        Raw bytes of the source code.
    language:
        The tree-sitter language name natively supported (e.g., 'python', 'javascript').
    file_path:
        Optional file path to associate with the parsed results.

    Returns
    -------
    ParsedFile
        Populated with functions, classes, variables, imports, and call sites.
    """
    file_hash = hashlib.sha256(source_bytes).hexdigest()

    ts_parser = _get_ts_parser(language)
    if ts_parser is None:
        return ParsedFile(
            path=file_path,
            language=language,
            hash=file_hash,
            parse_error="tree-sitter parser unavailable for this language",
        )

    try:
        tree = ts_parser.parse(source_bytes)
    except Exception as exc:
        return ParsedFile(
            path=file_path,
            language=language,
            hash=file_hash,
            parse_error=f"Parse error: {exc}",
        )

    root = tree.root_node
    lang_obj = _get_ts_language(language)
    queries = _QUERIES.get(language, {})

    result = ParsedFile(path=file_path, language=language, hash=file_hash)

    # ------------------------------------------------------------------ classes
    class_ranges: list[tuple[int, int, str]] = []
    class_q = queries.get("classes", "")
    if class_q:
        for caps in _safe_query_matches(lang_obj, class_q, root):
            def_node = _first_node(caps, "class.def")
            name_node = _first_node(caps, "class.name")
            if def_node is None or name_node is None:
                continue
            class_name = _text(name_node)
            if not class_name:
                continue
            bases_node = _first_node(caps, "class.bases")
            bases: list[str] = []
            if bases_node:
                raw = _text(bases_node).strip("()")
                for b in raw.split(","):
                    b = b.strip()
                    if b and b not in ("object", ""):
                        bases.append(b)
            docstring = _extract_docstring(def_node)
            parsed_cls = ParsedClass(
                name=class_name,
                file_path=file_path,
                line_start=def_node.start_point[0] + 1,
                line_end=def_node.end_point[0] + 1,
                docstring=docstring,
                bases=bases,
            )
            result.classes.append(parsed_cls)
            class_ranges.append((
                def_node.start_point[0],
                def_node.end_point[0],
                class_name,
            ))

    # ---------------------------------------------------------------- functions
    func_q = queries.get("functions", "")
    if func_q:
        for caps in _safe_query_matches(lang_obj, func_q, root):
            def_node = _first_node(caps, "func.def")
            name_node = _first_node(caps, "func.name")
            if def_node is None or name_node is None:
                continue
            func_name = _text(name_node)
            if not func_name:
                continue
            parent_class = _find_parent_class(
                def_node.start_point[0],
                def_node.end_point[0],
                class_ranges,
            )
            params_node = _first_node(caps, "func.params")
            params = _extract_params(params_node)
            ret_node = _first_node(caps, "func.ret")
            return_type = _text(ret_node).lstrip(": ").strip() if ret_node else ""
            docstring = _extract_docstring(def_node)
            result.functions.append(ParsedFunction(
                name=func_name,
                file_path=file_path,
                line_start=def_node.start_point[0] + 1,
                line_end=def_node.end_point[0] + 1,
                docstring=docstring,
                params=params,
                return_type=return_type,
                parent_class=parent_class,
            ))

    # ----------------------------------------------------------------- imports
    import_q = queries.get("imports", "")
    if import_q:
        seen_imports: set[str] = set()
        for caps in _safe_query_matches(lang_obj, import_q, root):
            mod_node = _first_node(caps, "import.mod")
            if mod_node is None:
                continue
            mod_name = _text(mod_node).strip("\"'<> ")
            if not mod_name or mod_name in seen_imports:
                continue
            seen_imports.add(mod_name)
            result.imports.append(ParsedImport(
                source_file=file_path,
                imported_name=mod_name,
            ))

    # -------------------------------------------------------------------- calls
    call_q = queries.get("calls", "")
    if call_q:
        func_line_map = _build_func_line_map(result.functions)
        seen_calls: set[tuple[str, str, int]] = set()
        for caps in _safe_query_matches(lang_obj, call_q, root):
            callee_node = _first_node(caps, "call.name", "call.method")
            if callee_node is None:
                continue
            callee = _text(callee_node)
            if not callee:
                continue
            line = callee_node.start_point[0] + 1
            caller = func_line_map.get(line, "<module>")
            key = (caller, callee, line)
            if key in seen_calls:
                continue
            seen_calls.add(key)
            result.calls.append(ParsedCall(
                caller_function=caller,
                callee_name=callee,
                file_path=file_path,
                line=line,
            ))

    return result

def parse_file(file_path: str) -> ParsedFile:
    """
    Parse a single source file and return all structural information.

    Handles parse errors gracefully: on failure, returns a ParsedFile with
    `parse_error` set and empty symbol lists so callers can skip or log.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the source file.

    Returns
    -------
    ParsedFile
        Populated with functions, classes, variables, imports, and call sites.
    """
    language = detect_language(file_path)
    if language is None:
        return ParsedFile(
            path=file_path,
            language="unknown",
            hash="",
            parse_error="Unsupported file extension",
        )

    try:
        with open(file_path, "rb") as fh:
            source_bytes = fh.read()
    except OSError as exc:
        return ParsedFile(
            path=file_path,
            language=language,
            hash="",
            parse_error=f"Cannot read file: {exc}",
        )

    return parse_code(source_bytes, language, file_path)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_file_hash(file_path: str) -> str:
    """
    Compute the SHA-256 hex digest of a file's contents.

    Parameters
    ----------
    file_path:
        Path to the file.

    Returns
    -------
    str
        64-character hex digest, or empty string if the file cannot be read.
    """
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()
