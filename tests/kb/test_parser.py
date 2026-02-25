"""
Unit tests for multi_agent_coder.kb.local.parser

Tests language detection, file parsing, and graceful error handling.
Uses temporary files with inline source code so tree-sitter is exercised
against real syntax without depending on project files.
"""

from __future__ import annotations

import os
import textwrap
import tempfile
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_py(tmp_path):
    """Write a simple Python file and return its path."""
    src = textwrap.dedent("""\
        \"\"\"Module docstring.\"\"\"

        import os
        import sys
        from pathlib import Path

        CONSTANT = 42


        class Animal:
            \"\"\"Base animal class.\"\"\"

            def __init__(self, name: str) -> None:
                self.name = name

            def speak(self) -> str:
                return "..."


        class Dog(Animal):
            \"\"\"A dog.\"\"\"

            def speak(self) -> str:
                return "Woof"

            def fetch(self, item: str) -> None:
                print(item)


        def standalone(x: int, y: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return x + y


        result = standalone(1, 2)
    """)
    p = tmp_path / "sample.py"
    p.write_text(src, encoding="utf-8")
    return str(p)


@pytest.fixture()
def tmp_js(tmp_path):
    """Write a simple JavaScript file and return its path."""
    src = textwrap.dedent("""\
        const fs = require('fs');

        class Greeter {
          constructor(name) {
            this.name = name;
          }

          greet() {
            return `Hello, ${this.name}`;
          }
        }

        function add(a, b) {
          return a + b;
        }

        module.exports = { Greeter, add };
    """)
    p = tmp_path / "sample.js"
    p.write_text(src, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_python(self):
        from multi_agent_coder.kb.local.parser import detect_language
        assert detect_language("foo.py") == "python"

    def test_javascript(self):
        from multi_agent_coder.kb.local.parser import detect_language
        assert detect_language("app.js") == "javascript"

    def test_typescript(self):
        from multi_agent_coder.kb.local.parser import detect_language
        assert detect_language("app.ts") == "typescript"

    def test_tsx(self):
        from multi_agent_coder.kb.local.parser import detect_language
        assert detect_language("component.tsx") == "typescript"

    def test_java(self):
        from multi_agent_coder.kb.local.parser import detect_language
        assert detect_language("Main.java") == "java"

    def test_go(self):
        from multi_agent_coder.kb.local.parser import detect_language
        assert detect_language("main.go") == "go"

    def test_rust(self):
        from multi_agent_coder.kb.local.parser import detect_language
        assert detect_language("lib.rs") == "rust"

    def test_cpp(self):
        from multi_agent_coder.kb.local.parser import detect_language
        assert detect_language("foo.cpp") == "cpp"
        assert detect_language("foo.cc") == "cpp"

    def test_csharp(self):
        from multi_agent_coder.kb.local.parser import detect_language
        assert detect_language("Program.cs") == "c_sharp"

    def test_ruby(self):
        from multi_agent_coder.kb.local.parser import detect_language
        assert detect_language("app.rb") == "ruby"

    def test_unsupported(self):
        from multi_agent_coder.kb.local.parser import detect_language
        assert detect_language("data.csv") is None
        assert detect_language("README.md") is None

    def test_case_insensitive(self):
        from multi_agent_coder.kb.local.parser import detect_language
        assert detect_language("APP.PY") == "python"


# ---------------------------------------------------------------------------
# Python parsing (requires tree-sitter-languages)
# ---------------------------------------------------------------------------

class TestPythonParsing:
    """Tests that require tree-sitter-languages to be installed."""

    def setup_method(self):
        pytest.importorskip("tree_sitter_languages")

    def test_returns_parsed_file(self, tmp_py):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_py)
        assert result.language == "python"
        assert result.parse_error is None or result.functions  # either no error or has data
        assert result.hash != ""

    def test_extracts_classes(self, tmp_py):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_py)
        class_names = {c.name for c in result.classes}
        assert "Animal" in class_names
        assert "Dog" in class_names

    def test_class_line_numbers(self, tmp_py):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_py)
        animal = next(c for c in result.classes if c.name == "Animal")
        assert animal.line_start > 0
        assert animal.line_end >= animal.line_start

    def test_class_inheritance(self, tmp_py):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_py)
        dog = next(c for c in result.classes if c.name == "Dog")
        assert "Animal" in dog.bases

    def test_extracts_functions(self, tmp_py):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_py)
        func_names = {f.name for f in result.functions}
        assert "standalone" in func_names
        assert "__init__" in func_names
        assert "speak" in func_names
        assert "fetch" in func_names

    def test_method_parent_class(self, tmp_py):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_py)
        inits = [f for f in result.functions if f.name == "__init__"]
        assert len(inits) >= 1
        assert inits[0].parent_class is not None

    def test_standalone_has_no_parent(self, tmp_py):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_py)
        standalone_fns = [f for f in result.functions if f.name == "standalone"]
        assert any(f.parent_class is None for f in standalone_fns)

    def test_function_params(self, tmp_py):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_py)
        standalone_fns = [f for f in result.functions if f.name == "standalone"]
        assert standalone_fns
        fn = standalone_fns[0]
        assert "x" in fn.params
        assert "y" in fn.params

    def test_return_type(self, tmp_py):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_py)
        standalone_fns = [f for f in result.functions if f.name == "standalone"]
        assert standalone_fns
        fn = standalone_fns[0]
        assert "int" in fn.return_type

    def test_extracts_imports(self, tmp_py):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_py)
        imported = {i.imported_name for i in result.imports}
        assert any("os" in m or "sys" in m or "pathlib" in m for m in imported)

    def test_hash_is_sha256(self, tmp_py):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_py)
        # SHA-256 hex digest is 64 chars
        assert len(result.hash) == 64

    def test_hash_changes_when_file_changes(self, tmp_py):
        from multi_agent_coder.kb.local.parser import parse_file
        r1 = parse_file(tmp_py)
        # Append a comment
        with open(tmp_py, "a") as fh:
            fh.write("\n# changed\n")
        r2 = parse_file(tmp_py)
        assert r1.hash != r2.hash


# ---------------------------------------------------------------------------
# JavaScript parsing
# ---------------------------------------------------------------------------

class TestJavaScriptParsing:
    def setup_method(self):
        pytest.importorskip("tree_sitter_languages")

    def test_detects_js(self, tmp_js):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_js)
        assert result.language == "javascript"

    def test_extracts_class(self, tmp_js):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_js)
        class_names = {c.name for c in result.classes}
        assert "Greeter" in class_names

    def test_extracts_function(self, tmp_js):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(tmp_js)
        func_names = {f.name for f in result.functions}
        assert "add" in func_names


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_nonexistent_file(self):
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file("/tmp/does_not_exist_xyz.py")
        assert result.parse_error is not None
        assert result.functions == []

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b,c\n1,2,3\n")
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(str(p))
        assert result.language == "unknown"
        assert result.parse_error is not None

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.py"
        p.write_text("")
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(str(p))
        # Should not raise; empty file is valid
        assert result.functions == []
        assert result.classes == []

    def test_binary_file(self, tmp_path):
        """Parser should not crash on binary content."""
        p = tmp_path / "binary.py"
        p.write_bytes(bytes(range(256)))
        from multi_agent_coder.kb.local.parser import parse_file
        result = parse_file(str(p))
        # May have parse errors but should not raise
        assert result is not None


# ---------------------------------------------------------------------------
# compute_file_hash
# ---------------------------------------------------------------------------

class TestComputeFileHash:
    def test_consistent_hash(self, tmp_path):
        from multi_agent_coder.kb.local.parser import compute_file_hash
        p = tmp_path / "foo.py"
        p.write_text("x = 1\n")
        h1 = compute_file_hash(str(p))
        h2 = compute_file_hash(str(p))
        assert h1 == h2
        assert len(h1) == 64

    def test_different_content_different_hash(self, tmp_path):
        from multi_agent_coder.kb.local.parser import compute_file_hash
        p1 = tmp_path / "a.py"
        p2 = tmp_path / "b.py"
        p1.write_text("x = 1\n")
        p2.write_text("x = 2\n")
        assert compute_file_hash(str(p1)) != compute_file_hash(str(p2))

    def test_missing_file(self):
        from multi_agent_coder.kb.local.parser import compute_file_hash
        h = compute_file_hash("/tmp/missing_file_xyz.py")
        assert h == ""
