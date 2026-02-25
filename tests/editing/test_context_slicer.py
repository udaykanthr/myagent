"""Tests for the ContextSlicer."""

import os
import pytest
import tempfile

from multi_agent_coder.editing.context_slicer import ContextSlicer, FileSlice
from multi_agent_coder.editing.scope_resolver import EditScope, SymbolRange


def _write_temp_file(content: str, suffix=".py") -> str:
    """Write content to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


SAMPLE_PYTHON = """\
import os
import sys
from datetime import datetime

# Module-level comment


class UserService:
    \"\"\"Handles user operations.\"\"\"

    def __init__(self, db):
        self.db = db

    def authenticate_user(self, username: str, password: str) -> bool:
        \"\"\"Authenticate a user.\"\"\"
        if not username or not password:
            return False
        user = self.db.find_user(username)
        if user is None:
            return False
        return user.check_password(password)

    def get_user(self, user_id: int):
        \"\"\"Get user by ID.\"\"\"
        return self.db.find_by_id(user_id)


def helper_function():
    \"\"\"A standalone helper.\"\"\"
    return 42
"""


class TestSliceFile:
    def test_basic_slice_with_primary_symbol(self):
        path = _write_temp_file(SAMPLE_PYTHON)
        try:
            scope = EditScope(
                primary_symbols=[
                    SymbolRange(
                        symbol_name="authenticate_user",
                        symbol_type="method",
                        file_path=path,
                        line_start=15,
                        line_end=22,
                        editable=True,
                        parent_class="UserService",
                    )
                ],
                affected_files=[path],
            )

            slicer = ContextSlicer()
            result = slicer.slice_file(path, scope, context_lines=2)

            assert result.file_path == path
            assert result.language == "python"
            assert result.total_lines > 0
            assert len(result.slices) >= 1
            # Must have imports
            assert "import os" in result.imports_block
            # Must have class signature
            assert result.class_signature is not None
            assert "UserService" in result.class_signature

            # Check editable block
            editable_blocks = [b for b in result.slices if b.editable]
            assert len(editable_blocks) >= 1
            assert "authenticate_user" in editable_blocks[0].content
            assert "EDITABLE" in editable_blocks[0].annotation
        finally:
            os.unlink(path)

    def test_context_symbol_is_read_only(self):
        # Use a file with symbols far enough apart that they don't merge
        content = """\
import os

def target_func():
    return 1

# padding line 6
# padding line 7
# padding line 8
# padding line 9
# padding line 10
# padding line 11
# padding line 12
# padding line 13
# padding line 14
# padding line 15
# padding line 16
# padding line 17
# padding line 18
# padding line 19
# padding line 20

def context_func():
    \"\"\"A context-only function.\"\"\"
    return 2
"""
        path = _write_temp_file(content)
        try:
            scope = EditScope(
                primary_symbols=[
                    SymbolRange(
                        symbol_name="target_func",
                        symbol_type="function",
                        file_path=path,
                        line_start=3,
                        line_end=4,
                        editable=True,
                    )
                ],
                context_symbols=[
                    SymbolRange(
                        symbol_name="context_func",
                        symbol_type="function",
                        file_path=path,
                        line_start=22,
                        line_end=24,
                        editable=False,
                    )
                ],
                affected_files=[path],
            )

            slicer = ContextSlicer()
            result = slicer.slice_file(path, scope, context_lines=2)

            context_blocks = [b for b in result.slices if not b.editable]
            assert len(context_blocks) >= 1
            assert "CONTEXT ONLY" in context_blocks[0].annotation
        finally:
            os.unlink(path)

    def test_imports_extraction(self):
        path = _write_temp_file(SAMPLE_PYTHON)
        try:
            scope = EditScope(
                primary_symbols=[
                    SymbolRange(
                        symbol_name="helper_function",
                        symbol_type="function",
                        file_path=path,
                        line_start=29,
                        line_end=31,
                        editable=True,
                    )
                ],
                affected_files=[path],
            )

            slicer = ContextSlicer()
            result = slicer.slice_file(path, scope)

            assert "import os" in result.imports_block
            assert "import sys" in result.imports_block
            assert "from datetime" in result.imports_block
        finally:
            os.unlink(path)


class TestFormatForPrompt:
    def test_formatted_output_structure(self):
        path = _write_temp_file(SAMPLE_PYTHON)
        try:
            scope = EditScope(
                primary_symbols=[
                    SymbolRange(
                        symbol_name="authenticate_user",
                        symbol_type="method",
                        file_path=path,
                        line_start=15,
                        line_end=22,
                        editable=True,
                        parent_class="UserService",
                    )
                ],
                affected_files=[path],
            )

            slicer = ContextSlicer()
            slices = {path: slicer.slice_file(path, scope)}
            formatted = slicer.format_for_prompt(slices)

            assert "=== FILE:" in formatted
            assert "lines total) ===" in formatted
            assert "Language: python" in formatted
            assert "[IMPORTS]" in formatted
            assert "=== END FILE ===" in formatted
            assert "EDITABLE" in formatted
        finally:
            os.unlink(path)

    def test_omission_markers(self):
        # Create a file with symbols far apart
        lines = ["import os\n"] + [f"# line {i}\n" for i in range(50)]
        lines.append("def foo():\n    pass\n")
        lines.extend([f"# filler {i}\n" for i in range(50)])
        lines.append("def bar():\n    pass\n")
        content = "".join(lines)
        path = _write_temp_file(content)
        try:
            scope = EditScope(
                primary_symbols=[
                    SymbolRange(
                        symbol_name="foo",
                        symbol_type="function",
                        file_path=path,
                        line_start=52,
                        line_end=53,
                        editable=True,
                    ),
                    SymbolRange(
                        symbol_name="bar",
                        symbol_type="function",
                        file_path=path,
                        line_start=104,
                        line_end=105,
                        editable=True,
                    ),
                ],
                affected_files=[path],
            )

            slicer = ContextSlicer()
            slices = {path: slicer.slice_file(path, scope, context_lines=2)}
            formatted = slicer.format_for_prompt(slices)

            assert "lines omitted" in formatted
        finally:
            os.unlink(path)


class TestSliceFiles:
    def test_multi_file_slicing(self):
        path1 = _write_temp_file("import os\ndef foo():\n    pass\n")
        path2 = _write_temp_file("import sys\ndef bar():\n    pass\n")
        try:
            scope1 = EditScope(
                primary_symbols=[
                    SymbolRange("foo", "function", path1, 2, 3, True),
                ],
                affected_files=[path1],
            )
            scope2 = EditScope(
                primary_symbols=[
                    SymbolRange("bar", "function", path2, 2, 3, True),
                ],
                affected_files=[path2],
            )

            slicer = ContextSlicer()
            result = slicer.slice_files({path1: scope1, path2: scope2})

            assert path1 in result
            assert path2 in result
        finally:
            os.unlink(path1)
            os.unlink(path2)


class TestLanguageDetection:
    @pytest.mark.parametrize("ext,expected", [
        (".py", "python"),
        (".js", "javascript"),
        (".ts", "typescript"),
        (".java", "java"),
        (".go", "go"),
        (".rs", "rust"),
        (".xyz", "unknown"),
    ])
    def test_detect_language(self, ext, expected):
        slicer = ContextSlicer()
        assert slicer._detect_language(f"test{ext}") == expected
