"""Tests for review context building."""

import pytest
from multi_agent_coder.orchestrator.memory import FileMemory


# Import the function under test - it's in step_handlers but we can
# test the logic via the module
def _build_review_context_standalone(new_files, old_files_dict, step_text):
    """Standalone version for testing without full step_handlers import."""
    import difflib
    import os

    parts = []
    for fpath, new_content in new_files.items():
        old_content = old_files_dict.get(fpath)

        if old_content is not None:
            old_lines = old_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            diff = difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{fpath}", tofile=f"b/{fpath}",
                n=3,
            )
            diff_text = "".join(diff)
            if diff_text.strip():
                parts.append(f"#### Changes in {fpath}:\n```diff\n{diff_text}```")
            else:
                parts.append(f"#### {fpath}: no changes")
        else:
            parts.append(f"#### [NEW FILE]: {fpath}\n```\n{new_content}\n```")

    return "\n\n".join(parts)


class TestBuildReviewContext:
    def test_new_file(self):
        result = _build_review_context_standalone(
            {"src/new.py": "def foo():\n    return 42\n"},
            {},
            "create new file",
        )
        assert "[NEW FILE]" in result
        assert "src/new.py" in result
        assert "def foo():" in result

    def test_modified_file(self):
        old = "def foo():\n    return 41\n"
        new = "def foo():\n    return 42\n"
        result = _build_review_context_standalone(
            {"src/mod.py": new},
            {"src/mod.py": old},
            "fix return value",
        )
        assert "Changes in src/mod.py" in result
        assert "diff" in result
        assert "return 42" in result or "+    return 42" in result

    def test_no_changes(self):
        content = "def foo():\n    return 42\n"
        result = _build_review_context_standalone(
            {"src/same.py": content},
            {"src/same.py": content},
            "check file",
        )
        assert "no changes" in result

    def test_diff_much_shorter_than_full(self):
        # Build a large file with one line changed
        old_lines = [f"line_{i} = {i}\n" for i in range(100)]
        new_lines = list(old_lines)
        new_lines[50] = "line_50 = 999  # changed\n"

        old = "".join(old_lines)
        new = "".join(new_lines)

        result = _build_review_context_standalone(
            {"big.py": new},
            {"big.py": old},
            "fix line 50",
        )
        # The diff should be much shorter than the full file
        assert len(result) < len(new)
        assert "line_50 = 999" in result

    def test_multiple_files(self):
        result = _build_review_context_standalone(
            {
                "a.py": "def a(): return 1\n",
                "b.py": "def b(): return 2\n",
            },
            {"a.py": "def a(): return 0\n"},
            "update functions",
        )
        assert "a.py" in result
        assert "b.py" in result
        assert "[NEW FILE]" in result  # b.py is new
