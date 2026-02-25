"""Tests for the PatchApplier."""

import os
import tempfile

import pytest

from multi_agent_coder.editing.diff_parser import DiffHunk, FilePatch, ParsedDiff
from multi_agent_coder.editing.patch_applier import PatchApplier, ApplyResult


def _write_temp(content: str, suffix=".py") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


SAMPLE_FILE = """\
import os
import sys

def authenticate_user(username, password):
    user = db.find(username)
    return user.check_password(password)

def helper():
    return 42
"""


class TestApplySingleHunk:
    def test_exact_match_replacement(self):
        path = _write_temp(SAMPLE_FILE)
        try:
            diff = ParsedDiff(
                file_patches=[
                    FilePatch(
                        file_path=path,
                        hunks=[
                            DiffHunk(
                                line_number=4,
                                original_lines=[
                                    "def authenticate_user(username, password):",
                                    "    user = db.find(username)",
                                    "    return user.check_password(password)",
                                ],
                                replacement_lines=[
                                    "def authenticate_user(username, password):",
                                    "    if not username or not password:",
                                    "        return False",
                                    "    user = db.find(username)",
                                    "    if user is None:",
                                    "        return False",
                                    "    return user.check_password(password)",
                                ],
                            )
                        ],
                    )
                ]
            )

            applier = PatchApplier(validate_syntax=False)
            result = applier.apply(diff)

            assert result.success is True
            assert result.hunks_applied == 1
            assert result.hunks_failed == 0
            assert path in result.files_modified

            with open(path, "r") as f:
                content = f.read()
            assert "if not username or not password:" in content
            assert "if user is None:" in content
            # Original imports preserved
            assert "import os" in content
            assert "import sys" in content
            # Helper preserved
            assert "def helper():" in content
        finally:
            os.unlink(path)

    def test_deletion_hunk(self):
        path = _write_temp(SAMPLE_FILE)
        try:
            diff = ParsedDiff(
                file_patches=[
                    FilePatch(
                        file_path=path,
                        hunks=[
                            DiffHunk(
                                line_number=2,
                                original_lines=["import sys"],
                                replacement_lines=[],
                            )
                        ],
                    )
                ]
            )

            applier = PatchApplier(validate_syntax=False)
            result = applier.apply(diff)

            assert result.success is True
            with open(path) as f:
                content = f.read()
            assert "import sys" not in content
            assert "import os" in content
        finally:
            os.unlink(path)


class TestMultiHunkOrdering:
    def test_bottom_up_application(self):
        """Hunks at higher line numbers should be applied first."""
        content = "line1\nline2\nline3\nline4\nline5\n"
        path = _write_temp(content, suffix=".txt")
        try:
            diff = ParsedDiff(
                file_patches=[
                    FilePatch(
                        file_path=path,
                        hunks=[
                            DiffHunk(
                                line_number=2,
                                original_lines=["line2"],
                                replacement_lines=["LINE_TWO"],
                            ),
                            DiffHunk(
                                line_number=4,
                                original_lines=["line4"],
                                replacement_lines=["LINE_FOUR"],
                            ),
                        ],
                    )
                ]
            )

            applier = PatchApplier(validate_syntax=False)
            result = applier.apply(diff)

            assert result.success is True
            assert result.hunks_applied == 2

            with open(path) as f:
                lines = f.readlines()
            assert lines[1].strip() == "LINE_TWO"
            assert lines[3].strip() == "LINE_FOUR"
        finally:
            os.unlink(path)


class TestFuzzyMatch:
    def test_fuzzy_match_within_window(self):
        """Hunk should match even if off by a few lines."""
        content = "# header\nimport os\n\ndef foo():\n    return 1\n"
        path = _write_temp(content, suffix=".py")
        try:
            diff = ParsedDiff(
                file_patches=[
                    FilePatch(
                        file_path=path,
                        hunks=[
                            DiffHunk(
                                line_number=3,  # actual is line 4
                                original_lines=["def foo():"],
                                replacement_lines=["def bar():"],
                            )
                        ],
                    )
                ]
            )

            applier = PatchApplier(
                fuzzy_match_window=3,
                validate_syntax=False,
            )
            result = applier.apply(diff)

            assert result.success is True
            assert result.hunks_applied == 1

            with open(path) as f:
                content = f.read()
            assert "def bar():" in content
        finally:
            os.unlink(path)

    def test_no_fuzzy_match_outside_window(self):
        content = "\n" * 10 + "def foo():\n    return 1\n"
        path = _write_temp(content, suffix=".txt")
        try:
            diff = ParsedDiff(
                file_patches=[
                    FilePatch(
                        file_path=path,
                        hunks=[
                            DiffHunk(
                                line_number=2,  # actual is line 11 — too far
                                original_lines=["def foo():"],
                                replacement_lines=["def bar():"],
                            )
                        ],
                    )
                ]
            )

            applier = PatchApplier(
                fuzzy_match_window=3,
                validate_syntax=False,
            )
            result = applier.apply(diff)

            # Hunk should fail, but apply still succeeds (just with failed hunks)
            assert result.hunks_failed == 1
        finally:
            os.unlink(path)


class TestMultiFileTransactional:
    def test_multi_file_all_succeed(self):
        path1 = _write_temp("def foo():\n    return 1\n")
        path2 = _write_temp("def bar():\n    return 2\n")
        try:
            diff = ParsedDiff(
                file_patches=[
                    FilePatch(
                        file_path=path1,
                        hunks=[DiffHunk(1, ["def foo():"], ["def foo_new():"])],
                    ),
                    FilePatch(
                        file_path=path2,
                        hunks=[DiffHunk(1, ["def bar():"], ["def bar_new():"])],
                    ),
                ]
            )

            applier = PatchApplier(validate_syntax=False)
            result = applier.apply(diff)

            assert result.success is True
            assert len(result.files_modified) == 2
        finally:
            os.unlink(path1)
            os.unlink(path2)


class TestApplyResult:
    def test_default_result(self):
        result = ApplyResult()
        assert result.success is False
        assert result.files_modified == []
        assert result.hunks_applied == 0
        assert result.hunks_failed == 0

    def test_empty_diff(self):
        diff = ParsedDiff(file_patches=[])
        applier = PatchApplier(validate_syntax=False)
        result = applier.apply(diff)
        assert result.success is False
        assert "No patches" in result.error
