"""Tests for the DiffParser."""

import pytest

from multi_agent_coder.editing.diff_parser import (
    DiffParser, ParsedDiff, FilePatch, DiffHunk,
)


VALID_SINGLE_FILE_DIFF = """\
Here is the diff:

@@DIFF_START@@
FILE: src/auth.py
<<<<<<< ORIGINAL (line 15)
        user = self.db.find(username)
        if user is None:
            return False
=======
    def authenticate_user(self, username, password):
        if not username or not password:
            return False
        user = self.db.find(username)
        if user is None:
            return False
        return user.check_password(password)
>>>>>>> UPDATED
@@DIFF_END@@
"""

VALID_MULTI_FILE_DIFF = """\
@@DIFF_START@@
FILE: src/auth.py
<<<<<<< ORIGINAL (line 1)
def old_validate(data):
    return True
=======
def validate_input(data):
    if not data:
        raise ValueError("empty data")
    return True
>>>>>>> UPDATED
FILE: src/api.py
<<<<<<< ORIGINAL (line 5)
    result = process(payload)
=======
    result = validate_input(payload)
>>>>>>> UPDATED
@@DIFF_END@@
"""

MULTI_HUNK_DIFF = """\
@@DIFF_START@@
FILE: src/auth.py
<<<<<<< ORIGINAL (line 5)
import os
=======
import hashlib
import hmac
>>>>>>> UPDATED
<<<<<<< ORIGINAL (line 20)
    return hash(password)
=======
    return hmac.new(key, password, hashlib.sha256).hexdigest()
>>>>>>> UPDATED
@@DIFF_END@@
"""


class TestParse:
    def test_valid_single_file(self):
        parser = DiffParser()
        result = parser.parse(VALID_SINGLE_FILE_DIFF)

        assert result is not None
        assert result.parse_successful is True
        assert len(result.file_patches) == 1
        assert result.file_patches[0].file_path == "src/auth.py"
        assert len(result.file_patches[0].hunks) == 1

        hunk = result.file_patches[0].hunks[0]
        assert hunk.line_number == 15
        assert len(hunk.original_lines) == 3   # 3 original lines to replace
        assert len(hunk.replacement_lines) == 7  # 7 new replacement lines
        assert hunk.is_insertion is False
        assert hunk.is_deletion is False

    def test_valid_multi_file(self):
        parser = DiffParser()
        result = parser.parse(VALID_MULTI_FILE_DIFF)

        assert result is not None
        assert result.parse_successful is True
        assert len(result.file_patches) == 2
        assert result.file_patches[0].file_path == "src/auth.py"
        assert result.file_patches[1].file_path == "src/api.py"

    def test_multi_hunk_same_file(self):
        parser = DiffParser()
        result = parser.parse(MULTI_HUNK_DIFF)

        assert result is not None
        assert result.parse_successful is True
        assert len(result.file_patches) == 1
        assert len(result.file_patches[0].hunks) == 2
        assert result.file_patches[0].hunks[0].line_number == 5
        assert result.file_patches[0].hunks[1].line_number == 20

    def test_missing_markers_returns_none(self):
        parser = DiffParser()
        result = parser.parse("Here is the new file:\n```\ndef foo():\n    pass\n```")

        assert result is None

    def test_missing_end_marker_returns_none(self):
        parser = DiffParser()
        result = parser.parse("@@DIFF_START@@\nFILE: foo.py\nsome stuff")

        assert result is None

    def test_empty_diff_block_returns_none(self):
        parser = DiffParser()
        result = parser.parse("@@DIFF_START@@\n@@DIFF_END@@")

        assert result is None

    def test_deletion_hunk(self):
        diff = """\
@@DIFF_START@@
FILE: src/auth.py
<<<<<<< ORIGINAL (line 10)
    old_code()
    deprecated_call()
=======
>>>>>>> UPDATED
@@DIFF_END@@
"""
        parser = DiffParser()
        result = parser.parse(diff)

        assert result is not None
        assert result.parse_successful is True
        hunk = result.file_patches[0].hunks[0]
        assert hunk.is_deletion is True
        assert len(hunk.original_lines) == 2
        assert len(hunk.replacement_lines) == 0


class TestValidate:
    def test_valid_hunks_pass(self):
        parser = DiffParser()
        parsed = ParsedDiff(
            file_patches=[
                FilePatch(
                    file_path="test.py",
                    hunks=[
                        DiffHunk(
                            line_number=2,
                            original_lines=["    return 1"],
                            replacement_lines=["    return 2"],
                        )
                    ],
                )
            ]
        )

        file_contents = {
            "test.py": ["def foo():\n", "    return 1\n", "\n"],
        }

        result = parser.validate(parsed, file_contents)
        assert result is not None
        assert result.parse_successful is True
        assert len(result.file_patches) == 1

    def test_invalid_hunks_removed(self):
        parser = DiffParser()
        parsed = ParsedDiff(
            file_patches=[
                FilePatch(
                    file_path="test.py",
                    hunks=[
                        DiffHunk(
                            line_number=2,
                            original_lines=["    WRONG LINE"],
                            replacement_lines=["    return 2"],
                        ),
                        DiffHunk(
                            line_number=1,
                            original_lines=["def foo():"],
                            replacement_lines=["def bar():"],
                        ),
                    ],
                )
            ]
        )

        file_contents = {
            "test.py": ["def foo():\n", "    return 1\n", "\n"],
        }

        result = parser.validate(parsed, file_contents)
        assert result is not None
        # One hunk passes, one fails → less than 50% invalid → OK
        assert len(result.file_patches[0].hunks) == 1
        assert result.file_patches[0].hunks[0].line_number == 1

    def test_majority_invalid_returns_none(self):
        parser = DiffParser()
        parsed = ParsedDiff(
            file_patches=[
                FilePatch(
                    file_path="test.py",
                    hunks=[
                        DiffHunk(
                            line_number=1,
                            original_lines=["WRONG"],
                            replacement_lines=["x"],
                        ),
                        DiffHunk(
                            line_number=2,
                            original_lines=["ALSO WRONG"],
                            replacement_lines=["y"],
                        ),
                    ],
                )
            ]
        )

        file_contents = {
            "test.py": ["def foo():\n", "    return 1\n"],
        }

        result = parser.validate(parsed, file_contents)
        assert result is None

    def test_insertion_hunk_valid(self):
        parser = DiffParser()
        parsed = ParsedDiff(
            file_patches=[
                FilePatch(
                    file_path="test.py",
                    hunks=[
                        DiffHunk(
                            line_number=2,
                            original_lines=[],
                            replacement_lines=["    # new comment"],
                        )
                    ],
                )
            ]
        )

        file_contents = {
            "test.py": ["def foo():\n", "    pass\n"],
        }

        result = parser.validate(parsed, file_contents)
        assert result is not None
        assert result.parse_successful is True
