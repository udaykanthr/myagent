"""Tests for the chunk editor module."""

import pytest
from multi_agent_coder.editing.chunk_editor import (
    ChunkEditor, FileChunk, ChunkEditResponse,
)


SAMPLE_PYTHON = """\
import os
from datetime import datetime

GLOBAL_VAR = 42


class UserService:
    def __init__(self, db):
        self.db = db

    def authenticate(self, username, password):
        user = self.db.find(username)
        if user is None:
            return False
        return user.check_password(password)

    def get_user(self, user_id):
        return self.db.get(user_id)


def helper_function():
    return "hello"


def another_helper(x, y):
    return x + y
"""


SAMPLE_JS = """\
const express = require('express');
const { UserService } = require('./services');

class AppController {
    constructor(service) {
        this.service = service;
    }

    async handleLogin(req, res) {
        const { username, password } = req.body;
        const result = await this.service.authenticate(username, password);
        res.json({ success: result });
    }
}

function createApp() {
    const app = express();
    return app;
}

module.exports = { AppController, createApp };
"""


class TestChunkFile:
    def test_chunks_python_file(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)

        # Should have: imports, class UserService, methods, helper functions
        chunk_ids = [c.chunk_id for c in chunks]
        assert any("imports" in cid for cid in chunk_ids)
        assert any("UserService" in cid for cid in chunk_ids)
        assert any("helper_function" in cid for cid in chunk_ids)
        assert any("another_helper" in cid for cid in chunk_ids)

    def test_chunks_js_file(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("app.js", SAMPLE_JS)

        chunk_ids = [c.chunk_id for c in chunks]
        assert any("AppController" in cid for cid in chunk_ids)
        assert any("createApp" in cid for cid in chunk_ids)

    def test_chunks_preserve_all_lines(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)

        # All non-empty lines should be covered by some chunk
        all_lines = SAMPLE_PYTHON.splitlines()
        covered = set()
        for c in chunks:
            for ln in range(c.line_start, c.line_end + 1):
                covered.add(ln)

        for i, line in enumerate(all_lines, 1):
            if line.strip():
                assert i in covered, f"Line {i} not covered: {line!r}"

    def test_empty_file(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("empty.py", "")
        assert chunks == []

    def test_imports_only(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("imports.py", "import os\nimport sys\n")
        assert len(chunks) >= 1
        assert any(c.chunk_type == "imports" for c in chunks)


class TestIdentifyTargetChunks:
    def test_exact_name_match(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)
        targets = editor.identify_target_chunks(chunks, "fix the authenticate method")
        assert any("authenticate" in t for t in targets)

    def test_no_match(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)
        targets = editor.identify_target_chunks(chunks, "do something completely unrelated xyz")
        # May return empty or low-relevance matches
        # The key is it doesn't crash
        assert isinstance(targets, list)

    def test_multiple_matches(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)
        targets = editor.identify_target_chunks(chunks, "update helper functions")
        assert any("helper" in t for t in targets)


class TestFormatChunksForPrompt:
    def test_format_with_targets(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)
        targets = editor.identify_target_chunks(chunks, "fix authenticate")

        formatted = editor.format_chunks_for_prompt(chunks, targets)
        assert "EDITABLE" in formatted
        assert "CONTEXT ONLY" in formatted
        assert "test.py" in formatted

    def test_format_all_editable(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)
        formatted = editor.format_chunks_for_prompt(chunks, target_chunk_ids=None)
        # When no targets specified, all should be editable
        assert "CONTEXT ONLY" not in formatted or "EDITABLE" in formatted


class TestParseChunkResponse:
    def test_parse_edit_marker(self):
        editor = ChunkEditor()
        response = """Here are the changes:

#### [EDIT]: test.py:authenticate (lines 10-15)
```python
def authenticate(self, username, password):
    if not username or not password:
        return False
    user = self.db.find(username)
    return user is not None and user.check_password(password)
```
"""
        edits = editor.parse_chunk_response(response)
        assert edits is not None
        assert len(edits) == 1
        assert edits[0].file_path == "test.py"
        assert edits[0].chunk_id == "authenticate"
        assert edits[0].line_start == 10
        assert edits[0].line_end == 15
        assert "username" in edits[0].new_content

    def test_parse_new_marker(self):
        editor = ChunkEditor()
        response = """
#### [NEW]: test.py (after line 25)
```python
def validate_email(email):
    return "@" in email
```
"""
        edits = editor.parse_chunk_response(response)
        assert edits is not None
        assert len(edits) == 1
        assert edits[0].is_new is True
        assert edits[0].insert_after == 25

    def test_parse_multiple_edits(self):
        editor = ChunkEditor()
        response = """
#### [EDIT]: test.py:func_a (lines 10-15)
```python
def func_a():
    return 1
```

#### [EDIT]: test.py:func_b (lines 20-25)
```python
def func_b():
    return 2
```
"""
        edits = editor.parse_chunk_response(response)
        assert edits is not None
        assert len(edits) == 2

    def test_fallback_on_full_file_format(self):
        editor = ChunkEditor()
        response = """
#### [FILE]: test.py
```python
import os

def func():
    pass
```
"""
        edits = editor.parse_chunk_response(response)
        assert edits is None  # Should signal fallback

    def test_no_edits(self):
        editor = ChunkEditor()
        response = "No changes needed, the code looks correct."
        edits = editor.parse_chunk_response(response)
        assert edits is None


class TestApplyChunkEdits:
    def test_single_edit(self):
        editor = ChunkEditor()
        original = "line1\nline2\nline3\nline4\nline5\n"
        edits = [ChunkEditResponse(
            file_path="test.py",
            chunk_id="test",
            line_start=2,
            line_end=3,
            new_content="new_line2\nnew_line3\n",
        )]
        result = editor.apply_chunk_edits(original, edits)
        lines = result.splitlines()
        assert lines[0] == "line1"
        assert lines[1] == "new_line2"
        assert lines[2] == "new_line3"
        assert lines[3] == "line4"

    def test_new_insertion(self):
        editor = ChunkEditor()
        original = "line1\nline2\nline3\n"
        edits = [ChunkEditResponse(
            file_path="test.py",
            chunk_id="new",
            line_start=3,
            line_end=3,
            new_content="inserted\n",
            is_new=True,
            insert_after=2,
        )]
        result = editor.apply_chunk_edits(original, edits)
        lines = result.splitlines()
        assert "inserted" in lines
        assert lines.index("inserted") == 2  # after line2

    def test_multiple_edits_reverse_order(self):
        editor = ChunkEditor()
        original = "a\nb\nc\nd\ne\n"
        edits = [
            ChunkEditResponse("t.py", "c1", 2, 2, "B\n"),
            ChunkEditResponse("t.py", "c2", 4, 4, "D\n"),
        ]
        result = editor.apply_chunk_edits(original, edits)
        lines = result.splitlines()
        assert lines == ["a", "B", "c", "D", "e"]
