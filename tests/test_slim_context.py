"""Tests for slim context and file skeleton extraction."""

import pytest
from multi_agent_coder.orchestrator.memory import (
    FileMemory, _extract_file_skeleton, _estimate_tokens,
)


SAMPLE_PYTHON = """\
import os
from datetime import datetime
from typing import Optional

GLOBAL_VAR = 42


class UserService:
    def __init__(self, db):
        self.db = db

    def authenticate(self, username: str, password: str) -> bool:
        user = self.db.find(username)
        if user is None:
            return False
        return user.check_password(password)

    def get_user(self, user_id: int) -> Optional[dict]:
        return self.db.get(user_id)


def helper_function():
    return "hello"
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


class TestExtractFileSkeleton:
    def test_python_skeleton_has_imports(self):
        skeleton = _extract_file_skeleton(SAMPLE_PYTHON, "service.py")
        assert "import os" in skeleton
        assert "from datetime import datetime" in skeleton
        assert "[IMPORTS]" in skeleton

    def test_python_skeleton_has_symbols(self):
        skeleton = _extract_file_skeleton(SAMPLE_PYTHON, "service.py")
        assert "[SYMBOLS]" in skeleton
        assert "UserService" in skeleton
        assert "authenticate" in skeleton
        assert "helper_function" in skeleton

    def test_python_skeleton_has_line_numbers(self):
        skeleton = _extract_file_skeleton(SAMPLE_PYTHON, "service.py")
        assert "(line " in skeleton

    def test_python_skeleton_has_file_info(self):
        skeleton = _extract_file_skeleton(SAMPLE_PYTHON, "service.py")
        assert "[FILE_STRUCTURE]: service.py" in skeleton
        assert "lines)" in skeleton

    def test_js_skeleton(self):
        skeleton = _extract_file_skeleton(SAMPLE_JS, "app.js")
        assert "[IMPORTS]" in skeleton
        assert "require" in skeleton
        assert "AppController" in skeleton or "createApp" in skeleton

    def test_skeleton_much_shorter_than_original(self):
        skeleton = _extract_file_skeleton(SAMPLE_PYTHON, "service.py")
        # Skeleton should be significantly shorter
        assert len(skeleton) < len(SAMPLE_PYTHON)

    def test_empty_file(self):
        skeleton = _extract_file_skeleton("", "empty.py")
        assert "[FILE_STRUCTURE]: empty.py (0 lines)" in skeleton

    def test_no_imports(self):
        code = "def foo():\n    return 42\n"
        skeleton = _extract_file_skeleton(code, "simple.py")
        assert "[IMPORTS]" not in skeleton
        assert "foo" in skeleton


class TestRelatedContextSlim:
    def test_slim_returns_skeletons(self):
        mem = FileMemory()
        mem._files = {
            "src/service.py": SAMPLE_PYTHON,
            "src/app.js": SAMPLE_JS,
        }
        result = mem.related_context_slim("fix the service authentication")
        assert "[FILE_STRUCTURE]" in result
        assert "service.py" in result

    def test_slim_respects_token_budget(self):
        mem = FileMemory()
        # Add many files
        for i in range(20):
            mem._files[f"src/module_{i}.py"] = SAMPLE_PYTHON
        result = mem.related_context_slim("update module", max_tokens=100)
        # Should be truncated
        tokens = _estimate_tokens(result)
        assert tokens <= 200  # some tolerance

    def test_slim_empty_memory(self):
        mem = FileMemory()
        result = mem.related_context_slim("anything")
        assert result == ""

    def test_slim_vs_full_token_reduction(self):
        mem = FileMemory()
        mem._files = {
            "src/service.py": SAMPLE_PYTHON,
            "src/controller.py": SAMPLE_PYTHON * 3,  # larger file
        }
        full = mem._substring_context("service", None)
        slim = mem.related_context_slim("service")

        full_tokens = _estimate_tokens(full)
        slim_tokens = _estimate_tokens(slim)

        # Slim should use significantly fewer tokens
        assert slim_tokens < full_tokens
        # Slim should be meaningfully smaller
        if full_tokens > 0:
            reduction = (full_tokens - slim_tokens) / full_tokens
            assert reduction > 0.2  # at least 20% reduction
