"""
Unit tests for multi_agent_coder.kb.context_builder

Tests the ContextBuilder class: intent detection, build_context(),
format_context_for_prompt(), and token budget management.
All KB dependencies (searcher, graph, global store) are mocked.
"""

from __future__ import annotations

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

from multi_agent_coder.kb.context_builder import (
    ContextBuilder, KBContext,
    _ERROR_KEYWORDS, _REVIEW_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder(tmp_path):
    """Return a ContextBuilder pointed at a tmp directory."""
    return ContextBuilder(project_root=str(tmp_path))


# ---------------------------------------------------------------------------
# Intent detection tests
# ---------------------------------------------------------------------------

class TestIntentDetection:

    def test_error_intent_positive(self):
        assert ContextBuilder._detect_error_intent("fix the login error")
        assert ContextBuilder._detect_error_intent("There is an exception in auth")
        assert ContextBuilder._detect_error_intent("Debug the crash")
        assert ContextBuilder._detect_error_intent("not working properly")

    def test_error_intent_negative(self):
        assert not ContextBuilder._detect_error_intent("add a new feature")
        assert not ContextBuilder._detect_error_intent("refactor auth module")

    def test_review_intent_positive(self):
        assert ContextBuilder._detect_review_intent("review the auth module")
        assert ContextBuilder._detect_review_intent("refactor the database layer")
        assert ContextBuilder._detect_review_intent("optimize the query")

    def test_review_intent_negative(self):
        assert not ContextBuilder._detect_review_intent("fix the login error")
        assert not ContextBuilder._detect_review_intent("create a new API endpoint")

    def test_language_detection(self):
        assert ContextBuilder._detect_language("src/auth.py") == "python"
        assert ContextBuilder._detect_language("app.js") == "javascript"
        assert ContextBuilder._detect_language("main.go") == "go"
        assert ContextBuilder._detect_language("Makefile") is None
        assert ContextBuilder._detect_language(None) is None


# ---------------------------------------------------------------------------
# Token estimation tests
# ---------------------------------------------------------------------------

class TestTokenEstimation:

    def test_estimate_tokens(self):
        assert ContextBuilder._estimate_tokens("") == 0
        assert ContextBuilder._estimate_tokens("a" * 100) == 25
        assert ContextBuilder._estimate_tokens("hello world") == 2


# ---------------------------------------------------------------------------
# KBContext dataclass tests
# ---------------------------------------------------------------------------

class TestKBContext:

    def test_defaults(self):
        ctx = KBContext()
        assert ctx.local_symbols == []
        assert ctx.related_symbols == []
        assert ctx.error_fixes == []
        assert ctx.global_patterns == []
        assert ctx.behavioral_instructions == []
        assert ctx.token_count == 0
        assert ctx.kb_available is False
        assert ctx.sources_used == []


# ---------------------------------------------------------------------------
# build_context tests
# ---------------------------------------------------------------------------

class TestBuildContext:

    def test_no_index_returns_unavailable(self, builder):
        """When no index exists, kb_available should be False."""
        ctx = builder.build_context("add login feature")
        assert ctx.kb_available is False
        assert ctx.local_symbols == []

    @patch("multi_agent_coder.kb.context_builder.ContextBuilder._ensure_local")
    @patch("multi_agent_coder.kb.context_builder.ContextBuilder._ensure_global")
    def test_build_context_with_mocked_local(self, mock_global, mock_local, builder):
        """Test build_context when local KB is available."""
        # Mock local to return True and set up searcher
        mock_local.return_value = True

        # Create a fake SearchResult
        fake_result = MagicMock()
        fake_result.symbol_name = "login"
        fake_result.symbol_type = "function"
        fake_result.file = "src/auth.py"
        fake_result.line_start = 10
        fake_result.line_end = 25
        fake_result.code_snippet = "def login(user, pwd): pass"
        fake_result.score = 0.95
        fake_result.related_symbols = []

        builder._searcher = MagicMock()
        builder._searcher.search.return_value = [fake_result]
        builder._graph = MagicMock()
        builder._graph.get_related_symbols.return_value = []

        ctx = builder.build_context("fix the login error")
        # local search should have been called
        builder._searcher.search.assert_called_once()
        assert len(ctx.local_symbols) == 1
        assert ctx.local_symbols[0].symbol_name == "login"

    @patch("multi_agent_coder.kb.context_builder.ContextBuilder._ensure_local")
    @patch("multi_agent_coder.kb.context_builder.ContextBuilder._ensure_global")
    def test_error_intent_triggers_error_lookup(self, mock_global, mock_local, builder):
        """Error-related tasks should trigger error_dict lookup."""
        mock_local.return_value = False

        fake_fix = MagicMock()
        fake_fix.error_type = "AttributeError"
        fake_fix.cause = "None attribute access"
        fake_fix.fix_template = "Check for None"
        fake_fix.tags = ""

        builder._global_store = MagicMock()
        builder._global_store.search_errors.return_value = [fake_fix]
        builder._global_store.get_behavioral_instructions.return_value = []

        ctx = builder.build_context("fix the AttributeError exception")
        builder._global_store.search_errors.assert_called_once()
        assert len(ctx.error_fixes) == 1
        assert ctx.error_fixes[0].error_type == "AttributeError"

    @patch("multi_agent_coder.kb.context_builder.ContextBuilder._ensure_local")
    @patch("multi_agent_coder.kb.context_builder.ContextBuilder._ensure_global")
    def test_review_intent_triggers_pattern_search(self, mock_global, mock_local, builder):
        """Review-related tasks should trigger global pattern search."""
        mock_local.return_value = False

        fake_pattern = MagicMock()
        fake_pattern.title = "SOLID Principles"
        fake_pattern.content = "Use dependency injection"
        fake_pattern.category = "pattern"

        builder._global_store = MagicMock()
        builder._global_store.search.return_value = [fake_pattern]
        builder._global_store.get_behavioral_instructions.return_value = []

        ctx = builder.build_context("review the auth module for patterns")
        # search should be called with pattern/adr categories
        assert len(ctx.global_patterns) == 1
        assert ctx.global_patterns[0].title == "SOLID Principles"

    @patch("multi_agent_coder.kb.context_builder.ContextBuilder._ensure_local")
    @patch("multi_agent_coder.kb.context_builder.ContextBuilder._ensure_global")
    def test_behavioral_always_included(self, mock_global, mock_local, builder):
        """Behavioral instructions should always be fetched."""
        mock_local.return_value = False

        fake_behavioral = MagicMock()
        fake_behavioral.title = "Always use type hints"
        fake_behavioral.content = "Add type hints to all functions"

        builder._global_store = MagicMock()
        builder._global_store.get_behavioral_instructions.return_value = [fake_behavioral]

        ctx = builder.build_context("add a new feature")
        builder._global_store.get_behavioral_instructions.assert_called_once()
        assert len(ctx.behavioral_instructions) == 1

    @patch("multi_agent_coder.kb.context_builder.ContextBuilder._ensure_local")
    @patch("multi_agent_coder.kb.context_builder.ContextBuilder._ensure_global")
    def test_exception_does_not_crash(self, mock_global, mock_local, builder):
        """KB exceptions should be caught, not crash the build."""
        mock_local.side_effect = Exception("boom")

        # Should not raise
        ctx = builder.build_context("any task")
        assert isinstance(ctx, KBContext)


# ---------------------------------------------------------------------------
# Token budget tests
# ---------------------------------------------------------------------------

class TestTokenBudget:

    def test_trim_low_priority_first(self, builder):
        """When over budget, related_symbols and extra local_symbols are trimmed first."""
        ctx = KBContext()

        # Create fake items with enough "tokens"
        for i in range(10):
            r = MagicMock()
            r.code_snippet = "x" * 400  # ~100 tokens each
            r.symbol_name = f"sym_{i}"
            ctx.local_symbols.append(r)

        for i in range(5):
            ctx.related_symbols.append({"name": f"rel_{i}", "x": "y" * 400})

        # Low budget
        result = builder._apply_token_budget(ctx, max_tokens=500)
        # local_symbols should be trimmed to 3
        assert len(result.local_symbols) <= 3
        # related_symbols should be empty
        assert len(result.related_symbols) == 0


# ---------------------------------------------------------------------------
# format_context_for_prompt tests
# ---------------------------------------------------------------------------

class TestFormatContext:

    def test_empty_context_returns_empty(self, builder):
        ctx = KBContext()
        result = builder.format_context_for_prompt(ctx)
        assert result == ""

    def test_format_with_local_symbols(self, builder):
        ctx = KBContext(kb_available=True)

        result_obj = MagicMock()
        result_obj.file = "src/auth.py"
        result_obj.line_start = 10
        result_obj.line_end = 25
        result_obj.code_snippet = "def login(): pass"
        result_obj.related_symbols = [{"name": "validate"}]
        ctx.local_symbols = [result_obj]

        output = builder.format_context_for_prompt(ctx)
        assert "KNOWLEDGE BASE CONTEXT" in output
        assert "RELEVANT CODE FROM THIS PROJECT" in output
        assert "src/auth.py" in output
        assert "def login(): pass" in output

    def test_format_with_error_fixes(self, builder):
        ctx = KBContext(kb_available=True)

        fix = MagicMock()
        fix.error_type = "AttributeError"
        fix.cause = "None access"
        fix.fix_template = "Check for None first"
        ctx.error_fixes = [fix]

        output = builder.format_context_for_prompt(ctx)
        assert "ERROR FIX PATTERNS" in output
        assert "AttributeError" in output
        assert "Check for None first" in output

    def test_format_with_behavioral(self, builder):
        ctx = KBContext()  # kb_available=False but has behavioral

        bi = MagicMock()
        bi.content = "Always validate inputs"
        bi.title = "Input Validation"
        ctx.behavioral_instructions = [bi]

        output = builder.format_context_for_prompt(ctx)
        assert "BEHAVIORAL INSTRUCTIONS" in output
        assert "Always validate inputs" in output

    def test_format_with_patterns(self, builder):
        ctx = KBContext(kb_available=True)

        pattern = MagicMock()
        pattern.title = "Repository Pattern"
        pattern.content = "Use repository pattern for data access"
        ctx.global_patterns = [pattern]

        output = builder.format_context_for_prompt(ctx)
        assert "CODING PATTERNS" in output
        assert "Repository Pattern" in output
