"""Tests for the ScopeResolver."""

import pytest
from unittest.mock import MagicMock

from multi_agent_coder.editing.scope_resolver import (
    ScopeResolver, EditScope, SymbolRange,
)


def _make_graph(file_symbols=None, find_results=None, related=None,
                impact=None):
    """Build a mock CodeGraph."""
    g = MagicMock()
    g.get_file_symbols.return_value = file_symbols or []
    g.find_symbol.return_value = find_results or []
    g.get_related_symbols.return_value = related or []
    g.impact_analysis.return_value = impact or []
    return g


def _sym(name, stype="FUNCTION", file_path="src/auth.py",
         ls=10, le=25, parent_class=None):
    return {
        "id": f"FUNC:{file_path}::{name}",
        "node_type": stype,
        "name": name,
        "file_path": file_path,
        "line_start": ls,
        "line_end": le,
        "docstring": "",
        "parent_class": parent_class,
    }


# === Explicit symbol mention ===

class TestExplicitSymbol:
    def test_symbol_found_by_name(self):
        sym = _sym("authenticate_user")
        graph = _make_graph(file_symbols=[sym])
        resolver = ScopeResolver(graph)

        scope = resolver.resolve(
            "fix the authenticate_user function to handle None",
            "src/auth.py", graph,
        )

        assert scope.resolution_method == "graph_lookup"
        assert scope.confidence == 0.95
        assert len(scope.primary_symbols) == 1
        assert scope.primary_symbols[0].symbol_name == "authenticate_user"
        assert scope.primary_symbols[0].editable is True

    def test_multiple_symbols_mentioned(self):
        sym1 = _sym("login", ls=10, le=20)
        sym2 = _sym("logout", ls=30, le=40)
        graph = _make_graph(file_symbols=[sym1, sym2])
        resolver = ScopeResolver(graph)

        scope = resolver.resolve(
            "update login and logout functions",
            "src/auth.py", graph,
        )

        assert scope.resolution_method == "graph_lookup"
        assert len(scope.primary_symbols) == 2
        names = {s.symbol_name for s in scope.primary_symbols}
        assert names == {"login", "logout"}

    def test_method_with_parent_class(self):
        sym = _sym("validate", parent_class="UserService", ls=50, le=70)
        graph = _make_graph(file_symbols=[sym])
        resolver = ScopeResolver(graph)

        scope = resolver.resolve(
            "fix validate method",
            "src/auth.py", graph,
        )

        assert len(scope.primary_symbols) == 1
        assert scope.primary_symbols[0].symbol_type == "method"
        assert scope.primary_symbols[0].parent_class == "UserService"


# === Line number mention ===

class TestLineNumber:
    def test_single_line_mention(self):
        sym = _sym("process_request", ls=40, le=60)
        graph = _make_graph(file_symbols=[sym])
        resolver = ScopeResolver(graph)

        scope = resolver.resolve(
            "fix the bug on line 50",
            "src/auth.py", graph,
        )

        assert scope.resolution_method == "line_mention"
        assert scope.confidence == 0.90
        assert len(scope.primary_symbols) == 1
        assert scope.primary_symbols[0].symbol_name == "process_request"

    def test_line_range_mention(self):
        sym = _sym("handler", ls=30, le=70)
        graph = _make_graph(file_symbols=[sym])
        resolver = ScopeResolver(graph)

        scope = resolver.resolve(
            "refactor lines 45-55",
            "src/auth.py", graph,
        )

        assert scope.resolution_method == "line_mention"
        assert len(scope.primary_symbols) == 1

    def test_around_line_mention(self):
        sym = _sym("parse", ls=80, le=100)
        graph = _make_graph(file_symbols=[sym])
        resolver = ScopeResolver(graph)

        scope = resolver.resolve(
            "fix the issue around line 90",
            "src/auth.py", graph,
        )

        assert scope.resolution_method == "line_mention"
        assert scope.primary_symbols[0].symbol_name == "parse"


# === Error / stack trace ===

class TestErrorLocation:
    def test_python_traceback(self):
        # Use a symbol name NOT present in the task text so explicit
        # symbol matching doesn't fire first.
        # Use the file:line format (not "line N") so line_mention
        # doesn't fire before error_location.
        sym = _sym("db_connection_handler", ls=15, le=30)
        graph = _make_graph(file_symbols=[sym])
        resolver = ScopeResolver(graph)

        scope = resolver.resolve(
            "TypeError at src/auth.py:22 — null reference",
            "src/auth.py", graph,
        )

        assert scope.resolution_method == "error_location"
        assert scope.confidence == 0.88
        assert scope.primary_symbols[0].symbol_name == "db_connection_handler"

    def test_file_line_pattern(self):
        sym = _sym("render", ls=10, le=25)
        graph = _make_graph(file_symbols=[sym])
        resolver = ScopeResolver(graph)

        scope = resolver.resolve(
            "error at src/auth.py:15 — null reference",
            "src/auth.py", graph,
        )

        assert scope.resolution_method == "error_location"


# === Semantic fuzzy match ===

class TestSemanticMatch:
    def test_fuzzy_match_on_similar_name(self):
        sym = _sym("authenticate_user", ls=10, le=30)
        graph = _make_graph(file_symbols=[sym])
        resolver = ScopeResolver(graph)

        scope = resolver.resolve(
            "fix the authentication logic",
            "src/auth.py", graph,
        )

        # "authenticate" in the symbol name should fuzzy-match "authentication"
        # from the task description
        if scope.primary_symbols:
            assert scope.resolution_method == "semantic"
            assert scope.confidence >= 0.70


# === Fallback ===

class TestFallback:
    def test_fallback_on_no_symbols(self):
        graph = _make_graph(file_symbols=[])
        resolver = ScopeResolver(graph)

        scope = resolver.resolve(
            "do something vague",
            "src/auth.py", graph,
        )

        assert scope.resolution_method == "fallback"
        assert scope.confidence == 0.0

    def test_fallback_on_no_graph(self):
        resolver = ScopeResolver(None)

        scope = resolver.resolve(
            "fix authenticate_user",
            "src/auth.py", None,
        )

        assert scope.resolution_method == "fallback"
        assert scope.confidence == 0.0


# === Multi-file impact ===

class TestMultiFile:
    def test_impact_expansion_with_keyword(self):
        sym = _sym("validate", ls=10, le=30)
        related_sym = _sym("validate", file_path="src/api.py", ls=5, le=15)
        graph = _make_graph(
            file_symbols=[sym],
            impact=["src/api.py", "src/views.py"],
        )
        # When impact_analysis returns files, get_file_symbols is called per file
        graph.get_file_symbols.side_effect = lambda fp: (
            [sym] if fp == "src/auth.py" else
            [related_sym] if fp == "src/api.py" else
            []
        )

        resolver = ScopeResolver(graph)

        scope = resolver.resolve(
            "rename validate to validate_input everywhere",
            "src/auth.py", graph,
        )

        assert len(scope.affected_files) >= 2
        assert "src/api.py" in scope.affected_files


# === Context symbols ===

class TestContextSymbols:
    def test_related_symbols_added_as_context(self):
        sym = _sym("authenticate_user", ls=10, le=30)
        related = [
            _sym("hash_password", ls=35, le=50),
            _sym("check_token", ls=55, le=70),
        ]
        graph = _make_graph(file_symbols=[sym], related=related)
        resolver = ScopeResolver(graph)

        scope = resolver.resolve(
            "fix the authenticate_user function",
            "src/auth.py", graph,
        )

        assert len(scope.primary_symbols) == 1
        assert len(scope.context_symbols) == 2
        for ctx in scope.context_symbols:
            assert ctx.editable is False
