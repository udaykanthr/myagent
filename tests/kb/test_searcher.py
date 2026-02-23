"""
Unit tests for multi_agent_coder.kb.local.searcher

Tests SearchResult dataclass, keyword fallback search, and the
semantic search path — Qdrant and OpenAI are mocked throughout.
"""

from __future__ import annotations

import os
import tempfile
import time
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph():
    """Build a synthetic CodeGraph with authentication-related symbols."""
    from multi_agent_coder.kb.local.graph import CodeGraph
    from multi_agent_coder.kb.local.parser import (
        ParsedFile, ParsedFunction, ParsedClass,
    )

    g = CodeGraph()
    pf = ParsedFile(
        path="src/auth.py",
        language="python",
        hash="aabbcc",
        functions=[
            ParsedFunction(
                name="login",
                file_path="src/auth.py",
                line_start=10,
                line_end=25,
                docstring="Authenticate a user.",
                params=["username", "password"],
                return_type="bool",
            ),
            ParsedFunction(
                name="logout",
                file_path="src/auth.py",
                line_start=27,
                line_end=35,
                docstring="Log out the current user.",
            ),
        ],
        classes=[
            ParsedClass(
                name="AuthService",
                file_path="src/auth.py",
                line_start=5,
                line_end=40,
                docstring="Handles authentication.",
            ),
        ],
    )
    g.add_parsed_file(pf)
    return g


def _make_manifest(tmp_path):
    from multi_agent_coder.kb.local.manifest import Manifest

    m = Manifest(str(tmp_path / "index.db"))
    m.upsert_file("src/auth.py", "aabbcc", "python", time.time(), [])
    return m


# ---------------------------------------------------------------------------
# Tests: SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_dataclass_fields(self):
        from multi_agent_coder.kb.local.searcher import SearchResult

        r = SearchResult(
            symbol_name="login",
            symbol_type="function",
            file="src/auth.py",
            line_start=10,
            line_end=25,
            code_snippet="def login(username, password):\n    ...",
            score=0.92,
            related_symbols=[{"name": "AuthService", "node_type": "CLASS"}],
        )
        assert r.symbol_name == "login"
        assert r.score == 0.92
        assert len(r.related_symbols) == 1

    def test_related_symbols_default_empty(self):
        from multi_agent_coder.kb.local.searcher import SearchResult

        r = SearchResult(
            symbol_name="x", symbol_type="function", file="a.py",
            line_start=1, line_end=2, code_snippet="", score=0.5,
        )
        assert r.related_symbols == []


# ---------------------------------------------------------------------------
# Tests: _read_snippet
# ---------------------------------------------------------------------------

class TestReadSnippet:
    def test_reads_lines(self, tmp_path):
        from multi_agent_coder.kb.local.searcher import _read_snippet

        src = tmp_path / "auth.py"
        src.write_text("line1\nline2\nline3\nline4\n")

        snippet = _read_snippet(str(tmp_path), "auth.py", 2, 3)
        assert "line2" in snippet
        assert "line3" in snippet
        assert "line1" not in snippet

    def test_missing_file_returns_empty(self, tmp_path):
        from multi_agent_coder.kb.local.searcher import _read_snippet

        result = _read_snippet(str(tmp_path), "nonexistent.py", 1, 5)
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: keyword fallback search
# ---------------------------------------------------------------------------

class TestGraphKeywordSearch:
    def test_finds_matching_symbols(self, tmp_path):
        from multi_agent_coder.kb.local.searcher import _graph_keyword_search

        g = _make_graph()
        m = _make_manifest(tmp_path)

        results = _graph_keyword_search(
            "login authentication", g, m, str(tmp_path), top_k=5, filters=None
        )
        names = [r.symbol_name for r in results]
        assert "login" in names

    def test_no_results_for_unrelated_query(self, tmp_path):
        from multi_agent_coder.kb.local.searcher import _graph_keyword_search

        g = _make_graph()
        m = _make_manifest(tmp_path)

        # Use a very specific query unlikely to match anything
        results = _graph_keyword_search(
            "zzznonexistentzzzquery", g, m, str(tmp_path), top_k=5, filters=None
        )
        assert results == []

    def test_returns_at_most_top_k(self, tmp_path):
        from multi_agent_coder.kb.local.searcher import _graph_keyword_search

        g = _make_graph()
        m = _make_manifest(tmp_path)

        results = _graph_keyword_search(
            "auth", g, m, str(tmp_path), top_k=1, filters=None
        )
        assert len(results) <= 1

    def test_sorted_by_score_descending(self, tmp_path):
        from multi_agent_coder.kb.local.searcher import _graph_keyword_search

        g = _make_graph()
        m = _make_manifest(tmp_path)

        results = _graph_keyword_search(
            "auth", g, m, str(tmp_path), top_k=10, filters=None
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Tests: Searcher with mocked Qdrant
# ---------------------------------------------------------------------------

class TestSearcher:
    def _make_searcher(self, tmp_path, qdrant_running=True):
        from multi_agent_coder.kb.local.searcher import Searcher
        from multi_agent_coder.kb.local.vector_store import QdrantStore

        g = _make_graph()
        m = _make_manifest(tmp_path)
        vs = MagicMock(spec=QdrantStore)
        vs.search.return_value = [
            {
                "score": 0.93,
                "payload": {
                    "symbol_name": "login",
                    "symbol_type": "function",
                    "file": "src/auth.py",
                    "line_start": 10,
                    "line_end": 25,
                    "language": "python",
                },
            }
        ]

        return Searcher(
            graph=g,
            manifest=m,
            vector_store=vs,
            project_root=str(tmp_path),
        ), vs

    def test_search_returns_results(self, tmp_path):
        from multi_agent_coder.kb.local.searcher import Searcher

        searcher, vs = self._make_searcher(tmp_path)

        with patch(
            "multi_agent_coder.kb.local.searcher.is_qdrant_running", return_value=True
        ), patch(
            "multi_agent_coder.kb.local.searcher._embed_query",
            return_value=[0.1] * 1536,
        ):
            results = searcher.search("find authentication functions", top_k=5)

        assert len(results) >= 1
        assert results[0].symbol_name == "login"
        assert results[0].score == pytest.approx(0.93)

    def test_search_falls_back_when_qdrant_down(self, tmp_path):
        from multi_agent_coder.kb.local.searcher import Searcher

        searcher, vs = self._make_searcher(tmp_path)

        with patch(
            "multi_agent_coder.kb.local.searcher.is_qdrant_running", return_value=False
        ):
            results = searcher.search("authentication login", top_k=10)

        # Fallback returns graph keyword results
        assert isinstance(results, list)
        # Qdrant search should not have been called
        vs.search.assert_not_called()

    def test_search_deduplicates_results(self, tmp_path):
        from multi_agent_coder.kb.local.searcher import Searcher
        from multi_agent_coder.kb.local.vector_store import QdrantStore

        g = _make_graph()
        m = _make_manifest(tmp_path)
        vs = MagicMock(spec=QdrantStore)

        # Duplicate result in Qdrant response
        hit = {
            "score": 0.9,
            "payload": {
                "symbol_name": "login",
                "symbol_type": "function",
                "file": "src/auth.py",
                "line_start": 10,
                "line_end": 25,
                "language": "python",
            },
        }
        vs.search.return_value = [hit, hit]  # same result twice

        searcher = Searcher(g, m, vs, str(tmp_path))

        with patch(
            "multi_agent_coder.kb.local.searcher.is_qdrant_running", return_value=True
        ), patch(
            "multi_agent_coder.kb.local.searcher._embed_query",
            return_value=[0.1] * 1536,
        ):
            results = searcher.search("login", top_k=10)

        # Should deduplicate to 1 result
        assert len(results) == 1

    def test_search_applies_file_filter(self, tmp_path):
        from multi_agent_coder.kb.local.searcher import Searcher
        from multi_agent_coder.kb.local.vector_store import QdrantStore

        g = _make_graph()
        m = _make_manifest(tmp_path)
        vs = MagicMock(spec=QdrantStore)

        # Result in wrong file
        vs.search.return_value = [
            {
                "score": 0.9,
                "payload": {
                    "symbol_name": "something",
                    "symbol_type": "function",
                    "file": "src/other.py",
                    "line_start": 1,
                    "line_end": 5,
                    "language": "python",
                },
            }
        ]

        searcher = Searcher(g, m, vs, str(tmp_path))

        with patch(
            "multi_agent_coder.kb.local.searcher.is_qdrant_running", return_value=True
        ), patch(
            "multi_agent_coder.kb.local.searcher._embed_query",
            return_value=[0.1] * 1536,
        ):
            results = searcher.search(
                "something", filters={"file": "src/auth"}, top_k=10
            )

        # src/other.py does not match the "src/auth" filter
        assert all("auth" in r.file for r in results)

    def test_search_related_symbols_populated(self, tmp_path):
        from multi_agent_coder.kb.local.searcher import Searcher
        from multi_agent_coder.kb.local.vector_store import QdrantStore

        g = _make_graph()
        m = _make_manifest(tmp_path)
        vs = MagicMock(spec=QdrantStore)
        vs.search.return_value = [
            {
                "score": 0.9,
                "payload": {
                    "symbol_name": "AuthService",
                    "symbol_type": "class",
                    "file": "src/auth.py",
                    "line_start": 5,
                    "line_end": 40,
                    "language": "python",
                },
            }
        ]

        searcher = Searcher(g, m, vs, str(tmp_path))

        with patch(
            "multi_agent_coder.kb.local.searcher.is_qdrant_running", return_value=True
        ), patch(
            "multi_agent_coder.kb.local.searcher._embed_query",
            return_value=[0.1] * 1536,
        ):
            results = searcher.search("AuthService", top_k=5)

        assert len(results) >= 1
        # related_symbols should be list of dicts (from graph.get_related_symbols)
        assert isinstance(results[0].related_symbols, list)
        for rel in results[0].related_symbols:
            assert isinstance(rel, dict)
            assert "node_type" in rel
            assert "name" in rel
