"""
Unit tests for multi_agent_coder.kb.local.vector_store

Tests collection naming, slugification, and Qdrant connectivity detection.
Qdrant client calls are mocked — no real Docker container needed.
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Tests: collection_name / slugify
# ---------------------------------------------------------------------------

class TestCollectionName:
    def test_basic(self):
        from multi_agent_coder.kb.local.vector_store import collection_name

        result = collection_name("/home/user/my-project")
        assert result == "local_my_project"

    def test_uppercase_slugified(self):
        from multi_agent_coder.kb.local.vector_store import collection_name

        result = collection_name("/workspace/MyApp")
        assert result == "local_myapp"

    def test_special_chars_slugified(self):
        from multi_agent_coder.kb.local.vector_store import collection_name

        result = collection_name("/home/user/agent chanti!")
        assert result.startswith("local_")
        # Should not contain spaces or exclamation marks
        assert " " not in result
        assert "!" not in result

    def test_trailing_slash_ignored(self):
        from multi_agent_coder.kb.local.vector_store import collection_name

        r1 = collection_name("/home/user/myproject")
        r2 = collection_name("/home/user/myproject/")
        assert r1 == r2

    def test_root_fallback(self):
        from multi_agent_coder.kb.local.vector_store import _slugify

        assert _slugify("!!!") == "project"

    def test_consistent_for_same_root(self):
        from multi_agent_coder.kb.local.vector_store import collection_name

        r1 = collection_name("/home/user/agentchanti")
        r2 = collection_name("/home/user/agentchanti")
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests: is_qdrant_running (mocked)
# ---------------------------------------------------------------------------

class TestIsQdrantRunning:
    def test_returns_false_when_connection_refused(self):
        from multi_agent_coder.kb.local.vector_store import is_qdrant_running

        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            assert is_qdrant_running() is False

    def test_returns_true_when_reachable(self):
        from multi_agent_coder.kb.local.vector_store import is_qdrant_running

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200

        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert is_qdrant_running() is True


# ---------------------------------------------------------------------------
# Tests: QdrantStore — mocked client
# ---------------------------------------------------------------------------

class TestQdrantStore:
    def _make_store(self, project_root="/tmp/test_project"):
        from multi_agent_coder.kb.local.vector_store import QdrantStore
        return QdrantStore(project_root)

    def test_collection_name_property(self):
        store = self._make_store("/home/user/agentchanti")
        assert store._collection == "local_agentchanti"

    def test_upsert_calls_client(self):
        from multi_agent_coder.kb.local.vector_store import QdrantStore

        store = QdrantStore("/tmp/proj")
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        store._client = mock_client

        with patch(
            "multi_agent_coder.kb.local.vector_store.is_qdrant_running",
            return_value=True,
        ):
            store.upsert([("uuid-1", [0.1] * 1536, {"file": "src/auth.py"})])

        mock_client.upsert.assert_called_once()

    def test_upsert_empty_is_noop(self):
        from multi_agent_coder.kb.local.vector_store import QdrantStore

        store = QdrantStore("/tmp/proj")
        mock_client = MagicMock()
        store._client = mock_client
        store.upsert([])
        mock_client.upsert.assert_not_called()

    def test_search_calls_client(self):
        from multi_agent_coder.kb.local.vector_store import QdrantStore

        store = QdrantStore("/tmp/proj")
        mock_hit = MagicMock()
        mock_hit.score = 0.95
        mock_hit.payload = {
            "file": "src/auth.py",
            "symbol_name": "login",
            "symbol_type": "function",
            "line_start": 10,
            "line_end": 25,
        }

        mock_result = MagicMock()
        mock_result.points = [mock_hit]

        mock_client = MagicMock()
        mock_client.query_points.return_value = mock_result
        store._client = mock_client

        with patch(
            "multi_agent_coder.kb.local.vector_store.is_qdrant_running",
            return_value=True,
        ):
            results = store.search([0.1] * 1536, top_k=5)

        assert len(results) == 1
        assert results[0]["score"] == 0.95
        assert results[0]["payload"]["symbol_name"] == "login"

    def test_get_client_raises_when_not_running(self):
        from multi_agent_coder.kb.local.vector_store import QdrantStore

        store = QdrantStore("/tmp/proj")
        with patch(
            "multi_agent_coder.kb.local.vector_store.is_qdrant_running",
            return_value=False,
        ):
            with pytest.raises(ConnectionError, match="Qdrant is not running"):
                store._get_client()

    def test_collection_info_returns_none_on_error(self):
        from multi_agent_coder.kb.local.vector_store import QdrantStore

        store = QdrantStore("/tmp/proj")
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("not found")
        store._client = mock_client

        result = store.collection_info()
        assert result is None
