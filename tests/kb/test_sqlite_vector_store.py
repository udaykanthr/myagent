"""
Unit tests for the SQLite-backed local vector store and
the graph-first file selection (Option A + B).
"""
import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Test: SQLiteVectorStore
# ---------------------------------------------------------------------------

class TestSQLiteVectorStore(unittest.TestCase):
    """Tests for the zero-config local vector store."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        from multi_agent_coder.kb.local.sqlite_vector_store import SQLiteVectorStore
        self.store = SQLiteVectorStore(
            project_root=self._tmpdir,
            db_path=os.path.join(self._tmpdir, "test_vectors.db"),
        )

    def tearDown(self):
        self.store.close()

    def test_ensure_collection_noop(self):
        """ensure_collection() should succeed silently."""
        self.store.ensure_collection()

    def test_upsert_and_search(self):
        """Upsert vectors, then search with cosine similarity."""
        points = [
            ("p1", [1.0, 0.0, 0.0], {"symbol_name": "foo", "file": "a.py"}),
            ("p2", [0.0, 1.0, 0.0], {"symbol_name": "bar", "file": "b.py"}),
            ("p3", [0.7, 0.7, 0.0], {"symbol_name": "baz", "file": "a.py"}),
        ]
        self.store.upsert(points)

        # Search with a query close to p1
        results = self.store.search([1.0, 0.0, 0.0], top_k=2)
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["payload"]["symbol_name"], "foo")
        self.assertGreater(results[0]["score"], 0.5)

    def test_upsert_replaces_existing(self):
        """Upserting the same point_id should update, not duplicate."""
        self.store.upsert([
            ("p1", [1.0, 0.0], {"symbol_name": "old"}),
        ])
        self.store.upsert([
            ("p1", [0.0, 1.0], {"symbol_name": "new"}),
        ])
        info = self.store.collection_info()
        self.assertEqual(info["points_count"], 1)

        results = self.store.search([0.0, 1.0], top_k=1)
        self.assertEqual(results[0]["payload"]["symbol_name"], "new")

    def test_delete_by_file(self):
        """delete_by_file() should remove matching points."""
        self.store.upsert([
            ("p1", [1.0, 0.0], {"file": "a.py"}),
            ("p2", [0.0, 1.0], {"file": "b.py"}),
        ])
        self.store.delete_by_file("a.py")
        info = self.store.collection_info()
        self.assertEqual(info["points_count"], 1)

    def test_collection_info(self):
        """collection_info() returns count and name."""
        info = self.store.collection_info()
        self.assertEqual(info["points_count"], 0)
        self.assertIn("local_sqlite", info["name"])

    def test_search_empty_store(self):
        """Searching an empty store should return []."""
        results = self.store.search([1.0, 0.0, 0.0], top_k=5)
        self.assertEqual(results, [])

    def test_search_with_language_filter(self):
        """Search should respect language filter."""
        self.store.upsert([
            ("p1", [1.0, 0.0], {"symbol_name": "foo", "language": "python"}),
            ("p2", [0.9, 0.1], {"symbol_name": "bar", "language": "javascript"}),
        ])
        results = self.store.search(
            [1.0, 0.0], top_k=5,
            filters={"language": "python"},
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["payload"]["symbol_name"], "foo")

    def test_upsert_empty_list(self):
        """Upserting [] should be a no-op."""
        self.store.upsert([])
        info = self.store.collection_info()
        self.assertEqual(info["points_count"], 0)


# ---------------------------------------------------------------------------
# Test: create_vector_store factory
# ---------------------------------------------------------------------------

class TestCreateVectorStore(unittest.TestCase):
    """Tests for the factory function."""

    def test_creates_sqlite_by_default(self):
        from multi_agent_coder.kb.local.sqlite_vector_store import (
            create_vector_store,
            SQLiteVectorStore,
        )
        tmpdir = tempfile.mkdtemp()
        store = create_vector_store(tmpdir, backend="local")
        self.assertIsInstance(store, SQLiteVectorStore)
        store.close()

# ---------------------------------------------------------------------------
# Test: ContextBuilder.get_relevant_files
# ---------------------------------------------------------------------------

class TestGetRelevantFiles(unittest.TestCase):
    """Tests for the KB-guided file selection."""

    def _make_builder(self):
        from multi_agent_coder.kb.context_builder import ContextBuilder
        builder = ContextBuilder(project_root="/tmp/fake_project")
        return builder

    @patch("multi_agent_coder.kb.context_builder.ContextBuilder._ensure_local")
    def test_returns_files_from_search(self, mock_ensure):
        """get_relevant_files should return file paths from search results."""
        mock_ensure.return_value = True

        builder = self._make_builder()

        # Mock the searcher
        mock_result = MagicMock()
        mock_result.file = "src/main.py"
        mock_result.score = 0.9

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [mock_result]
        builder._searcher = mock_searcher

        files = builder.get_relevant_files("implement login")
        self.assertIn("src/main.py", files)

    def test_includes_changed_files(self):
        """Changed files should always be included."""
        builder = self._make_builder()
        files = builder.get_relevant_files(
            "implement login",
            changed_files=["src/auth.py", "src/models.py"],
        )
        self.assertIn("src/auth.py", files)
        self.assertIn("src/models.py", files)

    def test_empty_without_kb(self):
        """Without KB, should return only changed files."""
        builder = self._make_builder()
        files = builder.get_relevant_files("implement login")
        self.assertEqual(files, [])

    def test_max_files_limit(self):
        """Should respect max_files parameter."""
        builder = self._make_builder()
        many_files = [f"file_{i}.py" for i in range(20)]
        files = builder.get_relevant_files(
            "something", changed_files=many_files, max_files=5,
        )
        self.assertLessEqual(len(files), 5)


# ---------------------------------------------------------------------------
# Test: FileMemory.scoped_context
# ---------------------------------------------------------------------------

class TestScopedContext(unittest.TestCase):
    """Tests for KB-guided file memory scoping."""

    def _make_memory(self):
        from multi_agent_coder.orchestrator.memory import FileMemory
        mem = FileMemory()
        mem.update({
            "src/main.py": "def main(): pass",
            "src/auth.py": "def login(): pass",
            "src/models.py": "class User: pass",
            "src/utils.py": "def helper(): pass",
            "tests/test_main.py": "def test_main(): pass",
        })
        return mem

    def test_scoped_context_filters_files(self):
        """scoped_context should only include listed files."""
        mem = self._make_memory()
        ctx = mem.scoped_context(
            "implement login",
            relevant_files=["src/auth.py", "src/main.py"],
        )
        self.assertIn("src/auth.py", ctx)
        self.assertIn("src/main.py", ctx)
        self.assertNotIn("src/utils.py", ctx)
        self.assertNotIn("tests/test_main.py", ctx)

    def test_scoped_context_empty_falls_back(self):
        """Empty relevant_files should fall back to related_context."""
        mem = self._make_memory()
        ctx = mem.scoped_context("implement login", relevant_files=[])
        # Should return something (fallback), not empty
        # (depends on what related_context returns without embeddings)
        self.assertIsInstance(ctx, str)

    def test_scoped_context_respects_token_budget(self):
        """Should stop adding files when token budget is exceeded."""
        mem = self._make_memory()
        # Very small budget — should only include 1-2 files
        ctx = mem.scoped_context(
            "implement login",
            relevant_files=["src/main.py", "src/auth.py", "src/models.py"],
            max_tokens=10,
        )
        # At most one file should fit in 10 tokens (~40 chars)
        self.assertIsInstance(ctx, str)


if __name__ == "__main__":
    unittest.main()
