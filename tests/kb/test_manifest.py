"""
Unit tests for multi_agent_coder.kb.local.manifest
"""

from __future__ import annotations

import time
import pytest


class TestManifest:
    def setup_method(self, tmp_path=None):
        pass

    def test_upsert_and_get_file(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest, SymbolRecord
        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file(
            path="src/foo.py",
            hash_="abc123",
            language="python",
            last_modified=time.time(),
            symbols=[SymbolRecord("my_func", "function", 1, 10)],
        )
        rec = m.get_file("src/foo.py")
        assert rec is not None
        assert rec.hash == "abc123"
        assert rec.language == "python"

    def test_is_file_changed_new_file(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest
        m = Manifest(str(tmp_path / "index.db"))
        assert m.is_file_changed("src/new.py", "hash1") is True

    def test_is_file_changed_same_hash(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest, SymbolRecord
        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("src/foo.py", "hash1", "python", 0.0, [])
        assert m.is_file_changed("src/foo.py", "hash1") is False

    def test_is_file_changed_different_hash(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest, SymbolRecord
        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("src/foo.py", "hash1", "python", 0.0, [])
        assert m.is_file_changed("src/foo.py", "hash2") is True

    def test_remove_file(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest
        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("src/foo.py", "h1", "python", 0.0, [])
        m.remove_file("src/foo.py")
        assert m.get_file("src/foo.py") is None

    def test_remove_nonexistent_file_is_safe(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest
        m = Manifest(str(tmp_path / "index.db"))
        m.remove_file("nonexistent.py")  # should not raise

    def test_get_all_indexed_paths(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest
        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("a.py", "h1", "python", 0.0, [])
        m.upsert_file("b.py", "h2", "python", 0.0, [])
        paths = m.get_all_indexed_paths()
        assert set(paths) == {"a.py", "b.py"}

    def test_get_symbols_for_file(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest, SymbolRecord
        m = Manifest(str(tmp_path / "index.db"))
        symbols = [
            SymbolRecord("Foo", "class", 1, 20),
            SymbolRecord("bar", "function", 5, 15),
        ]
        m.upsert_file("src/foo.py", "h1", "python", 0.0, symbols)
        stored = m.get_symbols_for_file("src/foo.py")
        assert {s.name for s in stored} == {"Foo", "bar"}

    def test_stats(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest, SymbolRecord
        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("a.py", "h1", "python", 0.0, [SymbolRecord("f", "function", 1, 5)])
        m.upsert_file("b.js", "h2", "javascript", 0.0, [])
        stats = m.stats()
        assert stats["file_count"] == 2
        assert stats["symbol_count"] == 1
        assert "python" in stats["languages"]

    def test_clear(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest
        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("a.py", "h1", "python", 0.0, [])
        m.clear()
        assert m.get_all_indexed_paths() == []

    def test_find_symbol(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest, SymbolRecord
        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("a.py", "h1", "python", 0.0, [
            SymbolRecord("MyClass", "class", 1, 30),
            SymbolRecord("my_func", "function", 5, 10),
        ])
        results = m.find_symbol("MyClass")
        assert len(results) == 1
        assert results[0]["symbol_type"] == "class"

    def test_find_symbol_with_type_filter(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest, SymbolRecord
        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("a.py", "h1", "python", 0.0, [
            SymbolRecord("run", "function", 1, 5),
        ])
        m.upsert_file("b.py", "h2", "python", 0.0, [
            SymbolRecord("run", "class", 1, 20),
        ])
        results = m.find_symbol("run", symbol_type="function")
        assert all(r["symbol_type"] == "function" for r in results)

    def test_upsert_replaces_symbols(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest, SymbolRecord
        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("a.py", "h1", "python", 0.0, [
            SymbolRecord("old_func", "function", 1, 5),
        ])
        m.upsert_file("a.py", "h2", "python", 0.0, [
            SymbolRecord("new_func", "function", 1, 5),
        ])
        symbols = m.get_symbols_for_file("a.py")
        names = {s.name for s in symbols}
        assert "new_func" in names
        assert "old_func" not in names
