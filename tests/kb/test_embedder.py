"""
Unit tests for multi_agent_coder.kb.local.embedder

Tests chunking logic, UUID generation, text formatting, and
incremental skip logic — without calling the OpenAI API.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph_with_symbols():
    """Build a small synthetic CodeGraph with one function and one class."""
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
                name="get_token",
                file_path="src/auth.py",
                line_start=27,
                line_end=35,
                parent_class="AuthService",
            ),
        ],
        classes=[
            ParsedClass(
                name="AuthService",
                file_path="src/auth.py",
                line_start=5,
                line_end=40,
                docstring="Handles authentication.",
                bases=["BaseService"],
            ),
        ],
    )
    g.add_parsed_file(pf)
    return g


# ---------------------------------------------------------------------------
# Tests: make_point_id
# ---------------------------------------------------------------------------

class TestMakePointId:
    def test_deterministic(self):
        from multi_agent_coder.kb.local.embedder import make_point_id

        id1 = make_point_id("src/auth.py", "login", 10)
        id2 = make_point_id("src/auth.py", "login", 10)
        assert id1 == id2

    def test_different_inputs_produce_different_ids(self):
        from multi_agent_coder.kb.local.embedder import make_point_id

        id1 = make_point_id("src/auth.py", "login", 10)
        id2 = make_point_id("src/auth.py", "login", 11)
        id3 = make_point_id("src/other.py", "login", 10)
        assert id1 != id2
        assert id1 != id3

    def test_valid_uuid_format(self):
        from multi_agent_coder.kb.local.embedder import make_point_id

        result = make_point_id("src/auth.py", "login", 10)
        # Should be parseable as UUID
        parsed = uuid.UUID(result)
        assert str(parsed) == result


# ---------------------------------------------------------------------------
# Tests: text formatters
# ---------------------------------------------------------------------------

class TestTextFormatters:
    def test_function_text_includes_fields(self):
        from multi_agent_coder.kb.local.embedder import _function_text

        text = _function_text(
            language="python",
            file_path="src/auth.py",
            name="login",
            params=["username", "password"],
            return_type="bool",
            docstring="Authenticate a user.",
            body_lines=["    return True"],
        )
        assert "Language: python" in text
        assert "Function: login" in text
        assert "Parameters: username, password" in text
        assert "Returns: bool" in text
        assert "Docstring: Authenticate a user." in text
        assert "return True" in text

    def test_function_text_empty_params(self):
        from multi_agent_coder.kb.local.embedder import _function_text

        text = _function_text("go", "main.go", "main", [], "", "", [])
        assert "Parameters: none" in text
        assert "Returns: none" in text

    def test_class_text_includes_fields(self):
        from multi_agent_coder.kb.local.embedder import _class_text

        text = _class_text(
            language="python",
            file_path="src/auth.py",
            name="AuthService",
            bases=["BaseService"],
            docstring="Handles authentication.",
            method_names=["login", "logout"],
        )
        assert "Language: python" in text
        assert "Class: AuthService" in text
        assert "Inherits: BaseService" in text
        assert "Docstring: Handles authentication." in text
        assert "login, logout" in text

    def test_class_text_no_bases(self):
        from multi_agent_coder.kb.local.embedder import _class_text

        text = _class_text("python", "a.py", "Foo", [], "", [])
        assert "Inherits: none" in text
        assert "Methods: none" in text


# ---------------------------------------------------------------------------
# Tests: extract_symbol_chunks
# ---------------------------------------------------------------------------

class TestExtractSymbolChunks:
    def test_returns_chunks_for_function_and_class(self, tmp_path):
        from multi_agent_coder.kb.local.embedder import extract_symbol_chunks

        g = _make_graph_with_symbols()
        chunks = extract_symbol_chunks(g, str(tmp_path))

        names = {c.symbol_name for c in chunks}
        # Should include: login, get_token (function/method), AuthService (class)
        assert "login" in names
        assert "get_token" in names
        assert "AuthService" in names

    def test_function_chunk_type(self, tmp_path):
        from multi_agent_coder.kb.local.embedder import extract_symbol_chunks

        g = _make_graph_with_symbols()
        chunks = extract_symbol_chunks(g, str(tmp_path))

        login_chunks = [c for c in chunks if c.symbol_name == "login"]
        assert len(login_chunks) == 1
        assert login_chunks[0].symbol_type == "function"
        assert login_chunks[0].file_path == "src/auth.py"

    def test_method_chunk_type(self, tmp_path):
        from multi_agent_coder.kb.local.embedder import extract_symbol_chunks

        g = _make_graph_with_symbols()
        chunks = extract_symbol_chunks(g, str(tmp_path))

        method_chunks = [c for c in chunks if c.symbol_name == "get_token"]
        assert len(method_chunks) == 1
        assert method_chunks[0].symbol_type == "method"
        assert method_chunks[0].parent_class == "AuthService"

    def test_class_chunk_type(self, tmp_path):
        from multi_agent_coder.kb.local.embedder import extract_symbol_chunks

        g = _make_graph_with_symbols()
        chunks = extract_symbol_chunks(g, str(tmp_path))

        class_chunks = [c for c in chunks if c.symbol_name == "AuthService"]
        assert len(class_chunks) == 1
        assert class_chunks[0].symbol_type == "class"

    def test_class_text_contains_method_names(self, tmp_path):
        from multi_agent_coder.kb.local.embedder import extract_symbol_chunks

        g = _make_graph_with_symbols()
        chunks = extract_symbol_chunks(g, str(tmp_path))

        class_chunks = [c for c in chunks if c.symbol_name == "AuthService"]
        assert len(class_chunks) == 1
        # Method name should appear in class text
        assert "get_token" in class_chunks[0].text

    def test_point_ids_are_unique(self, tmp_path):
        from multi_agent_coder.kb.local.embedder import extract_symbol_chunks

        g = _make_graph_with_symbols()
        chunks = extract_symbol_chunks(g, str(tmp_path))

        ids = [c.point_id for c in chunks]
        assert len(ids) == len(set(ids)), "Point IDs must be unique"

    def test_no_variable_chunks(self, tmp_path):
        from multi_agent_coder.kb.local.embedder import extract_symbol_chunks
        from multi_agent_coder.kb.local.graph import CodeGraph
        from multi_agent_coder.kb.local.parser import ParsedFile, ParsedVariable

        g = CodeGraph()
        pf = ParsedFile(
            path="src/config.py",
            language="python",
            hash="xyz",
            variables=[ParsedVariable(name="DEBUG", file_path="src/config.py", scope="module")],
        )
        g.add_parsed_file(pf)

        chunks = extract_symbol_chunks(g, str(tmp_path))
        assert all(c.symbol_type != "variable" for c in chunks)


# ---------------------------------------------------------------------------
# Tests: _read_lines
# ---------------------------------------------------------------------------

class TestReadLines:
    def test_reads_specified_range(self, tmp_path):
        from multi_agent_coder.kb.local.embedder import _read_lines

        src = tmp_path / "test.py"
        src.write_text("line1\nline2\nline3\nline4\nline5\n")

        lines = _read_lines(str(src), 2, 4)
        assert lines == ["line2", "line3", "line4"]

    def test_missing_file_returns_empty(self, tmp_path):
        from multi_agent_coder.kb.local.embedder import _read_lines

        lines = _read_lines(str(tmp_path / "nonexistent.py"), 1, 5)
        assert lines == []


# ---------------------------------------------------------------------------
# Tests: manifest embedded hash helpers
# ---------------------------------------------------------------------------

class TestManifestEmbeddedHash:
    def test_get_embedded_hash_initially_none(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest, SymbolRecord

        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("src/auth.py", "hash1", "python", time.time(), [])
        assert m.get_embedded_hash("src/auth.py") is None

    def test_set_and_get_embedded_hash(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest

        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("src/auth.py", "hash1", "python", time.time(), [])
        m.set_embedded_hash("src/auth.py", "hash1")
        assert m.get_embedded_hash("src/auth.py") == "hash1"

    def test_get_files_needing_embed_new_file(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest

        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("src/auth.py", "hash1", "python", time.time(), [])

        needing = m.get_files_needing_embed()
        assert ("src/auth.py", "hash1") in needing

    def test_get_files_needing_embed_up_to_date(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest

        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("src/auth.py", "hash1", "python", time.time(), [])
        m.set_embedded_hash("src/auth.py", "hash1")

        needing = m.get_files_needing_embed()
        assert all(p != "src/auth.py" for p, _ in needing)

    def test_get_files_needing_embed_changed_hash(self, tmp_path):
        from multi_agent_coder.kb.local.manifest import Manifest

        m = Manifest(str(tmp_path / "index.db"))
        m.upsert_file("src/auth.py", "hash1", "python", time.time(), [])
        m.set_embedded_hash("src/auth.py", "hash1")
        # Simulate file change
        m.upsert_file("src/auth.py", "hash2", "python", time.time(), [])

        needing = m.get_files_needing_embed()
        assert ("src/auth.py", "hash2") in needing


# ---------------------------------------------------------------------------
# Tests: migration (existing DB without last_embedded_hash column)
# ---------------------------------------------------------------------------

class TestManifestMigration:
    def test_migration_adds_column_to_existing_db(self, tmp_path):
        """Simulate a Phase 1 DB without last_embedded_hash, then upgrade."""
        import sqlite3

        db_path = str(tmp_path / "index.db")
        # Create an old-style DB without the new column
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                hash TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT '',
                last_modified REAL NOT NULL DEFAULT 0.0,
                indexed_at REAL NOT NULL DEFAULT 0.0
            )
        """)
        conn.execute("""
            CREATE TABLE symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                line_start INTEGER NOT NULL DEFAULT 0,
                line_end INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("INSERT INTO files (path,hash,language,last_modified,indexed_at) VALUES (?,?,?,?,?)",
                     ("src/old.py", "oldhash", "python", 0.0, 0.0))
        conn.commit()
        conn.close()

        # Now open via Manifest (should apply migration)
        from multi_agent_coder.kb.local.manifest import Manifest
        m = Manifest(db_path)

        # Should be able to access the new column without error
        result = m.get_embedded_hash("src/old.py")
        assert result is None
