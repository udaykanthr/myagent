"""
SQLite-backed file/symbol manifest for the Local Knowledge Base.

Tracks which files have been indexed, their hashes (for change detection),
and which symbols each file defines.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    path                TEXT    UNIQUE NOT NULL,
    hash                TEXT    NOT NULL,
    language            TEXT    NOT NULL DEFAULT '',
    last_modified       REAL    NOT NULL DEFAULT 0.0,
    indexed_at          REAL    NOT NULL DEFAULT 0.0,
    last_embedded_hash  TEXT    DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS symbols (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name        TEXT    NOT NULL,
    symbol_type TEXT    NOT NULL,
    line_start  INTEGER NOT NULL DEFAULT 0,
    line_end    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_files_path   ON files(path);
"""

# Applied to existing databases missing newer columns (Phase 2 migration).
_MIGRATIONS = [
    "ALTER TABLE files ADD COLUMN last_embedded_hash TEXT DEFAULT NULL",
]


@dataclass
class FileRecord:
    """Stored metadata for a single indexed file."""
    path: str
    hash: str
    language: str
    last_modified: float
    indexed_at: float


@dataclass
class SymbolRecord:
    """A symbol entry associated with a file."""
    name: str
    symbol_type: str   # "function" | "class" | "variable" | "module"
    line_start: int
    line_end: int


class Manifest:
    """
    SQLite-backed index tracking which files have been parsed and what
    symbols they define.  Supports change detection via file hashes.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Will be created if absent.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self):
        """Yield a connected SQLite connection with WAL mode for concurrency."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables and run any pending schema migrations."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Apply migrations idempotently (ignore "duplicate column" errors).
            for stmt in _MIGRATIONS:
                try:
                    conn.execute(stmt)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_file(self, path: str) -> Optional[FileRecord]:
        """
        Return the stored record for *path*, or None if not indexed.

        Parameters
        ----------
        path:
            Relative or absolute file path (stored as provided).
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT path, hash, language, last_modified, indexed_at "
                "FROM files WHERE path = ?",
                (path,),
            ).fetchone()
        if row is None:
            return None
        return FileRecord(
            path=row["path"],
            hash=row["hash"],
            language=row["language"],
            last_modified=row["last_modified"],
            indexed_at=row["indexed_at"],
        )

    def is_file_changed(self, path: str, current_hash: str) -> bool:
        """
        Return True if *path* is new or its stored hash differs from
        *current_hash*.

        Parameters
        ----------
        path:
            File path to check.
        current_hash:
            The hash computed from the file's current contents.
        """
        record = self.get_file(path)
        if record is None:
            return True
        return record.hash != current_hash

    def upsert_file(
        self,
        path: str,
        hash_: str,
        language: str,
        last_modified: float,
        symbols: list[SymbolRecord],
    ) -> None:
        """
        Insert or update the manifest entry for *path* and its symbols.

        Replaces all existing symbols for the file with the new list.

        Parameters
        ----------
        path:
            File path (used as unique key).
        hash_:
            SHA-256 hash of file contents.
        language:
            Detected language string.
        last_modified:
            File's mtime (seconds since epoch).
        symbols:
            List of symbol records defined in this file.
        """
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (path, hash, language, last_modified, indexed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    hash          = excluded.hash,
                    language      = excluded.language,
                    last_modified = excluded.last_modified,
                    indexed_at    = excluded.indexed_at
                """,
                (path, hash_, language, last_modified, now),
            )
            file_id = conn.execute(
                "SELECT id FROM files WHERE path = ?", (path,)
            ).fetchone()["id"]
            # Replace symbols for this file
            conn.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))
            conn.executemany(
                "INSERT INTO symbols (file_id, name, symbol_type, line_start, line_end) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (file_id, s.name, s.symbol_type, s.line_start, s.line_end)
                    for s in symbols
                ],
            )

    def remove_file(self, path: str) -> None:
        """
        Remove all manifest data for *path* (file + its symbols).

        Safe to call even if *path* is not in the manifest.

        Parameters
        ----------
        path:
            File path to remove.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM files WHERE path = ?", (path,))

    def get_all_indexed_paths(self) -> list[str]:
        """Return the paths of every file currently in the manifest."""
        with self._connect() as conn:
            rows = conn.execute("SELECT path FROM files").fetchall()
        return [r["path"] for r in rows]

    def get_symbols_for_file(self, path: str) -> list[SymbolRecord]:
        """
        Return all symbols recorded for *path*.

        Parameters
        ----------
        path:
            File path to look up.
        """
        with self._connect() as conn:
            file_row = conn.execute(
                "SELECT id FROM files WHERE path = ?", (path,)
            ).fetchone()
            if file_row is None:
                return []
            rows = conn.execute(
                "SELECT name, symbol_type, line_start, line_end "
                "FROM symbols WHERE file_id = ?",
                (file_row["id"],),
            ).fetchall()
        return [
            SymbolRecord(
                name=r["name"],
                symbol_type=r["symbol_type"],
                line_start=r["line_start"],
                line_end=r["line_end"],
            )
            for r in rows
        ]

    def get_symbol_occurrences(self) -> list[tuple[str, str, str]]:
        """
        Return all symbols with their type and the file path they occur in.
        
        Returns
        -------
        list[tuple[str, str, str]]
            List of (symbol_name, symbol_type, file_path)
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.name, s.symbol_type, f.path "
                "FROM symbols s JOIN files f ON s.file_id = f.id"
            ).fetchall()
        return [(r["name"], r["symbol_type"], r["path"]) for r in rows]

    def get_embedded_hash(self, path: str) -> Optional[str]:
        """
        Return the last embedded hash for *path*, or None if not set.

        Parameters
        ----------
        path:
            File path to look up.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_embedded_hash FROM files WHERE path = ?", (path,)
            ).fetchone()
        if row is None:
            return None
        return row["last_embedded_hash"]

    def set_embedded_hash(self, path: str, hash_: str) -> None:
        """
        Update the last_embedded_hash for *path*.

        Parameters
        ----------
        path:
            File path to update.
        hash_:
            The file content hash at the time of embedding.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE files SET last_embedded_hash = ? WHERE path = ?",
                (hash_, path),
            )

    def get_files_needing_embed(self) -> list[tuple[str, str]]:
        """
        Return (path, hash) pairs for files whose hash differs from
        last_embedded_hash (i.e., need re-embedding).
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT path, hash FROM files
                WHERE last_embedded_hash IS NULL OR last_embedded_hash != hash
                """
            ).fetchall()
        return [(r["path"], r["hash"]) for r in rows]

    def find_symbol(self, name: str, symbol_type: Optional[str] = None) -> list[dict]:
        """
        Search the manifest for symbols matching *name*.

        Parameters
        ----------
        name:
            Symbol name to search for (case-sensitive).
        symbol_type:
            Optional filter: "function", "class", "variable".

        Returns
        -------
        list[dict]
            Each dict contains: name, symbol_type, file_path, line_start, line_end.
        """
        with self._connect() as conn:
            if symbol_type:
                rows = conn.execute(
                    """
                    SELECT s.name, s.symbol_type, f.path, s.line_start, s.line_end
                    FROM symbols s JOIN files f ON s.file_id = f.id
                    WHERE s.name = ? AND s.symbol_type = ?
                    """,
                    (name, symbol_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT s.name, s.symbol_type, f.path, s.line_start, s.line_end
                    FROM symbols s JOIN files f ON s.file_id = f.id
                    WHERE s.name = ?
                    """,
                    (name,),
                ).fetchall()
        return [
            {
                "name": r["name"],
                "symbol_type": r["symbol_type"],
                "file_path": r["path"],
                "line_start": r["line_start"],
                "line_end": r["line_end"],
            }
            for r in rows
        ]

    def stats(self) -> dict:
        """
        Return aggregate statistics about the manifest.

        Returns
        -------
        dict
            Keys: file_count, symbol_count, languages (dict[str, int]).
        """
        with self._connect() as conn:
            file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            symbol_count = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
            lang_rows = conn.execute(
                "SELECT language, COUNT(*) AS cnt FROM files GROUP BY language"
            ).fetchall()
        languages = {r["language"]: r["cnt"] for r in lang_rows}
        return {
            "file_count": file_count,
            "symbol_count": symbol_count,
            "languages": languages,
        }

    def clear(self) -> None:
        """Delete all data from the manifest (files + symbols)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM symbols")
            conn.execute("DELETE FROM files")
