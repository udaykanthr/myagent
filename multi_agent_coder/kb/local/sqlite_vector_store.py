"""
SQLite-backed local vector store for the Local Knowledge Base.

Stores embedding vectors in SQLite and computes cosine similarity
using numpy.  Zero-config — no Docker, no external services required.

The KB stack (embedder, searcher, context_builder) all use this store
for vector operations.

Storage: ``.agentchanti/kb/local/vectors.db``
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_SIZE = 1536  # OpenAI text-embedding-ada-002 dimension

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS vectors (
    point_id   TEXT PRIMARY KEY,
    vector     BLOB NOT NULL,
    payload    TEXT NOT NULL DEFAULT '{}'
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_vectors_file
ON vectors(json_extract(payload, '$.file'));
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec_to_bytes(vec: list[float]) -> bytes:
    """Serialise a float list to compact bytes via numpy."""
    if _HAS_NUMPY:
        return np.array(vec, dtype=np.float32).tobytes()
    # Pure-Python fallback (slower but works without numpy)
    import struct
    return struct.pack(f"{len(vec)}f", *vec)


def _bytes_to_vec(buf: bytes) -> "np.ndarray | list[float]":
    """Deserialise bytes back to a vector."""
    if _HAS_NUMPY:
        return np.frombuffer(buf, dtype=np.float32).copy()
    import struct
    n = len(buf) // 4
    return list(struct.unpack(f"{n}f", buf))


def _cosine_similarity_batch(
    query: "np.ndarray", matrix: "np.ndarray"
) -> "np.ndarray":
    """Compute cosine similarity between *query* (1-D) and each row of *matrix*."""
    # Avoid division by zero
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(matrix.shape[0])
    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms[row_norms == 0] = 1.0
    return (matrix @ query) / (row_norms * query_norm)


def _cosine_similarity_single(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity for one pair (fallback)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# SQLiteVectorStore
# ---------------------------------------------------------------------------

class SQLiteVectorStore:
    """Local vector store backed by SQLite + numpy cosine similarity.

    Zero-config vector store — no Docker or external services required.

    Parameters
    ----------
    project_root:
        Absolute path to the project root directory.
    db_path:
        Override the default database path.
    """

    def __init__(
        self,
        project_root: str,
        db_path: str | None = None,
    ) -> None:
        self._project_root = project_root
        if db_path is None:
            kb_dir = os.path.join(project_root, ".agentchanti", "kb", "local")
            os.makedirs(kb_dir, exist_ok=True)
            db_path = os.path.join(kb_dir, "vectors.db")
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the database and table if missing."""
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        conn = self._get_conn()
        conn.executescript(_CREATE_TABLE)
        try:
            conn.execute(_CREATE_INDEX)
        except sqlite3.OperationalError:
            # json_extract not available on older SQLite builds — harmless
            pass
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Thread-safe lazy connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self._db_path, check_same_thread=False
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """No-op — the table is auto-created in ``_init_db``."""
        pass

    def upsert(self, points: list[tuple[str, list[float], dict]]) -> None:
        """Upsert vector points.

        Parameters
        ----------
        points:
            List of ``(point_id, vector, payload)`` tuples.
        """
        if not points:
            return
        with self._lock:
            conn = self._get_conn()
            for point_id, vector, payload in points:
                vec_bytes = _vec_to_bytes(vector)
                payload_json = json.dumps(payload, default=str)
                conn.execute(
                    "INSERT OR REPLACE INTO vectors (point_id, vector, payload) "
                    "VALUES (?, ?, ?)",
                    (point_id, vec_bytes, payload_json),
                )
            conn.commit()
        logger.debug(
            "[SQLiteVectorStore] Upserted %d points", len(points)
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Cosine-similarity search.

        Parameters
        ----------
        query_vector:
            The query embedding vector.
        top_k:
            Number of results to return.
        filters:
            Optional payload filters (``language``, ``symbol_type``).

        Returns
        -------
        list[dict]
            Each dict has ``score`` (float) and ``payload`` (dict).
        """
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT point_id, vector, payload FROM vectors"
            ).fetchall()

        if not rows:
            return []

        # Decode payloads and apply pre-filters
        filtered: list[tuple[str, bytes, dict]] = []
        for pid, vec_bytes, payload_json in rows:
            try:
                payload = json.loads(payload_json)
            except (json.JSONDecodeError, TypeError):
                payload = {}
            if filters:
                if "language" in filters and payload.get("language") != filters["language"]:
                    continue
                if "symbol_type" in filters and payload.get("symbol_type") != filters["symbol_type"]:
                    continue
            filtered.append((pid, vec_bytes, payload))

        if not filtered:
            return []

        # Compute similarities
        if _HAS_NUMPY:
            query_arr = np.array(query_vector, dtype=np.float32)
            matrix = np.stack(
                [np.frombuffer(row[1], dtype=np.float32).copy() for row in filtered]
            )
            scores = _cosine_similarity_batch(query_arr, matrix)

            # Get top-k indices
            if len(scores) <= top_k:
                top_indices = np.argsort(scores)[::-1]
            else:
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            results = []
            for idx in top_indices:
                score = float(scores[idx])
                if score <= 0:
                    continue
                results.append({
                    "score": score,
                    "payload": filtered[idx][2],
                })
            return results
        else:
            # Pure-Python fallback
            scored: list[tuple[float, dict]] = []
            for pid, vec_bytes, payload in filtered:
                vec = _bytes_to_vec(vec_bytes)
                sim = _cosine_similarity_single(query_vector, vec)  # type: ignore[arg-type]
                if sim > 0:
                    scored.append((sim, payload))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {"score": s, "payload": p}
                for s, p in scored[:top_k]
            ]

    def delete_by_file(self, file_path: str) -> None:
        """Delete all points whose payload ``file`` matches *file_path*.

        Parameters
        ----------
        file_path:
            Relative file path.
        """
        with self._lock:
            conn = self._get_conn()
            # Fetch all, filter by file in Python (works on all SQLite versions)
            rows = conn.execute(
                "SELECT point_id, payload FROM vectors"
            ).fetchall()
            to_delete: list[str] = []
            for pid, payload_json in rows:
                try:
                    payload = json.loads(payload_json)
                except (json.JSONDecodeError, TypeError):
                    continue
                if payload.get("file") == file_path:
                    to_delete.append(pid)
            if to_delete:
                placeholders = ",".join("?" for _ in to_delete)
                conn.execute(
                    f"DELETE FROM vectors WHERE point_id IN ({placeholders})",
                    to_delete,
                )
                conn.commit()
                logger.debug(
                    "[SQLiteVectorStore] Deleted %d points for file %s",
                    len(to_delete), file_path,
                )

    def collection_info(self) -> Optional[dict]:
        """Return basic info about the collection.

        Returns
        -------
        Optional[dict]
            Keys: ``name``, ``points_count``.
        """
        try:
            with self._lock:
                conn = self._get_conn()
                row = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
            count = row[0] if row else 0
            return {
                "name": f"local_sqlite_{os.path.basename(self._project_root)}",
                "points_count": count,
            }
        except sqlite3.Error:
            return None


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_vector_store(
    project_root: str,
    backend: str = "local",
) -> SQLiteVectorStore:
    """Create the local vector store.

    Parameters
    ----------
    project_root:
        Absolute path to the project root.
    backend:
        Kept for backward compatibility. Always uses SQLite.

    Returns
    -------
    SQLiteVectorStore
    """
    return SQLiteVectorStore(project_root)
