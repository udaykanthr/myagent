"""
Qdrant vector store wrapper for the Local Knowledge Base — Phase 2.

Manages the local Qdrant Docker container and provides a thin client
wrapper for upsert and search operations.

Collection name: "local_{project_name_slugified}"
Vector size: 1536  (text-embedding-3-small)
Distance: Cosine
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
QDRANT_CONTAINER_NAME = "agentchanti-qdrant"
VECTOR_SIZE = 1536
DISTANCE = "Cosine"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    """Convert *name* to a lowercase alphanumeric slug (hyphens allowed)."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug or "project"


def collection_name(project_root: str) -> str:
    """
    Derive the Qdrant collection name from *project_root*.

    Parameters
    ----------
    project_root:
        Absolute path to the project root directory.

    Returns
    -------
    str
        ``"local_{slugified_directory_name}"``
    """
    dir_name = os.path.basename(os.path.abspath(project_root))
    return f"local_{_slugify(dir_name)}"


def is_qdrant_running() -> bool:
    """
    Return True if Qdrant is reachable at the configured port.

    Uses a lightweight HTTP GET to the Qdrant root endpoint.
    """
    try:
        import urllib.request
        req = urllib.request.Request(f"{QDRANT_URL}/", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Docker management helpers
# ---------------------------------------------------------------------------

def qdrant_start(project_root: str) -> bool:
    """
    Start the Qdrant Docker container, mounting *project_root*'s data dir.

    Parameters
    ----------
    project_root:
        Absolute path to the project root.  The Qdrant storage directory
        is placed at ``{project_root}/.agentchanti/kb/local/qdrant``.

    Returns
    -------
    bool
        True if the container started successfully.
    """
    if is_qdrant_running():
        print("Qdrant is already running.")
        return True

    storage_dir = os.path.join(
        project_root, ".agentchanti", "kb", "local", "qdrant"
    )
    os.makedirs(storage_dir, exist_ok=True)

    cmd = [
        "docker", "run", "-d",
        "--name", QDRANT_CONTAINER_NAME,
        "-p", f"{QDRANT_PORT}:{QDRANT_PORT}",
        "-v", f"{storage_dir}:/qdrant/storage",
        "qdrant/qdrant",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Already exists but stopped: try to restart it.
            if "already in use" in stderr or "already exists" in stderr:
                print(f"Container '{QDRANT_CONTAINER_NAME}' already exists. Starting it...")
                restart = subprocess.run(
                    ["docker", "start", QDRANT_CONTAINER_NAME],
                    capture_output=True, text=True, timeout=30,
                )
                if restart.returncode != 0:
                    print(f"Failed to start existing container: {restart.stderr.strip()}")
                    return False
            else:
                print(f"Failed to start Qdrant: {stderr}")
                return False

        # Wait up to 10 seconds for Qdrant to become ready.
        for _ in range(20):
            if is_qdrant_running():
                print(f"Qdrant started successfully on port {QDRANT_PORT}.")
                return True
            time.sleep(0.5)

        print("Qdrant container started but not responding yet. Give it a moment.")
        return False
    except FileNotFoundError:
        print(
            "Docker is not installed or not in PATH. "
            "Install Docker to use Qdrant: https://docs.docker.com/get-docker/"
        )
        return False
    except Exception as exc:
        print(f"Unexpected error starting Qdrant: {exc}")
        return False


def qdrant_stop() -> bool:
    """
    Stop and remove the Qdrant Docker container.

    Returns
    -------
    bool
        True if stopped successfully.
    """
    try:
        result = subprocess.run(
            ["docker", "stop", QDRANT_CONTAINER_NAME],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"Failed to stop Qdrant: {result.stderr.strip()}")
            return False
        print(f"Qdrant container '{QDRANT_CONTAINER_NAME}' stopped.")
        return True
    except FileNotFoundError:
        print("Docker is not installed or not in PATH.")
        return False
    except Exception as exc:
        print(f"Unexpected error stopping Qdrant: {exc}")
        return False


def qdrant_status() -> None:
    """Print a summary of the Qdrant container and connectivity status."""
    running = is_qdrant_running()
    print(f"Qdrant HTTP endpoint : {QDRANT_URL}")
    print(f"Qdrant status        : {'RUNNING' if running else 'NOT RUNNING'}")
    if not running:
        print(
            "\nTo start Qdrant, run:\n"
            "  agentchanti kb qdrant start"
        )
        return

    # List collections if running
    try:
        import urllib.request
        import json
        req = urllib.request.Request(f"{QDRANT_URL}/collections", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        collections = data.get("result", {}).get("collections", [])
        if collections:
            print(f"\nCollections ({len(collections)}):")
            for col in collections:
                print(f"  - {col.get('name', '?')}")
        else:
            print("\nNo collections found.")
    except Exception as exc:
        print(f"\nCould not list collections: {exc}")


# ---------------------------------------------------------------------------
# QdrantStore
# ---------------------------------------------------------------------------

class QdrantStore:
    """
    Thin wrapper around the Qdrant HTTP client for the Local KB.

    Parameters
    ----------
    project_root:
        Project root directory (used to derive collection name).
    """

    def __init__(self, project_root: str) -> None:
        self._project_root = os.path.abspath(project_root)
        self._collection = collection_name(project_root)
        self._client = None  # lazy init

    # ------------------------------------------------------------------
    # Client & collection setup
    # ------------------------------------------------------------------

    def _get_client(self):
        """Return a Qdrant client, raising if not installed or not running."""
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required. "
                "Install it with: pip install 'multi_agent_coder[semantic]'"
            ) from exc

        if not is_qdrant_running():
            raise ConnectionError(
                "Qdrant is not running. "
                "Start it with: agentchanti kb qdrant start"
            )

        self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        return self._client

    def ensure_collection(self) -> None:
        """
        Create the Qdrant collection if it does not yet exist.

        Uses Cosine distance and ``VECTOR_SIZE`` dimensions.
        """
        from qdrant_client.models import Distance, VectorParams  # type: ignore

        client = self._get_client()
        existing = [c.name for c in client.get_collections().collections]
        if self._collection not in existing:
            client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", self._collection)
        else:
            logger.debug("Qdrant collection already exists: %s", self._collection)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(self, points: list[tuple[str, list[float], dict]]) -> None:
        """
        Upsert vector points into the collection.

        Parameters
        ----------
        points:
            List of ``(point_id, vector, payload)`` tuples.
            ``point_id`` is a UUID string; ``vector`` is a list of floats;
            ``payload`` is the Qdrant document payload dict.
        """
        if not points:
            return

        from qdrant_client.models import PointStruct  # type: ignore

        client = self._get_client()
        self.ensure_collection()

        qdrant_points = [
            PointStruct(id=pid, vector=vec, payload=payload)
            for pid, vec, payload in points
        ]
        client.upsert(collection_name=self._collection, points=qdrant_points)
        logger.debug("Upserted %d points into %s", len(points), self._collection)

    def delete_by_file(self, file_path: str) -> None:
        """
        Delete all points whose payload ``file`` matches *file_path*.

        Called when a file is deleted or re-indexed.

        Parameters
        ----------
        file_path:
            Relative file path.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore

        client = self._get_client()
        self.ensure_collection()
        client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="file", match=MatchValue(value=file_path))]
            ),
        )
        logger.debug("Deleted Qdrant points for file: %s", file_path)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Perform a cosine-similarity search.

        Parameters
        ----------
        query_vector:
            The embedded query vector (1536 dimensions).
        top_k:
            Number of results to return.
        filters:
            Optional payload filters.  Supported keys: ``"file"``,
            ``"language"``, ``"symbol_type"``.

        Returns
        -------
        list[dict]
            Each dict has ``score`` (float) and ``payload`` (dict).
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore

        client = self._get_client()

        qdrant_filter: Optional[object] = None
        if filters:
            conditions = []
            for key in ("file", "language", "symbol_type"):
                value = filters.get(key)
                if value:
                    # Support prefix matching for "file" (e.g. "src/auth")
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            if conditions:
                qdrant_filter = Filter(must=conditions)

        results = client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {"score": hit.score, "payload": hit.payload or {}}
            for hit in results.points
        ]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def collection_info(self) -> Optional[dict]:
        """
        Return basic info about the collection, or None if not found.

        Returns
        -------
        Optional[dict]
            Keys: ``name``, ``points_count``.
        """
        try:
            client = self._get_client()
            info = client.get_collection(self._collection)
            return {
                "name": self._collection,
                "points_count": info.points_count,
            }
        except Exception:
            return None
