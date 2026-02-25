"""
Symbol embedder for the Local Knowledge Base — Phase 2 Semantic Layer.

Extracts FUNCTION and CLASS nodes from the Phase 1 code graph, formats
them into natural-language text chunks, and calls the OpenAI Embeddings
API (text-embedding-3-small) to produce 1536-dimensional vectors.

Never uses token windows — always chunks at symbol boundaries.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .graph import CodeGraph
    from .manifest import Manifest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 1536
BATCH_SIZE = 100
MAX_RETRIES = 3

# NodeType strings (mirror graph.NodeType to avoid circular imports)
_FUNCTION = "FUNCTION"
_CLASS = "CLASS"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SymbolChunk:
    """A symbol extracted from the graph, ready for embedding."""

    # Identification
    file_path: str          # relative path
    symbol_name: str
    symbol_type: str        # "function" | "class" | "method"
    parent_class: Optional[str]
    line_start: int
    line_end: int
    language: str

    # Embedding text
    text: str

    # Point ID for Qdrant (deterministic UUID)
    point_id: str


# ---------------------------------------------------------------------------
# Text formatters
# ---------------------------------------------------------------------------

def _function_text(
    language: str,
    file_path: str,
    name: str,
    params: list[str],
    return_type: str,
    docstring: str,
    body_lines: list[str],
) -> str:
    """Format a FUNCTION node into embeddable text."""
    params_str = ", ".join(params) if params else "none"
    return_str = return_type if return_type else "none"
    doc_str = docstring.strip() if docstring else "none"
    body_str = "\n".join(body_lines).strip() if body_lines else ""
    return (
        f"Language: {language}\n"
        f"File: {file_path}\n"
        f"Function: {name}\n"
        f"Parameters: {params_str}\n"
        f"Returns: {return_str}\n"
        f"Docstring: {doc_str}\n"
        f"Body:\n{body_str}"
    )


def _class_text(
    language: str,
    file_path: str,
    name: str,
    bases: list[str],
    docstring: str,
    method_names: list[str],
) -> str:
    """Format a CLASS node into embeddable text."""
    inherits_str = ", ".join(bases) if bases else "none"
    doc_str = docstring.strip() if docstring else "none"
    methods_str = ", ".join(method_names) if method_names else "none"
    return (
        f"Language: {language}\n"
        f"File: {file_path}\n"
        f"Class: {name}\n"
        f"Inherits: {inherits_str}\n"
        f"Docstring: {doc_str}\n"
        f"Methods: {methods_str}"
    )


# ---------------------------------------------------------------------------
# Deterministic UUID for idempotent Qdrant upserts
# ---------------------------------------------------------------------------

def make_point_id(file_path: str, symbol_name: str, line_start: int) -> str:
    """
    Generate a deterministic UUID from ``{file_path}:{symbol_name}:{line_start}``.

    Parameters
    ----------
    file_path:
        Relative file path.
    symbol_name:
        Name of the symbol.
    line_start:
        Starting line number.

    Returns
    -------
    str
        UUID5 string, guaranteed stable for the same inputs.
    """
    key = f"{file_path}:{symbol_name}:{line_start}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


# ---------------------------------------------------------------------------
# Graph → SymbolChunk extractor
# ---------------------------------------------------------------------------

def extract_symbol_chunks(
    graph: "CodeGraph",
    project_root: str,
) -> list[SymbolChunk]:
    """
    Walk the code graph and produce one :class:`SymbolChunk` per FUNCTION
    or CLASS node.

    Parameters
    ----------
    graph:
        Loaded :class:`~agentchanti.kb.local.graph.CodeGraph`.
    project_root:
        Absolute path to the project root (used to read source lines).

    Returns
    -------
    list[SymbolChunk]
        One chunk per embeddable symbol.
    """
    chunks: list[SymbolChunk] = []
    # Build a map: (file_path, class_name) → list of method names
    # so we can populate the CLASS text with method names only.
    class_methods: dict[tuple[str, str], list[str]] = {}
    for nid, attrs in graph._g.nodes(data=True):
        if attrs.get("node_type") != _FUNCTION:
            continue
        parent = attrs.get("parent_class")
        if parent:
            fp = attrs.get("file_path", "")
            key = (fp, parent)
            class_methods.setdefault(key, []).append(attrs.get("name", ""))

    for nid, attrs in graph._g.nodes(data=True):
        node_type = attrs.get("node_type", "")
        if node_type not in (_FUNCTION, _CLASS):
            continue

        file_path: str = attrs.get("file_path", "")
        name: str = attrs.get("name", "")
        line_start: int = attrs.get("line_start", 0)
        line_end: int = attrs.get("line_end", 0)
        language: str = ""
        parent_class: Optional[str] = attrs.get("parent_class")

        # Determine language from the FILE node
        fid = f"FILE:{file_path}"
        if graph._g.has_node(fid):
            language = graph._g.nodes[fid].get("language", "")

        # Read source body
        body_lines = _read_lines(
            os.path.join(project_root, file_path), line_start, line_end
        )

        if node_type == _FUNCTION:
            params: list[str] = attrs.get("params") or []
            return_type: str = attrs.get("return_type") or ""
            docstring: str = attrs.get("docstring") or ""
            text = _function_text(
                language, file_path, name, params, return_type, docstring, body_lines
            )
            sym_type = "method" if parent_class else "function"
        else:  # CLASS
            bases: list[str] = attrs.get("bases") or []
            docstring = attrs.get("docstring") or ""
            method_names = class_methods.get((file_path, name), [])
            text = _class_text(language, file_path, name, bases, docstring, method_names)
            sym_type = "class"

        point_id = make_point_id(file_path, name, line_start)
        chunks.append(
            SymbolChunk(
                file_path=file_path,
                symbol_name=name,
                symbol_type=sym_type,
                parent_class=parent_class,
                line_start=line_start,
                line_end=line_end,
                language=language,
                text=text,
                point_id=point_id,
            )
        )

    return chunks


def _read_lines(abs_path: str, line_start: int, line_end: int) -> list[str]:
    """
    Read *line_start* to *line_end* (1-indexed, inclusive) from *abs_path*.

    Returns an empty list on any I/O error.
    """
    if not abs_path or not os.path.exists(abs_path):
        return []
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
        start = max(0, line_start - 1)
        end = line_end if line_end and line_end > 0 else len(all_lines)
        return [l.rstrip("\n") for l in all_lines[start:end]]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# OpenAI embedding helpers
# ---------------------------------------------------------------------------

def _get_openai_client():
    """Return an openai.OpenAI client, raising ImportError if not installed."""
    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai package is required for embedding. "
            "Install it with: pip install 'multi_agent_coder[semantic]'"
        ) from exc
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set."
        )
    return openai.OpenAI(api_key=api_key)


def _embed_batch(client, texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts via the OpenAI Embeddings API.

    Retries up to MAX_RETRIES times with exponential back-off on failure.

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` client instance.
    texts:
        List of text strings to embed.

    Returns
    -------
    list[list[float]]
        Embedding vectors in the same order as *texts*.

    Raises
    ------
    RuntimeError
        If all retries are exhausted.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                logger.warning(
                    "Embedding API error (attempt %d/%d): %s — retrying in %ds",
                    attempt, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Embedding API failed after {MAX_RETRIES} attempts: {exc}"
                ) from exc
    return []  # unreachable


# ---------------------------------------------------------------------------
# Main embedding orchestrator
# ---------------------------------------------------------------------------

def embed_project(
    graph: "CodeGraph",
    manifest: "Manifest",
    vector_store,
    project_root: str,
    incremental: bool = False,
) -> dict:
    """
    Embed all (or only changed) symbols and upsert them into *vector_store*.

    Parameters
    ----------
    graph:
        Loaded code graph from Phase 1.
    manifest:
        Loaded manifest instance.
    vector_store:
        A :class:`~agentchanti.kb.local.vector_store.QdrantStore` instance.
    project_root:
        Absolute path to the project root.
    incremental:
        If True, skip files whose hash matches their last_embedded_hash.

    Returns
    -------
    dict
        Keys: total_symbols, embedded, skipped, errors.
    """
    try:
        from tqdm import tqdm  # type: ignore
        _tqdm = tqdm
    except ImportError:
        _tqdm = None  # type: ignore

    client = _get_openai_client()
    all_chunks = extract_symbol_chunks(graph, project_root)

    # For incremental mode, collect hashes of already-embedded files.
    embedded_hashes: dict[str, str] = {}
    if incremental:
        for path, hash_ in manifest.get_files_needing_embed():
            embedded_hashes[path] = hash_
        # Chunks for files that do NOT need re-embedding are skipped.
        chunks_to_embed: list[SymbolChunk] = []
        chunks_skipped: list[SymbolChunk] = []
        files_needing = {p for p, _ in manifest.get_files_needing_embed()}
        for chunk in all_chunks:
            if chunk.file_path in files_needing:
                chunks_to_embed.append(chunk)
            else:
                chunks_skipped.append(chunk)
    else:
        chunks_to_embed = all_chunks
        chunks_skipped = []

    total_symbols = len(all_chunks)
    embedded_count = 0
    error_count = 0
    batches_processed = 0

    # Group into batches of BATCH_SIZE
    iterator = range(0, len(chunks_to_embed), BATCH_SIZE)
    if _tqdm:
        iterator = _tqdm(
            iterator,
            desc="Embedding symbols",
            unit="batch",
            total=(len(chunks_to_embed) + BATCH_SIZE - 1) // BATCH_SIZE,
            postfix={"embedded": 0},
        )

    for batch_start in iterator:
        batch = chunks_to_embed[batch_start: batch_start + BATCH_SIZE]
        texts = [c.text for c in batch]

        try:
            vectors = _embed_batch(client, texts)
        except RuntimeError as exc:
            logger.warning("Skipping batch starting at %d: %s", batch_start, exc)
            error_count += len(batch)
            continue

        # Build Qdrant points
        import datetime
        points = []
        for chunk, vector in zip(batch, vectors):
            try:
                last_mod = os.path.getmtime(
                    os.path.join(project_root, chunk.file_path)
                )
                last_mod_str = datetime.datetime.utcfromtimestamp(last_mod).isoformat()
            except Exception:
                last_mod_str = ""

            loc = (chunk.line_end - chunk.line_start + 1) if (
                chunk.line_end and chunk.line_start
            ) else 0

            payload = {
                "file": chunk.file_path,
                "language": chunk.language,
                "symbol_type": chunk.symbol_type,
                "symbol_name": chunk.symbol_name,
                "parent_class": chunk.parent_class,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "last_modified": last_mod_str,
                "loc": loc,
            }
            points.append((chunk.point_id, vector, payload))

        try:
            vector_store.upsert(points)
            embedded_count += len(batch)
            batches_processed += 1
            if _tqdm and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"embedded": embedded_count})
        except Exception as exc:
            logger.warning("Qdrant upsert failed for batch: %s", exc)
            error_count += len(batch)

    # After embedding, update last_embedded_hash for each file in batch
    if embedded_count > 0:
        embedded_files: set[str] = {c.file_path for c in chunks_to_embed}
        for path in embedded_files:
            record = manifest.get_file(path)
            if record:
                manifest.set_embedded_hash(path, record.hash)

    return {
        "total_symbols": total_symbols,
        "embedded": embedded_count,
        "skipped": len(chunks_skipped),
        "errors": error_count,
    }


def embed_file_symbols(
    file_path: str,
    graph: "CodeGraph",
    manifest: "Manifest",
    vector_store,
    project_root: str,
) -> None:
    """
    Re-embed only the symbols belonging to *file_path*.

    Called by the file watcher after an incremental graph update.

    Parameters
    ----------
    file_path:
        Relative file path that was modified.
    graph:
        Updated code graph.
    manifest:
        Manifest instance.
    vector_store:
        Qdrant store instance.
    project_root:
        Absolute path to the project root.
    """
    client = _get_openai_client()
    all_chunks = extract_symbol_chunks(graph, project_root)
    file_chunks = [c for c in all_chunks if c.file_path == file_path]

    if not file_chunks:
        logger.debug("No embeddable symbols found in %s", file_path)
        return

    texts = [c.text for c in file_chunks]
    try:
        vectors = _embed_batch(client, texts)
    except RuntimeError as exc:
        logger.warning("Embedding failed for file %s: %s", file_path, exc)
        return

    import datetime
    points = []
    for chunk, vector in zip(file_chunks, vectors):
        try:
            last_mod = os.path.getmtime(os.path.join(project_root, chunk.file_path))
            last_mod_str = datetime.datetime.utcfromtimestamp(last_mod).isoformat()
        except Exception:
            last_mod_str = ""

        loc = (chunk.line_end - chunk.line_start + 1) if (
            chunk.line_end and chunk.line_start
        ) else 0

        payload = {
            "file": chunk.file_path,
            "language": chunk.language,
            "symbol_type": chunk.symbol_type,
            "symbol_name": chunk.symbol_name,
            "parent_class": chunk.parent_class,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "last_modified": last_mod_str,
            "loc": loc,
        }
        points.append((chunk.point_id, vector, payload))

    try:
        vector_store.upsert(points)
        record = manifest.get_file(file_path)
        if record:
            manifest.set_embedded_hash(file_path, record.hash)
        logger.info("[embedder] Re-embedded %d symbols for %s", len(points), file_path)
    except Exception as exc:
        logger.warning("[embedder] Qdrant upsert failed for %s: %s", file_path, exc)
