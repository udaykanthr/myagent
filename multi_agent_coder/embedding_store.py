"""
Embedding Store — in-memory vector storage with cosine similarity search.

Stores embeddings for code files and retrieves the most semantically
relevant files for a given query (e.g. a step description).
"""

import math
from typing import List, Optional, Tuple
from .llm.base import LLMClient
from .cli_display import log

# Approximate chars-per-token ratio for chunking
_CHARS_PER_CHUNK = 6000       # ~1500 tokens
_CHUNK_OVERLAP_CHARS = 400    # overlap between chunks


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors without numpy."""
    if not a or not b or len(a) != len(b):
        return 0.0
    try:
        dot = sum(x * y for x, y in zip(a, b) if x is not None and y is not None)
        norm_a = math.sqrt(sum(x * x for x in a if x is not None))
        norm_b = math.sqrt(sum(x * x for x in b if x is not None))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
    except (TypeError, ValueError):
        return 0.0


def _chunk_text(text: str, chunk_size: int = _CHARS_PER_CHUNK,
                overlap: int = _CHUNK_OVERLAP_CHARS) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


class EmbeddingStore:
    """
    In-memory vector store backed by an LLM embedding endpoint.

    Usage:
        store = EmbeddingStore(llm_client, embed_model="nomic-embed-text")
        store.add("src/utils.py", file_contents)
        results = store.search("parse JSON config", top_k=3)
        # results = [("src/utils.py", 0.87), ...]
    """

    def __init__(self, llm_client: LLMClient, embed_model: Optional[str] = None):
        self.llm_client = llm_client
        self.embed_model = embed_model
        # key -> list of (chunk_index, embedding_vector)
        self._vectors: dict[str, List[Tuple[int, List[float]]]] = {}
        # Track keys that have already failed to avoid spamming warnings
        self._failed_keys: set[str] = set()

    def add(self, key: str, text: str) -> bool:
        """
        Embed *text* and store under *key*.
        Long texts are chunked automatically.
        Returns True if at least one chunk was embedded successfully.
        """
        chunks = _chunk_text(text)
        stored_any = False
        chunk_vectors: List[Tuple[int, List[float]]] = []
        for idx, chunk in enumerate(chunks):
            vec = self.llm_client.generate_embedding(chunk, model=self.embed_model)
            if vec and all(v is not None for v in vec):
                chunk_vectors.append((idx, vec))
                stored_any = True
        if stored_any:
            self._vectors[key] = chunk_vectors
            self._failed_keys.discard(key)
            log.debug(f"[EmbeddingStore] Stored {len(chunk_vectors)} chunk(s) for '{key}'")
        else:
            if key not in self._failed_keys:
                log.warning(f"[EmbeddingStore] Failed to embed '{key}' "
                            f"(falling back to substring matching)")
                self._failed_keys.add(key)
            else:
                log.debug(f"[EmbeddingStore] Still failing to embed '{key}'")
        return stored_any

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Embed *query* and return the top-k keys ranked by cosine similarity.
        For files with multiple chunks, the best (highest) chunk score is used.
        Returns list of (key, score) tuples, descending by score.
        """
        query_vec = self.llm_client.generate_embedding(query, model=self.embed_model)
        if not query_vec or any(v is None for v in query_vec):
            log.debug("[EmbeddingStore] Could not embed query, falling back to substring match")
            return []

        scores: dict[str, float] = {}
        for key, chunk_vectors in self._vectors.items():
            best = max(
                (_cosine_similarity(query_vec, vec) for _, vec in chunk_vectors),
                default=0.0,
            )
            scores[key] = best

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @property
    def size(self) -> int:
        """Number of keys stored."""
        return len(self._vectors)

    def has_key(self, key: str) -> bool:
        return key in self._vectors
