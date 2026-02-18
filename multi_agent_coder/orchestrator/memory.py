"""
FileMemory — thread-safe file content tracker with context-window budget.
"""

import threading

from ..embedding_store import EmbeddingStore
from ..cli_display import log


def _estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token)."""
    return len(text) // 4


class FileMemory:
    """Tracks every file's path and current contents across all steps.

    When an EmbeddingStore is provided, context retrieval uses semantic
    similarity instead of simple filename matching.
    """

    def __init__(self, embedding_store: EmbeddingStore | None = None,
                 top_k: int = 5):
        self._files: dict[str, str] = {}   # filepath -> contents
        self._store = embedding_store
        self._top_k = top_k
        self._lock = threading.Lock()

    def update(self, files: dict[str, str]):
        """Store or overwrite file contents and update embeddings."""
        with self._lock:
            self._files.update(files)
            if self._store:
                for fpath, content in files.items():
                    self._store.add(fpath, content)

    def get(self, filepath: str) -> str | None:
        with self._lock:
            return self._files.get(filepath)

    def all_files(self) -> dict[str, str]:
        with self._lock:
            return dict(self._files)

    def as_dict(self) -> dict[str, str]:
        """Snapshot for checkpoint serialization."""
        with self._lock:
            return dict(self._files)

    def related_context(self, step_text: str, max_tokens: int | None = None) -> str:
        """Build a compact context string with the most relevant files.

        Uses semantic search when embeddings are available, otherwise
        falls back to filename substring matching.

        When *max_tokens* is given, files are accumulated until the
        budget is reached.
        """
        with self._lock:
            if self._store and self._store.size > 0:
                return self._semantic_context(step_text, max_tokens)
            return self._substring_context(step_text, max_tokens)

    def _semantic_context(self, step_text: str, max_tokens: int | None) -> str:
        results = self._store.search(step_text, top_k=self._top_k)
        parts: list[str] = []
        budget = max_tokens or float("inf")
        used = 0
        for fpath, score in results:
            content = self._files.get(fpath, "")
            if not content:
                continue
            entry = (
                f"#### [FILE]: {fpath} (relevance: {score:.2f})\n"
                f"```\n{content}\n```"
            )
            entry_tokens = _estimate_tokens(entry)
            if used + entry_tokens > budget:
                break
            parts.append(entry)
            used += entry_tokens
        log.debug(f"[FileMemory] Semantic search returned {len(parts)} files "
                  f"({used} est. tokens)")
        return "\n\n".join(parts)

    def _substring_context(self, step_text: str, max_tokens: int | None) -> str:
        parts: list[str] = []
        budget = max_tokens or float("inf")
        used = 0
        for fpath, content in self._files.items():
            basename = fpath.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
            if basename in step_text or fpath in step_text:
                entry = f"#### [FILE]: {fpath}\n```\n{content}\n```"
                entry_tokens = _estimate_tokens(entry)
                if used + entry_tokens > budget:
                    break
                parts.append(entry)
                used += entry_tokens
        return "\n\n".join(parts)

    def summary(self) -> str:
        with self._lock:
            if not self._files:
                return "(no files yet)"
            return ", ".join(self._files.keys())
