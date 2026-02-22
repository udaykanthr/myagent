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
        """Store or overwrite file contents and update embeddings.

        Protected manifest files (package.json, go.mod, etc.) are skipped
        when they already exist on disk to prevent LLM-generated corruption.
        """
        import os
        from ..executor import Executor

        with self._lock:
            for fpath, content in files.items():
                # Guard: don't store LLM-generated protected files if they
                # already exist on disk (the LLM's version is almost always
                # a stripped-down, corrupted subset)
                basename = os.path.basename(fpath)
                if basename in Executor._PROTECTED_FILENAMES and os.path.isfile(fpath):
                    log.warning(f"[FileMemory] Skipping protected file update: "
                                f"{fpath} (already exists on disk)")
                    continue

                self._files[fpath] = content
                if self._store:
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
        """Score-based fallback matching when embeddings are unavailable.

        Scoring:
        1. Exact path/filename match in step text  → 100 pts
        2. File name keyword match (stem words)     → 50 pts
        3. Extension keyword match (e.g. "HTML")    → 10 pts
        """
        import os
        import re
        step_lower = step_text.lower()

        scored: list[tuple[int, str, str]] = []  # (score, fpath, content)
        for fpath, content in self._files.items():
            score = 0
            basename = fpath.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
            stem, ext = os.path.splitext(basename)

            # Exact path or basename match
            if fpath in step_text or basename in step_text:
                score += 100

            # Keyword match: stem parts in step text
            # e.g. "login_page.html" → ["login", "page"]
            stem_parts = [p for p in stem.replace("-", "_").split("_") if len(p) > 2]
            for part in stem_parts:
                if part.lower() in step_lower:
                    score += 50
                    break  # one keyword match is enough

            # Extension keyword match: e.g. "HTML" in step text matches .html
            # Only match as a standalone word (not inside filenames like "utils.py")
            if ext:
                ext_name = ext.lstrip(".").lower()
                if re.search(r'\b' + re.escape(ext_name) + r'\b', step_lower):
                    # Exclude matches that are part of a filename (e.g. "utils.py")
                    # by checking if the match is preceded by a dot
                    matches = list(re.finditer(r'\b' + re.escape(ext_name) + r'\b', step_lower))
                    has_standalone = any(
                        m.start() == 0 or step_lower[m.start() - 1] != '.'
                        for m in matches
                    )
                    if has_standalone:
                        score += 10

            if score > 0:
                scored.append((score, fpath, content))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        parts: list[str] = []
        budget = max_tokens or float("inf")
        used = 0
        for _score, fpath, content in scored:
            entry = f"#### [FILE]: {fpath}\n```\n{content}\n```"
            entry_tokens = _estimate_tokens(entry)
            if used + entry_tokens > budget:
                break
            parts.append(entry)
            used += entry_tokens
        log.debug(f"[FileMemory] Substring fallback returned {len(parts)} files "
                  f"({used} est. tokens)")
        return "\n\n".join(parts)

    def summary(self) -> str:
        with self._lock:
            if not self._files:
                return "(no files yet)"
            return ", ".join(self._files.keys())
