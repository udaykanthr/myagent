"""
Unified global knowledge base query interface.

Provides a single ``search()`` method that queries both the Qdrant
``global_kb`` collection and the SQLite ``errors.db``, returning
ranked results across all categories.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from .error_dict import ErrorDict, ErrorFix

logger = logging.getLogger(__name__)

_GLOBAL_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.join(_GLOBAL_DIR, "core")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GlobalKBResult:
    """A single result from the global knowledge base."""

    title: str
    category: str  # "pattern" | "adr" | "doc" | "behavioral" | "error"
    content: str
    file: str
    score: float
    tags: list[str] = field(default_factory=list)
    language: str = "all"


# ---------------------------------------------------------------------------
# GlobalKBStore
# ---------------------------------------------------------------------------

class GlobalKBStore:
    """
    Unified interface over all global KB content.

    Combines:
    - Qdrant ``global_kb`` collection for semantic search over documents
    - SQLite ``errors.db`` for deterministic error lookups

    Parameters
    ----------
    errors_db_path:
        Path to the errors.db SQLite database.  Defaults to
        ``kb/global/core/errors.db``.
    """

    def __init__(self, errors_db_path: Optional[str] = None) -> None:
        self._errors_db_path = errors_db_path or os.path.join(
            _CORE_DIR, "errors.db"
        )
        self._error_dict: Optional[ErrorDict] = None
        self._qdrant_store = None

    # ------------------------------------------------------------------
    # Lazy initializers
    # ------------------------------------------------------------------

    def _get_error_dict(self) -> ErrorDict:
        """Return the ErrorDict, creating it if needed."""
        if self._error_dict is None:
            self._error_dict = ErrorDict(self._errors_db_path)
        return self._error_dict

    def _get_qdrant_store(self):
        """Return the global Qdrant store (lazy)."""
        if self._qdrant_store is None:
            from .seeder import _GlobalQdrantStore
            self._qdrant_store = _GlobalQdrantStore()
        return self._qdrant_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        categories: Optional[list[str]] = None,
        language: Optional[str] = None,
        top_k: int = 5,
    ) -> list[GlobalKBResult]:
        """
        Semantic search across all global KB content.

        Parameters
        ----------
        query:
            Natural-language search query.
        categories:
            Filter by category (e.g. ``["pattern", "adr"]``).
        language:
            Filter by language (e.g. ``"python"``).
        top_k:
            Maximum number of results.

        Returns
        -------
        list[GlobalKBResult]
            Ranked results.
        """
        try:
            return self._qdrant_search(query, categories, language, top_k)
        except Exception as exc:
            logger.warning("Qdrant search failed, falling back to file search: %s", exc)
            return self._fallback_file_search(query, categories, language, top_k)

    def search_errors(
        self,
        error_message: str,
        language: Optional[str] = None,
    ) -> list[ErrorFix]:
        """
        Search for error fixes.

        Strategy:
        1. Exact/regex lookup via ErrorDict (fast, deterministic)
        2. Semantic search filtered to behavioral docs
        3. Error dict results ranked above semantic results

        Parameters
        ----------
        error_message:
            The error text to look up.
        language:
            Optional language filter.

        Returns
        -------
        list[ErrorFix]
            Matched error-fix records ranked by match quality.
        """
        edict = self._get_error_dict()
        return edict.lookup(error_message, language)

    def get_behavioral_instructions(
        self,
        context: str,
    ) -> list[GlobalKBResult]:
        """
        Retrieve behavioral instructions relevant to *context*.

        Used by Phase 4 context_builder to inject agent instructions.

        Parameters
        ----------
        context:
            Description of the current task or situation.

        Returns
        -------
        list[GlobalKBResult]
            Behavioral instruction results.
        """
        return self.search(
            query=context,
            categories=["behavioral"],
            top_k=3,
        )

    # ------------------------------------------------------------------
    # Qdrant search
    # ------------------------------------------------------------------

    def _qdrant_search(
        self,
        query: str,
        categories: Optional[list[str]],
        language: Optional[str],
        top_k: int,
    ) -> list[GlobalKBResult]:
        """Perform semantic search via Qdrant."""
        from ..local.embedder import _get_openai_client, _embed_batch

        client = _get_openai_client()
        vectors = _embed_batch(client, [query])
        if not vectors:
            return []
        query_vector = vectors[0]

        store = self._get_qdrant_store()

        # Build filters
        filters: Optional[dict] = {}
        if categories and len(categories) == 1:
            filters["category"] = categories[0]
        if language:
            filters["language"] = language
        if not filters:
            filters = None

        raw_results = store.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

        results: list[GlobalKBResult] = []
        for hit in raw_results:
            payload = hit.get("payload", {})
            cat = payload.get("category", "")

            # Category filter (multi-category)
            if categories and len(categories) > 1 and cat not in categories:
                continue

            results.append(GlobalKBResult(
                title=payload.get("title", ""),
                category=cat,
                content="",  # Content stored in files, not Qdrant payload
                file=payload.get("file", ""),
                score=hit.get("score", 0.0),
                tags=payload.get("tags", []),
                language=payload.get("language", "all"),
            ))

        return results

    # ------------------------------------------------------------------
    # Fallback file search (offline / no Qdrant)
    # ------------------------------------------------------------------

    def _fallback_file_search(
        self,
        query: str,
        categories: Optional[list[str]],
        language: Optional[str],
        top_k: int,
    ) -> list[GlobalKBResult]:
        """
        Simple keyword-based file search when Qdrant is unavailable.

        Scans registry markdown files for query terms.
        """
        from .seeder import _REGISTRY_DIR, _parse_frontmatter

        query_words = set(query.lower().split())
        results: list[tuple[float, GlobalKBResult]] = []

        category_dirs = {
            "pattern": "patterns",
            "adr": "adrs",
            "doc": "docs",
            "behavioral": "behavioral",
        }

        for cat, dirname in category_dirs.items():
            if categories and cat not in categories:
                continue

            cat_dir = os.path.join(_REGISTRY_DIR, dirname)
            if not os.path.isdir(cat_dir):
                continue

            for fname in os.listdir(cat_dir):
                if not fname.endswith(".md"):
                    continue

                filepath = os.path.join(cat_dir, fname)
                try:
                    with open(filepath, encoding="utf-8") as fh:
                        content = fh.read()
                except OSError:
                    continue

                meta = _parse_frontmatter(content)
                doc_lang = meta.get("language", "all")
                if language and doc_lang != "all" and doc_lang != language:
                    continue

                # Score: count matching words
                content_lower = content.lower()
                score = sum(
                    1.0 for w in query_words if w in content_lower
                ) / max(len(query_words), 1)

                if score > 0:
                    tags_str = meta.get("tags", "")
                    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                    rel_path = os.path.relpath(filepath, _GLOBAL_DIR)

                    results.append((score, GlobalKBResult(
                        title=meta.get("title", fname),
                        category=cat,
                        content=content[:500],
                        file=rel_path,
                        score=score,
                        tags=tags,
                        language=doc_lang,
                    )))

        results.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in results[:top_k]]
