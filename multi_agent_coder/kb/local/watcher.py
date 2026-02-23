"""
File watcher for incremental Knowledge Base updates.

Uses watchdog to monitor project files and trigger the Indexer
for any changed, created, or deleted source files.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class KBFileHandler:
    """
    Watchdog-compatible event handler that triggers incremental KB updates.

    After updating the graph for a changed file, optionally triggers
    incremental re-embedding of that file's symbols via the Phase 2
    embedder if *vector_store* is provided.

    Parameters
    ----------
    indexer:
        The :class:`~agentchanti.kb.local.indexer.Indexer` instance to call.
    project_root:
        Absolute path to the project root (used to compute relative paths).
    debounce_seconds:
        Minimum delay between processing the same file (prevents rapid
        re-indexing on editor auto-saves).
    vector_store:
        Optional :class:`~agentchanti.kb.local.vector_store.QdrantStore`.
        When provided, changed files are re-embedded after the graph update.
    """

    def __init__(
        self,
        indexer,
        project_root: str,
        debounce_seconds: float = 0.5,
        vector_store=None,
    ) -> None:
        self._indexer = indexer
        self._project_root = os.path.abspath(project_root)
        self._debounce = debounce_seconds
        self._last_event: dict[str, float] = {}
        self._lock = threading.Lock()
        self._vector_store = vector_store

    # ------------------------------------------------------------------
    # Watchdog event dispatch
    # ------------------------------------------------------------------

    def on_modified(self, event) -> None:  # type: ignore[override]
        """Handle a file modification event."""
        if not event.is_directory:
            self._handle_change(event.src_path)

    def on_created(self, event) -> None:  # type: ignore[override]
        """Handle a file creation event."""
        if not event.is_directory:
            self._handle_change(event.src_path)

    def on_deleted(self, event) -> None:  # type: ignore[override]
        """Handle a file deletion event."""
        if not event.is_directory:
            self._handle_delete(event.src_path)

    def on_moved(self, event) -> None:  # type: ignore[override]
        """Handle a file move/rename event."""
        if not event.is_directory:
            self._handle_delete(event.src_path)
            self._handle_change(event.dest_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rel_path(self, abs_path: str) -> Optional[str]:
        """Convert *abs_path* to a project-relative path, or None if outside."""
        try:
            return os.path.relpath(abs_path, self._project_root)
        except ValueError:
            return None

    def _should_ignore(self, abs_path: str) -> bool:
        """Return True if this file should not be processed."""
        from .parser import EXTENSION_TO_LANGUAGE
        from .indexer import _SKIP_DIRS

        # Check extension
        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in EXTENSION_TO_LANGUAGE:
            return True

        # Check if inside a skipped directory
        parts = abs_path.replace("\\", "/").split("/")
        for part in parts:
            if part in _SKIP_DIRS:
                return True

        return False

    def _is_debounced(self, abs_path: str) -> bool:
        """Return True if this file was recently processed (debounce)."""
        now = time.time()
        with self._lock:
            last = self._last_event.get(abs_path, 0.0)
            if now - last < self._debounce:
                return True
            self._last_event[abs_path] = now
        return False

    def _handle_change(self, abs_path: str) -> None:
        """Process a file modification or creation event."""
        if self._should_ignore(abs_path):
            return
        if self._is_debounced(abs_path):
            return
        rel_path = self._rel_path(abs_path)
        if rel_path is None:
            return

        logger.info("[KB watcher] Updated: %s", rel_path)
        try:
            self._indexer.update_file(rel_path)
        except Exception as exc:
            logger.warning("[KB watcher] Error processing %s: %s", rel_path, exc)
            return

        # Phase 2: re-embed this file's symbols if a vector store is attached.
        if self._vector_store is not None:
            self._trigger_incremental_embed(rel_path)

    def _handle_delete(self, abs_path: str) -> None:
        """Process a file deletion event."""
        if self._should_ignore(abs_path):
            return
        rel_path = self._rel_path(abs_path)
        if rel_path is None:
            return

        logger.info("[KB watcher] Deleted: %s", rel_path)
        try:
            self._indexer.remove_file(rel_path)
        except Exception as exc:
            logger.warning("[KB watcher] Error removing %s: %s", rel_path, exc)
            return

        # Phase 2: remove Qdrant points for this file.
        if self._vector_store is not None:
            try:
                self._vector_store.delete_by_file(rel_path)
                logger.info("[KB watcher] Removed Qdrant points for %s", rel_path)
            except Exception as exc:
                logger.warning(
                    "[KB watcher] Failed to remove Qdrant points for %s: %s",
                    rel_path, exc
                )

    def _trigger_incremental_embed(self, rel_path: str) -> None:
        """
        Re-embed the symbols for *rel_path* in a background thread.

        Parameters
        ----------
        rel_path:
            Relative file path that was just updated in the graph.
        """
        import threading

        def _do_embed() -> None:
            try:
                from .embedder import embed_file_symbols
                from .manifest import Manifest
                from .indexer import _manifest_path

                manifest = Manifest(_manifest_path(self._project_root))
                graph = self._indexer.load_graph()
                embed_file_symbols(
                    file_path=rel_path,
                    graph=graph,
                    manifest=manifest,
                    vector_store=self._vector_store,
                    project_root=self._project_root,
                )
            except Exception as exc:
                logger.warning(
                    "[KB watcher] Incremental embed failed for %s: %s",
                    rel_path, exc,
                )

        t = threading.Thread(target=_do_embed, daemon=True, name=f"kb-embed-{rel_path}")
        t.start()


class KBWatcher:
    """
    High-level wrapper around watchdog that monitors a project directory.

    Usage::

        watcher = KBWatcher(indexer, project_root="/path/to/project")
        watcher.start()   # blocking (call from a thread) or use start_background()
        watcher.stop()

    Parameters
    ----------
    indexer:
        Configured :class:`~agentchanti.kb.local.indexer.Indexer`.
    project_root:
        Directory to watch.
    vector_store:
        Optional :class:`~agentchanti.kb.local.vector_store.QdrantStore`.
        When provided, changed files are re-embedded automatically after
        each incremental graph update (Phase 2 integration).
    """

    def __init__(self, indexer, project_root: str, vector_store=None) -> None:
        self._indexer = indexer
        self._project_root = os.path.abspath(project_root)
        self._observer: Optional[object] = None
        self._handler = KBFileHandler(indexer, project_root, vector_store=vector_store)

    def start(self) -> None:
        """
        Start watching the project directory.

        Blocks until :meth:`stop` is called.  For non-blocking use, call
        :meth:`start_background` instead.
        """
        try:
            from watchdog.observers import Observer  # type: ignore
            from watchdog.events import FileSystemEventHandler  # type: ignore
        except ImportError:
            logger.error(
                "watchdog is not installed. Install with: pip install watchdog"
            )
            return

        # Adapt our handler to watchdog's interface
        class _WatchdogAdapter(FileSystemEventHandler):
            def __init__(self, handler: KBFileHandler) -> None:
                self._h = handler

            def on_modified(self, event):
                self._h.on_modified(event)

            def on_created(self, event):
                self._h.on_created(event)

            def on_deleted(self, event):
                self._h.on_deleted(event)

            def on_moved(self, event):
                self._h.on_moved(event)

        observer = Observer()
        observer.schedule(_WatchdogAdapter(self._handler), self._project_root, recursive=True)
        observer.start()
        self._observer = observer
        logger.info("[KB watcher] Watching %s", self._project_root)

        try:
            while observer.is_alive():
                observer.join(timeout=1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    def start_background(self) -> None:
        """
        Start the file watcher in a background daemon thread.
        """
        t = threading.Thread(target=self.start, daemon=True, name="kb-watcher")
        t.start()

    def stop(self) -> None:
        """Stop the file watcher observer."""
        if self._observer is not None:
            try:
                self._observer.stop()  # type: ignore[union-attr]
            except Exception:
                pass
            logger.info("[KB watcher] Stopped")
