"""
Runtime file watcher — auto-indexes files during agent execution.

Phase 4: Manages a background daemon thread that watches for file
changes (including new files created by the agent) and triggers
incremental KB indexing in real time.

Handles both:
- Existing projects (incremental mode from the start)
- Blank/new projects (waits for first file, triggers full index,
  then switches to incremental mode)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class RuntimeWatcher:
    """
    Manages the file watcher as a background daemon thread during
    AgentChanti agent execution.

    Handles both empty projects (initial index on first file creation)
    and existing projects (incremental updates).
    """

    def __init__(self, debounce_seconds: float = 1.0) -> None:
        self._debounce = debounce_seconds
        self._observer: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._project_root: Optional[str] = None
        self._running = False
        self._stop_event = threading.Event()

    def start(self, project_root: str) -> None:
        """
        Start the runtime file watcher in a background daemon thread.

        Parameters
        ----------
        project_root:
            Absolute path to the project root directory.
        """
        self._project_root = os.path.abspath(project_root)

        has_index = self._has_local_index()

        if has_index:
            # Existing project with index — start incremental watcher
            logger.info("[KB] Existing index detected. Starting incremental watcher.")
            self._start_incremental_watcher()
        else:
            # Blank/new project — start creation watcher
            logger.info(
                "[KB] New project detected. Will auto-index on first file creation."
            )
            self._start_creation_watcher()

    def stop(self) -> None:
        """Gracefully stop the file watcher."""
        self._stop_event.set()
        self._running = False

        if self._observer is not None:
            try:
                self._observer.stop()  # type: ignore[union-attr]
                self._observer.join(timeout=5)  # type: ignore[union-attr]
            except Exception:
                pass
            self._observer = None

        logger.info("[KB] RuntimeWatcher stopped.")

    @property
    def is_running(self) -> bool:
        """Return True if the watcher is active."""
        return self._running

    # ------------------------------------------------------------------
    # Internal: check index existence
    # ------------------------------------------------------------------

    def _has_local_index(self) -> bool:
        """Check if a local KB index (graph_meta.json) exists."""
        if not self._project_root:
            return False
        meta_path = os.path.join(
            self._project_root, ".agentchanti", "kb", "local", "graph_meta.json"
        )
        return os.path.isfile(meta_path)

    # ------------------------------------------------------------------
    # Internal: incremental watcher (existing project)
    # ------------------------------------------------------------------

    def _start_incremental_watcher(self) -> None:
        """Start the Phase 1 KBWatcher in a background daemon thread."""
        def _run() -> None:
            try:
                from .local.indexer import Indexer
                from .local.watcher import KBWatcher

                indexer = Indexer(self._project_root)
                watcher = KBWatcher(indexer, self._project_root)
                self._observer = watcher._observer  # will be set after start
                self._running = True

                # Use start_background which creates its own daemon thread
                # but we just call start() in our own thread
                watcher.start()
            except Exception as exc:
                logger.warning("[KB] Incremental watcher failed to start: %s", exc)
                self._running = False

        t = threading.Thread(target=_run, daemon=True, name="kb-runtime-watcher")
        t.start()
        self._thread = t
        self._running = True

    # ------------------------------------------------------------------
    # Internal: creation watcher (blank project)
    # ------------------------------------------------------------------

    def _start_creation_watcher(self) -> None:
        """
        Watch for the first file creation, then trigger a full index
        and switch to incremental mode.
        """
        def _run() -> None:
            try:
                from watchdog.observers import Observer  # type: ignore
                from watchdog.events import FileSystemEventHandler  # type: ignore
            except ImportError:
                logger.warning(
                    "[KB] watchdog not installed — auto-indexing disabled."
                )
                return

            handler = _FirstFileHandler(
                project_root=self._project_root,
                debounce_seconds=self._debounce,
                on_first_index_done=self._switch_to_incremental,
                stop_event=self._stop_event,
            )

            observer = Observer()
            observer.schedule(handler, self._project_root, recursive=True)
            observer.daemon = True
            observer.start()
            self._observer = observer
            self._running = True

            try:
                while not self._stop_event.is_set() and observer.is_alive():
                    self._stop_event.wait(timeout=1)
            except Exception:
                pass
            finally:
                observer.stop()
                observer.join(timeout=5)

        t = threading.Thread(target=_run, daemon=True, name="kb-creation-watcher")
        t.start()
        self._thread = t
        self._running = True

    def _switch_to_incremental(self) -> None:
        """Stop creation watcher and start incremental watcher."""
        logger.info("[KB] Switching to incremental watcher after initial index.")
        # Stop current observer
        if self._observer is not None:
            try:
                self._observer.stop()  # type: ignore[union-attr]
            except Exception:
                pass
            self._observer = None
        # Start incremental
        self._start_incremental_watcher()


class _FirstFileHandler:
    """
    Watchdog handler that triggers a full index on the first file
    creation event, then invokes a callback to switch modes.
    """

    # Skip directories that the indexer would also skip
    _SKIP_DIRS = frozenset({
        "node_modules", "dist", "build", "__pycache__",
        ".git", "vendor", ".agentchanti",
        ".venv", "venv", "env", ".env",
    })

    def __init__(
        self,
        project_root: str,
        debounce_seconds: float,
        on_first_index_done,
        stop_event: threading.Event,
    ) -> None:
        self._project_root = project_root
        self._debounce = debounce_seconds
        self._on_done = on_first_index_done
        self._stop_event = stop_event
        self._triggered = False
        self._lock = threading.Lock()
        self._pending_events: list[str] = []
        self._timer: Optional[threading.Timer] = None

    def dispatch(self, event) -> None:
        """Watchdog event dispatch."""
        if event.is_directory:
            return
        event_type = getattr(event, "event_type", "")
        if event_type in ("created", "modified"):
            self._handle_event(event.src_path)

    def on_created(self, event) -> None:
        """Handle file creation."""
        if not event.is_directory:
            self._handle_event(event.src_path)

    def on_modified(self, event) -> None:
        """Handle file modification."""
        if not event.is_directory:
            self._handle_event(event.src_path)

    def _should_ignore(self, path: str) -> bool:
        """Return True if the path should be ignored."""
        parts = path.replace("\\", "/").split("/")
        for part in parts:
            if part in self._SKIP_DIRS:
                return True
        return False

    def _handle_event(self, abs_path: str) -> None:
        """Buffer events and debounce."""
        if self._should_ignore(abs_path):
            return

        with self._lock:
            if self._triggered:
                return
            self._pending_events.append(abs_path)

            # Reset debounce timer
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(
                self._debounce + 1.0,  # extra second for agent to finish writing
                self._trigger_full_index,
            )
            self._timer.daemon = True
            self._timer.start()

    def _trigger_full_index(self) -> None:
        """Run full index + embed in a background thread."""
        with self._lock:
            if self._triggered:
                return
            self._triggered = True
            events = list(self._pending_events)
            self._pending_events.clear()

        if not events:
            return

        logger.info(
            "[KB] First file(s) detected (%d events). Triggering full index.",
            len(events),
        )

        try:
            from .local.indexer import Indexer

            indexer = Indexer(self._project_root)
            summary = indexer.full_index()
            logger.info(
                "[KB] Auto-index complete: %d files, %d symbols.",
                summary.get("file_count", 0),
                summary.get("symbol_count", 0),
            )

            # Try to embed if Qdrant is running
            try:
                from .local.vector_store import is_qdrant_running
                if is_qdrant_running():
                    from .local.embedder import embed_project
                    from .local.indexer import _manifest_path
                    from .local.manifest import Manifest
                    from .local.vector_store import QdrantStore

                    graph = indexer.load_graph()
                    manifest = Manifest(_manifest_path(self._project_root))
                    vector_store = QdrantStore(self._project_root)
                    embed_project(
                        graph=graph,
                        manifest=manifest,
                        vector_store=vector_store,
                        project_root=self._project_root,
                    )
                    logger.info("[KB] Auto-embed complete.")
            except Exception as exc:
                logger.debug("[KB] Auto-embed skipped: %s", exc)

        except Exception as exc:
            logger.warning("[KB] Auto-index failed: %s", exc)

        # Switch to incremental mode
        try:
            self._on_done()
        except Exception as exc:
            logger.warning("[KB] Failed to switch to incremental mode: %s", exc)
