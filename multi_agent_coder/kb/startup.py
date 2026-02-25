"""
KB Startup Manager — smart decisions about what KB operations to run.

Runs at every AgentChanti CLI start. Checks Qdrant, global KB, and
local KB state, then takes the minimum action needed.

Target: < 10ms for the common case (nothing to do).  All heavy work
(indexing, embedding) runs in background daemon threads so the CLI
is never blocked.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Startup report
# ---------------------------------------------------------------------------

@dataclass
class KBStartupReport:
    """Summarises what the startup manager did (or chose not to do)."""

    qdrant_started: bool = False
    global_kb_seeded: bool = False
    local_index_triggered: bool = False
    local_incremental_triggered: bool = False
    background: bool = False
    skipped_reason: Optional[str] = None
    # "blank_project" | "large_project_needs_manual_index"

    def anything_happened(self) -> bool:
        """Return True if any visible action was taken."""
        return any([
            self.qdrant_started,
            self.global_kb_seeded,
            self.local_index_triggered,
        ])
        # Note: incremental + background = silent, no summary needed

    def print_summary(self) -> None:
        """Print a compact summary of actions taken (only if visible)."""
        lines = ["[KB] Startup:"]
        if self.qdrant_started:
            lines.append("  + Qdrant started")
        if self.global_kb_seeded:
            lines.append("  + Global KB initialized")
        if self.local_index_triggered:
            suffix = " (background)" if self.background else ""
            lines.append(f"  + Local KB indexing started{suffix}")
        print("\n".join(lines))


# ---------------------------------------------------------------------------
# Startup manager
# ---------------------------------------------------------------------------

class KBStartupManager:
    """
    Runs at every AgentChanti CLI start.
    Makes smart decisions about what KB operations are needed.

    Decision table
    ──────────────────────────────────────────────────────────────
    Condition                            Action
    ──────────────────────────────────────────────────────────────
    Qdrant not running                 → Auto-start Docker (blocking, fast)
    global_kb collection missing       → seed_all() (blocking, one-time)
    global_kb exists                   → Nothing
    No local index + blank project     → Nothing (RuntimeWatcher handles)
    No local index + <= 50 files       → Full index+embed in background
    No local index + > 50 files        → Warn user, suggest manual kb index
    0 files changed                    → Nothing (< 10ms)
    1-10 files changed                 → Incremental update in background
    11-50 files changed                → Incremental update in background
    > 50 files changed OR > 60m stale  → Full re-index in background
    ──────────────────────────────────────────────────────────────
    """

    def run(self, project_root: str) -> KBStartupReport:
        """
        Execute the startup check sequence.

        Parameters
        ----------
        project_root:
            Absolute path to the project root directory.

        Returns
        -------
        KBStartupReport
        """
        report = KBStartupReport()

        # 1. CHECK QDRANT
        try:
            if not self._qdrant_running():
                self._start_qdrant(project_root)
                report.qdrant_started = True
        except Exception as exc:
            logger.debug("[KB] Qdrant auto-start failed: %s", exc)

        # 2. CHECK GLOBAL KB (seed)
        try:
            if not self._global_kb_exists():
                logger.info("[KB] Initializing Global KB for first time...")
                self._seed_global_kb(project_root)
                report.global_kb_seeded = True
        except Exception as exc:
            logger.debug("[KB] Global KB seed failed: %s", exc)

        # 3. CHECK LOCAL KB (index + embed)
        try:
            self._check_local_kb(project_root, report)
        except Exception as exc:
            logger.debug("[KB] Local KB check failed: %s", exc)

        # 4. PRINT STARTUP SUMMARY (only if something happened)
        if report.anything_happened():
            report.print_summary()

        return report

    # ------------------------------------------------------------------
    # Qdrant helpers
    # ------------------------------------------------------------------

    def _qdrant_running(self) -> bool:
        """Check if Qdrant is reachable."""
        from .local.vector_store import is_qdrant_running
        return is_qdrant_running()

    def _start_qdrant(self, project_root: str) -> None:
        """Auto-start Qdrant Docker container."""
        from .local.vector_store import qdrant_start
        qdrant_start(os.path.abspath(project_root))

    # ------------------------------------------------------------------
    # Global KB helpers
    # ------------------------------------------------------------------

    def _global_kb_exists(self) -> bool:
        """
        Check if the global KB has been seeded.

        Checks errors.db has records — this is the most reliable indicator
        because errors.db is always seeded and doesn't depend on Qdrant.
        """
        try:
            from .global_kb.error_dict import ErrorDict
            from .global_kb.seeder import _errors_db_path

            db_path = _errors_db_path()
            if not os.path.exists(db_path):
                return False
            edict = ErrorDict(db_path)
            return edict.count() > 0
        except Exception:
            return False

    def _seed_global_kb(self, project_root: str) -> None:
        """Run the global KB seeder."""
        from .global_kb.seeder import seed
        seed(embed=self._qdrant_running(), project_root=project_root)

    # ------------------------------------------------------------------
    # Local KB: the main decision logic
    # ------------------------------------------------------------------

    def _check_local_kb(self, project_root: str, report: KBStartupReport) -> None:
        """Apply the local KB decision table."""
        meta = self._read_graph_meta(project_root)

        # CASE A: No index at all — new project or first run
        if meta is None:
            file_count = self._count_project_files(project_root)

            if file_count == 0:
                # Blank project — RuntimeWatcher handles this
                logger.debug(
                    "[KB] Blank project, skipping index. "
                    "Will auto-index when files are created."
                )
                report.skipped_reason = "blank_project"
                return

            if file_count <= 50:
                # Small project — index + embed silently in background
                logger.info(
                    "[KB] New project detected, indexing in background..."
                )
                self._run_background(self._full_index_and_embed, project_root)
                report.local_index_triggered = True
                report.background = True
            else:
                # Large project — warn user, don't block startup
                logger.info(
                    "[KB] Project has %d files but no KB index.\n"
                    "     Run 'agentchanti kb index' to enable code intelligence.\n"
                    "     (This is a one-time operation)",
                    file_count,
                )
                report.skipped_reason = "large_project_needs_manual_index"
            return

        # CASE B: Index exists — check if stale
        age_minutes = self._index_age_minutes(meta)
        changed_files = self._count_changed_files(project_root, meta)

        if changed_files == 0:
            # Nothing changed — skip entirely
            logger.debug("[KB] Local KB is up to date, skipping.")
            return

        if changed_files <= 10:
            # Small change — incremental update in background
            logger.debug(
                "[KB] %d files changed, incremental update in background...",
                changed_files,
            )
            self._run_background(self._incremental_update, project_root)
            report.local_incremental_triggered = True
            report.background = True
            return

        if age_minutes > 60 or changed_files > 50:
            # Significantly stale — full re-index in background
            logger.info(
                "[KB] KB index is stale (%d files changed, %dm old). "
                "Re-indexing in background...",
                changed_files,
                age_minutes,
            )
            self._run_background(self._full_index_and_embed, project_root)
            report.local_index_triggered = True
            report.background = True
            return

        # CASE C: Moderate changes (11-50 files) — incremental in background
        self._run_background(self._incremental_update, project_root)
        report.local_incremental_triggered = True
        report.background = True

    # ------------------------------------------------------------------
    # Meta & file counting
    # ------------------------------------------------------------------

    def _read_graph_meta(self, project_root: str) -> Optional[dict]:
        """Read .agentchanti/kb/local/graph_meta.json."""
        from .local.indexer import read_meta
        return read_meta(project_root)

    def _count_project_files(self, project_root: str) -> int:
        """Count indexable source files in the project."""
        from .local.indexer import _walk_source_files
        return len(_walk_source_files(project_root))

    def _index_age_minutes(self, meta: dict) -> int:
        """Return how many minutes old the index is."""
        last_indexed = meta.get("last_indexed")
        if not last_indexed:
            return 9999  # treat as very stale
        try:
            indexed_time = datetime.fromisoformat(
                last_indexed.replace("Z", "+00:00")
            )
            now = datetime.now(timezone.utc)
            return int((now - indexed_time).total_seconds() / 60)
        except Exception:
            return 9999

    def _count_changed_files(self, project_root: str, meta: dict) -> int:
        """
        Count files whose on-disk hash differs from the indexed hash.

        Compares current source files against the manifest (index.db).
        """
        try:
            from .local.indexer import _walk_source_files, _manifest_path
            from .local.manifest import Manifest
            from .local.parser import compute_file_hash

            manifest = Manifest(_manifest_path(project_root))
            source_files = _walk_source_files(project_root)
            indexed_paths = set(manifest.get_all_indexed_paths())

            changed = 0

            # Check for new or modified files
            for rel_path in source_files:
                abs_path = os.path.join(project_root, rel_path)
                if rel_path not in indexed_paths:
                    changed += 1
                    continue
                try:
                    current_hash = compute_file_hash(abs_path)
                    if manifest.is_file_changed(rel_path, current_hash):
                        changed += 1
                except Exception:
                    changed += 1

            # Check for deleted files
            current_set = set(source_files)
            for indexed_path in indexed_paths:
                if indexed_path not in current_set:
                    changed += 1

            return changed
        except Exception as exc:
            logger.debug("[KB] Failed to count changed files: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Background runner
    # ------------------------------------------------------------------

    def _run_background(self, fn, *args) -> None:
        """Run *fn* in a daemon thread — never blocks the CLI."""
        thread = threading.Thread(
            target=self._safe_run,
            args=(fn, *args),
            daemon=True,
            name="kb-startup",
        )
        thread.start()

    def _safe_run(self, fn, *args) -> None:
        """Execute *fn* and swallow any exceptions."""
        try:
            fn(*args)
        except Exception as exc:
            logger.debug("[KB] Background startup task failed: %s", exc)

    # ------------------------------------------------------------------
    # Index / embed operations
    # ------------------------------------------------------------------

    def _full_index_and_embed(self, project_root: str) -> None:
        """Run a full index, then embed if Qdrant is running."""
        from .local.indexer import Indexer, _manifest_path

        indexer = Indexer(project_root)
        summary = indexer.full_index()
        logger.info(
            "[KB] Background full index complete: %d files, %d symbols.",
            summary.get("file_count", 0),
            summary.get("symbol_count", 0),
        )

        # Embed if Qdrant is available
        try:
            if self._qdrant_running():
                from .local.embedder import embed_project
                from .local.manifest import Manifest
                from .local.vector_store import QdrantStore

                graph = indexer.load_graph()
                manifest = Manifest(_manifest_path(project_root))
                vector_store = QdrantStore(project_root)
                embed_project(
                    graph=graph,
                    manifest=manifest,
                    vector_store=vector_store,
                    project_root=project_root,
                )
                logger.info("[KB] Background embed complete.")
        except Exception as exc:
            logger.debug("[KB] Background embed skipped: %s", exc)

    def _incremental_update(self, project_root: str) -> None:
        """
        Incrementally update changed files in the index.

        Walks the project, finds files whose hash differs from the
        manifest, and re-indexes only those files.
        """
        from .local.indexer import Indexer, _walk_source_files, _manifest_path
        from .local.manifest import Manifest
        from .local.parser import compute_file_hash

        indexer = Indexer(project_root)
        manifest = Manifest(_manifest_path(project_root))
        source_files = _walk_source_files(project_root)
        indexed_paths = set(manifest.get_all_indexed_paths())

        updated = 0

        # Update new or modified files
        for rel_path in source_files:
            abs_path = os.path.join(project_root, rel_path)
            if rel_path not in indexed_paths:
                indexer.update_file(rel_path)
                updated += 1
                continue
            try:
                current_hash = compute_file_hash(abs_path)
                if manifest.is_file_changed(rel_path, current_hash):
                    indexer.update_file(rel_path)
                    updated += 1
            except Exception:
                pass

        # Remove deleted files
        current_set = set(source_files)
        for indexed_path in indexed_paths:
            if indexed_path not in current_set:
                indexer.remove_file(indexed_path)
                updated += 1

        logger.info("[KB] Background incremental update: %d files processed.", updated)
