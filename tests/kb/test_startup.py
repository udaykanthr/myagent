"""
Unit tests for multi_agent_coder.kb.startup

Tests the KBStartupManager decision logic: each branch in the decision
table gets its own test.  All external dependencies (Qdrant, indexer,
seeder, filesystem) are mocked.
"""

from __future__ import annotations

import os
import threading
import time
import pytest
from unittest.mock import MagicMock, patch, call

from multi_agent_coder.kb.startup import KBStartupManager, KBStartupReport


# ---------------------------------------------------------------------------
# KBStartupReport tests
# ---------------------------------------------------------------------------

class TestKBStartupReport:

    def test_defaults(self):
        report = KBStartupReport()
        assert not report.qdrant_started
        assert not report.global_kb_seeded
        assert not report.local_index_triggered
        assert not report.local_incremental_triggered
        assert not report.background
        assert report.skipped_reason is None

    def test_anything_happened_false_by_default(self):
        report = KBStartupReport()
        assert not report.anything_happened()

    def test_anything_happened_qdrant(self):
        report = KBStartupReport(qdrant_started=True)
        assert report.anything_happened()

    def test_anything_happened_global_kb(self):
        report = KBStartupReport(global_kb_seeded=True)
        assert report.anything_happened()

    def test_anything_happened_local_index(self):
        report = KBStartupReport(local_index_triggered=True)
        assert report.anything_happened()

    def test_anything_happened_incremental_only_is_silent(self):
        """Incremental + background should NOT count as 'anything happened'."""
        report = KBStartupReport(
            local_incremental_triggered=True, background=True
        )
        assert not report.anything_happened()

    def test_print_summary_qdrant(self, capsys):
        report = KBStartupReport(qdrant_started=True)
        report.print_summary()
        out = capsys.readouterr().out
        assert "Qdrant started" in out

    def test_print_summary_global(self, capsys):
        report = KBStartupReport(global_kb_seeded=True)
        report.print_summary()
        out = capsys.readouterr().out
        assert "Global KB initialized" in out

    def test_print_summary_local_background(self, capsys):
        report = KBStartupReport(
            local_index_triggered=True, background=True
        )
        report.print_summary()
        out = capsys.readouterr().out
        assert "Local KB indexing started" in out
        assert "background" in out


# ---------------------------------------------------------------------------
# KBStartupManager — Qdrant checks
# ---------------------------------------------------------------------------

class TestStartupQdrant:

    @patch.object(KBStartupManager, "_check_local_kb")
    @patch.object(KBStartupManager, "_global_kb_exists", return_value=True)
    @patch.object(KBStartupManager, "_qdrant_running", return_value=True)
    def test_qdrant_already_running(self, mock_qr, mock_gk, mock_lk):
        """When Qdrant is running, qdrant_started stays False."""
        report = KBStartupManager().run("/tmp/project")
        assert not report.qdrant_started

    @patch.object(KBStartupManager, "_check_local_kb")
    @patch.object(KBStartupManager, "_global_kb_exists", return_value=True)
    @patch.object(KBStartupManager, "_start_qdrant")
    @patch.object(KBStartupManager, "_qdrant_running", return_value=False)
    def test_qdrant_auto_started(self, mock_qr, mock_start, mock_gk, mock_lk):
        """When Qdrant is not running, it should be auto-started."""
        report = KBStartupManager().run("/tmp/project")
        assert report.qdrant_started
        mock_start.assert_called_once_with("/tmp/project")

    @patch.object(KBStartupManager, "_check_local_kb")
    @patch.object(KBStartupManager, "_global_kb_exists", return_value=True)
    @patch.object(
        KBStartupManager, "_start_qdrant",
        side_effect=Exception("Docker not found"),
    )
    @patch.object(KBStartupManager, "_qdrant_running", return_value=False)
    def test_qdrant_start_failure_swallowed(
        self, mock_qr, mock_start, mock_gk, mock_lk
    ):
        """Qdrant start failure should be swallowed, not crash."""
        report = KBStartupManager().run("/tmp/project")
        assert not report.qdrant_started


# ---------------------------------------------------------------------------
# KBStartupManager — Global KB checks
# ---------------------------------------------------------------------------

class TestStartupGlobalKB:

    @patch.object(KBStartupManager, "_check_local_kb")
    @patch.object(KBStartupManager, "_qdrant_running", return_value=True)
    @patch.object(KBStartupManager, "_global_kb_exists", return_value=True)
    def test_global_kb_exists_no_seed(self, mock_gk, mock_qr, mock_lk):
        """When global KB exists, no seeding should happen."""
        report = KBStartupManager().run("/tmp/project")
        assert not report.global_kb_seeded

    @patch.object(KBStartupManager, "_check_local_kb")
    @patch.object(KBStartupManager, "_qdrant_running", return_value=True)
    @patch.object(KBStartupManager, "_seed_global_kb")
    @patch.object(KBStartupManager, "_global_kb_exists", return_value=False)
    def test_global_kb_missing_triggers_seed(
        self, mock_gk, mock_seed, mock_qr, mock_lk
    ):
        """When global KB doesn't exist, seeder should run."""
        report = KBStartupManager().run("/tmp/project")
        assert report.global_kb_seeded
        mock_seed.assert_called_once_with("/tmp/project")

    @patch.object(KBStartupManager, "_check_local_kb")
    @patch.object(KBStartupManager, "_qdrant_running", return_value=True)
    @patch.object(
        KBStartupManager, "_seed_global_kb",
        side_effect=Exception("seed fail"),
    )
    @patch.object(KBStartupManager, "_global_kb_exists", return_value=False)
    def test_global_kb_seed_failure_swallowed(
        self, mock_gk, mock_seed, mock_qr, mock_lk
    ):
        """Global KB seed failure should be swallowed."""
        report = KBStartupManager().run("/tmp/project")
        assert not report.global_kb_seeded


# ---------------------------------------------------------------------------
# KBStartupManager — Local KB: _check_local_kb branches
# ---------------------------------------------------------------------------

class TestStartupLocalKB:
    """Tests for each branch of _check_local_kb."""

    def _make_manager(self, **overrides):
        """Create a KBStartupManager with mocked dependencies."""
        mgr = KBStartupManager()
        # Defaults
        mgr._qdrant_running = MagicMock(return_value=True)
        mgr._global_kb_exists = MagicMock(return_value=True)
        mgr._run_background = MagicMock()
        for k, v in overrides.items():
            setattr(mgr, k, v)
        return mgr

    # -- CASE A: No index --

    def test_no_index_blank_project(self):
        """No index + 0 files → skip, reason='blank_project'."""
        mgr = self._make_manager(
            _read_graph_meta=MagicMock(return_value=None),
            _count_project_files=MagicMock(return_value=0),
        )
        report = KBStartupReport()
        mgr._check_local_kb("/tmp/project", report)

        assert report.skipped_reason == "blank_project"
        assert not report.local_index_triggered
        mgr._run_background.assert_not_called()

    def test_no_index_small_project(self):
        """No index + <= 50 files → background full index."""
        mgr = self._make_manager(
            _read_graph_meta=MagicMock(return_value=None),
            _count_project_files=MagicMock(return_value=25),
        )
        report = KBStartupReport()
        mgr._check_local_kb("/tmp/project", report)

        assert report.local_index_triggered
        assert report.background
        mgr._run_background.assert_called_once()
        # Verify the function called is _full_index_and_embed
        args = mgr._run_background.call_args
        assert args[0][0] == mgr._full_index_and_embed

    def test_no_index_exactly_50_files(self):
        """No index + exactly 50 files → background full index (boundary)."""
        mgr = self._make_manager(
            _read_graph_meta=MagicMock(return_value=None),
            _count_project_files=MagicMock(return_value=50),
        )
        report = KBStartupReport()
        mgr._check_local_kb("/tmp/project", report)

        assert report.local_index_triggered
        assert report.background

    def test_no_index_large_project(self):
        """No index + > 50 files → skip, reason='large_project_needs_manual_index'."""
        mgr = self._make_manager(
            _read_graph_meta=MagicMock(return_value=None),
            _count_project_files=MagicMock(return_value=200),
        )
        report = KBStartupReport()
        mgr._check_local_kb("/tmp/project", report)

        assert report.skipped_reason == "large_project_needs_manual_index"
        assert not report.local_index_triggered
        mgr._run_background.assert_not_called()

    def test_no_index_exactly_51_files(self):
        """No index + 51 files → large project (boundary)."""
        mgr = self._make_manager(
            _read_graph_meta=MagicMock(return_value=None),
            _count_project_files=MagicMock(return_value=51),
        )
        report = KBStartupReport()
        mgr._check_local_kb("/tmp/project", report)

        assert report.skipped_reason == "large_project_needs_manual_index"

    # -- CASE B: Index exists, check staleness --

    def test_index_exists_nothing_changed(self):
        """Index exists + 0 changed → no action (common case)."""
        mgr = self._make_manager(
            _read_graph_meta=MagicMock(return_value={"last_indexed": "2025-01-01T00:00:00Z"}),
            _index_age_minutes=MagicMock(return_value=5),
            _count_changed_files=MagicMock(return_value=0),
        )
        report = KBStartupReport()
        mgr._check_local_kb("/tmp/project", report)

        assert not report.local_index_triggered
        assert not report.local_incremental_triggered
        assert report.skipped_reason is None
        mgr._run_background.assert_not_called()

    def test_index_exists_small_change(self):
        """Index exists + 1-10 changed → incremental background."""
        mgr = self._make_manager(
            _read_graph_meta=MagicMock(return_value={"last_indexed": "2025-01-01T00:00:00Z"}),
            _index_age_minutes=MagicMock(return_value=5),
            _count_changed_files=MagicMock(return_value=5),
        )
        report = KBStartupReport()
        mgr._check_local_kb("/tmp/project", report)

        assert report.local_incremental_triggered
        assert report.background
        assert not report.local_index_triggered
        mgr._run_background.assert_called_once()
        args = mgr._run_background.call_args
        assert args[0][0] == mgr._incremental_update

    def test_index_exists_10_files_changed(self):
        """Index exists + exactly 10 changed → incremental (boundary)."""
        mgr = self._make_manager(
            _read_graph_meta=MagicMock(return_value={"last_indexed": "2025-01-01T00:00:00Z"}),
            _index_age_minutes=MagicMock(return_value=5),
            _count_changed_files=MagicMock(return_value=10),
        )
        report = KBStartupReport()
        mgr._check_local_kb("/tmp/project", report)

        assert report.local_incremental_triggered
        assert not report.local_index_triggered

    def test_index_exists_moderate_change(self):
        """Index exists + 11-50 changed, < 60 min → incremental background."""
        mgr = self._make_manager(
            _read_graph_meta=MagicMock(return_value={"last_indexed": "2025-01-01T00:00:00Z"}),
            _index_age_minutes=MagicMock(return_value=30),
            _count_changed_files=MagicMock(return_value=25),
        )
        report = KBStartupReport()
        mgr._check_local_kb("/tmp/project", report)

        assert report.local_incremental_triggered
        assert report.background
        assert not report.local_index_triggered

    def test_index_exists_many_changes(self):
        """Index exists + > 50 changed → full re-index background."""
        mgr = self._make_manager(
            _read_graph_meta=MagicMock(return_value={"last_indexed": "2025-01-01T00:00:00Z"}),
            _index_age_minutes=MagicMock(return_value=30),
            _count_changed_files=MagicMock(return_value=60),
        )
        report = KBStartupReport()
        mgr._check_local_kb("/tmp/project", report)

        assert report.local_index_triggered
        assert report.background
        assert not report.local_incremental_triggered
        args = mgr._run_background.call_args
        assert args[0][0] == mgr._full_index_and_embed

    def test_index_exists_very_stale(self):
        """Index exists + > 60 min old → full re-index even with few changes."""
        mgr = self._make_manager(
            _read_graph_meta=MagicMock(return_value={"last_indexed": "2025-01-01T00:00:00Z"}),
            _index_age_minutes=MagicMock(return_value=120),
            _count_changed_files=MagicMock(return_value=15),
        )
        report = KBStartupReport()
        mgr._check_local_kb("/tmp/project", report)

        assert report.local_index_triggered
        assert report.background

    def test_index_exists_exactly_61_min_stale(self):
        """Index exists + 61 min old → full re-index (boundary)."""
        mgr = self._make_manager(
            _read_graph_meta=MagicMock(return_value={"last_indexed": "2025-01-01T00:00:00Z"}),
            _index_age_minutes=MagicMock(return_value=61),
            _count_changed_files=MagicMock(return_value=15),
        )
        report = KBStartupReport()
        mgr._check_local_kb("/tmp/project", report)

        assert report.local_index_triggered

    def test_index_exists_exactly_60_min_moderate_changes(self):
        """Index exists + exactly 60 min + 25 changes → incremental (boundary)."""
        mgr = self._make_manager(
            _read_graph_meta=MagicMock(return_value={"last_indexed": "2025-01-01T00:00:00Z"}),
            _index_age_minutes=MagicMock(return_value=60),
            _count_changed_files=MagicMock(return_value=25),
        )
        report = KBStartupReport()
        mgr._check_local_kb("/tmp/project", report)

        # 60 min is NOT > 60, and 25 is NOT > 50 → falls through to Case C
        assert report.local_incremental_triggered
        assert not report.local_index_triggered


# ---------------------------------------------------------------------------
# KBStartupManager — helper methods
# ---------------------------------------------------------------------------

class TestStartupHelpers:

    def test_index_age_minutes_valid(self):
        mgr = KBStartupManager()
        meta = {"last_indexed": "2025-01-01T00:00:00Z"}
        age = mgr._index_age_minutes(meta)
        # Should be a very large number since this is years ago
        assert age > 0

    def test_index_age_minutes_missing_field(self):
        mgr = KBStartupManager()
        meta = {}
        age = mgr._index_age_minutes(meta)
        assert age == 9999

    def test_index_age_minutes_bad_format(self):
        mgr = KBStartupManager()
        meta = {"last_indexed": "not-a-date"}
        age = mgr._index_age_minutes(meta)
        assert age == 9999


# ---------------------------------------------------------------------------
# KBStartupManager — background runner
# ---------------------------------------------------------------------------

class TestBackgroundRunner:

    def test_run_background_calls_function(self):
        """_run_background should call the function in a daemon thread."""
        mgr = KBStartupManager()
        results = []

        def tracker(val):
            results.append(val)

        mgr._run_background(tracker, "hello")
        time.sleep(0.5)
        assert results == ["hello"]

    def test_safe_run_swallows_exceptions(self):
        """_safe_run should not raise even if fn throws."""
        mgr = KBStartupManager()

        def exploder():
            raise RuntimeError("boom")

        # Should not raise
        mgr._safe_run(exploder)


# ---------------------------------------------------------------------------
# KBStartupManager — full integration (mocked externals)
# ---------------------------------------------------------------------------

class TestStartupIntegration:

    @patch.object(KBStartupManager, "_check_local_kb")
    @patch.object(KBStartupManager, "_global_kb_exists", return_value=True)
    @patch.object(KBStartupManager, "_qdrant_running", return_value=True)
    def test_common_case_nothing_to_do(self, mock_qr, mock_gk, mock_lk):
        """Common case: everything ready, nothing to do → fast path."""
        report = KBStartupManager().run("/tmp/project")

        assert not report.qdrant_started
        assert not report.global_kb_seeded
        assert not report.anything_happened()

    @patch.object(KBStartupManager, "_run_background")
    @patch.object(KBStartupManager, "_count_project_files", return_value=10)
    @patch.object(KBStartupManager, "_read_graph_meta", return_value=None)
    @patch.object(KBStartupManager, "_seed_global_kb")
    @patch.object(KBStartupManager, "_global_kb_exists", return_value=False)
    @patch.object(KBStartupManager, "_start_qdrant")
    @patch.object(KBStartupManager, "_qdrant_running", return_value=False)
    def test_fresh_install_small_project(
        self, mock_qr, mock_start, mock_gk, mock_seed,
        mock_meta, mock_files, mock_bg, capsys,
    ):
        """Fresh install + small project → start qdrant, seed, index bg."""
        report = KBStartupManager().run("/tmp/project")

        assert report.qdrant_started
        assert report.global_kb_seeded
        assert report.local_index_triggered
        assert report.background
        mock_start.assert_called_once()
        mock_seed.assert_called_once()
        mock_bg.assert_called_once()

    @patch.object(KBStartupManager, "_run_background")
    @patch.object(KBStartupManager, "_count_project_files", return_value=200)
    @patch.object(KBStartupManager, "_read_graph_meta", return_value=None)
    @patch.object(KBStartupManager, "_global_kb_exists", return_value=True)
    @patch.object(KBStartupManager, "_qdrant_running", return_value=True)
    def test_fresh_install_large_project(
        self, mock_qr, mock_gk, mock_meta, mock_files, mock_bg,
    ):
        """Large project without index → warn but don't index."""
        report = KBStartupManager().run("/tmp/project")

        assert report.skipped_reason == "large_project_needs_manual_index"
        assert not report.local_index_triggered
        mock_bg.assert_not_called()

    @patch.object(KBStartupManager, "_run_background")
    @patch.object(KBStartupManager, "_count_changed_files", return_value=0)
    @patch.object(KBStartupManager, "_index_age_minutes", return_value=5)
    @patch.object(
        KBStartupManager, "_read_graph_meta",
        return_value={"last_indexed": "2025-01-01T00:00:00Z"},
    )
    @patch.object(KBStartupManager, "_global_kb_exists", return_value=True)
    @patch.object(KBStartupManager, "_qdrant_running", return_value=True)
    def test_existing_project_nothing_changed(
        self, mock_qr, mock_gk, mock_meta, mock_age, mock_changed, mock_bg,
    ):
        """Existing project, 0 changes → silent, < 10ms."""
        report = KBStartupManager().run("/tmp/project")

        assert not report.local_index_triggered
        assert not report.local_incremental_triggered
        assert not report.anything_happened()
        mock_bg.assert_not_called()

    @patch.object(KBStartupManager, "_run_background")
    @patch.object(KBStartupManager, "_count_changed_files", return_value=5)
    @patch.object(KBStartupManager, "_index_age_minutes", return_value=10)
    @patch.object(
        KBStartupManager, "_read_graph_meta",
        return_value={"last_indexed": "2025-01-01T00:00:00Z"},
    )
    @patch.object(KBStartupManager, "_global_kb_exists", return_value=True)
    @patch.object(KBStartupManager, "_qdrant_running", return_value=True)
    def test_existing_project_few_changes(
        self, mock_qr, mock_gk, mock_meta, mock_age, mock_changed, mock_bg,
    ):
        """5 files changed → silent incremental background update."""
        report = KBStartupManager().run("/tmp/project")

        assert report.local_incremental_triggered
        assert report.background
        assert not report.anything_happened()  # incremental is silent
        mock_bg.assert_called_once()

    @patch.object(KBStartupManager, "_run_background")
    @patch.object(KBStartupManager, "_count_changed_files", return_value=60)
    @patch.object(KBStartupManager, "_index_age_minutes", return_value=120)
    @patch.object(
        KBStartupManager, "_read_graph_meta",
        return_value={"last_indexed": "2025-01-01T00:00:00Z"},
    )
    @patch.object(KBStartupManager, "_global_kb_exists", return_value=True)
    @patch.object(KBStartupManager, "_qdrant_running", return_value=True)
    def test_existing_project_very_stale(
        self, mock_qr, mock_gk, mock_meta, mock_age, mock_changed, mock_bg,
    ):
        """Very stale index (> 60 min, > 50 changes) → full re-index bg."""
        report = KBStartupManager().run("/tmp/project")

        assert report.local_index_triggered
        assert report.background
        assert report.anything_happened()

    def test_all_exceptions_swallowed(self):
        """KBStartupManager.run should never raise, even if everything fails."""
        mgr = KBStartupManager()
        mgr._qdrant_running = MagicMock(side_effect=Exception("fail1"))
        mgr._global_kb_exists = MagicMock(side_effect=Exception("fail2"))
        mgr._check_local_kb = MagicMock(side_effect=Exception("fail3"))

        # Should not raise
        report = mgr.run("/tmp/project")
        assert isinstance(report, KBStartupReport)
