"""
Unit tests for multi_agent_coder.kb.runtime_watcher

Tests the RuntimeWatcher class: start/stop lifecycle, index detection,
and the first-file handler. watchdog is mocked throughout.
"""

from __future__ import annotations

import os
import threading
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

from multi_agent_coder.kb.runtime_watcher import (
    RuntimeWatcher, _FirstFileHandler,
)


# ---------------------------------------------------------------------------
# RuntimeWatcher lifecycle tests
# ---------------------------------------------------------------------------

class TestRuntimeWatcher:

    def test_initial_state(self):
        watcher = RuntimeWatcher()
        assert not watcher.is_running
        assert watcher._observer is None

    def test_stop_when_not_started(self):
        """stop() should be safe to call before start()."""
        watcher = RuntimeWatcher()
        watcher.stop()  # should not raise
        assert not watcher.is_running

    @patch("multi_agent_coder.kb.runtime_watcher.RuntimeWatcher._has_local_index")
    @patch("multi_agent_coder.kb.runtime_watcher.RuntimeWatcher._start_incremental_watcher")
    def test_start_with_existing_index(self, mock_incremental, mock_has_index, tmp_path):
        """When index exists, start() should use incremental mode."""
        mock_has_index.return_value = True
        watcher = RuntimeWatcher()
        watcher.start(str(tmp_path))
        mock_incremental.assert_called_once()

    @patch("multi_agent_coder.kb.runtime_watcher.RuntimeWatcher._has_local_index")
    @patch("multi_agent_coder.kb.runtime_watcher.RuntimeWatcher._start_creation_watcher")
    def test_start_without_index(self, mock_creation, mock_has_index, tmp_path):
        """When no index exists, start() should use creation mode."""
        mock_has_index.return_value = False
        watcher = RuntimeWatcher()
        watcher.start(str(tmp_path))
        mock_creation.assert_called_once()

    def test_has_local_index_true(self, tmp_path):
        """_has_local_index returns True when graph_meta.json exists."""
        meta_dir = tmp_path / ".agentchanti" / "kb" / "local"
        meta_dir.mkdir(parents=True)
        (meta_dir / "graph_meta.json").write_text("{}")

        watcher = RuntimeWatcher()
        watcher._project_root = str(tmp_path)
        assert watcher._has_local_index()

    def test_has_local_index_false(self, tmp_path):
        """_has_local_index returns False when meta file is absent."""
        watcher = RuntimeWatcher()
        watcher._project_root = str(tmp_path)
        assert not watcher._has_local_index()


# ---------------------------------------------------------------------------
# _FirstFileHandler tests
# ---------------------------------------------------------------------------

class TestFirstFileHandler:

    def test_ignore_git_directory(self):
        handler = _FirstFileHandler(
            project_root="/tmp/project",
            debounce_seconds=0.1,
            on_first_index_done=MagicMock(),
            stop_event=threading.Event(),
        )
        assert handler._should_ignore("/tmp/project/.git/HEAD")
        assert handler._should_ignore("/tmp/project/node_modules/foo.js")
        assert not handler._should_ignore("/tmp/project/src/app.py")

    def test_debounce_prevents_duplicate_triggers(self):
        """Multiple rapid events should only trigger one index."""
        callback = MagicMock()
        handler = _FirstFileHandler(
            project_root="/tmp/project",
            debounce_seconds=5.0,  # long debounce so timer doesn't fire
            on_first_index_done=callback,
            stop_event=threading.Event(),
        )

        # Simulate multiple file creation events
        handler._handle_event("/tmp/project/src/app.py")
        handler._handle_event("/tmp/project/src/utils.py")
        handler._handle_event("/tmp/project/src/main.py")

        # All events should be buffered, not yet triggered
        assert len(handler._pending_events) == 3
        assert not handler._triggered

        # Cancel the timer to prevent background execution
        if handler._timer:
            handler._timer.cancel()

    def test_triggered_flag_prevents_re_trigger(self):
        """Once triggered, subsequent events should be ignored."""
        handler = _FirstFileHandler(
            project_root="/tmp/project",
            debounce_seconds=0.1,
            on_first_index_done=MagicMock(),
            stop_event=threading.Event(),
        )
        handler._triggered = True

        handler._handle_event("/tmp/project/src/app.py")
        assert len(handler._pending_events) == 0
