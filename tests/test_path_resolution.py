"""Tests for sub-project path resolution fix.

Covers:
- _normalize_fix_paths: remapping truncated paths to full sub-project paths
- _detect_subproject_root: detecting common subdirectory prefix in memory
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from multi_agent_coder.orchestrator.step_handlers import (
    _normalize_fix_paths,
    _detect_subproject_root,
)
from multi_agent_coder.orchestrator.memory import FileMemory


# ─── _normalize_fix_paths ───────────────────────────────────────────


def _make_memory(files: dict[str, str]) -> FileMemory:
    """Create a FileMemory with given files (no embeddings)."""
    mem = FileMemory(embedding_store=None)
    mem.update(files)
    return mem


class TestNormalizeFixPaths:
    def test_remaps_suffix_to_full_path(self):
        """src/App.js → my-app/src/App.js when memory knows the full path."""
        memory = _make_memory({
            "my-app/src/App.js": "export default App;",
            "my-app/src/index.js": "import App from './App';",
            "my-app/package.json": "{}",
        })

        fix_files = {"src/App.js": "// updated App"}
        result = _normalize_fix_paths(fix_files, memory)

        assert "my-app/src/App.js" in result
        assert "src/App.js" not in result
        assert result["my-app/src/App.js"] == "// updated App"

    def test_keeps_exact_match(self):
        """If the path already matches a known file, keep it as-is."""
        memory = _make_memory({
            "my-app/src/App.js": "export default App;",
        })

        fix_files = {"my-app/src/App.js": "// updated"}
        result = _normalize_fix_paths(fix_files, memory)

        assert "my-app/src/App.js" in result
        assert len(result) == 1

    def test_keeps_unknown_path(self):
        """If the path doesn't match any known file, keep it unchanged."""
        memory = _make_memory({
            "my-app/src/App.js": "export default App;",
        })

        fix_files = {"src/NewComponent.js": "// new"}
        result = _normalize_fix_paths(fix_files, memory)

        assert "src/NewComponent.js" in result
        assert len(result) == 1

    def test_no_remap_with_empty_memory(self):
        """No remapping when memory is empty."""
        memory = _make_memory({})

        fix_files = {"src/App.js": "// code"}
        result = _normalize_fix_paths(fix_files, memory)

        assert result == fix_files

    def test_multiple_files_remapped(self):
        """Multiple truncated paths get remapped correctly."""
        memory = _make_memory({
            "my-app/src/App.js": "A",
            "my-app/src/index.js": "B",
            "my-app/public/index.html": "C",
        })

        fix_files = {
            "src/App.js": "A updated",
            "src/index.js": "B updated",
        }
        result = _normalize_fix_paths(fix_files, memory)

        assert "my-app/src/App.js" in result
        assert "my-app/src/index.js" in result
        assert len(result) == 2


# ─── _detect_subproject_root ────────────────────────────────────────


class TestDetectSubprojectRoot:
    def test_detects_common_prefix(self, tmp_path):
        """When all files share a common first directory, detect it."""
        # Create the sub-project directory on disk
        subdir = tmp_path / "my-app"
        subdir.mkdir()

        memory = _make_memory({
            "my-app/src/App.js": "A",
            "my-app/src/index.js": "B",
            "my-app/package.json": "{}",
        })

        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)

        assert result == "my-app"

    def test_returns_none_for_multiple_roots(self):
        """When files span multiple top-level dirs, no subproject detected."""
        memory = _make_memory({
            "frontend/src/App.js": "A",
            "backend/src/server.js": "B",
        })

        result = _detect_subproject_root(memory)
        assert result is None

    def test_returns_none_for_flat_files(self):
        """Files without subdirectories → no subproject."""
        memory = _make_memory({
            "index.js": "A",
            "package.json": "{}",
        })

        result = _detect_subproject_root(memory)
        assert result is None

    def test_ignores_cmd_output(self):
        """Internal _cmd_output files should be excluded."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": "$ npx create-react-app my-app",
            "my-app/src/App.js": "A",
            "my-app/package.json": "{}",
        })

        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)

        assert result == "my-app"

    def test_returns_none_for_empty_memory(self):
        """Empty memory → no subproject."""
        memory = _make_memory({})
        result = _detect_subproject_root(memory)
        assert result is None

    def test_returns_none_if_dir_doesnt_exist(self):
        """Even if prefix is consistent, dir must exist on disk."""
        memory = _make_memory({
            "nonexistent-app/src/App.js": "A",
            "nonexistent-app/package.json": "{}",
        })

        with patch("os.path.isdir", return_value=False):
            result = _detect_subproject_root(memory)

        assert result is None

    def test_detects_subproject_via_manifest_fallback(self):
        """When memory has multiple top-level dirs, use the one with package.json."""
        memory = _make_memory({
            "dashboard-app/src/App.js": "A",
            "dashboard-app/package.json": "{}",
            "other-folder/data.txt": "some data",
        })

        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)

        assert result == "dashboard-app"

    def test_returns_none_if_multiple_manifests(self):
        """If multiple directories have manifests, fallback cannot cleanly resolve."""
        memory = _make_memory({
            "frontend/package.json": "{}",
            "backend/package.json": "{}",
        })

        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)

        assert result is None

    def test_ignores_fix_output(self):
        """Internal _fix_output files should be excluded like _cmd_output."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": "$ npx create-react-app my-app",
            "_fix_output/step_2.txt": "$ npm install --save-dev jest",
            "my-app/src/App.js": "A",
            "my-app/src/index.js": "B",
        })

        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)

        assert result == "my-app"

    def test_ignores_all_underscore_prefixed_paths(self):
        """Any path starting with _ should be treated as internal."""
        memory = _make_memory({
            "_internal/tracking.txt": "data",
            "_cmd_output/step_1.txt": "output",
            "_fix_output/step_2.txt": "fix",
            "dashboard-app/src/App.jsx": "A",
            "dashboard-app/src/index.js": "B",
        })

        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)

        assert result == "dashboard-app"

    def test_disk_manifest_fallback(self, tmp_path):
        """When package.json is not in memory (protected), detect via disk."""
        # Create subproject dir with package.json on disk
        subdir = tmp_path / "dashboard-app"
        subdir.mkdir()
        (subdir / "package.json").write_text("{}")

        memory = _make_memory({
            "dashboard-app/src/App.jsx": "A",
            "dashboard-app/src/__tests__/App.test.jsx": "test",
            "other/config.txt": "cfg",
        })

        orig_isdir = os.path.isdir
        orig_isfile = os.path.isfile

        def mock_isdir(path):
            if path == "dashboard-app":
                return True
            if path == "other":
                return True
            return orig_isdir(path)

        def mock_isfile(path):
            if path == os.path.join("dashboard-app", "package.json"):
                return True
            return orig_isfile(path)

        with patch("os.path.isdir", side_effect=mock_isdir), \
             patch("os.path.isfile", side_effect=mock_isfile):
            result = _detect_subproject_root(memory)

        assert result == "dashboard-app"

    def test_majority_fallback(self):
        """When files span dirs but one has >70%, pick the majority."""
        files = {f"my-app/src/file{i}.js": f"code{i}" for i in range(8)}
        files["other/readme.txt"] = "readme"
        memory = _make_memory(files)

        with patch("os.path.isdir", return_value=True), \
             patch("os.path.isfile", return_value=False):
            result = _detect_subproject_root(memory)

        assert result == "my-app"
