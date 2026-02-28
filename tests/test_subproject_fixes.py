"""Tests for _prefix_subproject_paths and coder prompt file extensions.

Covers:
- _prefix_subproject_paths: prefixing new files with sub-project root
- CoderAgent prompt example using correct file extensions
- CMD-output-based sub-project detection
- _NON_SUBPROJECT_DIRS blocklist
"""

import os
import re
from unittest.mock import patch

from multi_agent_coder.orchestrator.step_handlers import (
    _prefix_subproject_paths,
    _detect_subproject_root,
)
from multi_agent_coder.orchestrator.memory import FileMemory
from multi_agent_coder.agents.coder import _LANG_TO_EXT


# ─── Helpers ────────────────────────────────────────────────────────


def _make_memory(files: dict[str, str]) -> FileMemory:
    """Create a FileMemory with given files (no embeddings)."""
    mem = FileMemory(embedding_store=None)
    mem.update(files)
    return mem


# ─── _prefix_subproject_paths ───────────────────────────────────────


class TestPrefixSubprojectPaths:
    def test_prefixes_new_files(self):
        """New files without sub-project prefix get prefixed."""
        memory = _make_memory({
            "my-app/src/App.js": "A",
            "my-app/package.json": "{}",
        })

        files = {
            "components/Header.tsx": "header code",
            "components/Footer.tsx": "footer code",
        }
        result = _prefix_subproject_paths(files, "my-app", memory)

        assert "my-app/components/Header.tsx" in result
        assert "my-app/components/Footer.tsx" in result
        assert "components/Header.tsx" not in result
        assert "components/Footer.tsx" not in result

    def test_does_not_double_prefix(self):
        """Files already prefixed with sub-project root are unchanged."""
        memory = _make_memory({
            "my-app/src/App.js": "A",
        })

        files = {"my-app/src/App.js": "updated A"}
        result = _prefix_subproject_paths(files, "my-app", memory)

        assert "my-app/src/App.js" in result
        assert len(result) == 1

    def test_skips_internal_files(self):
        """Internal tracking files (_cmd_output/) are not prefixed."""
        memory = _make_memory({
            "my-app/src/App.js": "A",
        })

        files = {
            "_cmd_output/step_1.txt": "output",
            "src/NewFile.js": "new code",
        }
        result = _prefix_subproject_paths(files, "my-app", memory)

        assert "_cmd_output/step_1.txt" in result
        assert "my-app/src/NewFile.js" in result
        assert "my-app/_cmd_output/step_1.txt" not in result

    def test_skips_known_memory_files(self):
        """Files that already exist in memory by exact path are untouched."""
        memory = _make_memory({
            "src/utils.js": "utility code",
            "my-app/src/App.js": "A",
        })

        files = {"src/utils.js": "updated utility"}
        result = _prefix_subproject_paths(files, "my-app", memory)

        assert "src/utils.js" in result
        assert "my-app/src/utils.js" not in result

    def test_no_subproject_returns_unchanged(self):
        """When subproject is empty string, files are unchanged."""
        memory = _make_memory({})
        files = {"src/App.js": "code"}
        result = _prefix_subproject_paths(files, "", memory)

        assert result == files

    def test_handles_trailing_slash(self):
        """Sub-project root with trailing slash doesn't double-slash."""
        memory = _make_memory({
            "my-app/src/App.js": "A",
        })

        files = {"src/index.js": "new code"}
        result = _prefix_subproject_paths(files, "my-app/", memory)

        assert "my-app/src/index.js" in result
        assert "my-app//src/index.js" not in result


# ─── CMD-output-based sub-project detection ─────────────────────────


class TestCmdOutputSubprojectDetection:
    """Tests for _detect_subproject_root's CMD output parsing (Fallback 0).

    This is the critical fix: when only _cmd_output/ entries exist in memory
    (no source files yet), the function must still detect the sub-project
    by parsing the create command from the CMD output.
    """

    def test_detects_create_next_app(self):
        """Detect sub-project from 'npx create-next-app my-app'."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": (
                "$ npx create-next-app@latest my-bootstrap-website --typescript --yes\n"
                "Creating a new Next.js app...\n"
                "Success!"
            ),
        })
        with patch("os.path.isdir", return_value=True), \
             patch("os.path.isfile", return_value=True):
            result = _detect_subproject_root(memory)
        assert result == "my-bootstrap-website"

    def test_detects_create_react_app(self):
        """Detect sub-project from 'npx create-react-app my-app'."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": (
                "$ npx create-react-app my-react-app\n"
                "Creating a new React app..."
            ),
        })
        with patch("os.path.isdir", return_value=True), \
             patch("os.path.isfile", return_value=True):
            result = _detect_subproject_root(memory)
        assert result == "my-react-app"

    def test_ignores_dot_slash_current_dir(self):
        """When create-next-app uses ./ (current dir), return None."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": (
                "$ npx create-next-app ./ --typescript --yes\n"
                "Creating a new Next.js app..."
            ),
        })
        result = _detect_subproject_root(memory)
        assert result is None

    def test_ignores_dot_current_dir(self):
        """When create-next-app uses . (current dir), return None."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": (
                "$ npx create-next-app . --typescript --yes\n"
                "Creating a new Next.js app..."
            ),
        })
        result = _detect_subproject_root(memory)
        assert result is None

    def test_requires_manifest_on_disk(self):
        """Directory must have a manifest file to be considered a sub-project."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": (
                "$ npx create-next-app my-app --yes\n"
                "Creating..."
            ),
        })
        # Dir exists but no manifest file
        with patch("os.path.isdir", return_value=True), \
             patch("os.path.isfile", return_value=False):
            result = _detect_subproject_root(memory)
        assert result is None

    def test_detects_from_fix_output(self):
        """Also parse _fix_output/ entries (failed + retried commands)."""
        memory = _make_memory({
            "_cmd_output/step_1.txt": "$ cd foo\n",
            "_fix_output/step_1.txt": (
                "$ npx create-next-app ./new-project --yes\n"
                "Success!"
            ),
        })
        with patch("os.path.isdir", return_value=True), \
             patch("os.path.isfile", return_value=True):
            result = _detect_subproject_root(memory)
        assert result == "new-project"


# ─── Non-subproject directory blocklist ──────────────────────────────


class TestNonSubprojectDirsBlocklist:
    """Tests that common source dirs (src, app, components, etc.) are NOT
    incorrectly identified as sub-project roots."""

    def test_src_is_not_subproject(self):
        """src/ should never be returned as a sub-project root."""
        memory = _make_memory({
            "src/components/Header.tsx": "A",
            "src/components/Footer.tsx": "B",
        })
        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)
        assert result is None

    def test_components_is_not_subproject(self):
        """components/ should never be returned as a sub-project root."""
        memory = _make_memory({
            "components/Header.tsx": "A",
            "components/Footer.tsx": "B",
        })
        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)
        assert result is None

    def test_pages_is_not_subproject(self):
        """pages/ should never be returned as a sub-project root."""
        memory = _make_memory({
            "pages/_app.tsx": "A",
            "pages/index.tsx": "B",
        })
        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)
        assert result is None

    def test_real_project_name_is_detected(self):
        """A real project directory name (my-app) should still be detected."""
        memory = _make_memory({
            "my-app/src/App.tsx": "A",
            "my-app/package.json": "{}",
        })
        with patch("os.path.isdir", return_value=True):
            result = _detect_subproject_root(memory)
        assert result == "my-app"


# ─── Coder prompt file extension ────────────────────────────────────


class TestCoderPromptExtension:
    def test_lang_to_ext_has_all_common_languages(self):
        """Mapping covers the most common languages."""
        assert _LANG_TO_EXT["python"] == ".py"
        assert _LANG_TO_EXT["javascript"] == ".js"
        assert _LANG_TO_EXT["typescript"] == ".ts"
        assert _LANG_TO_EXT["go"] == ".go"
        assert _LANG_TO_EXT["rust"] == ".rs"
        assert _LANG_TO_EXT["java"] == ".java"

    def test_no_dotpython_extension(self):
        """The mapping should NEVER produce .python as extension."""
        for lang, ext in _LANG_TO_EXT.items():
            assert ext != ".python", f"Language {lang} maps to .python"
            assert ext != ".javascript", f"Language {lang} maps to .javascript"
            assert ext != ".typescript", f"Language {lang} maps to .typescript"

    def test_all_extensions_are_short(self):
        """File extensions should be short (max 6 chars incl. dot)."""
        for lang, ext in _LANG_TO_EXT.items():
            assert ext.startswith("."), f"Extension for {lang} missing dot: {ext}"
            assert len(ext) <= 6, f"Extension for {lang} too long: {ext}"

