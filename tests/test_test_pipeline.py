"""
Tests for the test pipeline safety improvements:
- Path normalization (_normalize_fix_paths)
- Test-only file filtering (_filter_test_only_files)
- Duplicate directory prefix stripping (_sanitize_filename)
"""
import pytest
from unittest.mock import MagicMock

from multi_agent_coder.executor import Executor
from multi_agent_coder.orchestrator.step_handlers import (
    _normalize_fix_paths,
    _filter_test_only_files,
)


# ── Path normalization tests ─────────────────────────────────


def _make_memory(files: dict[str, str]) -> MagicMock:
    """Create a mock FileMemory with the given files."""
    mem = MagicMock()
    mem.all_files.return_value = dict(files)
    return mem


def test_normalize_fix_paths_exact_match():
    """Exact path match should be kept unchanged."""
    memory = _make_memory({"my-app/src/index.js": "code"})
    fix = {"my-app/src/index.js": "fixed code"}
    result = _normalize_fix_paths(fix, memory)
    assert "my-app/src/index.js" in result
    assert result["my-app/src/index.js"] == "fixed code"


def test_normalize_fix_paths_suffix_remapping():
    """src/index.js should be remapped to my-app/src/index.js."""
    memory = _make_memory({"my-app/src/index.js": "code"})
    fix = {"src/index.js": "fixed code"}
    result = _normalize_fix_paths(fix, memory)
    assert "my-app/src/index.js" in result
    assert "src/index.js" not in result
    assert result["my-app/src/index.js"] == "fixed code"


def test_normalize_fix_paths_no_match():
    """Unknown paths should be kept as-is."""
    memory = _make_memory({"my-app/src/index.js": "code"})
    fix = {"lib/utils.js": "new code"}
    result = _normalize_fix_paths(fix, memory)
    assert "lib/utils.js" in result


def test_normalize_fix_paths_empty_memory():
    """With empty memory, all paths pass through unchanged."""
    memory = _make_memory({})
    fix = {"src/app.py": "code"}
    result = _normalize_fix_paths(fix, memory)
    assert result == fix


# ── Test-only file filter tests ──────────────────────────────


def test_filter_allows_test_files():
    """Files in test directories should be allowed."""
    memory = _make_memory({"src/app.py": "code"})
    test_files = {"__tests__/app.test.js": "test code"}
    fix = {"__tests__/app.test.js": "fixed test"}
    result = _filter_test_only_files(fix, test_files, memory)
    assert "__tests__/app.test.js" in result


def test_filter_blocks_package_json():
    """package.json should be blocked."""
    memory = _make_memory({"src/app.py": "code"})
    test_files = {"tests/test_app.py": "test code"}
    fix = {
        "tests/test_app.py": "fixed test",
        "package.json": '{"name": "broken"}',
    }
    result = _filter_test_only_files(fix, test_files, memory)
    assert "tests/test_app.py" in result
    assert "package.json" not in result


def test_filter_blocks_source_file_overwrite():
    """Known source files should be blocked during test fixes."""
    memory = _make_memory({"src/calculator.py": "original code"})
    test_files = {"tests/test_calc.py": "test code"}
    fix = {
        "tests/test_calc.py": "fixed test",
        "src/calculator.py": "corrupted code",
    }
    result = _filter_test_only_files(fix, test_files, memory)
    assert "tests/test_calc.py" in result
    assert "src/calculator.py" not in result


def test_filter_allows_test_naming_patterns():
    """Files with test naming patterns should be allowed."""
    memory = _make_memory({"src/app.js": "code"})
    test_files = {}
    fix = {
        "test_utils.py": "test helper code",
        "helpers.test.js": "test code",
        "spec/app_spec.rb": "test code",
    }
    result = _filter_test_only_files(fix, test_files, memory)
    assert "test_utils.py" in result
    assert "helpers.test.js" in result
    assert "spec/app_spec.rb" in result


def test_filter_blocks_protected_manifests():
    """All protected manifest files should be blocked."""
    memory = _make_memory({})
    test_files = {}
    fix = {
        "package-lock.json": "lock content",
        "yarn.lock": "yarn content",
        "requirements.txt": "deps",
        "go.mod": "module content",
        "Cargo.toml": "cargo content",
    }
    result = _filter_test_only_files(fix, test_files, memory)
    assert len(result) == 0


# ── Sanitize filename duplicate prefix tests ─────────────────


def test_sanitize_strips_duplicate_prefix():
    """my-app/my-app/src/file.js → my-app/src/file.js"""
    result = Executor._sanitize_filename("my-app/my-app/src/file.js")
    assert result == "my-app/src/file.js"


def test_sanitize_no_duplicate_prefix():
    """Normal paths should not be changed."""
    result = Executor._sanitize_filename("my-app/src/file.js")
    assert result == "my-app/src/file.js"


def test_sanitize_single_segment():
    """Single filename should not be changed."""
    result = Executor._sanitize_filename("file.js")
    assert result == "file.js"


def test_sanitize_different_prefixes():
    """Different first two segments should not be collapsed."""
    result = Executor._sanitize_filename("src/lib/utils.py")
    assert result == "src/lib/utils.py"


# ── Write files path conflict detection ──────────────────────


def test_write_files_detects_path_conflict(tmp_path):
    """Writing same basename to different dirs should log a warning."""
    files = {
        "src/app.py": "code1",
        "lib/app.py": "code2",
    }
    written = Executor.write_files(files, base_dir=str(tmp_path))
    # Both files + auto-created __init__.py files
    assert (tmp_path / "src" / "app.py").exists()
    assert (tmp_path / "lib" / "app.py").exists()
    # At least the 2 main files should be written
    main_files = [w for w in written if not w.endswith("__init__.py")]
    assert len(main_files) == 2


def test_write_files_protects_existing_package_json(tmp_path):
    """Existing package.json should not be overwritten."""
    pkg_path = tmp_path / "package.json"
    pkg_path.write_text('{"name": "original"}')

    files = {"package.json": '{"name": "corrupted"}'}
    written = Executor.write_files(files, base_dir=str(tmp_path))

    assert len(written) == 0
    assert pkg_path.read_text() == '{"name": "original"}'
