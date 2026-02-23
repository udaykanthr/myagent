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


# ── ReferenceError detection tests ───────────────────────────


def test_detect_missing_packages_reference_error_expect():
    """ReferenceError: expect is not defined → @jest/globals."""
    output = "ReferenceError: expect is not defined"
    packages = Executor.detect_missing_packages(output)
    assert "@jest/globals" in packages


def test_detect_missing_packages_reference_error_describe():
    """ReferenceError: describe is not defined → @jest/globals."""
    output = "ReferenceError: describe is not defined"
    packages = Executor.detect_missing_packages(output)
    assert "@jest/globals" in packages


def test_detect_missing_packages_reference_error_dedup():
    """Multiple ReferenceErrors for Jest globals → single @jest/globals entry."""
    output = (
        "ReferenceError: expect is not defined\n"
        "ReferenceError: describe is not defined\n"
        "ReferenceError: test is not defined"
    )
    packages = Executor.detect_missing_packages(output)
    assert packages.count("@jest/globals") == 1


def test_detect_missing_packages_reference_error_unknown():
    """ReferenceError for unknown names should NOT produce packages."""
    output = "ReferenceError: myCustomVar is not defined"
    packages = Executor.detect_missing_packages(output)
    assert len(packages) == 0


def test_detect_missing_packages_mixed_errors():
    """Mix of ModuleNotFoundError and ReferenceError."""
    output = (
        "ModuleNotFoundError: No module named 'flask'\n"
        "ReferenceError: expect is not defined"
    )
    packages = Executor.detect_missing_packages(output)
    assert "flask" in packages
    assert "@jest/globals" in packages


# ── JS project environment detection tests ───────────────────

from multi_agent_coder.orchestrator.step_handlers import _read_js_project_env
import json


def test_js_env_detection_esm(tmp_path, monkeypatch):
    """Detect ESM project from package.json."""
    pkg = {"name": "test-app", "type": "module"}
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    monkeypatch.chdir(tmp_path)

    env = _read_js_project_env()
    assert env["is_esm"] is True
    assert env["module_type"] == "module"


def test_js_env_detection_cjs(tmp_path, monkeypatch):
    """Detect CJS project (no type field or type=commonjs)."""
    pkg = {"name": "test-app"}
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    monkeypatch.chdir(tmp_path)

    env = _read_js_project_env()
    assert env["is_esm"] is False
    assert env["module_type"] == "commonjs"


def test_js_env_detection_jest_in_deps(tmp_path, monkeypatch):
    """Detect jest in devDependencies."""
    pkg = {"name": "test-app", "devDependencies": {"jest": "^29.0.0"}}
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    monkeypatch.chdir(tmp_path)

    env = _read_js_project_env()
    assert env["has_jest"] is True


def test_js_env_detection_jest_globals(tmp_path, monkeypatch):
    """Detect @jest/globals in devDependencies."""
    pkg = {"name": "test-app", "devDependencies": {
        "jest": "^29.0.0", "@jest/globals": "^29.0.0"
    }}
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    monkeypatch.chdir(tmp_path)

    env = _read_js_project_env()
    assert env["has_jest_globals"] is True


def test_js_env_detection_jest_config(tmp_path, monkeypatch):
    """Detect jest.config.js presence."""
    pkg = {"name": "test-app"}
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    (tmp_path / "jest.config.js").write_text("module.exports = {};")
    monkeypatch.chdir(tmp_path)

    env = _read_js_project_env()
    assert env["has_jest_config"] is True


def test_js_env_detection_no_package_json(tmp_path, monkeypatch):
    """No package.json → all defaults."""
    monkeypatch.chdir(tmp_path)

    env = _read_js_project_env()
    assert env["is_esm"] is False
    assert env["has_jest"] is False
    assert env["has_jest_config"] is False


# ── ESM-aware JS test rules tests ────────────────────────────

from multi_agent_coder.agents.tester import TesterAgent as AppTesterAgent
from multi_agent_coder.language import get_test_framework


def test_js_test_rules_esm_includes_jest_globals_import():
    """ESM projects should include @jest/globals import instruction."""
    fw = get_test_framework("javascript")
    env = {"is_esm": True}
    rules = AppTesterAgent._js_test_rules("javascript", fw, env_info=env)
    assert "@jest/globals" in rules
    assert "import" in rules.lower()
    assert "require" not in rules or "Do NOT use `require()`" in rules


def test_js_test_rules_cjs_no_globals_import():
    """CJS projects should say globals are available without import."""
    fw = get_test_framework("javascript")
    env = {"is_esm": False}
    rules = AppTesterAgent._js_test_rules("javascript", fw, env_info=env)
    assert "globally" in rules.lower() or "available" in rules.lower()
    assert "require" in rules


def test_js_test_rules_no_env_defaults_to_cjs():
    """No env_info should default to CJS behavior."""
    fw = get_test_framework("javascript")
    rules = AppTesterAgent._js_test_rules("javascript", fw)
    assert "require" in rules


# ── Protected file stripping tests ───────────────────────────

from multi_agent_coder.orchestrator.step_handlers import _strip_protected_files
from multi_agent_coder.orchestrator.memory import FileMemory


def test_strip_protected_files_blocks_existing_package_json_no_additions(tmp_path, monkeypatch):
    """package.json with no new deps should not be written (no changes to merge)."""
    existing = '{"name": "my-app", "dependencies": {"react": "^18.0.0"}}'
    (tmp_path / "package.json").write_text(existing)
    monkeypatch.chdir(tmp_path)

    files = {
        "package.json": '{"name": "my-app", "dependencies": {"react": "^18.0.0"}}',
        "src/App.jsx": "export default function App() {}",
    }
    result = _strip_protected_files(files)
    # No new additions → skipped
    assert "package.json" not in result
    assert "src/App.jsx" in result


def test_strip_protected_files_allows_new_package_json(tmp_path, monkeypatch):
    """package.json should be allowed through for NEW projects (no file on disk)."""
    monkeypatch.chdir(tmp_path)

    files = {
        "package.json": '{"name": "new-app"}',
        "index.js": "console.log('hello')",
    }
    result = _strip_protected_files(files)
    assert "package.json" in result
    assert "index.js" in result


def test_strip_protected_files_blocks_lock_files(tmp_path, monkeypatch):
    """Lock files should always be fully blocked."""
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 2}')
    (tmp_path / "yarn.lock").write_text("# yarn lock")
    monkeypatch.chdir(tmp_path)

    files = {
        "package-lock.json": '{"corrupted": true}',
        "yarn.lock": "corrupted lock",
        "main.js": "console.log('hello')",
    }
    result = _strip_protected_files(files)
    assert "package-lock.json" not in result
    assert "yarn.lock" not in result
    assert "main.js" in result


def test_strip_protected_files_empty_input():
    """Empty dict should return empty dict."""
    assert _strip_protected_files({}) == {}


# ── Smart merge: package.json tests ──────────────────────────

from multi_agent_coder.orchestrator.step_handlers import (
    _smart_merge_json_manifest,
    _smart_merge_requirements_txt,
)


def test_smart_merge_adds_new_dependency():
    """New dependency in LLM output should be merged into existing."""
    existing = json.dumps({
        "name": "my-app",
        "dependencies": {"react": "^18.0.0"}
    }, indent=2)
    llm_output = json.dumps({
        "name": "my-app",
        "dependencies": {"react": "^18.0.0", "axios": "^1.4.0"}
    }, indent=2)

    result = _smart_merge_json_manifest(existing, llm_output, "package.json")
    assert result is not None
    merged = json.loads(result)
    assert merged["dependencies"]["react"] == "^18.0.0"
    assert merged["dependencies"]["axios"] == "^1.4.0"


def test_smart_merge_adds_new_dev_dependency():
    """New devDependency should be merged."""
    existing = json.dumps({
        "name": "my-app",
        "dependencies": {"express": "^4.0.0"}
    }, indent=2)
    llm_output = json.dumps({
        "name": "my-app",
        "dependencies": {"express": "^4.0.0"},
        "devDependencies": {"jest": "^29.0.0"}
    }, indent=2)

    result = _smart_merge_json_manifest(existing, llm_output, "package.json")
    assert result is not None
    merged = json.loads(result)
    assert merged["dependencies"]["express"] == "^4.0.0"
    assert merged["devDependencies"]["jest"] == "^29.0.0"


def test_smart_merge_blocks_removed_dependency():
    """Removing a dependency should be blocked (kept in result)."""
    existing = json.dumps({
        "name": "my-app",
        "dependencies": {"react": "^18.0.0", "lodash": "^4.17.0"}
    }, indent=2)
    llm_output = json.dumps({
        "name": "my-app",
        "dependencies": {"react": "^18.0.0"}  # lodash removed
    }, indent=2)

    result = _smart_merge_json_manifest(existing, llm_output, "package.json")
    # No new additions, so result equals existing
    assert result == existing


def test_smart_merge_blocks_version_change():
    """Version changes should be blocked (keep original version)."""
    existing = json.dumps({
        "name": "my-app",
        "dependencies": {"react": "^18.0.0"}
    }, indent=2)
    llm_output = json.dumps({
        "name": "my-app",
        "dependencies": {"react": "^17.0.0"}  # downgrade attempt
    }, indent=2)

    result = _smart_merge_json_manifest(existing, llm_output, "package.json")
    # No new additions, so result equals existing
    assert result == existing


def test_smart_merge_adds_new_script():
    """New script entry should be merged."""
    existing = json.dumps({
        "name": "my-app",
        "scripts": {"start": "node index.js"}
    }, indent=2)
    llm_output = json.dumps({
        "name": "my-app",
        "scripts": {"start": "node index.js", "test": "jest"}
    }, indent=2)

    result = _smart_merge_json_manifest(existing, llm_output, "package.json")
    assert result is not None
    merged = json.loads(result)
    assert merged["scripts"]["start"] == "node index.js"
    assert merged["scripts"]["test"] == "jest"


def test_smart_merge_preserves_existing_keys():
    """Top-level keys like name, version should stay from existing file."""
    existing = json.dumps({
        "name": "my-app",
        "version": "1.0.0",
        "dependencies": {"react": "^18.0.0"}
    }, indent=2)
    llm_output = json.dumps({
        "name": "corrupted-name",
        "dependencies": {"react": "^18.0.0", "axios": "^1.0.0"}
    }, indent=2)

    result = _smart_merge_json_manifest(existing, llm_output, "package.json")
    assert result is not None
    merged = json.loads(result)
    # name stays from existing (not in merge_sections)
    assert merged["name"] == "my-app"
    # version preserved
    assert merged["version"] == "1.0.0"
    # new dep added
    assert merged["dependencies"]["axios"] == "^1.0.0"


def test_smart_merge_fallback_on_invalid_json():
    """Invalid JSON from LLM should return None (trigger fallback block)."""
    existing = '{"name": "my-app"}'
    llm_output = "this is not json at all"

    result = _smart_merge_json_manifest(existing, llm_output, "package.json")
    assert result is None


# ── Smart merge: requirements.txt tests ──────────────────────


def test_smart_merge_requirements_adds_new_package():
    """New package in LLM output should be appended."""
    existing = "flask==2.3.0\nrequests==2.31.0\n"
    llm_output = "flask==2.3.0\nrequests==2.31.0\nnumpy==1.25.0\n"

    result = _smart_merge_requirements_txt(existing, llm_output, "requirements.txt")
    assert result is not None
    assert "numpy==1.25.0" in result
    assert "flask==2.3.0" in result
    assert "requests==2.31.0" in result


def test_smart_merge_requirements_blocks_version_change():
    """Version changes in requirements.txt should be blocked."""
    existing = "flask==2.3.0\n"
    llm_output = "flask==3.0.0\n"  # version changed

    result = _smart_merge_requirements_txt(existing, llm_output, "requirements.txt")
    # No new packages → returns existing unchanged
    assert result == existing


def test_smart_merge_requirements_blocks_removal():
    """Removed packages should stay (returns existing if no additions)."""
    existing = "flask==2.3.0\nrequests==2.31.0\n"
    llm_output = "flask==2.3.0\n"  # requests removed

    result = _smart_merge_requirements_txt(existing, llm_output, "requirements.txt")
    assert result == existing


# ── Smart merge integration with _strip_protected_files ──────


def test_strip_merges_new_dependency_into_package_json(tmp_path, monkeypatch):
    """_strip_protected_files should merge new deps into existing package.json."""
    existing = json.dumps({
        "name": "my-app",
        "dependencies": {"react": "^18.0.0"}
    }, indent=2)
    (tmp_path / "package.json").write_text(existing)
    monkeypatch.chdir(tmp_path)

    llm_version = json.dumps({
        "name": "my-app",
        "dependencies": {"react": "^18.0.0", "axios": "^1.4.0"}
    }, indent=2)

    files = {
        "package.json": llm_version,
        "src/App.jsx": "code",
    }
    result = _strip_protected_files(files)
    # package.json should be in result with merged content
    assert "package.json" in result
    merged = json.loads(result["package.json"])
    assert merged["dependencies"]["axios"] == "^1.4.0"
    assert merged["dependencies"]["react"] == "^18.0.0"
    assert "src/App.jsx" in result


def test_strip_merges_new_package_into_requirements_txt(tmp_path, monkeypatch):
    """_strip_protected_files should merge new packages into requirements.txt."""
    (tmp_path / "requirements.txt").write_text("flask==2.3.0\n")
    monkeypatch.chdir(tmp_path)

    files = {
        "requirements.txt": "flask==2.3.0\nnumpy==1.25.0\n",
        "app.py": "import flask",
    }
    result = _strip_protected_files(files)
    assert "requirements.txt" in result
    assert "numpy==1.25.0" in result["requirements.txt"]
    assert "flask==2.3.0" in result["requirements.txt"]


def test_memory_update_skips_existing_protected_files(tmp_path, monkeypatch):
    """FileMemory.update() should skip package.json when it already exists on disk."""
    pkg_path = tmp_path / "package.json"
    pkg_path.write_text('{"name": "original"}')
    monkeypatch.chdir(tmp_path)

    mem = FileMemory()
    mem.update({
        "package.json": '{"name": "corrupted"}',
        "src/index.js": "console.log('hello')",
    })

    assert mem.get("package.json") is None  # blocked
    assert mem.get("src/index.js") == "console.log('hello')"  # allowed


def test_memory_update_allows_new_protected_files(tmp_path, monkeypatch):
    """FileMemory.update() should allow package.json for new projects."""
    monkeypatch.chdir(tmp_path)

    mem = FileMemory()
    mem.update({
        "package.json": '{"name": "new-app"}',
    })

    assert mem.get("package.json") == '{"name": "new-app"}'


