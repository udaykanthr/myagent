"""
Step handlers — CMD, CODE, and TEST step execution logic.
"""
# Separate retry limit for test generation (lower than code to avoid pipeline halts)
MAX_TEST_GEN_RETRIES = 2

import ast
import json
import os
import re
import shutil

from ..config import Config
from ..agents.coder import CoderAgent
from ..agents.reviewer import ReviewerAgent
from ..agents.tester import TesterAgent
from ..executor import Executor
from ..cli_display import CLIDisplay, token_tracker, log
from ..language import (
    get_code_block_lang, get_test_framework, detect_test_runner,
    detect_language_from_files, EXTENSION_MAP,
)
from .test_analyzer import perform_baseline_test_analysis, _count_test_failures, _identify_test_files

from .memory import FileMemory
from .classification import (
    _extract_command_from_step, _extract_commands_from_text,
    _looks_like_command, _cleanup_shell_command, resolve_cmd_placeholders,
)

from ..diff_display import show_diffs, prompt_diff_approval, _detect_hazards


MAX_STEP_RETRIES = 2  # Used for code steps and test run/fix attempts

# Map test runner binary → install command
_RUNNER_INSTALL = {
    "pytest": "pip install pytest",
    "jest": "npm install --save-dev jest",
    "npx": "npm install --save-dev jest",
    "mocha": "npm install --save-dev mocha",
    "vitest": "npm install --save-dev vitest",
    "go": None,  # built-in, no install needed
    "cargo": None,
    "rspec": "gem install rspec",
    "phpunit": "composer require --dev phpunit/phpunit",
}

# Manifest file → (package_manager_name, install_all_cmd, install_pkgs_prefix)
# Ordered: most specific manifests first (pyproject.toml before setup.py, etc.)
_MANIFEST_TO_PM: list[tuple[str, str, str, str]] = [
    # manifest filename  , pm name   , install-all cmd              , add-package prefix
    ("package.json",       "npm",      "npm install",                 "npm install"),
    ("pyproject.toml",     "pip",      "pip install -e .",            "pip install"),
    ("requirements.txt",   "pip",      "pip install -r requirements.txt", "pip install"),
    ("setup.py",           "pip",      "pip install -e .",            "pip install"),
    ("Gemfile",            "bundler",  "bundle install",              "gem install"),
    ("go.mod",             "go",       "go mod download",             "go get"),
    ("Cargo.toml",         "cargo",    "cargo build --quiet",         "cargo add"),
    ("composer.json",      "composer", "composer install",            "composer require"),
]

# Package manager name → default add-package prefix (fallback when no manifest found)
_PM_INSTALL_PREFIX: dict[str, str] = {
    "pip":      "pip install",
    "npm":      "npm install",
    "bundler":  "gem install",
    "gem":      "gem install",
    "go":       "go get",
    "cargo":    "cargo add",
    "composer": "composer require",
}


def _detect_package_manager(cwd: str | None = None) -> tuple[str | None, str | None, str | None]:
    """Detect the package manager in *cwd* by scanning manifest files.

    Returns (pm_name, install_all_cmd, install_pkgs_prefix) or (None, None, None).
    """
    root = cwd or "."
    for manifest, pm, install_all, install_prefix in _MANIFEST_TO_PM:
        if os.path.isfile(os.path.join(root, manifest)):
            return pm, install_all, install_prefix
    # Fallback: check for .csproj files (C# / .NET)
    try:
        for fname in os.listdir(root):
            if fname.endswith(".csproj"):
                return "dotnet", "dotnet restore", "dotnet add package"
    except OSError:
        pass
    return None, None, None


_DEV_ONLY_PACKAGES = re.compile(
    r'^(@types/|eslint|prettier|@eslint|stylelint|'
    r'jest|vitest|mocha|chai|sinon|nyc|c8|cypress|playwright|'
    r'@testing-library/|@jest/|ts-jest|babel-jest|'
    r'webpack-dev-server|@vitejs/plugin-|vite$|'
    r'typescript$|ts-node|tsx$|nodemon|concurrently)',
    re.IGNORECASE,
)


def _ensure_packages_installed(
    project_context,
    executor: Executor,
    memory: FileMemory,
    display: CLIDisplay,
    step_idx: int,
    subproject_cwd: str | None = None,
    language: str | None = None,
) -> None:
    """Proactively install missing packages before code/test/server steps.

    Reads the project manifest (``package.json`` or ``requirements.txt``)
    from the **subproject root** to determine what is currently installed,
    then compares against ``project_context.required_packages`` and runs
    bulk installs for anything missing.

    Designed to be called just-in-time — i.e. after the project scaffold
    has been created by earlier CMD steps but before the first CODE, TEST,
    or dev-server step runs.

    Key behaviours:
    - **One-time**: sets ``memory._packages_preinstalled`` to skip re-runs.
    - **Non-fatal**: failures are logged as warnings, never halt the pipeline.
    - **Subproject-aware**: installs inside the correct ``cwd``.
    - **npm**: splits into ``npm install`` (production) and
      ``npm install --save-dev`` (dev-only) so packages land in the
      correct section of ``package.json``.
    """
    # Already ran for this pipeline execution
    if getattr(memory, '_packages_preinstalled', False):
        return

    if project_context is None:
        return

    required = getattr(project_context, 'required_packages', [])
    if not required:
        return

    # Detect the project's package manager from manifest files.
    root = subproject_cwd or "."
    pm_name, _, install_prefix = _detect_package_manager(root)

    if not install_prefix:
        # No manifest found yet — project may not be scaffolded
        return

    currently_installed: set[str] = set()

    # Read already-installed packages from the manifest to avoid reinstalling.
    pkg_json_path = os.path.join(root, "package.json")
    reqs_txt_path = os.path.join(root, "requirements.txt")
    gemfile_lock_path = os.path.join(root, "Gemfile.lock")
    cargo_lock_path = os.path.join(root, "Cargo.lock")
    composer_lock_path = os.path.join(root, "composer.lock")

    if os.path.isfile(pkg_json_path):
        try:
            with open(pkg_json_path, "r", encoding="utf-8") as f:
                pkg = json.loads(f.read())
            currently_installed.update(pkg.get("dependencies", {}).keys())
            currently_installed.update(pkg.get("devDependencies", {}).keys())
        except Exception:
            pass
    elif os.path.isfile(reqs_txt_path):
        try:
            with open(reqs_txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        pkg_name = re.split(r'[>=<~!\[]', line)[0].strip()
                        if pkg_name:
                            currently_installed.add(pkg_name.lower())
        except Exception:
            pass
    elif os.path.isfile(gemfile_lock_path):
        # Gemfile.lock: lines like "    rack (2.2.3)" inside GEM section
        try:
            with open(gemfile_lock_path, "r", encoding="utf-8") as f:
                for line in f:
                    m = re.match(r'^\s{4}(\S+)\s+\(', line)
                    if m:
                        currently_installed.add(m.group(1).lower())
        except Exception:
            pass
    elif os.path.isfile(cargo_lock_path):
        # Cargo.lock: lines like 'name = "serde"'
        try:
            with open(cargo_lock_path, "r", encoding="utf-8") as f:
                for line in f:
                    m = re.match(r'^name\s*=\s*"(.+)"', line)
                    if m:
                        currently_installed.add(m.group(1).lower())
        except Exception:
            pass
    elif os.path.isfile(composer_lock_path):
        # composer.lock: JSON with packages[].name
        try:
            with open(composer_lock_path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            for pkg in data.get("packages", []) + data.get("packages-dev", []):
                if "name" in pkg:
                    currently_installed.add(pkg["name"].lower())
        except Exception:
            pass

    install_tool = install_prefix

    # Determine what's truly missing right now (case-insensitive for pip/gem/cargo)
    missing = [
        pkg for pkg in required
        if pkg.lower() not in currently_installed
        and pkg not in ('--save-dev', '--save', '-D', '-g')
    ]

    if not missing:
        memory._packages_preinstalled = True
        return

    log.info(f"[Pre-install] Installing {len(missing)} missing package(s): "
             f"{', '.join(missing)}")
    display.step_info(step_idx,
                      f"Pre-installing {len(missing)} missing package(s)...")

    # For npm: split into production deps and dev-only deps so they land
    # in the correct section of package.json.
    if pm_name == "npm":
        prod_pkgs = [p for p in missing if not _DEV_ONLY_PACKAGES.match(p)]
        dev_pkgs = [p for p in missing if _DEV_ONLY_PACKAGES.match(p)]

        all_ok = True
        combined_output = ""

        if prod_pkgs:
            cmd = f"npm install {' '.join(prod_pkgs)}"
            log.info(f"[Pre-install] Production: {cmd}")
            ok, output = executor.run_command(cmd, cwd=subproject_cwd)
            all_ok &= ok
            combined_output += output + "\n"

        if dev_pkgs:
            cmd = f"npm install --save-dev {' '.join(dev_pkgs)}"
            log.info(f"[Pre-install] Dev: {cmd}")
            ok, output = executor.run_command(cmd, cwd=subproject_cwd)
            all_ok &= ok
            combined_output += output + "\n"

        ok = all_ok
        output = combined_output
    else:
        cmd = f"{install_tool} {' '.join(missing)}"
        ok, output = executor.run_command(cmd, cwd=subproject_cwd)

    if ok:
        log.info(f"[Pre-install] Successfully installed: {', '.join(missing)}")
        display.step_info(step_idx, f"Pre-installed {len(missing)} package(s)")
        # Update project_context so downstream agents see the new state
        for pkg in missing:
            if pkg not in project_context.installed_packages:
                project_context.installed_packages.append(pkg)
        project_context.missing_packages = [
            p for p in project_context.missing_packages
            if p not in missing
        ]
        # Refresh version snapshot so agents target the installed versions
        from .api_grounding import refresh_installed_versions
        try:
            refresh_installed_versions(
                project_context, executor=executor, cwd=subproject_cwd,
                language=language)
        except Exception as _rv_exc:
            log.debug(f"[ApiGrounding] Version refresh failed: {_rv_exc}")
    else:
        log.warning(f"[Pre-install] Bulk install failed (non-fatal): "
                    f"{output[:300]}")
        display.step_info(step_idx, "Pre-install failed (non-fatal)")

    # Mark as done regardless of success — individual install steps will retry
    memory._packages_preinstalled = True


def _get_runner_install_cmd(runner: str, cwd: str | None = None) -> str | None:
    """Return the install command for a test runner binary.

    Looks up the explicit map first, then falls back to the package manager
    detected in *cwd* so that non-Python runners use the right tool.
    Returns ``None`` for tools that must be installed manually (e.g. ``go``,
    ``cargo``).
    """
    if runner in _RUNNER_INSTALL:
        return _RUNNER_INSTALL[runner]
    # Detect the project's package manager and use its prefix
    pm, _, install_prefix = _detect_package_manager(cwd)
    if install_prefix:
        return f"{install_prefix} {runner}"
    # Ultimate fallback: pip (Python project assumed)
    return f"pip install {runner}"


# `python -m unittest ...` anywhere in a plan-declared acceptance command.
_UNITTEST_GATE_RE = re.compile(
    r'\bpython3?(?:\.exe)?\s+-m\s+unittest\b', re.IGNORECASE)


def _plan_declared_test_runner(plan_step, language: str | None) -> str | None:
    """The test runner the plan's own gate uses, when it overrides the default.

    Returns ``None`` when the plan says nothing useful, so the caller keeps
    the language default.

    Only unittest is honoured, and deliberately so: it ships with CPython,
    so adopting it can never leave the step waiting on an install, and
    pytest-flavoured tests do not satisfy a `python -m unittest` gate
    (bare test functions and fixtures are invisible to it) while
    unittest-flavoured tests satisfy both.
    """
    if plan_step is None or (language or "python") != "python":
        return None
    gate = getattr(plan_step, "verify_cmd", "") or ""
    if _UNITTEST_GATE_RE.search(gate):
        return "python -m unittest discover -v"
    return None


def _read_js_project_env(cwd: str | None = None) -> dict:
    """Read package.json and project config to detect JS/TS environment.

    Returns a dict with:
        is_esm: bool       — True if package.json has "type": "module"
        has_jest: bool      — True if jest in dependencies/devDependencies
        has_jest_globals: bool — True if @jest/globals is installed
        has_jest_config: bool — True if jest.config.* exists
        module_type: str    — "module" or "commonjs"
        test_runner: str    — "vitest", "jest", or "jest" (default)
        has_vitest: bool    — True if vitest in devDependencies
        has_vitest_config: bool — True if vitest.config.* exists
        has_tsx: bool       — True if .tsx files exist in project
        has_jsx: bool       — True if .jsx files exist in project
    """
    env = {
        "is_esm": False,
        "has_jest": False,
        "has_jest_globals": False,
        "has_jest_config": False,
        "module_type": "commonjs",
        "test_runner": "jest",
        "has_vitest": False,
        "has_vitest_config": False,
        "has_tsx": False,
        "has_jsx": False,
        "test_script": None,     # raw "scripts.test" from package.json
    }

    # Read package.json
    pkg_path = os.path.join(cwd, "package.json") if cwd else "package.json"
    if os.path.isfile(pkg_path):
        try:
            with open(pkg_path, "r", encoding="utf-8") as f:
                pkg = json.load(f)
        except (json.JSONDecodeError, OSError):
            return env

        # ESM detection
        if pkg.get("type") == "module":
            env["is_esm"] = True
            env["module_type"] = "module"

        # Read scripts.test — the canonical test command defined by the project
        _scripts = pkg.get("scripts", {})
        _test_script = _scripts.get("test", "")
        if _test_script:
            env["test_script"] = _test_script

        # Dependency detection
        all_deps = {}
        all_deps.update(pkg.get("dependencies", {}))
        all_deps.update(pkg.get("devDependencies", {}))
        env["has_jest"] = "jest" in all_deps
        env["has_jest_globals"] = "@jest/globals" in all_deps
        env["has_vitest"] = "vitest" in all_deps

    # Vitest config detection
    for config_name in ("vitest.config.ts", "vitest.config.js",
                        "vitest.config.mts", "vitest.config.mjs"):
        cfg_path = os.path.join(cwd, config_name) if cwd else config_name
        if os.path.isfile(cfg_path):
            env["has_vitest_config"] = True
            break

    # Vite config with embedded test section (most common Vite+Vitest setup)
    if not env["has_vitest_config"]:
        for vite_cfg in ("vite.config.ts", "vite.config.js",
                         "vite.config.mts", "vite.config.mjs"):
            vite_path = os.path.join(cwd, vite_cfg) if cwd else vite_cfg
            if os.path.isfile(vite_path):
                try:
                    with open(vite_path, "r", encoding="utf-8") as f:
                        vite_content = f.read()
                    # Check for test config inside vite.config
                    if "test:" in vite_content or "test :" in vite_content:
                        env["has_vitest_config"] = True
                        log.info(f"Detected Vitest config embedded in {vite_cfg}")
                except OSError:
                    pass
                break  # stop after first vite.config found

    # Vite project detection: if project uses Vite, prefer Vitest
    env["has_vite"] = False
    if os.path.isfile(os.path.join(cwd, "package.json") if cwd else "package.json"):
        try:
            with open(os.path.join(cwd, "package.json") if cwd else "package.json",
                      "r", encoding="utf-8") as f:
                _pkg = json.load(f)
            _all = {}
            _all.update(_pkg.get("dependencies", {}))
            _all.update(_pkg.get("devDependencies", {}))
            env["has_vite"] = ("vite" in _all or
                               any(k.startswith("@vitejs/") for k in _all))
        except (json.JSONDecodeError, OSError):
            pass

    # Jest config detection
    for config_name in ("jest.config.js", "jest.config.ts", "jest.config.mjs",
                        "jest.config.cjs", "jest.config.json"):
        cfg_path = os.path.join(cwd, config_name) if cwd else config_name
        if os.path.isfile(cfg_path):
            env["has_jest_config"] = True
            break

    # Determine test runner: prefer scripts.test > config files > dependencies
    # The scripts.test field is the most authoritative signal — it's what
    # `npm test` actually runs, so it covers Angular (ng test), Karma,
    # Mocha, and any other runner without hardcoded framework checks.
    _ts = (env.get("test_script") or "").lower()
    if _ts:
        if "vitest" in _ts:
            env["test_runner"] = "vitest"
        elif "ng test" in _ts or "karma" in _ts:
            env["test_runner"] = "ng"
        elif "mocha" in _ts:
            env["test_runner"] = "mocha"
        elif "jest" in _ts:
            env["test_runner"] = "jest"
        # else: fall through to config/dependency heuristics below

    if env["test_runner"] == "jest":
        # Config / dependency heuristics (fallback when scripts.test is absent
        # or doesn't name a recognisable runner).
        if env["has_vitest_config"] or env["has_vitest"]:
            env["test_runner"] = "vitest"
        elif env.get("has_vite") and not env["has_jest_config"]:
            # Vite project without explicit Jest config → default to Vitest
            env["test_runner"] = "vitest"
        elif env["has_jest_config"] or env["has_jest"]:
            env["test_runner"] = "jest"

    # Check for .tsx/.jsx files (useful for React testing guidance)
    scan_dir = cwd or "."
    if os.path.isdir(os.path.join(scan_dir, "src")):
        for root, _dirs, files in os.walk(os.path.join(scan_dir, "src")):
            if any(f.endswith(".tsx") for f in files):
                env["has_tsx"] = True
            if any(f.endswith(".jsx") for f in files):
                env["has_jsx"] = True
            if env["has_tsx"] and env["has_jsx"]:
                break

    return env


# ── Deterministic vitest environment bootstrap ────────────────────────────
# The planner handles vitest setup inconsistently (observed across runs:
# a Windows-broken echo chain, then omitting it entirely) and the BulkTest
# fix loop's destructive-change guard used to block the LLM from creating
# the missing config. The environment is the same ~10 lines every time —
# write it deterministically, no LLM involved.

_VITEST_SETUP_FILE = "vitest.setup.js"

_VITEST_SETUP_TEMPLATE = "import '@testing-library/jest-dom/vitest'\n"

_VITEST_DOM_DEPS = ("vitest", "jsdom", "@testing-library/react",
                    "@testing-library/dom", "@testing-library/jest-dom")


def _vitest_config_template(with_react_plugin: bool) -> str:
    plugin_import = ("import react from '@vitejs/plugin-react'\n"
                     if with_react_plugin else "")
    plugins_line = "  plugins: [react()],\n" if with_react_plugin else ""
    return (
        "import { defineConfig } from 'vitest/config'\n"
        f"{plugin_import}"
        "\n"
        "export default defineConfig({\n"
        f"{plugins_line}"
        "  test: {\n"
        "    environment: 'jsdom',\n"
        "    globals: true,\n"
        f"    setupFiles: './{_VITEST_SETUP_FILE}',\n"
        "  },\n"
        "})\n"
    )


def ensure_vitest_env(executor, cwd: str | None,
                      test_files: dict[str, str],
                      memory=None) -> None:
    """Make a vitest run actually possible: install the DOM-testing deps
    and write a jsdom-enabled ``vitest.config.js`` + setup file when the
    project has neither. Idempotent; no LLM calls.

    Only acts when the tests need a DOM (they import @testing-library) —
    pure-logic vitest suites run fine without any of this.
    """
    root = cwd or "."
    pkg_path = os.path.join(root, "package.json")
    if not os.path.isfile(pkg_path):
        return
    needs_dom = any("@testing-library" in (c or "")
                    for c in test_files.values())
    if not needs_dom:
        return

    try:
        with open(pkg_path, "r", encoding="utf-8") as f:
            pkg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    deps = {}
    deps.update(pkg.get("dependencies", {}))
    deps.update(pkg.get("devDependencies", {}))

    missing = [p for p in _VITEST_DOM_DEPS if p not in deps]
    if missing:
        log.info("[VitestEnv] Installing missing test deps: %s", missing)
        ok, out = executor.run_command(
            "npm install -D " + " ".join(missing), cwd=root)
        if not ok:
            log.warning("[VitestEnv] Dep install failed: %s", (out or "")[:300])

    env = _read_js_project_env(root)
    if env["has_vitest_config"]:
        return

    cfg_rel = os.path.join(cwd, "vitest.config.js") if cwd \
        else "vitest.config.js"
    setup_rel = os.path.join(cwd, _VITEST_SETUP_FILE) if cwd \
        else _VITEST_SETUP_FILE
    written: dict[str, str] = {}
    cfg_content = _vitest_config_template(
        with_react_plugin="@vitejs/plugin-react" in deps)
    try:
        with open(os.path.join(root, "vitest.config.js"), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write(cfg_content)
        written[cfg_rel.replace("\\", "/")] = cfg_content
        setup_abs = os.path.join(root, _VITEST_SETUP_FILE)
        if not os.path.isfile(setup_abs):
            with open(setup_abs, "w", encoding="utf-8", newline="\n") as f:
                f.write(_VITEST_SETUP_TEMPLATE)
            written[setup_rel.replace("\\", "/")] = _VITEST_SETUP_TEMPLATE
    except OSError as e:
        log.warning("[VitestEnv] Could not write config: %s", e)
        return
    log.info("[VitestEnv] Bootstrapped vitest environment: %s",
             list(written))
    if memory is not None:
        try:
            memory.update(written)
        except Exception as e:
            log.debug("[VitestEnv] memory update failed: %s", e)


import platform


# ---------------------------------------------------------------------------
# KB-driven content fixes
# ---------------------------------------------------------------------------


def _apply_content_fixes(
    files: dict[str, str],
    content_fixes: list | None = None,
) -> dict[str, str]:
    """Apply KB-driven content-fix rules to generated file content.

    Each rule (a :class:`ContentFix` from the global KB) specifies a
    ``file_glob``, a ``content_pattern`` regex, a ``replacement``, and
    optional ``ensure_content`` to prepend when missing.

    Returns *files* unchanged when *content_fixes* is ``None`` or empty.
    """
    import fnmatch as _fnmatch

    if not content_fixes:
        return files

    result = dict(files)
    for filepath, content in list(result.items()):
        basename = filepath.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        original = content

        for fix in content_fixes:
            if not _fnmatch.fnmatch(basename, fix.file_glob):
                continue

            flags = fix.compiled_flags()
            pattern = re.compile(fix.content_pattern, flags)
            if not pattern.search(content):
                continue

            content = pattern.sub(fix.replacement, content)

            if fix.ensure_content:
                if fix.ensure_content.strip() not in content:
                    content = fix.ensure_content + content

            if fix.collapse_blanks:
                content = re.sub(r"\n{3,}", "\n\n", content).strip() + "\n"

            log.info(
                "Content fix '%s': applied to %s (%s)",
                fix.name, filepath, fix.description or "no description",
            )

        if content != original:
            result[filepath] = content

    return result


def _inject_browser_router_if_needed(
    files: dict[str, str],
    memory: "FileMemory",
    all_plan_steps=None,
) -> dict[str, str]:
    """Wrap <App /> in <BrowserRouter> when Link/NavLink is used but BrowserRouter is absent.

    When the planner's inline code for main.jsx/index.jsx includes ``createRoot``
    but omits ``BrowserRouter``, any component that uses ``<Link>`` or ``<NavLink>``
    from ``react-router-dom`` will crash at runtime with:

        useHref() may only be used within a <Router> context.

    This helper scans the entry-point file written in *files*, detects the omission,
    then patches the file by:
      1. Adding ``import { BrowserRouter } from 'react-router-dom'``
      2. Wrapping the ``<App />`` JSX element with ``<BrowserRouter>``.

    Only runs when memory contains at least one file that imports from
    ``react-router-dom`` and references ``Link`` or ``NavLink``.
    """
    result = dict(files)

    for filepath, content in list(result.items()):
        basename = filepath.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if basename not in ("main.jsx", "main.tsx", "index.jsx", "index.tsx"):
            continue

        # Already has a Router — nothing to do
        if "BrowserRouter" in content or "HashRouter" in content:
            continue

        # Only applies to React 18 createRoot entry points
        if "createRoot" not in content:
            continue

        # Check if any memory file uses Link / NavLink from react-router-dom
        all_memory_files = memory.all_files()
        uses_router_link = any(
            "react-router-dom" in c and bool(
                re.search(r'\b(?:Link|NavLink)\b', c)
            )
            for c in all_memory_files.values()
        )
        if not uses_router_link:
            continue

        # Check if any OTHER memory file already wraps with BrowserRouter
        # (e.g. Root.jsx or App.jsx).  If so, skip injection — the router
        # is already provided deeper in the component tree.
        child_has_router = any(
            ("BrowserRouter" in c or "HashRouter" in c)
            and fpath_key != filepath
            for fpath_key, c in all_memory_files.items()
        )

        # Also check pending plan steps' inline code — App.jsx may not
        # be in memory yet (written in a later wave) but the plan already
        # contains its content with BrowserRouter.
        if not child_has_router and all_plan_steps:
            for ps in all_plan_steps:
                for _ps_path, _ps_content in (ps.inline_code or {}).items():
                    if _ps_path != filepath and (
                        "BrowserRouter" in _ps_content
                        or "HashRouter" in _ps_content
                    ):
                        child_has_router = True
                        break
                if child_has_router:
                    break

        if child_has_router:
            log.debug(
                "BrowserRouter injection skipped for %s — "
                "a child component already provides BrowserRouter",
                filepath,
            )
            continue

        patched = content

        # ── 1. Add BrowserRouter import ──
        rrd_import_re = re.compile(
            r"(import\s+\{)([^}]+?)(\}\s+from\s+['\"]react-router-dom['\"])"
        )
        rrd_match = rrd_import_re.search(patched)
        if rrd_match and "BrowserRouter" not in rrd_match.group(2):
            patched = rrd_import_re.sub(
                lambda m: m.group(1) + m.group(2).rstrip() + ", BrowserRouter" + m.group(3),
                patched,
                count=1,
            )
        elif not rrd_match:
            # Insert after last React-related import line
            react_imports = list(re.finditer(
                r"^import .+['\"]react(?:-dom)?['\"].*$", patched, re.MULTILINE
            ))
            if react_imports:
                insert_pos = react_imports[-1].end()
                patched = (
                    patched[:insert_pos]
                    + "\nimport { BrowserRouter } from 'react-router-dom'"
                    + patched[insert_pos:]
                )
            else:
                patched = "import { BrowserRouter } from 'react-router-dom'\n" + patched

        # ── 2. Wrap the root component with <BrowserRouter> ──
        # Detect the root component rendered inside createRoot().render()
        # It could be <App />, <Root />, or any PascalCase component.
        _root_comp_re = re.compile(
            r'(<(?:App|Root|[A-Z]\w*)\s*/>)'
        )
        patched = _root_comp_re.sub(
            r'<BrowserRouter>\n      \1\n    </BrowserRouter>',
            patched,
            count=1,
        )

        if patched != content:
            result[filepath] = patched
            log.info(
                "BrowserRouter injection: patched %s "
                "(Link/NavLink detected in project memory)",
                filepath,
            )

    return result


def _looks_like_file_path(s: str) -> bool:
    """Return True if *s* looks like a file path (has extension or path separator)."""
    return bool(re.search(r'\.\w{2,6}$', s) or '/' in s or os.sep in s)


def _strip_protected_files(files: dict[str, str], intent_spec=None) -> dict[str, str]:
    """Handle protected manifest files from parsed LLM output.

    For lock files (package-lock.json, yarn.lock, etc.) — fully blocked.
    For mergeable manifests (package.json, requirements.txt, etc.) — attempts
    smart merge: new dependencies/scripts are merged in, removals and version
    changes are blocked.  Falls back to full block on parse errors.

    Files are only protected when they already exist on disk (new projects
    need to create them initially).
    """
    if not files:
        return files

    # Lock files that should NEVER be touched — only package managers write these
    _LOCK_FILES: set[str] = {
        'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
        'go.sum', 'Cargo.lock', 'Gemfile.lock',
        'composer.lock', 'Pipfile.lock', 'poetry.lock',
        'Pipfile',  # managed by pipenv
    }

    # Manifests that support smart merge (additive deps only)
    _MERGEABLE_JSON: set[str] = {
        'package.json', 'composer.json',
    }
    _MERGEABLE_TEXT: set[str] = {
        'requirements.txt',
    }
    _MERGEABLE_TOML: set[str] = {
        'Cargo.toml',
    }
    _MERGEABLE_RUBY: set[str] = {
        'Gemfile',
    }
    _MERGEABLE_GO: set[str] = {
        'go.mod',
    }

    filtered: dict[str, str] = {}
    merged_count = 0
    stripped_count = 0

    for fpath, content in files.items():
        basename = os.path.basename(fpath)

        # Not a core protected file — pass through
        if basename not in Executor._PROTECTED_FILENAMES:
            filtered[fpath] = content
            continue

        # File doesn't exist on disk yet — allow creation
        if not os.path.isfile(fpath):
            filtered[fpath] = content
            continue

        # Lock files — always block
        if basename in _LOCK_FILES:
            log.warning(f"[Pipeline] Blocked lock file: {fpath} "
                        f"(only package managers should modify this)")
            stripped_count += 1
            continue

        # Read existing file content
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                existing_content = f.read()
        except OSError:
            log.warning(f"[Pipeline] Cannot read {fpath}, blocking write")
            stripped_count += 1
            continue

        # Attempt smart merge based on file type
        merged = None
        if basename in _MERGEABLE_JSON:
            merged = _smart_merge_json_manifest(existing_content, content, fpath)
        elif basename in _MERGEABLE_TEXT:
            merged = _smart_merge_requirements_txt(existing_content, content, fpath)
        elif basename in _MERGEABLE_GO:
            merged = _smart_merge_go_mod(existing_content, content, fpath)
        elif basename in _MERGEABLE_TOML:
            merged = _smart_merge_line_based(existing_content, content, fpath)
        elif basename in _MERGEABLE_RUBY:
            merged = _smart_merge_line_based(existing_content, content, fpath)

        if merged is not None and merged != existing_content:
            filtered[fpath] = merged
            merged_count += 1
            log.info(f"[Pipeline] Smart-merged additive changes into {fpath}")
        elif merged == existing_content:
            log.info(f"[Pipeline] No new additions for {fpath}, skipping write")
            stripped_count += 1
        else:
            log.warning(f"[Pipeline] Smart merge failed for {fpath}, "
                        f"blocking write (fallback)")
            stripped_count += 1

    if merged_count > 0:
        log.info(f"[Pipeline] Smart-merged {merged_count} protected file(s)")
    if stripped_count > 0:
        log.info(f"[Pipeline] Blocked {stripped_count} protected file(s)")
    return filtered


def _smart_merge_json_manifest(existing: str, llm_output: str,
                                filepath: str) -> str | None:
    """Merge additive changes from LLM output into an existing JSON manifest.

    Merges new keys in ``dependencies``, ``devDependencies``, ``scripts``,
    and safe top-level keys (e.g. ``"type"``, ``"main"``, ``"exports"``).
    Blocks removals and version changes.  Returns merged JSON string, or
    ``None`` on parse failure.
    """
    try:
        old_data = json.loads(existing)
        new_data = json.loads(llm_output)
    except (json.JSONDecodeError, TypeError):
        log.warning(f"[SmartMerge] JSON parse failed for {filepath}")
        return None

    if not isinstance(old_data, dict) or not isinstance(new_data, dict):
        return None

    changed = False
    # Sections where we allow additive merges
    merge_sections = ['dependencies', 'devDependencies', 'scripts',
                      'peerDependencies', 'optionalDependencies']

    for section in merge_sections:
        old_section = old_data.get(section, {})
        new_section = new_data.get(section, {})
        if not isinstance(old_section, dict) or not isinstance(new_section, dict):
            continue

        for key, value in new_section.items():
            if key not in old_section:
                # New key — merge it in
                if section not in old_data:
                    old_data[section] = {}
                old_data[section][key] = value
                changed = True
                log.info(f"[SmartMerge] Added {section}.{key} = {value!r} "
                         f"to {filepath}")
            elif old_section[key] != value:
                # Changed value — block, keep original
                log.info(f"[SmartMerge] Blocked change to {section}.{key} "
                         f"in {filepath}: {old_section[key]!r} → {value!r}")

        # Check for removals — log but don't apply
        for key in old_section:
            if key not in new_section:
                log.info(f"[SmartMerge] Blocked removal of {section}.{key} "
                         f"from {filepath}")

    # Safe top-level keys: allow adding new ones or overwriting with same type.
    # These control module system, entry points, and project metadata — not
    # dependency resolution — so they are safe to update without pkg-manager help.
    _SAFE_TOPLEVEL_KEYS = {
        'type', 'main', 'module', 'browser', 'exports', 'imports',
        'bin', 'man', 'files', 'sideEffects', 'private',
        'description', 'keywords', 'author', 'license', 'homepage',
        'repository', 'bugs', 'engines', 'os', 'cpu',
        'publishConfig', 'workspaces', 'volta',
    }
    for key, value in new_data.items():
        if key in merge_sections:
            continue  # already handled above
        if key not in _SAFE_TOPLEVEL_KEYS:
            continue
        if key not in old_data:
            old_data[key] = value
            changed = True
            log.info(f"[SmartMerge] Added top-level {key!r} = {value!r} "
                     f"to {filepath}")
        elif old_data[key] != value:
            old_data[key] = value
            changed = True
            log.info(f"[SmartMerge] Updated top-level {key!r}: "
                     f"{old_data[key]!r} → {value!r} in {filepath}")

    if not changed:
        return existing

    # Preserve original formatting indent
    indent = 2  # default
    for line in existing.splitlines()[1:5]:
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            break

    return json.dumps(old_data, indent=indent, ensure_ascii=False) + "\n"


def _smart_merge_requirements_txt(existing: str, llm_output: str,
                                   filepath: str) -> str | None:
    """Merge new packages from LLM output into existing requirements.txt.

    Appends packages that don't exist yet.  Blocks removals and version
    changes.  Returns merged content string.
    """
    import re

    def _parse_req_name(line: str) -> str | None:
        """Extract package name from a requirements.txt line."""
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            return None
        # Handle: package==1.0, package>=1.0, package~=1.0, package[extra]
        m = re.match(r'^([A-Za-z0-9_][A-Za-z0-9._-]*)', line)
        return m.group(1).lower() if m else None

    existing_lines = existing.splitlines()
    new_lines = llm_output.splitlines()

    # Build map of existing packages: name → full line
    existing_pkgs: dict[str, str] = {}
    for line in existing_lines:
        name = _parse_req_name(line)
        if name:
            existing_pkgs[name] = line.strip()

    # Find new packages to add
    additions: list[str] = []
    for line in new_lines:
        name = _parse_req_name(line)
        if name is None:
            continue
        if name not in existing_pkgs:
            additions.append(line.strip())
            log.info(f"[SmartMerge] Adding new package: {line.strip()} "
                     f"to {filepath}")
        elif existing_pkgs[name] != line.strip():
            log.info(f"[SmartMerge] Blocked version change for {name} "
                     f"in {filepath}: {existing_pkgs[name]} → {line.strip()}")

    if not additions:
        return existing

    # Append new packages at the end
    result = existing.rstrip('\n')
    result += '\n' + '\n'.join(additions) + '\n'
    return result


def _smart_merge_go_mod(existing: str, llm_output: str,
                         filepath: str) -> str | None:
    """Merge new require directives from LLM output into existing go.mod.

    Only adds new ``require`` lines that don't exist yet.
    """
    import re

    def _parse_requires(content: str) -> dict[str, str]:
        """Extract module → version from require directives."""
        reqs: dict[str, str] = {}
        # Single-line: require module/path v1.2.3
        for m in re.finditer(r'^\s*require\s+(\S+)\s+(\S+)', content, re.MULTILINE):
            reqs[m.group(1)] = m.group(2)
        # Block: require ( ... )
        for block in re.finditer(r'require\s*\((.*?)\)', content, re.DOTALL):
            for line in block.group(1).splitlines():
                line = line.strip()
                if line and not line.startswith('//'):
                    parts = line.split()
                    if len(parts) >= 2:
                        reqs[parts[0]] = parts[1]
        return reqs

    existing_reqs = _parse_requires(existing)
    new_reqs = _parse_requires(llm_output)

    additions: list[str] = []
    for mod, ver in new_reqs.items():
        if mod not in existing_reqs:
            additions.append(f"\trequire {mod} {ver}")
            log.info(f"[SmartMerge] Adding require {mod} {ver} to {filepath}")

    if not additions:
        return existing

    result = existing.rstrip('\n')
    result += '\n' + '\n'.join(additions) + '\n'
    return result


def _smart_merge_line_based(existing: str, llm_output: str,
                             filepath: str) -> str | None:
    """Generic line-based merge for Cargo.toml, Gemfile, etc.

    Appends lines from LLM output that don't exist (case-sensitive) in
    the existing file.  This is a conservative catch-all for formats
    we don't deeply parse.
    """
    existing_lines_set = set(existing.splitlines())

    additions: list[str] = []
    for line in llm_output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if line not in existing_lines_set and stripped not in {
            l.strip() for l in existing_lines_set
        }:
            additions.append(line)

    if not additions:
        return existing

    log.info(f"[SmartMerge] Appending {len(additions)} new line(s) to {filepath}")
    result = existing.rstrip('\n')
    result += '\n' + '\n'.join(additions) + '\n'
    return result


def _shell_instructions() -> str:
    """Return OS-aware shell command guidance for LLM prompts."""
    if os.name == 'nt':
        base = (
            "Use plain CMD commands that work in Windows cmd.exe.\n"
            "For listing files use: dir /s /b\n"
            "For reading a file use: type <path>\n"
            "For creating a directory use: mkdir <path>\n"
            "For installing Python packages use: pip install <package>\n"
            "For activating a virtual environment use: call venv\\Scripts\\activate\n"
            "Do NOT use Unix commands: source, mkdir -p, touch, rm -rf, cat, ls, chmod, export.\n"
            "Do NOT use PowerShell cmdlets like Get-ChildItem, Select-Object, etc.\n"
            "Do NOT use bash-style line continuation characters (`\\`). Write multiline commands on a single line using `&&`.\n"
        )
    else:
        _sysname = platform.system()
        _os_label = "macOS" if _sysname == "Darwin" else _sysname
        base = (
            f"Use standard shell commands for {_os_label}.\n"
            "For listing files use: find . -type f\n"
            "For reading a file use: cat <path>\n"
            "For creating a directory use: mkdir -p <path>\n"
            "For installing Python packages use: pip install <package>\n"
            "For activating a virtual environment use: source venv/bin/activate\n"
        )
    base += (
        "\nCRITICAL: Commands run non-interactively (no terminal input available).\n"
        "NEVER use commands that prompt for user input. Always add non-interactive flags:\n"
        "  - npx create-next-app: add --yes\n"
        "  - npm init / yarn init: add --yes or -y\n"
        "  - Angular CLI (ng new): add --defaults\n"
        "  - Composer: add --no-interaction\n"
        "  - Any tool with prompts: use --yes, --default, -y, or equivalent flag.\n"
    )
    return base


def _shell_examples() -> str:
    """Return OS-aware example commands for the planner prompt."""
    if os.name == 'nt':
        return "  1. List all project files with `dir /s /b`"
    else:
        return "  1. List all project files with `find . -type f`"


# File extensions and names that don't need code review
_NON_CODE_EXTENSIONS = {
    '.md', '.txt', '.rst', '.log', '.csv',
    '.yml', '.yaml', '.toml', '.ini', '.cfg',
    '.env', '.env.example', '.gitignore', '.dockerignore',
    '.editorconfig',
}
_NON_CODE_FILENAMES = {
    'README', 'README.md', 'README.rst', 'README.txt',
    'LICENSE', 'LICENSE.md', 'LICENSE.txt',
    'CHANGELOG', 'CHANGELOG.md',
    'CONTRIBUTING', 'CONTRIBUTING.md',
    'Makefile', 'Dockerfile', 'Procfile',
    '.gitignore', '.dockerignore', '.editorconfig',
    'requirements.txt', 'setup.cfg',
}


def _all_non_code_files(filenames: list[str]) -> bool:
    """Return True if every file in the list is non-functional (docs, config, etc.)."""
    if not filenames:
        return False
    for f in filenames:
        basename = f.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        _, ext = os.path.splitext(basename)
        if basename not in _NON_CODE_FILENAMES and ext.lower() not in _NON_CODE_EXTENSIONS:
            return False
    return True


def _build_prior_steps_context(memory: FileMemory, step_idx: int) -> str:
    """Collect outputs of prior steps (CMD + SEARCH) from memory for context."""
    parts: list[str] = []
    all_files = memory.all_files()
    for i in range(step_idx):
        # CMD step output
        cmd_key = f"_cmd_output/step_{i+1}.txt"
        if cmd_key in all_files:
            parts.append(f"Step {i+1} output:\n{all_files[cmd_key]}")
        # SEARCH step output — propagate search results to dependent steps
        search_key = f"_search_context/step_{i+1}.txt"
        if search_key in all_files:
            parts.append(
                f"Step {i+1} search results:\n{all_files[search_key]}")
    if not parts:
        return ""
    return "Previously executed steps:\n" + "\n\n".join(parts) + "\n\n"


_MANIFEST_NAMES = (
    'package.json', 'Cargo.toml', 'go.mod', 'requirements.txt', 'Gemfile',
    'pyproject.toml', 'composer.json', 'manage.py', 'setup.py', 'setup.cfg',
)

# Directory names that mark a directory as its OWN build/runtime context.
_OWN_ENV_MARKERS = ('venv', '.venv', 'node_modules', 'Pipfile', '.python-version')


def _memory_has_manifest(all_files, subproject: str) -> bool:
    """True when a project manifest is tracked under *subproject* in memory.

    Protected files (package.json, etc.) are frequently absent from the
    on-disk check at detection time but present in memory, so this is an
    equally valid self-containment signal.
    """
    prefix = subproject.replace('\\', '/').rstrip('/') + '/'
    for p in all_files:
        norm = p.replace('\\', '/')
        if norm.startswith(prefix) and os.path.basename(norm) in _MANIFEST_NAMES:
            return True
    return False


def _dir_has_own_env(subproject: str) -> bool:
    """True when *subproject* holds its own venv/interpreter/deps on disk."""
    for marker in _OWN_ENV_MARKERS:
        if os.path.exists(os.path.join(subproject, marker)):
            return True
    return False


def _dir_imported_as_package(all_files, subproject: str) -> bool:
    """True when files under *subproject* import it as a top-level package.

    ``brick_breaker/test_main.py`` doing ``import brick_breaker.main`` means
    the directory must sit on ``sys.path`` via its PARENT (the repo root) —
    i.e. it is a package under the root project, not a self-contained
    sub-project. Treating it as one prepends ``cd {sub}`` to its gates,
    which breaks that very import and the root-relative venv path.
    """
    pkg = subproject.replace('\\', '/').rstrip('/').split('/')[-1]
    if not pkg.isidentifier():
        return False
    prefix = subproject.replace('\\', '/').rstrip('/') + '/'
    pat = re.compile(rf"(?m)^\s*(?:from|import)\s+{re.escape(pkg)}\b")
    for p, content in all_files.items():
        norm = p.replace('\\', '/')
        if norm.startswith(prefix) and content and pat.search(content):
            return True
    return False


def _step_belongs_to_subproject(plan_step, subproject: str,
                                memory: FileMemory) -> bool | None:
    """Whether *plan_step*'s files live under *subproject*, or None if unknown.

    A detected sub-project root is a claim about ONE tree, and the whole
    pipeline treats it as the tree: CMD commands are re-rooted into it,
    gates get a ``cd {sub} &&`` prefix, unprefixed paths get prefixed.
    That is right for `npx create-next-app my-app` and wrong the moment
    a plan declares a second root beside it.

    Measured 2026-08-19 on a "React frontend + Express backend" task. Step
    1.1 scaffolded ``frontend/``; step 1.2's ``mkdir backend && cd backend
    && npm init -y`` matched the ``npm init`` routing pattern and ran with
    cwd=``frontend``, so the backend was built at ``frontend/backend/``.
    Every later backend gate was prefixed ``cd frontend &&`` and read the
    same wrong copy, so the misroute was self-consistent and nothing
    disagreed — all four backend gates passed green. The run ended with
    the backend on disk twice: ``frontend/backend/`` holding the installed
    ``node_modules``, and the declared ``backend/`` holding no
    dependencies at all and unable to start.

    The discriminator is the plan's own vocabulary. ``_plan_declared_roots``
    holds the top-level directories the plan names, and *subproject* being
    one of them is the proof that these paths are repo-root-relative: a
    planner that writes ``frontend/src/App.jsx`` is not also writing
    ``backend/`` to mean ``frontend/backend/``. Only then is a step's
    declared target trusted to say which root it belongs to.

    Returns None — never a guess — when the plan does not name the
    sub-project or the step declares no targets, which is the ordinary
    single-scaffold case and keeps its existing behaviour exactly.
    """
    sibs = _plan_sibling_roots(memory, subproject)
    if sibs is None:
        return None
    targets = list(getattr(plan_step, "target_files", None) or [])         if plan_step is not None else []
    if not targets:
        return None
    prefix = subproject.replace("\\", "/").rstrip("/") + "/"
    # EVERY target must live under the sub-project, not merely one of
    # them. Re-rooting is only correct when the whole step does, because
    # `cd {sub}` breaks every path that does not — while a root-relative
    # command is correct for all of them.
    #
    # Measured 2026-08-19 run 13, step 8.1 with
    # `target: backend/.env.example, frontend/.env.example, README.md`.
    # Under `any()` it read as "belongs to frontend", so its gate ran as
    # `cd frontend && node -e "...readFileSync('backend/.env.example')
    # ...readFileSync('README.md')..."` — pointing at two files that do
    # not exist. The agent satisfied it by CREATING them:
    # `frontend/backend/.env.example` and `frontend/README.md`, both
    # reported by the ghost as unplanned writes. That is the same
    # manufacture-the-path-the-gate-names failure as run 2's
    # `frontend/frontend/package.json`.
    return all(_norm_rel(t).startswith(prefix) for t in targets)


def _norm_rel(path: str) -> str:
    """``path`` with backslashes and a leading ``./`` normalised away."""
    norm = (path or "").replace("\\", "/").lstrip("/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def _plan_sibling_roots(memory: FileMemory,
                        subproject: str | None) -> set[str] | None:
    """Top-level roots the plan declares BESIDE *subproject*, or None.

    None means "the plan's paths are not known to be repo-root-relative",
    which is the ordinary single-scaffold case: a planner that writes
    ``src/App.jsx`` for a project living under ``my-app/`` expects the
    prefix to be applied, and nothing here should second-guess it. The
    proof that a plan is root-relative is that it names *subproject*
    itself as a target root — see `_step_belongs_to_subproject`.
    """
    if not subproject:
        return None
    roots = getattr(memory, "_plan_declared_roots", None)
    sub = subproject.replace("\\", "/").rstrip("/")
    if not roots or sub not in roots:
        return None
    return {r for r in roots if r != sub}


def _detect_subproject_root(memory: FileMemory) -> str | None:
    """Detect if all project files share a common subdirectory prefix.

    When an earlier CMD step created a project in a subdirectory (e.g.
    ``npx create-next-app my-app``), subsequent files in memory will all
    live under ``my-app/``.  This function finds that common root.

    Detection strategies (in order):
    1. Parse CMD outputs for project-creation commands that specify a
       subdirectory (e.g. ``npx create-next-app my-app``).
    2. Check if all source files share a single first directory component.
    3. Look for project manifests in memory or on disk.
    4. Majority vote among first-level directories.

    Returns the subdirectory name (e.g. ``my-app``) or ``None``.
    """
    # ── Fast path: a prior scaffold CMD already identified the subproject ──
    scaffolded = getattr(memory, '_scaffolded_subproject', None)
    if scaffolded and os.path.isdir(scaffolded):
        return scaffolded

    all_files = memory.all_files()

    # ── Fallback 0: Parse CMD outputs for project-creation commands ──
    # This is the EARLIEST detection — it works even when no source files
    # have been written to memory yet, only _cmd_output/ entries exist.
    # We look for commands like "npx create-next-app my-app" and extract
    # the directory name from the command itself.
    _PROJECT_CREATE_PATTERNS = [
        # npx create-next-app <dir>
        re.compile(r'create-next-app(?:@\S+)?\s+(\S+)'),
        # npx create-react-app <dir>
        re.compile(r'create-react-app\s+(\S+)'),
        # npx create-vite <dir>  OR  npm create vite@latest <dir>
        re.compile(r'create-vite(?:@\S+)?\s+(\S+)'),
        re.compile(r'npm\s+create\s+vite(?:@\S+)?\s+(\S+)'),
        # npx create-vue <dir>  OR  npm create vue@latest <dir>
        re.compile(r'create-vue(?:@\S+)?\s+(\S+)'),
        re.compile(r'npm\s+create\s+vue(?:@\S+)?\s+(\S+)'),
        # vue create <dir>
        re.compile(r'vue\s+create\s+(\S+)'),
        # ng new <dir>
        re.compile(r'ng\s+new\s+(\S+)'),
        # rails new <dir>
        re.compile(r'rails\s+new\s+(\S+)'),
        # cargo new <dir>
        re.compile(r'cargo\s+new\s+(\S+)'),
        # mkdir <dir> && ... django  (Django scaffold creates dir separately)
        re.compile(r'mkdir\s+(-p\s+)?(\S+)\s*&&.*django'),
        # django-admin startproject <name> <dir>  (explicit target dir)
        re.compile(r'django-admin\s+startproject\s+\S+\s+(\S+)'),
    ]

    for fpath, content in all_files.items():
        if not fpath.startswith('_cmd_output/') and not fpath.startswith('_fix_output/'):
            continue
        # Only scan the first line (the command itself, prefixed with $)
        first_line = content.split('\n')[0] if content else ''
        for pattern in _PROJECT_CREATE_PATTERNS:
            m = pattern.search(first_line)
            if m:
                candidate = m.group(m.lastindex).strip().rstrip('/')
                # Skip if the command used ./ (current directory)
                if candidate in ('.', './', ''):
                    continue
                # Strip leading ./ if present
                if candidate.startswith('./'):
                    candidate = candidate[2:]
                if not candidate:
                    continue
                # Verify the directory actually exists on disk with a manifest
                if os.path.isdir(candidate):
                    for manifest in ('package.json', 'Cargo.toml', 'go.mod',
                                     'requirements.txt', 'Gemfile', 'pyproject.toml',
                                     'composer.json', 'manage.py'):
                        if os.path.isfile(os.path.join(candidate, manifest)):
                            log.info(f"[SubProject] Detected sub-project root "
                                     f"from CMD output ({manifest}): {candidate}/")
                            return candidate

    # A sub-project root can only exist if some file lives in a
    # SUBDIRECTORY, so root-level files tell us nothing here and the
    # `'/' in p` filter is the real question being asked.
    # Internal tracking paths use underscore-prefixed directories
    # (_cmd_output/, _fix_output/, _search_context/) and are excluded;
    # directories like __tests__/ and __mocks__/ are legitimate.
    _internal = ('_cmd_output/', '_fix_output/', '_search_context/')
    nested_paths = [
        p for p in all_files
        if not p.startswith(_internal) and '/' in p
    ]
    if not nested_paths:
        # This is the normal, correct outcome for a flat project — not a
        # failure. The previous wording ("No source files in memory")
        # contradicted the memory keys printed alongside it, which read as
        # a bug in the pipeline every time a flat project ran.
        log.debug(
            "[SubProject] Every file is at the project root (no path has a "
            "directory component), so there is no sub-project — scanning "
            "disk for a nested manifest before giving up. Memory keys: %s",
            list(all_files.keys())[:10],
        )
        # Last-resort: scan immediate subdirectories on disk for project
        # manifests. Catches Django projects created with:
        #   mkdir <dir> && cd <dir> && django-admin startproject config .
        # where the CMD pattern couldn't extract the directory name.
        _fs_manifests = (
            'manage.py', 'package.json', 'requirements.txt',
            'go.mod', 'Cargo.toml', 'Gemfile', 'pyproject.toml',
        )
        try:
            for _entry in sorted(os.scandir('.'), key=lambda e: e.name):
                if not _entry.is_dir() or _entry.name.startswith('.'):
                    continue
                if _entry.name in _internal:
                    continue
                for _manifest in _fs_manifests:
                    if os.path.isfile(os.path.join(_entry.name, _manifest)):
                        log.info(
                            "[SubProject] Detected sub-project root via "
                            "filesystem scan (%s): %s/",
                            _manifest, _entry.name,
                        )
                        return _entry.name
        except OSError:
            pass
        return None

    # Directories that are NOT sub-project roots — they are conventional
    # source subdirectories within a project.  If the common first component
    # is one of these, we should NOT treat it as a sub-project generator root.
    _NON_SUBPROJECT_DIRS = {
        'src', 'lib', 'app', 'apps', 'api', 'pkg',
        'components', 'pages', 'routes', 'views',
        'utils', 'helpers', 'hooks', 'styles', 'css',
        'public', 'static', 'assets', 'images', 'fonts',
        'tests', 'test', '__tests__', 'spec', 'specs',
        'docs', 'documentation', 'scripts', 'config',
        'dist', 'build', 'out', 'output',
        'models', 'controllers', 'services', 'repositories',
        'migrations', 'fixtures', 'seeds',
        'middleware', 'decorators', 'validators',
    }

    # Extract first path component from each file
    first_components: set[str] = set()
    for p in nested_paths:
        parts = p.replace('\\', '/').split('/')
        if len(parts) >= 2:  # must have at least dir/file
            first_components.add(parts[0])

    # If all files share the same single first directory component,
    # that's our sub-project root — unless it's a well-known source directory
    if len(first_components) == 1:
        subproject = first_components.pop()
        # Sanity check: directory must exist on disk and not be a common
        # source directory name (which would be a false positive)
        if os.path.isdir(subproject):
            # Even if the name is in _NON_SUBPROJECT_DIRS (e.g. "app"),
            # treat it as a real sub-project if it contains a manifest
            is_blocked_name = subproject in _NON_SUBPROJECT_DIRS

            # A directory that contains __init__.py is a Python package,
            # not a standalone sub-project — unless it ALSO contains a
            # project manifest (pyproject.toml, setup.py, etc.).
            _is_python_package = os.path.isfile(
                os.path.join(subproject, '__init__.py'))

            _has_manifest = False
            for manifest in ('package.json', 'Cargo.toml', 'go.mod',
                             'requirements.txt', 'Gemfile', 'pyproject.toml',
                             'composer.json', 'manage.py', 'setup.py',
                             'setup.cfg'):
                if os.path.isfile(os.path.join(subproject, manifest)):
                    _has_manifest = True
                    log.info(f"[SubProject] Detected sub-project root "
                             f"({'manifest override, ' if is_blocked_name else ''}"
                             f"{manifest}): {subproject}/")
                    return subproject

            # A manifest recorded in memory, or the directory's OWN
            # virtualenv/interpreter, is strong positive evidence of a
            # self-contained sub-project (as the on-disk manifest above).
            _self_contained = (
                _memory_has_manifest(all_files, subproject)
                or _dir_has_own_env(subproject))
            # Counter-evidence: the dir is imported as a top-level package
            # by its own files, so it lives under the root project.
            _imported_as_pkg = _dir_imported_as_package(all_files, subproject)

            if _is_python_package and not _self_contained:
                log.debug(
                    f"[SubProject] '{subproject}/' has __init__.py but no "
                    f"project manifest — treating as Python package, not "
                    f"sub-project")
            elif is_blocked_name:
                pass  # common source-dir name with no manifest — not a root
            elif _imported_as_pkg and not _self_contained:
                # Imported as a package from the repo root and no manifest
                # of its own — a package dir, not a sub-project. Treating it
                # as one prepends `cd {sub}` to its gates, breaking the very
                # import being probed and the root-relative venv path: a
                # false monotonic regression that rolls back working code.
                log.debug(
                    f"[SubProject] '{subproject}/' is imported as a top-level "
                    f"package by its own files and has no manifest — treating "
                    f"as a package under the project root, not a sub-project")
            else:
                log.info(f"[SubProject] Detected sub-project root: {subproject}/")
                return subproject

    # Fallback 1: if memory contains files from multiple top-level directories
    # (e.g. search provider added files), look for a known project manifest
    from ..executor import Executor
    manifest_dirs = set()
    for p in nested_paths:
        if os.path.basename(p) in Executor._PROTECTED_FILENAMES:
            dirname = os.path.dirname(p)
            if dirname:  # Must be a subdirectory
                manifest_dirs.add(dirname)

    if len(manifest_dirs) == 1:
        subproject = manifest_dirs.pop()
        if os.path.isdir(subproject):
            log.info(f"[SubProject] Detected sub-project root via manifest in memory: {subproject}/")
            return subproject

    # Fallback 2: scan immediate subdirectories on disk for project manifests.
    # Protected files (package.json, etc.) are often NOT in memory because
    # _strip_protected_files blocks them.  Check the filesystem directly.
    # Only consider directories that memory files reference.
    candidate_dirs = first_components if first_components else set()
    for candidate in candidate_dirs:
        if not os.path.isdir(candidate):
            continue
        for manifest in ('package.json', 'requirements.txt', 'go.mod',
                         'Cargo.toml', 'Gemfile', 'pyproject.toml',
                         'composer.json'):
            if os.path.isfile(os.path.join(candidate, manifest)):
                log.info(f"[SubProject] Detected sub-project root via "
                         f"disk manifest ({manifest}): {candidate}/")
                return candidate

    # Fallback 3: if memory has files under a common prefix but the primary
    # check failed (e.g. multiple first-components), pick the directory that
    # contains the majority of files.
    if len(first_components) > 1:
        from collections import Counter
        counts = Counter(
            p.replace('\\', '/').split('/')[0]
            for p in nested_paths
            if len(p.replace('\\', '/').split('/')) >= 2
        )
        if counts:
            best, best_count = counts.most_common(1)[0]
            total = sum(counts.values())
            # Only use majority if it covers >70% of files
            if best_count > total * 0.7 and os.path.isdir(best):
                if best in _NON_SUBPROJECT_DIRS:
                    # Blocklisted name — only allow if it contains a
                    # project manifest (same logic as single-component
                    # check at Fallback 0).
                    for manifest in ('package.json', 'Cargo.toml', 'go.mod',
                                     'requirements.txt', 'Gemfile',
                                     'pyproject.toml', 'composer.json',
                                     'manage.py'):
                        if os.path.isfile(os.path.join(best, manifest)):
                            log.info(
                                f"[SubProject] Detected sub-project root via "
                                f"majority ({best_count}/{total} files, "
                                f"manifest override {manifest}): {best}/")
                            return best
                    # No manifest — not a real sub-project
                    log.debug(
                        f"[SubProject] Majority candidate '{best}' is a "
                        f"common source directory name without a project "
                        f"manifest — skipping")
                else:
                    # A directory that contains __init__.py is a Python
                    # package, not a standalone sub-project — unless it
                    # ALSO has a project manifest.  This mirrors the
                    # single-component check in Fallback 0 above.
                    _is_py_pkg = os.path.isfile(
                        os.path.join(best, '__init__.py'))
                    if _is_py_pkg:
                        _has_mf = any(
                            os.path.isfile(os.path.join(best, m))
                            for m in ('package.json', 'Cargo.toml', 'go.mod',
                                      'requirements.txt', 'Gemfile',
                                      'pyproject.toml', 'composer.json',
                                      'manage.py', 'setup.py', 'setup.cfg')
                        )
                        if not _has_mf:
                            log.debug(
                                f"[SubProject] Majority candidate '{best}' "
                                f"has __init__.py but no project manifest "
                                f"— treating as Python package, not sub-project")
                        else:
                            log.info(
                                f"[SubProject] Detected sub-project root via "
                                f"majority ({best_count}/{total} files): {best}/")
                            return best
                    else:
                        log.info(
                            f"[SubProject] Detected sub-project root via "
                            f"majority ({best_count}/{total} files): {best}/")
                        return best

    return None


# Standard top-level directories inside a project root.  Used by the
# planner-leader detector below to avoid stripping legitimate paths
# like ``src/components/Header.jsx`` that share a leading dir name
# with no project at all.
_COMMON_PROJECT_TOP_DIRS = frozenset({
    "src", "lib", "app", "apps", "packages", "tests", "test",
    "public", "static", "assets", "dist", "build", "node_modules",
    "__tests__", "__mocks__", "bin", "scripts", "docs",
    "components", "pages", "styles", "utils", "hooks", "config",
})


def _normalize_proj_name(name: str) -> str:
    """Lower-case and strip non-alphanumerics so naming variants
    collapse to a single key.

    ``my-app``, ``my_app``, ``myApp``, ``MyApp``, and ``myapp`` all
    normalize to ``"myapp"``.  Used to detect when an inline-code path
    leader is the planner's name for a subproject that the actual
    scaffold created under a different (but equivalent) directory.
    """
    return "".join(c for c in name if c.isalnum()).lower()


def _prefix_subproject_paths(files: dict[str, str],
                             subproject: str,
                             memory: FileMemory) -> dict[str, str]:
    """Prefix file paths with the sub-project root when missing.

    When the LLM generates paths like ``components/Header.tsx`` but the
    project lives under ``my-app/``, this function rewrites them to
    ``my-app/components/Header.tsx``.

    Also detects when the LLM embeds the subproject name in the middle
    of the path (e.g. ``src/my-app/src/Header.jsx``) and reconstructs
    the correct path (``my-app/src/Header.jsx``).

    **Planner-leader rename**: when the planner emits a path like
    ``myapp/src/Header.jsx`` but the actual scaffold landed under a
    naming variant like ``my-app/`` (because the CMD-step diagnosis
    swapped the folder name), the leading ``myapp/`` is recognised as
    the planner's name for the same subproject and stripped before
    the real prefix is applied.  Without this rule the path would be
    blindly prepended to ``my-app/myapp/src/Header.jsx`` — a
    double-nested location that Vite never serves, leaving the user
    with the default scaffold in the browser while the test runner
    happily passes against the orphaned tree.

    Files that are already prefixed, already known in memory, or are
    internal tracking paths (``_cmd_output/`` etc.) are left unchanged.
    """
    if not subproject:
        return files

    prefix = subproject.rstrip('/') + '/'
    proj_name = subproject.rstrip('/')
    proj_name_norm = _normalize_proj_name(proj_name)
    known_paths = set(memory.all_files().keys())
    corrected: dict[str, str] = {}
    # Roots the plan names beside this one. Non-empty only when the plan
    # is provably root-relative, in which case `backend/app.js` means the
    # repo's backend and prefixing it to `frontend/backend/app.js` builds
    # a second, undeclared tree — the classic-path twin of the CMD
    # misroute described in `_step_belongs_to_subproject`.
    _sibling_roots = _plan_sibling_roots(memory, subproject) or set()

    # Internal tracking directories that should never be prefixed
    _INTERNAL_PREFIXES = ('_cmd_output/', '_fix_output/', '_search_context/')

    for fpath, content in files.items():
        # Skip internal tracking paths (but NOT __tests__, __mocks__, etc.)
        if fpath.startswith(_INTERNAL_PREFIXES):
            corrected[fpath] = content
            continue

        # Already has the sub-project prefix (normalise backslashes so
        # Windows paths like 'my-app\src\foo.js' match 'my-app/')
        norm_fpath = fpath.replace("\\", "/")
        if norm_fpath.startswith(prefix):
            corrected[fpath] = content
            continue

        # Declared under a sibling root the plan named itself — this path
        # is repo-root-relative and already where it belongs.
        _leader = norm_fpath.split('/')[0]
        if _leader in _sibling_roots:
            log.debug("[SubProject] Left '%s' at the repo root — '%s/' is a "
                      "plan-declared root beside '%s/'",
                      fpath, _leader, proj_name)
            corrected[fpath] = content
            continue

        # Already a known file in memory (exact match) — fpath does NOT
        # start with prefix (that case was caught above).
        if fpath in known_paths:
            # Tracked without the subproject prefix — check if the
            # prefixed version exists in memory or on disk.  If so,
            # the LLM wrote to the wrong (root-level) path in a
            # previous step; redirect this write and clean up the stale
            # memory entry so it no longer pollutes context.
            new_path = prefix + fpath
            if new_path in known_paths or os.path.isfile(new_path):
                log.warning(f"[SubProject] Redirected stale root-level path: "
                            f"'{fpath}' → '{new_path}'")
                corrected[new_path] = content
                memory.delete(fpath)
                continue
            # Legitimately at the project root (e.g. README.md written
            # before the scaffold CMD ran) — leave unchanged.
            corrected[fpath] = content
            continue

        # Detect embedded subproject name: the LLM sometimes generates
        # paths like "src/my-app/src/Header.jsx" where the project name
        # is in the middle.  Extract the suffix after the embedded name
        # and reconstruct the correct path.
        norm = fpath.replace("\\", "/")
        needle = '/' + proj_name + '/'
        embed_idx = norm.find(needle)
        if embed_idx != -1:
            suffix = norm[embed_idx + len(needle):]
            candidate = prefix + suffix
            if candidate in known_paths:
                log.warning(f"[SubProject] Fixed embedded subproject: "
                            f"'{fpath}' → '{candidate}'")
                corrected[candidate] = content
                continue

        # Planner-leader rename detection.  The planner may have used
        # one project-name spelling (e.g. ``myapp/...``) while the
        # actual scaffold landed under a normalisation-equivalent
        # variant (``my-app/``) — typically because a CMD-step
        # diagnosis fix renamed the directory.  When the inline path's
        # leading directory is a naming variant of the real subproject
        # AND the leading dir is not a standard top-level project
        # folder (``src/``, ``lib/``, etc.), strip it before applying
        # the real prefix.  Without this rule the result would be
        # double-nested at ``my-app/myapp/src/...`` and Vite would
        # serve the default scaffold instead of the user's homepage.
        if '/' in norm:
            leader, _, rest = norm.partition('/')
            if (leader.lower() not in _COMMON_PROJECT_TOP_DIRS
                    and rest
                    and _normalize_proj_name(leader) == proj_name_norm):
                new_path = prefix + rest
                log.warning(
                    f"[SubProject] Renamed planner-leader '{leader}/' → "
                    f"'{prefix}': '{fpath}' → '{new_path}' "
                    f"(prevents double-nesting)")
                corrected[new_path] = content
                continue

        # Bare filename (no directory component): the fuzzy parser may
        # extract just "Header.jsx" from LLM prose like "Here is Header.jsx:".
        # Before blindly prefixing to "subproject/Header.jsx", check memory
        # for a file whose basename matches — if exactly one exists, use it.
        norm_basename = os.path.basename(norm)
        if norm_basename == norm:  # no directory component
            basename_matches = [
                kp for kp in known_paths
                if os.path.basename(kp.replace("\\", "/")) == norm_basename
            ]
            if len(basename_matches) == 1:
                resolved = basename_matches[0]
                log.info(f"[SubProject] Resolved bare filename '{fpath}' → '{resolved}'")
                corrected[resolved] = content
                continue

        # Prefix with sub-project root
        new_path = prefix + fpath
        log.info(f"[SubProject] Prefixed '{fpath}' → '{new_path}'")
        corrected[new_path] = content

    return corrected


def _handle_search_step(step_text: str, search_agent,
                        memory: FileMemory, display: CLIDisplay,
                        step_idx: int,
                        language: str | None = None) -> tuple[bool, str]:
    """Handle a [SEARCH] step — search the web for documentation / info.

    Results are stored in memory under ``_search_context/step_N.txt`` so
    that subsequent CODE / CMD steps can reference them.  The step is always
    considered successful (best-effort, non-blocking).
    """
    if search_agent is None:
        display.step_info(step_idx, "Search agent not available, skipping.")
        log.warning(f"Step {step_idx+1}: SEARCH step but no search_agent configured.")
        return True, ""

    display.step_info(step_idx, f"Searching: {step_text[:80]}...")
    log.info(f"Step {step_idx+1}: SEARCH — {step_text}")

    try:
        result = search_agent.search_for_task(
            step_text, language=language,
            kb_context=getattr(memory, '_kb_context', ''))
    except Exception as exc:
        log.warning(f"Step {step_idx+1}: Search failed: {exc}")
        display.step_info(step_idx, "Search failed (non-blocking), continuing.")
        return True, ""

    if result:
        # Store in memory so downstream steps see the search context
        memory.update({
            f"_search_context/step_{step_idx+1}.txt": result,
        })
        display.step_info(step_idx, "Search results stored for downstream steps.")
        log.info(f"Step {step_idx+1}: Search returned {len(result)} chars of context.")
    else:
        display.step_info(step_idx, "No relevant results found.")
        log.info(f"Step {step_idx+1}: Search returned no results.")

    return True, ""


def _get_pip_cmd(cwd: str | None = None) -> str:
    """Return the venv pip if present in *cwd*, else fall back to 'pip'."""
    root = cwd or "."
    candidates = [
        os.path.join(root, "venv", "Scripts", "pip.exe"),   # Windows venv
        os.path.join(root, "venv", "bin", "pip"),            # Unix venv
        os.path.join(root, ".venv", "Scripts", "pip.exe"),
        os.path.join(root, ".venv", "bin", "pip"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return "pip"


# Shell metacharacters that mean a line is more than one command. Their
# presence makes token-level rewriting unsafe: the words after them belong
# to a different command, or are redirections, not package names.
_SHELL_OPERATOR_RE = re.compile(r'(&&|\|\||[|;<>])')


# A chain segment that upgrades pip, whether or not other packages ride
# along (`--upgrade pip`, `-U pip setuptools wheel`). The `pip`/`pip.exe`
# form is the one that breaks; `python -m pip` is already the fix.
_PIP_SELF_UPGRADE_RE = re.compile(
    r'^(?P<interp>[\w./\\]*pip3?(?:\.exe)?)\s+install\s+'
    r'(?P<rest>(?=(?:[^&|]*\s)?(?:-U|--upgrade)(?:\s|$))'
    r'(?=[^&|]*(?:\s|^)pip(?:\s|$))[^&|]*)$',
    re.IGNORECASE,
)


def _fix_pip_self_upgrade(cmd: str) -> str | None:
    """Route pip-self-upgrade segments through ``python -m pip``.

    Returns the corrected command, or ``None`` when there was nothing to
    correct.

    On Windows ``pip install --upgrade pip`` cannot replace the running
    ``pip.exe`` shim: pip refuses with "To modify pip, please run the
    following command: ...python.exe -m pip install --upgrade pip" and
    exits 1. Chained with ``&&`` that aborts every later segment — observed
    on a Pygame run where `python -m venv venv && ... && pip install
    --upgrade pip setuptools wheel && pip install pygame` left pygame
    uninstalled, and the recovery loop then spent 8 turns rediscovering the
    remedy pip had already printed verbatim.

    Rewriting rather than dropping is deliberate: `--upgrade pip setuptools
    wheel` also upgrades build tooling a later step may need, so removing
    the segment would lose real work. `python` after a `call ...activate`
    is the venv's interpreter, so the rewrite installs in the same place.
    """
    segments = [s.strip() for s in cmd.split('&&')]
    fixed: list[str] = []
    changed = False
    for seg in segments:
        m = _PIP_SELF_UPGRADE_RE.match(seg)
        if m:
            fixed.append(f"python -m pip install {m.group('rest').strip()}")
            changed = True
        else:
            fixed.append(seg)
    if not changed:
        return None
    return ' && '.join(fixed)


def _make_cmd_idempotent(
    cmd: str,
    executor: Executor,
    cwd: str | None = None,
) -> tuple[str | None, str]:
    """Check whether a setup command is redundant and skip or trim it.

    Returns ``(cmd_to_run, skip_reason)``:
    - ``cmd_to_run=None``  → skip entirely, *skip_reason* explains why.
    - ``cmd_to_run=str``   → run this (may be a trimmed version of *cmd*).
    - ``skip_reason=""``   → no skip, run normally.
    """
    stripped = cmd.strip()
    root = cwd or "."

    # ── pip self-upgrade inside a chain ──
    # `pip install --upgrade pip` cannot replace the running pip.exe on
    # Windows; route it through `python -m pip`, which is the remedy pip
    # itself prints. See _fix_pip_self_upgrade for the observed failure.
    _fixed_upgrade = _fix_pip_self_upgrade(stripped)
    if _fixed_upgrade is not None:
        log.info("[Cmd] Routing pip self-upgrade through `python -m pip` "
                 "(pip.exe cannot replace itself on Windows)")
        stripped = _fixed_upgrade
        cmd = _fixed_upgrade

    # ── venv activation — always a no-op in a subprocess ──
    if re.match(
        r'^(source\s+\S+/activate'
        r'|[\w./\\]+[/\\]Scripts[/\\]activate(\.bat)?'
        r'|\.\s+\S+/activate)$',
        stripped, re.IGNORECASE,
    ):
        return None, "venv activation is a no-op inside a subprocess"

    # ── python -m venv <dir> ──
    m = re.match(r'^python3?\s+-m\s+venv\s+(\S+)', stripped)
    if m:
        venv_dir = m.group(1)
        if os.path.isdir(os.path.join(root, venv_dir)):
            # Strip just the venv creation segment; keep the rest of a && chain
            segments = [s.strip() for s in stripped.split('&&')]
            remaining = [s for s in segments if not re.match(r'^python3?\s+-m\s+venv\s+', s)]
            if remaining:
                return ' && '.join(remaining), f"virtualenv '{venv_dir}' already exists, skipping creation"
            return None, f"virtualenv '{venv_dir}' already exists, skipping creation"
        return cmd, ""

    # ── git init ──
    if re.match(r'^git\s+init\b', stripped, re.IGNORECASE):
        if os.path.isdir(os.path.join(root, ".git")):
            return None, "git repository already initialised"
        return cmd, ""

    # ── mkdir / mkdir -p ──
    # executor.write_files() calls os.makedirs(..., exist_ok=True) for every
    # file it writes, so explicit mkdir steps are always redundant.  Skipping
    # them also avoids Windows cmd.exe treating forward-slash paths (e.g.
    # `mkdir src/components`) as invalid option flags.
    # Handles compound commands like `cd myapp && mkdir src/components` by
    # checking that every non-cd segment is a mkdir.
    _segments = [s.strip() for s in stripped.split('&&')]
    _non_cd = [s for s in _segments if not re.match(r'^cd\s+', s, re.IGNORECASE)]
    if _non_cd and all(re.match(r'^mkdir\b', s, re.IGNORECASE) for s in _non_cd):
        return None, "mkdir skipped — parent directories are created automatically on file write"

    # ── pip install <packages> ──
    # Only a STANDALONE install can be trimmed. The token scan below reads
    # everything after `install` as a flag or a package name, which is true
    # only when nothing else is on the line: given
    #
    #   pip install -U pygame && pip freeze > requirements.txt
    #
    # it classified `&&`, `pip`, `freeze` and `>` as packages and
    # `requirements.txt` as a flag, then reassembled the survivors into
    #
    #   pip install -U requirements.txt && freeze
    #
    # — a command that cannot succeed, and which became step 1's gate. The
    # agent loop then spent its whole budget trying to satisfy it (it even
    # read pip's own source to work out why `requirements.txt` was not a
    # package) while the real goal, installing pygame, had already been met.
    # The run died there, and the plan's command had been correct all along.
    if _SHELL_OPERATOR_RE.search(stripped):
        return cmd, ""
    m = re.match(
        r'^(pip3?|python3?\s+-m\s+pip)\s+install\s+(.*)',
        stripped, re.IGNORECASE,
    )
    if m:
        rest = m.group(2).strip()
        flags, pkgs = [], []
        for tok in rest.split():
            # Flags and -r/--requirement targets are not package names
            if tok.startswith("-") or tok.endswith(".txt") or tok.endswith(".cfg"):
                flags.append(tok)
            else:
                pkgs.append(tok)
        if not pkgs:
            return cmd, ""  # only flags (e.g. -r requirements.txt) — run as-is

        pip_cmd = _get_pip_cmd(root)
        missing = []
        for pkg in pkgs:
            pkg_name = re.split(r"[>=<~!\[]", pkg)[0].strip()
            if not pkg_name:
                continue
            ok, _ = executor.run_command(f'"{pip_cmd}" show {pkg_name}', cwd=cwd)
            if not ok:
                missing.append(pkg)

        if not missing:
            skipped = ", ".join(pkgs)
            return None, f"all packages already installed ({skipped}), skipping"

        if len(missing) < len(pkgs):
            skipped = ", ".join(p for p in pkgs if p not in missing)
            prefix_m = re.match(
                r'^(pip3?|python3?\s+-m\s+pip)\s+install', stripped, re.IGNORECASE
            )
            prefix = prefix_m.group(0) if prefix_m else "pip install"
            flag_str = (" " + " ".join(flags)) if flags else ""
            new_cmd = f"{prefix}{flag_str} {' '.join(missing)}"
            return new_cmd, f"skipping already-installed: {skipped}"

        return cmd, ""

    # ── npm install <specific packages> ──
    m = re.match(
        r'^npm\s+install\s+((?:--save-dev|-D|-P|--save)?\s*)(.+)',
        stripped, re.IGNORECASE,
    )
    if m:
        flags_part = m.group(1).strip()
        pkgs_part = m.group(2).strip()
        pkgs = [t for t in pkgs_part.split() if not t.startswith("-")]
        if not pkgs:
            return cmd, ""  # bare `npm install` — always run

        nm = os.path.join(root, "node_modules")
        if not os.path.isdir(nm):
            return cmd, ""  # node_modules absent — install everything

        missing = []
        for pkg in pkgs:
            # Handle scoped packages (@scope/name) and version suffixes (pkg@1.0)
            if pkg.startswith("@"):
                bare = pkg[1:].split("@")[0]
                pkg_dir = os.path.join(nm, "@" + bare.split("/")[0], bare.split("/")[1] if "/" in bare else "")
            else:
                pkg_dir = os.path.join(nm, pkg.split("@")[0])
            if not os.path.isdir(pkg_dir):
                missing.append(pkg)

        if not missing:
            return None, f"all npm packages already installed ({', '.join(pkgs)}), skipping"

        if len(missing) < len(pkgs):
            skipped = ", ".join(p for p in pkgs if p not in missing)
            flag_str = (" " + flags_part) if flags_part else ""
            new_cmd = f"npm install{flag_str} {' '.join(missing)}"
            return new_cmd, f"skipping already-installed: {skipped}"

        return cmd, ""

    return cmd, ""


_NOOP_SEGMENT_RE = re.compile(r'^(?:cd\s+\S.*|mkdir\b.*|[()\s]*)$',
                              re.IGNORECASE)


def _cmd_is_noop(cmd: str) -> bool:
    """True when every ``&&`` segment of *cmd* is cd/mkdir/blank/stray
    parens — a command that cannot create any file.

    Observed: a multi-line parenthesized echo block parsed down to
    ``cd app && (`` which exited 0 having written nothing; the missing
    vitest config only surfaced much later in BulkTest.
    """
    segs = [s.strip() for s in (cmd or "").split("&&")]
    return bool(segs) and all(_NOOP_SEGMENT_RE.match(s) for s in segs)


def _concrete_produces(plan_step) -> list[str]:
    """Declared produces that are literal checkable paths.

    Globs, entries with spaces/parens (planner annotations like
    ``vitest-report (console output only)``), and empties are skipped —
    planners routinely fabricate produces for real commands, so only a
    no-op command combined with missing concrete produces is proof of a
    silent failure.
    """
    out = []
    for pf in (getattr(plan_step, "target_files", None) or []):
        p = str(pf).replace("\\", "/").strip()
        if not p or "*" in p or "?" in p or " " in p or "(" in p:
            continue
        out.append(p)
    return out


def _handle_cmd_step(step_text: str, executor: Executor,
                     llm_client, memory: FileMemory,
                     display: CLIDisplay, step_idx: int,
                     language: str | None = None,
                     project_context=None,
                     plan_step=None,
                     intent_spec=None,
                     cfg: Config | None = None) -> tuple[bool, str]:
    # Prefer the structured plan's explicit command field
    cmd = (plan_step.command if plan_step is not None and plan_step.command
           else _extract_command_from_step(step_text))

    if cmd:
        pass  # use extracted command
    else:
        display.step_info(step_idx, "Generating command...")

        prior_context = _build_prior_steps_context(memory, step_idx)
        file_summary = memory.summary()

        gen_prompt = (
            "You are a shell command generator. Given a task step, output "
            "ONLY the shell command to accomplish it. No explanations, no "
            "markdown, no backticks — just the raw command.\n"
            f"{_shell_instructions()}\n"
        )
        # Tell the LLM about the sub-project directory so it runs commands there
        _subproject_for_prompt = _detect_subproject_root(memory)
        if _subproject_for_prompt:
            gen_prompt += (
                f"IMPORTANT: This project lives in the '{_subproject_for_prompt}/' subdirectory. "
                f"All npm/npx/yarn/pnpm/node commands MUST be prefixed with "
                f"'cd {_subproject_for_prompt} && ' to run in the correct directory.\n"
            )
        kb_ctx = getattr(memory, '_kb_context', '')
        if kb_ctx:
            gen_prompt += f"{kb_ctx}\n"
        if prior_context:
            gen_prompt += (
                f"{prior_context}"
                "IMPORTANT: Use the exact names, paths, and values from the "
                "previous steps above. Do NOT guess or use defaults.\n\n"
            )
        if file_summary != "(no files yet)":
            gen_prompt += f"Project files: {file_summary}\n\n"
        gen_prompt += f"Step: {step_text}\n\nCommand:"
        sent_before, recv_before = token_tracker.snapshot()

        cmd_response = llm_client.generate_response(gen_prompt).strip()

        sent_after, recv_after = token_tracker.snapshot()
        sent_delta = sent_after - sent_before
        recv_delta = recv_after - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        if cmd_response:
            display.add_llm_log(cmd_response, source="Coder")

        extracted = _extract_commands_from_text(cmd_response)
        if extracted:
            cmd = extracted[0]
        else:
            cmd = cmd_response.strip('`').strip()
            if not _looks_like_command(cmd):
                for line in cmd.splitlines():
                    line = line.strip('`').strip()
                    if _looks_like_command(line):
                        cmd = line
                        break
        if cmd.startswith('```'):
            cmd = cmd.split('\n', 1)[-1].rsplit('```', 1)[0].strip()

        if not cmd:
            display.step_info(step_idx, "Could not generate command, skipping.")
            log.warning(f"Step {step_idx+1}: LLM returned empty command.")
            return True, ""

    # ── Normalize command ──
    # Clean up bash-style line continuations and dangling operators
    cmd = _cleanup_shell_command(cmd)

    # ── Resolve LLM placeholders ──
    # Dumb/small LLMs sometimes output <project-name> instead of the real name.
    _task_desc = getattr(project_context, 'goal_summary', '') if project_context else ''
    cmd = resolve_cmd_placeholders(cmd, step_text=step_text, task=_task_desc,
                                   language=language)

    # ── Strip dead `cd <dir> &&` before scaffold commands ──
    # LLMs often generate `cd <name> && npm create vite@latest <name>` but the
    # directory doesn't exist yet — `npm create vite@latest` is what creates it.
    # Strip the leading `cd` so the scaffold command can run correctly.
    _SCAFFOLD_CD_RE = re.compile(
        r'^cd\s+\S+\s*&&\s*(npm\s+create\s+(?:vite|react-app|next-app|expo)\b)',
        re.IGNORECASE,
    )
    _m = _SCAFFOLD_CD_RE.match(cmd)
    if _m:
        cmd = cmd[_m.start(1):]
        log.info(f"Step {step_idx+1}: Stripped dead `cd` prefix from scaffold command")

    # ── Idempotency check ──
    # Detect the subproject root early so idempotency checks resolve
    # paths relative to the correct directory. It must be the SAME
    # directory the command will actually run in, decided below: this
    # check answers "is `npm install express` already satisfied?" by
    # looking for `node_modules/express`, and answering it about
    # `frontend/` while the command runs at the repo root would skip a
    # backend install because the frontend happened to have the package.
    # A wrong answer here does not merely mis-report — it SKIPS the work.
    _early_cwd = _detect_subproject_root(memory) or None
    if _early_cwd and _step_belongs_to_subproject(
            plan_step, _early_cwd, memory) is False:
        _early_cwd = None
    cmd, skip_reason = _make_cmd_idempotent(cmd, executor, cwd=_early_cwd)
    if cmd is None:
        log.info(f"Step {step_idx+1}: Skipping redundant command — {skip_reason}")
        display.step_info(step_idx, f"Skipped (already done): {skip_reason}")
        return True, skip_reason

    # Detect sub-project root so commands like `npm install` run in the
    # correct directory instead of the repo root.
    subproject_cwd = None
    subproject = _detect_subproject_root(memory)
    if subproject:
        import re as _re_sp
        # Commands that should run inside the sub-project directory
        _subproject_cmd_patterns = (
            r'\bnpm\s+(install|start|run|test|build|ci|pkg|exec|init)\b',
            r'\bnpx\s+',
            r'\byarn\s+(install|add|start|dev|build|test)\b',
            r'\bpnpm\s+(install|add|start|dev|build|test)\b',
            r'\bnode\s+',
            r'\bng\s+(serve|build|test|generate|add)\b',
        )
        needs_subproject = any(
            _re_sp.search(p, cmd, _re_sp.IGNORECASE)
            for p in _subproject_cmd_patterns
        )
        # Don't set cwd if the command already includes a `cd` to the subproject
        already_has_cd = f'cd {subproject}' in cmd or f'cd ./{subproject}' in cmd
        # A step the plan places under a SIBLING root must not be re-rooted
        # here. `mkdir backend && cd backend && npm init -y` matches the
        # `npm init` pattern above, and running it with cwd=frontend built
        # the whole Express backend inside the React app — see
        # `_step_belongs_to_subproject` for the measured run.
        _belongs = _step_belongs_to_subproject(plan_step, subproject, memory)
        if needs_subproject and _belongs is False:
            log.info(f"Step {step_idx+1}: NOT running in sub-project "
                     f"{subproject}/ — the plan declares this step's target(s) "
                     f"under a sibling root "
                     f"({', '.join(plan_step.target_files[:3])})")
        elif needs_subproject and not already_has_cd:
            subproject_cwd = subproject
            log.info(f"Step {step_idx+1}: Running command in sub-project: "
                     f"{subproject}/")

    # Detect if this should be a background command (e.g. starting a server).
    # Must be specific — broad keywords like "npm" or "run" cause false
    # positives that make install/build commands return before completing.
    import re as _re
    _bg_cmd_patterns = (
        r'\bnpm\s+start\b',               # npm start
        r'\bnpm\s+run\s+(dev|serve|start)\b',  # npm run dev/serve/start
        r'\bnpx\s+(next|vite|nuxt)\s+dev\b',   # npx next dev, npx vite dev
        r'\bnode\s+\S*server\S*',          # node server.js, node src/server.ts
        r'\bpython\s+\S*server\S*',        # python server.py
        r'\bpython\s+-m\s+(http\.server|flask)\b',
        r'\bflask\s+run\b',
        r'\brunserver\b',                   # manage.py runserver (Django)
        r'\buvicorn\b',
        r'\bgunicorn\b',
        r'\bng\s+serve\b',                 # Angular dev server
        r'\byarn\s+(start|dev)\b',
        r'\bpnpm\s+(start|dev)\b',
    )
    is_background = any(_re.search(p, cmd, _re.IGNORECASE) for p in _bg_cmd_patterns)

    # ── Proactive pre-install: ensure packages are installed ──
    # Runs once, just before the first server or test-suite command,
    # when the project scaffold already exists.
    if is_background and project_context is not None:
        _ensure_packages_installed(
            project_context, executor, memory, display, step_idx,
            subproject_cwd=subproject_cwd, language=language,
        )

    cwd_note = f" (in {subproject_cwd}/)" if subproject_cwd else ""
    if is_background:
        display.step_info(step_idx, f"Running background: {cmd}{cwd_note}")
        log.info(f"Step {step_idx+1}: Running background command: {cmd}")
    else:
        display.step_info(step_idx, f"Running: {cmd}{cwd_note}")
        log.info(f"Step {step_idx+1}: Running command: {cmd}")

    # Scaffold commands (ng new, create-vite, npm create, etc.) run
    # `npm install` internally, which can take 2-5 minutes on slow
    # networks.  Use a longer timeout so they don't get killed.
    _is_scaffold = bool(re.search(
        r'\b(ng\s+new|npm\s+create|npx\s+create-|npm\s+init'
        r'|npx\s+.*@angular/cli.*\s+new'    # npx @angular/cli@latest new ...
        r'|npm\s+install\b)',                 # npm install (can be slow)
        cmd, re.IGNORECASE,
    ))
    _cmd_timeout = 300 if _is_scaffold else 120

    success, output = executor.run_command(
        cmd, background=is_background, cwd=subproject_cwd,
        timeout=_cmd_timeout)
    log.info(f"Step {step_idx+1}: Command output:\n{output}")

    # ── Semantic failure check ──
    # Some CLIs exit 0 but print failure text (e.g. create-vite outputs
    # "Operation cancelled" when run in an existing non-empty directory).
    if success and output:
        _SEMANTIC_FAILURE_PATTERNS = [
            re.compile(r'Operation cancell?ed', re.IGNORECASE),
        ]
        for _sfp in _SEMANTIC_FAILURE_PATTERNS:
            if _sfp.search(output):
                log.warning(
                    f"Step {step_idx+1}: Exit code 0 but semantic failure "
                    f"detected in output ('{_sfp.pattern}'). Treating as failure."
                )
                success = False
                break

    if output:
        truncated = output[:4000] if len(output) > 4000 else output
        memory.update({
            f"_cmd_output/step_{step_idx+1}.txt": f"$ {cmd}\n\n{truncated}"
        })

    if success:
        display.step_info(step_idx, "Command succeeded.")

        # Package install commands change the API surface downstream CODE
        # steps must target — refresh the installed-version snapshot.  On a
        # blank project the initial snapshot predates the venv entirely.
        from .api_grounding import is_install_command, refresh_installed_versions
        if is_install_command(cmd):
            try:
                refresh_installed_versions(
                    project_context, executor=executor, cwd=subproject_cwd,
                    language=language)
            except Exception as _rv_exc:
                log.debug(f"[ApiGrounding] Version refresh failed: {_rv_exc}")

        # Mark scaffold files: when a project-creation command succeeds, record
        # the subproject so that _auto_fix_hazards can skip hazard checks for
        # npm/framework-generated template files on their first modification.
        # Also verify the expected sentinel file exists — some scaffold CLIs
        # (e.g. `npx create vite@latest`) exit 0 but produce nothing.
        _SCAFFOLD_PATTERNS = [
            (re.compile(r'create-next-app(?:@\S+)?\s+(\S+)'),        'package.json'),
            (re.compile(r'create-react-app\s+(\S+)'),                 'package.json'),
            (re.compile(r'create-vite(?:@\S+)?\s+(\S+)'),            'package.json'),
            (re.compile(r'npm\s+create\s+vite(?:@\S+)?\s+(\S+)'),    'package.json'),
            (re.compile(r'npx\s+create-vite(?:@\S+)?\s+(\S+)'),      'package.json'),
            (re.compile(r'create-vue(?:@\S+)?\s+(\S+)'),             'package.json'),
            (re.compile(r'npm\s+create\s+vue(?:@\S+)?\s+(\S+)'),     'package.json'),
            (re.compile(r'vue\s+create\s+(\S+)'),                     'package.json'),
            (re.compile(r'ng\s+new\s+(\S+)'),                         'package.json'),
            (re.compile(r'rails\s+new\s+(\S+)'),                      'Gemfile'),
            (re.compile(r'cargo\s+new\s+(\S+)'),                      'Cargo.toml'),
            (re.compile(r'django-admin\s+startproject\s+(\S+)'),      'manage.py'),
        ]
        for _pat, _sentinel in _SCAFFOLD_PATTERNS:
            _m = _pat.search(cmd)
            if _m:
                _candidate = _m.group(1).strip().rstrip('/')
                # `startproject NAME .` (and friends) place files in the
                # CURRENT directory, not NAME/ — an explicit `.` target
                # after the name overrides the name as the location.
                _after_toks = cmd[_m.end():].split()
                if _after_toks and _after_toks[0] in ('.', './'):
                    _candidate = '.'
                # The command may carry `cd <dir> &&` prefixes (e.g.
                # `mkdir user_portal && cd user_portal && django-admin ...`)
                # — the scaffold ran inside that directory, so the sentinel
                # lives there too. Without this the check looked for
                # ./config/manage.py while reality was user_portal/manage.py,
                # flagging a successful scaffold as failed on every run.
                _cd_dirs = [
                    d.strip().strip('"').rstrip('/\\')
                    for d in re.findall(
                        r'(?:^|&&)\s*cd\s+("[^"]+"|[^&|;]+?)\s*(?:&&|\|\||;)',
                        cmd[:_m.start()])
                ]
                _base = subproject_cwd or '.'
                for _cd in _cd_dirs:
                    if _cd not in ('.', './'):
                        _base = os.path.join(_base, _cd)
                if _candidate in ('.', './', ''):
                    _sentinel_path = os.path.join(_base, _sentinel)
                else:
                    _sentinel_path = os.path.join(_base, _candidate, _sentinel)
                if not os.path.exists(_sentinel_path):
                    log.warning(
                        f"[Scaffold] Exit code 0 but sentinel '{_sentinel}' not found "
                        f"at '{_sentinel_path}' — scaffold did not produce expected output. "
                        f"Treating as failure."
                    )
                    success = False
                elif not getattr(memory, '_scaffolded_subproject', None):
                    _marker_parts = [d for d in _cd_dirs if d not in ('.', './')]
                    if _candidate not in ('.', './', '') and not _candidate.startswith('./'):
                        _marker_parts.append(_candidate)
                    if _marker_parts:
                        _marker = "/".join(_marker_parts)
                        memory._scaffolded_subproject = _marker
                        log.info(f"[Scaffold] Marked '{_marker}' as freshly "
                                 f"scaffolded — hazard check skipped on first write")
                break

    # ── Silent no-op guard ──
    # A command that reduces to cd/mkdir/parens cannot have created the
    # files the plan declares it produces. Exit code 0 there is a silent
    # failure — surface it now, at the step where it happened, instead of
    # letting a later phase trip over the missing files.
    if success and plan_step is not None and _cmd_is_noop(cmd):
        _missing_products = [
            p for p in _concrete_produces(plan_step)
            if not os.path.exists(p)
        ]
        if _missing_products:
            log.warning(
                f"Step {step_idx+1}: command is a no-op ('{cmd}') but the "
                f"plan declares produces that don't exist: "
                f"{_missing_products} — treating as failure.")
            success = False
            output = (output or "") + (
                "\n[NoopCmd] The command had no effect (only cd/mkdir/"
                "parens) yet this step must create: "
                f"{', '.join(_missing_products)}. Create the file(s) with "
                "the correct content.")

    if success:
        display.step_info(step_idx, "Command succeeded.")
        return True, ""

    display.step_info(step_idx, "Command failed. See log.")
    log.warning(f"Step {step_idx+1}: Command failed.")
    error_info = f"Command `{cmd}` failed.\nOutput:\n{output}"

    # ── Agent-loop recovery (opt-in) ──────────────────────────────
    # The planned command is a fixed guess made at planning time; when
    # it fails, a bounded tool-loop can adapt (different command, install
    # a missing dependency) or report the blocker in plain terms.
    from .agent_loop import (
        RECOVERY_FAILED_MARKER, agent_loop_enabled, build_step_tools,
        reverifiable_cmd, run_recovery_loop,
    )
    if agent_loop_enabled(cfg, llm_client):
        display.step_info(step_idx, "Command failed — agent loop recovery")
        tools = build_step_tools(executor, memory)
        _task_goal = (getattr(project_context, "goal_summary", "")
                      if project_context else "") or step_text
        # The failed command itself is the ground truth for the recovery
        # when it is safe to re-run — without it the loop exits on the
        # model's word alone (observed: a summary describing a still-
        # failing `npm run build:css` was logged as "recovered").
        recovered, info = run_recovery_loop(
            llm_client, tools, step_text, _task_goal, error_info,
            display=display, step_idx=step_idx, language=language,
            max_turns=getattr(cfg, "AGENT_LOOP_MAX_TURNS", 8),
            verify_cmd=reverifiable_cmd(cmd),
            escalation_client=getattr(llm_client, "escalation_client",
                                      None))
        if recovered:
            log.info(f"Step {step_idx+1}: Agent loop recovered failed "
                     f"command: {info[:200]}")
            return True, ""
        return False, f"{RECOVERY_FAILED_MARKER} {error_info}\n\n{info}"

    return False, error_info


def _auto_fix_hazards(files: dict[str, str], coder: CoderAgent,
                      executor: Executor, display: CLIDisplay,
                      step_idx: int, step_text: str,
                      language: str | None = None,
                      base_dir: str = ".",
                      memory: "FileMemory | None" = None) -> dict[str, str]:
    """Scan generated files for hazardous diffs and auto-fix them.

    For each file where ``_detect_hazards`` flags problems (e.g. significant
    size reduction, dependency removal), the coder LLM is asked to produce
    a corrected version that preserves the existing content while applying
    only the intended changes.

    Returns the (potentially corrected) file dict.
    """
    fixed_files = dict(files)

    # Build the set of files our pipeline has already written, used below to
    # detect scaffold files that have never been touched by the pipeline.
    _pipeline_written: set[str] = set(memory.all_files().keys()) if memory else set()
    _scaffolded_root: str | None = getattr(memory, '_scaffolded_subproject', None)

    for filepath, new_content in list(files.items()):
        full_path = os.path.join(base_dir, filepath)
        if not os.path.isfile(full_path):
            continue  # new file, nothing to compare

        # Skip hazard check for npm/framework scaffold files on their FIRST
        # modification in this session.  When a project-creation command ran
        # (e.g. `npm create vite@latest my-app`), the generated template files
        # (App.jsx, main.jsx, index.css, …) are expected to be fully replaced.
        # Replacing a 3 KB template with a 300-byte component is intentional.
        if _scaffolded_root and filepath not in _pipeline_written:
            _norm = filepath.replace("\\", "/")
            _root_prefix = _scaffolded_root.rstrip("/") + "/"
            if _norm.startswith(_root_prefix) or _norm == _scaffolded_root:
                log.info(f"[Scaffold] Skipping hazard check for '{filepath}' "
                         f"(first write into scaffolded project '{_scaffolded_root}')")
                continue

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                old_content = f.read()
        except OSError:
            continue

        hazards = _detect_hazards(filepath, old_content, new_content)
        if not hazards:
            continue

        hazard_descriptions = "\n".join(f"- {msg}" for _, msg in hazards)
        log.warning(f"Step {step_idx+1}: Hazards detected in {filepath}:\n"
                    f"{hazard_descriptions}")
        display.step_info(step_idx, f"Hazard in {filepath}, auto-fixing...")

        fix_prompt = (
            f"You generated a new version of `{filepath}` but it has safety issues:\n"
            f"{hazard_descriptions}\n\n"
            f"EXISTING file content (DO NOT lose any of this):\n"
            f"```\n{old_content}\n```\n\n"
            f"YOUR generated version (has problems):\n"
            f"```\n{new_content}\n```\n\n"
            f"The step was: {step_text}\n\n"
            f"If this change is **intentional** (e.g. you are deliberately overwriting "
            f"a template or removing code as requested), reply with exactly `[INTENTIONAL]` "
            f"and nothing else.\n\n"
            f"Otherwise, if this is an **accidental error** (truncation, missing exports), "
            f"produce a CORRECTED version of `{filepath}` that:\n"
            f"1. Keeps ALL existing content (dependencies, imports, configs, etc.)\n"
            f"2. Only adds/changes what the step requires\n"
            f"3. Does NOT remove anything that was in the original file\n\n"
            f"#### [FILE]: {filepath}\n"
            f"```\n"
            f"(write the complete corrected file here)\n"
            f"```"
        )

        sent_before, recv_before = token_tracker.snapshot()

        _hazard_briefing = getattr(memory, '_task_briefing', '')
        _hazard_context = (
            "TASK BRIEFING (the overall goal — do not lose features listed under Preserve):\n"
            f"{_hazard_briefing}\n"
        ) if _hazard_briefing else ""
        fix_response = coder.process(fix_prompt, context=_hazard_context, language=language)

        sent_after, recv_after = token_tracker.snapshot()
        sent_delta = sent_after - sent_before
        recv_delta = recv_after - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        if "[INTENTIONAL]" in fix_response:
            log.info(f"Step {step_idx+1}: Hazard in {filepath} marked as intentional by LLM.")
            display.step_info(step_idx, f"Verified intentional change in {filepath}")
            continue

        explanation = CLIDisplay.extract_explanation(fix_response)
        if explanation:
            display.add_llm_log(explanation, source="Coder")

        fix_files = executor.parse_code_blocks(fix_response)
        if not fix_files:
            fix_files = executor.parse_code_blocks_fuzzy(fix_response)

        if filepath in fix_files:
            # Verify the fix resolved the hazard
            new_hazards = _detect_hazards(filepath, old_content, fix_files[filepath])
            if len(new_hazards) < len(hazards):
                fixed_files[filepath] = fix_files[filepath]
                log.info(f"Step {step_idx+1}: Auto-fixed hazard in {filepath} "
                         f"({len(hazards)} -> {len(new_hazards)} hazards)")
                display.step_info(step_idx, f"Fixed hazard in {filepath}")
            else:
                log.warning(f"Step {step_idx+1}: Auto-fix did not resolve hazard in "
                            f"{filepath}, keeping original generated version for user review")
        else:
            log.warning(f"Step {step_idx+1}: Auto-fix did not return {filepath}")

    return fixed_files


# ---------------------------------------------------------------------------
# Pre-install: scan imports and install missing npm packages before running
# ---------------------------------------------------------------------------

# Regex to extract npm package names from JS/TS import statements.
# Matches: import ... from 'pkg'  |  import 'pkg'  |  require('pkg')
_JS_IMPORT_PKG_RE = re.compile(
    r'(?:'
    r'import\s+.*?\s+from\s+["\']([^"\'./][^"\']*)["\']'
    r'|import\s+["\']([^"\'./][^"\']*)["\']'
    r'|require\s*\(\s*["\']([^"\'./][^"\']*)["\']'
    r')',
)


def _extract_npm_packages(files: dict[str, str]) -> set[str]:
    """Extract npm package names from import statements in JS/TS files.

    Returns the set of top-level package names (e.g. ``@testing-library/react``,
    ``react-router-dom``) referenced in the given file contents.
    Skips relative imports (starting with ``.`` or ``/``).
    """
    packages: set[str] = set()
    for content in files.values():
        for m in _JS_IMPORT_PKG_RE.finditer(content):
            raw = m.group(1) or m.group(2) or m.group(3)
            if not raw:
                continue
            # Normalize scoped packages: '@scope/pkg/subpath' → '@scope/pkg'
            if raw.startswith("@"):
                parts = raw.split("/")
                pkg = "/".join(parts[:2]) if len(parts) >= 2 else raw
            else:
                # Unscoped: 'react-router-dom/something' → 'react-router-dom'
                pkg = raw.split("/")[0]
            packages.add(pkg)
    return packages


def _get_installed_packages(subproject_cwd: str | None) -> set[str]:
    """Read package.json and return the set of declared dependency names."""
    import json as _json
    pkg_path = os.path.join(subproject_cwd, "package.json") if subproject_cwd else "package.json"
    if not os.path.isfile(pkg_path):
        return set()
    try:
        with open(pkg_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        installed = set()
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            installed.update(data.get(key, {}).keys())
        return installed
    except Exception:
        return set()


def _preinstall_missing_packages(
    test_files: dict[str, str],
    memory,
    executor,
    display,
    step_idx: int,
    language: str | None,
    subproject_cwd: str | None,
) -> None:
    """Scan test files and their imported source files for npm package
    imports, then install any that aren't in package.json.

    This prevents "Failed to resolve import" / "Cannot find package"
    errors from wasting a test run + LLM fix cycle.
    """
    if language not in ("javascript", "typescript"):
        return

    # 1. Collect all imports from test files
    all_imported = _extract_npm_packages(test_files)

    # 2. Also scan source files that tests import (transitive deps)
    source_imports = _extract_imported_sources(test_files, memory)
    if source_imports:
        all_imported |= _extract_npm_packages(source_imports)

    if not all_imported:
        return

    # 3. Compare against package.json
    installed = _get_installed_packages(subproject_cwd)

    # Built-in modules and test runner internals that don't need install
    builtins = {"vitest", "jest", "react", "react-dom", "path", "fs",
                "url", "util", "os", "child_process", "stream", "events",
                "crypto", "http", "https", "assert", "buffer", "net"}
    missing = all_imported - installed - builtins

    if not missing:
        return

    log.info(f"Step {step_idx+1}: Pre-install scan found missing packages: "
             f"{sorted(missing)}")
    display.step_info(
        step_idx,
        f"Pre-installing {len(missing)} missing packages: "
        f"{', '.join(sorted(missing))}...")

    ok, out = executor.install_packages(
        sorted(missing), tool="npm install --save-dev",
        cwd=subproject_cwd)
    if ok:
        display.step_info(step_idx, "Packages installed ✔")
    else:
        log.warning(f"Step {step_idx+1}: Pre-install failed: {out[:300]}")


# Regex to detect at least one test block in JS/TS/Python test files.
# Matches: describe(, it(, test(, def test_, @pytest, @Test
_HAS_TEST_BLOCK_RE = re.compile(
    r'(?:'
    r'\b(?:describe|it|test)\s*\('          # JS/TS: describe(, it(, test(
    r'|\bdef\s+test_'                       # Python: def test_
    r'|\b@(?:pytest\.mark|Test)\b'          # Python/Java annotations
    r'|\bcontext\s+["\']'                   # RSpec: context "..."
    r')',
)

# File names that are typically setup/config, not test suites
_SETUP_FILE_RE = re.compile(
    r'(setup|config|fixture|helper|util|mock|factory|conftest)'
    r'\.test\.',
    re.IGNORECASE,
)


def _strip_empty_test_files(files: dict[str, str]) -> dict[str, str]:
    """Remove test files that contain no actual test blocks.

    LLMs sometimes generate scaffold files like ``vitestSetup.test.js``
    that only contain configuration/setup code (imports, vi.mock calls,
    custom matchers) but no ``describe``/``it``/``test`` blocks.  These
    cause "No test suite found" errors from every test runner.

    Returns the dict with empty test files removed.  Non-test files
    (e.g. source files in a source-bug fix) are always kept.
    """
    result: dict[str, str] = {}
    for fpath, content in files.items():
        basename = fpath.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        # Only check files that look like test files
        is_test_file = (
            ".test." in basename
            or ".spec." in basename
            or basename.startswith("test_")
            or "_test." in basename
            or "_spec." in basename
        )

        if not is_test_file:
            # Not a test file (e.g. source fix) — keep as-is
            result[fpath] = content
            continue

        # Check if the file has at least one test block
        if _HAS_TEST_BLOCK_RE.search(content):
            result[fpath] = content
        else:
            log.warning(
                f"[TestFilter] Stripped '{fpath}' — no test blocks found "
                f"(setup/scaffold file)"
            )

    return result


def _quick_offline_lint(files: dict[str, str]) -> str:
    """Perform a quick offline syntax/linter check on generated files."""
    errors = []
    import os
    
    try:
        from syntax_checker import check_syntax
    except ImportError:
        check_syntax = None

    for filepath, content in files.items():
        ext = os.path.splitext(filepath)[1].lower()
        if ext.startswith('.'):
            ext = ext[1:]

        # ── Built-in Python syntax validation (Enhancement #7) ──
        # Use Python's compile() for .py files — catches syntax errors
        # before writing to disk, saving a full test run cycle.
        if ext == 'py':
            try:
                compile(content, filepath, 'exec')
            except SyntaxError as e:
                line_info = f":{e.lineno}" if e.lineno else ""
                errors.append(f"{filepath}{line_info}: SyntaxError: {e.msg}")
                continue  # skip syntax_checker for this file

        if check_syntax:
            supported_exts = {k.lstrip('.') for k in EXTENSION_MAP.keys()}
            if ext in supported_exts:
                try:
                    result = check_syntax(ext, content)
                    if hasattr(result, 'errors') and result.errors:
                        errors.append(f"{filepath} SyntaxError:\n{result.description}")
                except Exception as e:
                    errors.append(f"{filepath}: Error parsing: {str(e)}")
                    
    if errors:
        return "LINTER ERRORS FOUND:\n" + "\n".join(errors) + "\n\n"
    return ""


# ── Relative-import path resolution extensions ────────────────────
# For JS/TS files, import './Foo' can resolve to Foo.js, Foo.jsx,
# Foo.ts, Foo.tsx, Foo/index.js, Foo/index.jsx, etc.
_JS_IMPORT_EXTENSIONS = (
    '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',
    '/index.js', '/index.jsx', '/index.ts', '/index.tsx',
)

# Matches: import ... from '...', import '...', export ... from '...'
_JS_IMPORT_RE = re.compile(
    r'''(?:import|export)\s+.*?from\s+['"](\.\.?/[^'"]+)['"]'''
    r'''|import\s+['"](\.\.?/[^'"]+)['"]''',
    re.DOTALL,
)

# Matches ALL import sources (packages + relative) for duplicate detection.
# Captures the module string from: import X from 'mod', import 'mod', export from 'mod'
_JS_ANY_IMPORT_SOURCE_RE = re.compile(
    r'''(?:^|(?<=\n))\s*(?:import|export)\b[^;'"]*?from\s+['"]([^'"]+)['"]'''
    r'''|(?:^|(?<=\n))\s*import\s+['"]([^'"]+)['"]''',
    re.MULTILINE,
)

# Python relative imports: from .foo import bar, from ..pkg import baz
_PY_IMPORT_RE = re.compile(
    r'^from\s+(\.+\w[\w.]*)\s+import\b',
    re.MULTILINE,
)


# ── Duplicate-import auto-normalisation (zero LLM cost) ──────────────────────

def _parse_js_import_stmt(stmt: str):
    """Parse a single JS/TS import statement.

    Returns ``(module, default_name, named_set, namespace_name)`` or ``None``
    if the statement cannot be parsed (e.g. dynamic imports, re-exports).
    """
    # Normalise whitespace (handles multi-line imports)
    s = re.sub(r'\s+', ' ', stmt.strip().rstrip(';'))

    # Side-effect import: import 'mod'  or  import "mod"
    m = re.match(r'''import\s+['"]([^'"]+)['"]$''', s)
    if m:
        return m.group(1), None, set(), None

    # Must contain 'from'
    from_m = re.search(r'''\bfrom\s+['"]([^'"]+)['"]''', s)
    if not from_m:
        return None
    module = from_m.group(1)

    # bindings = text between 'import' and 'from ...'
    bindings = s[len('import '):s.rfind(' from ')].strip()

    default_name = None
    named_set: set[str] = set()
    namespace_name = None

    # Namespace: * as X
    ns_m = re.search(r'\*\s+as\s+(\w+)', bindings)
    if ns_m:
        namespace_name = ns_m.group(1)
        bindings = (bindings[:ns_m.start()] + bindings[ns_m.end():]).strip().strip(',').strip()

    # Named: { X, Y as Z, ... }
    named_m = re.search(r'\{([^}]*)\}', bindings)
    if named_m:
        for part in named_m.group(1).split(','):
            part = part.strip()
            if part:
                named_set.add(part)
        bindings = (bindings[:named_m.start()] + bindings[named_m.end():]).strip().strip(',').strip()

    # Whatever remains is the default binding (must be a single identifier)
    if re.match(r'^\w+$', bindings):
        default_name = bindings

    return module, default_name, named_set, namespace_name


def _render_js_import_stmt(module: str, default_name, named_set, namespace_name,
                            quote: str = '"') -> str:
    """Render a merged import statement as a string (no trailing newline)."""
    parts: list[str] = []
    if default_name:
        parts.append(default_name)
    if namespace_name:
        parts.append(f'* as {namespace_name}')
    if named_set:
        parts.append('{ ' + ', '.join(sorted(named_set)) + ' }')
    if not parts:
        return f'import {quote}{module}{quote};'
    return f'import {", ".join(parts)} from {quote}{module}{quote};'


def _dedup_js_imports(content: str) -> tuple[str, list[str]]:
    """Merge duplicate/split ``import`` statements in a JS/TS source file.

    Scans the leading import block (blank lines and ``//`` comments allowed
    between statements).  For each module imported more than once the
    bindings are merged into a single statement at the first occurrence;
    subsequent duplicates are removed.

    Returns ``(new_content, descriptions)`` where *descriptions* is a list
    of human-readable change summaries (empty when nothing changed).
    """
    from collections import OrderedDict

    lines = content.splitlines(keepends=True)
    n = len(lines)

    # ── Pass 1: collect import statement line-ranges from the top block ──
    import_ranges: list[tuple[int, int, str]] = []  # (start, end_excl, text)
    i = 0
    while i < n:
        stripped = lines[i].strip()
        # Allow blank lines and line comments between imports
        if not stripped or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            i += 1
            continue
        if not re.match(r'import\s', stripped):
            break  # first non-import, non-blank, non-comment line → stop

        start = i
        stmt_parts = [lines[i]]
        # Handle multi-line imports: accumulate until braces balance
        brace_depth = stripped.count('{') - stripped.count('}')
        i += 1
        while i < n and brace_depth > 0:
            stmt_parts.append(lines[i])
            brace_depth += lines[i].count('{') - lines[i].count('}')
            i += 1
        import_ranges.append((start, i, ''.join(stmt_parts)))

    if not import_ranges:
        return content, []

    # ── Pass 2: group by module ──
    by_module: dict[str, list] = OrderedDict()
    for start, end, stmt in import_ranges:
        parsed = _parse_js_import_stmt(stmt)
        if parsed is None:
            continue  # unparseable — leave untouched
        module = parsed[0]
        by_module.setdefault(module, []).append((start, end, stmt, parsed))

    if not any(len(v) > 1 for v in by_module.values()):
        return content, []  # nothing to do

    # ── Pass 3: compute replacements ──
    replacements: dict[int, tuple[int, str]] = {}  # start → (end, new_text)
    changes: list[str] = []

    for module, entries in by_module.items():
        if len(entries) == 1:
            continue

        # Detect quote style from the first statement
        first_stmt = entries[0][2]
        quote = '"' if f'"{module}"' in first_stmt else "'"

        # Merge all bindings
        merged_default = None
        merged_named: set[str] = set()
        merged_namespace = None
        for _, _, _, (_, default, named, namespace) in entries:
            if default and merged_default is None:
                merged_default = default
            merged_named |= named
            if namespace and merged_namespace is None:
                merged_namespace = namespace

        merged_line = _render_js_import_stmt(
            module, merged_default, merged_named, merged_namespace, quote)

        # Preserve indentation of the first statement
        raw_first = entries[0][2]
        indent = raw_first[:len(raw_first) - len(raw_first.lstrip())]

        # First occurrence → merged statement
        first_start, first_end = entries[0][0], entries[0][1]
        replacements[first_start] = (first_end, indent + merged_line + '\n')
        # Subsequent occurrences → delete
        for start, end, _, _ in entries[1:]:
            replacements[start] = (end, '')

        changes.append(
            f"Merged {len(entries)}× import from '{module}' → {merged_line}"
        )

    # ── Pass 4: reconstruct file ──
    result: list[str] = []
    i = 0
    while i < n:
        if i in replacements:
            end_idx, new_text = replacements[i]
            if new_text:
                result.append(new_text)
            i = end_idx
        else:
            result.append(lines[i])
            i += 1

    new_content = ''.join(result)
    # Clean up extra blank lines left by deletions
    new_content = re.sub(r'\n{3,}', '\n\n', new_content)
    return new_content, changes


def _auto_dedup_imports(files: dict[str, str], display, step_idx: int) -> dict[str, str]:
    """Run duplicate-import normalisation on all JS/TS files in *files*.

    Modifies content in-place (returns the same dict with updated values).
    Logs a step-info message for each file where changes were made.
    Zero LLM calls.
    """
    _JS_TS_EXTS = {'.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'}
    result = dict(files)
    for filepath, content in files.items():
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in _JS_TS_EXTS:
            continue
        new_content, changes = _dedup_js_imports(content)
        if changes:
            result[filepath] = new_content
            for desc in changes:
                log.info("[DedupeImports] %s: %s", filepath, desc)
            display.step_info(step_idx,
                              f"[DedupeImports] {os.path.basename(filepath)}: "
                              f"{len(changes)} import(s) merged")
    return result


def _validate_import_paths(
    files: dict[str, str],
    memory: "FileMemory",
) -> str:
    """Check that relative imports in generated files resolve to known files.

    Validates JS/TS/JSX/TSX and Python files.  Returns an error string
    describing broken imports, or empty string if all imports resolve.
    """
    # Build a set of all known file paths (memory + newly generated)
    all_paths: set[str] = set(memory.all_files().keys()) | set(files.keys())
    # Normalise to forward slashes for matching
    all_paths_norm: set[str] = {p.replace("\\", "/") for p in all_paths}

    errors: list[str] = []

    for filepath, content in files.items():
        ext = os.path.splitext(filepath)[1].lower()
        dir_of_file = os.path.dirname(filepath).replace("\\", "/")

        if ext in ('.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'):
            for m in _JS_IMPORT_RE.finditer(content):
                import_path = m.group(1) or m.group(2)
                if not import_path:
                    continue

                resolved = os.path.normpath(
                    os.path.join(dir_of_file, import_path)
                ).replace("\\", "/")

                # Check if it resolves directly (with extension already)
                if resolved in all_paths_norm:
                    continue

                # Also check disk — scaffold/static files (react.svg, hero.png,
                # etc.) are created by CMD steps and never tracked in memory.
                if os.path.isfile(resolved):
                    continue

                # Try appending standard extensions
                found = False
                for try_ext in _JS_IMPORT_EXTENSIONS:
                    if (resolved + try_ext) in all_paths_norm:
                        found = True
                        break
                    if os.path.isfile(resolved + try_ext):
                        found = True
                        break
                if found:
                    continue

                # Try to find the correct path by basename
                basename = os.path.basename(resolved)
                suggestions = [
                    p for p in all_paths_norm
                    if os.path.basename(p).startswith(basename + '.')
                    or os.path.basename(p) == basename
                    or p.endswith('/' + basename + '.jsx')
                    or p.endswith('/' + basename + '.js')
                    or p.endswith('/' + basename + '.tsx')
                    or p.endswith('/' + basename + '.ts')
                ]
                if suggestions:
                    # Calculate what the correct import should be
                    best = suggestions[0]
                    correct_rel = os.path.relpath(
                        best, dir_of_file
                    ).replace("\\", "/")
                    # Strip extension for JS imports
                    correct_rel_no_ext = re.sub(
                        r'\.(jsx?|tsx?|mjs|cjs)$', '', correct_rel
                    )
                    if not correct_rel_no_ext.startswith('.'):
                        correct_rel_no_ext = './' + correct_rel_no_ext
                    errors.append(
                        f"{filepath}: import '{import_path}' not found. "
                        f"Did you mean '{correct_rel_no_ext}'? "
                        f"(actual file: {best})"
                    )
                else:
                    errors.append(
                        f"{filepath}: import '{import_path}' does not "
                        f"resolve to any known file."
                    )

            # ── Duplicate import check ──────────────────────────────
            # Catch LLM-generated duplicate `import X from 'mod'` lines.
            # Count how many times each source module appears.
            from collections import Counter
            import_sources = [
                (m.group(1) or m.group(2)).strip()
                for m in _JS_ANY_IMPORT_SOURCE_RE.finditer(content)
                if (m.group(1) or m.group(2))
            ]
            for module, count in Counter(import_sources).items():
                if count > 1:
                    errors.append(
                        f"{filepath}: duplicate import from '{module}' "
                        f"({count} times) — merge into a single import statement."
                    )

        elif ext == '.py':
            for m in _PY_IMPORT_RE.finditer(content):
                rel_import = m.group(1)
                # Count leading dots
                dots = len(rel_import) - len(rel_import.lstrip('.'))
                module_path = rel_import[dots:]
                if not module_path:
                    continue

                # Walk up `dots` directories
                base = dir_of_file
                for _ in range(dots - 1):
                    base = os.path.dirname(base)

                py_path = os.path.join(
                    base, module_path.replace('.', '/')
                ).replace("\\", "/")

                # Check module.py or module/__init__.py
                if (py_path + '.py') in all_paths_norm:
                    continue
                if (py_path + '/__init__.py') in all_paths_norm:
                    continue

                errors.append(
                    f"{filepath}: relative import '{rel_import}' does not "
                    f"resolve to any known file."
                )

    if errors:
        return "IMPORT ERRORS FOUND:\n" + "\n".join(errors) + "\n\n"
    return ""


# ── Package import detection and auto-install ─────────────────────

# Matches ALL JS/TS imports (package AND relative):
#   import X from 'pkg'  |  import 'pkg'  |  export { X } from 'pkg'
#   Also handles dynamic: await import('pkg')
_JS_ALL_IMPORT_RE = re.compile(
    r'''(?:import|export)\s+.*?from\s+['"]([^'"]+)['"]'''
    r'''|import\s+['"]([^'"]+)['"]'''
    r'''|import\(\s*['"]([^'"]+)['"]\s*\)''',
    re.DOTALL,
)

# Node built-in modules — never need npm install
_NODE_BUILTINS = frozenset({
    'assert', 'async_hooks', 'buffer', 'child_process', 'cluster',
    'console', 'constants', 'crypto', 'dgram', 'diagnostics_channel',
    'dns', 'domain', 'events', 'fs', 'http', 'http2', 'https',
    'inspector', 'module', 'net', 'os', 'path', 'perf_hooks',
    'process', 'punycode', 'querystring', 'readline', 'repl',
    'stream', 'string_decoder', 'sys', 'timers', 'tls', 'trace_events',
    'tty', 'url', 'util', 'v8', 'vm', 'wasi', 'worker_threads', 'zlib',
    # node: protocol prefixed versions are caught by startswith check
})


def _extract_npm_package_name(specifier: str) -> str | None:
    """Extract the npm package name from an import specifier.

    Examples:
        'react'                       → 'react'
        'react-router-dom'            → 'react-router-dom'
        '@heroicons/react/outline'    → '@heroicons/react'
        '@testing-library/react'      → '@testing-library/react'
        './App'                       → None  (relative)
        'fs'                          → None  (built-in)
        'node:path'                   → None  (built-in)
    """
    if not specifier:
        return None

    # Skip relative imports
    if specifier.startswith('.') or specifier.startswith('/'):
        return None

    # Skip node: protocol
    if specifier.startswith('node:'):
        return None

    # Scoped packages: @scope/package/subpath → @scope/package
    if specifier.startswith('@'):
        parts = specifier.split('/')
        if len(parts) >= 2:
            pkg_name = parts[0] + '/' + parts[1]
        else:
            return None  # malformed
    else:
        # Regular packages: package/subpath → package
        pkg_name = specifier.split('/')[0]

    # Skip Node built-ins
    if pkg_name in _NODE_BUILTINS:
        return None

    return pkg_name


def _auto_install_code_imports(
    files: dict[str, str],
    executor: Executor,
    memory: "FileMemory",
    display: CLIDisplay,
    step_idx: int,
) -> None:
    """Scan generated JS/TS files for package imports and auto-install missing ones.

    After the coder generates code, it may import packages that weren't
    listed in ``required_packages`` (e.g. ``@heroicons/react``).  This
    function detects those imports, checks ``package.json``, and runs
    ``npm install`` for anything missing.
    """
    # Only process JS/TS files
    js_exts = {'.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'}
    imported_packages: set[str] = set()

    for filepath, content in files.items():
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in js_exts:
            continue

        for m in _JS_ALL_IMPORT_RE.finditer(content):
            specifier = m.group(1) or m.group(2) or m.group(3)
            pkg = _extract_npm_package_name(specifier)
            if pkg:
                imported_packages.add(pkg)

    if not imported_packages:
        return

    # Also scan existing memory files — packages already imported before
    # this step have presumably been installed already, so don't re-install
    for filepath, content in memory.all_files().items():
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in js_exts:
            continue
        for m in _JS_ALL_IMPORT_RE.finditer(content):
            specifier = m.group(1) or m.group(2) or m.group(3)
            pkg = _extract_npm_package_name(specifier)
            if pkg:
                imported_packages.add(pkg)

    # Read package.json to see what's already installed
    subproject_cwd = _detect_subproject_root(memory)
    root = subproject_cwd or "."
    pkg_json_path = os.path.join(root, "package.json")

    installed: set[str] = set()
    if os.path.isfile(pkg_json_path):
        try:
            with open(pkg_json_path, "r", encoding="utf-8") as f:
                pkg_data = json.loads(f.read())
            installed.update(pkg_data.get("dependencies", {}).keys())
            installed.update(pkg_data.get("devDependencies", {}).keys())
            installed.update(pkg_data.get("peerDependencies", {}).keys())
        except Exception:
            pass
    else:
        # No package.json — can't determine what's installed
        return

    missing = sorted(imported_packages - installed)
    if not missing:
        return

    cmd = f"npm install {' '.join(missing)}"
    cwd_note = f" (in {subproject_cwd}/)" if subproject_cwd else ""
    display.step_info(step_idx,
                      f"Auto-installing {len(missing)} imported package(s){cwd_note}")
    log.info(f"Step {step_idx+1}: Auto-installing packages from code imports: "
             f"{', '.join(missing)}")

    ok, output = executor.run_command(cmd, cwd=subproject_cwd)
    if ok:
        log.info(f"Step {step_idx+1}: Auto-install succeeded: {', '.join(missing)}")
        display.step_info(step_idx,
                          f"Auto-installed {len(missing)} package(s): "
                          f"{', '.join(missing)}")
    else:
        log.warning(f"Step {step_idx+1}: Auto-install failed (non-fatal): "
                    f"{output[:300]}")


# ---- ANSI code pattern (shared) ----
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def _parse_test_counts(output: str) -> tuple[int, int, list[dict]]:
    """Parse test runner output into (passed, total, failures).

    Works with pytest, Jest, Vitest, Go test, RSpec, and generic patterns.
    *failures* is a list of ``{"name": str, "message": str}`` dicts (max 5).
    """
    if not output:
        return 1, 1, []
    clean = _ANSI_RE.sub('', output)

    passed = failed = 0

    # ── Jest / Vitest: "Tests: 5 failed, 3 passed, 8 total" ──
    m = re.search(r'Tests:\s*(?:(\d+)\s+failed,\s*)?(\d+)\s+passed', clean)
    if m:
        failed = int(m.group(1) or 0)
        passed = int(m.group(2))
        total  = passed + failed
        if total:
            return passed, total, _extract_failure_names(clean)

    # ── pytest: "3 passed, 2 failed" or "5 passed" ──
    m_p = re.search(r'(\d+)\s+passed', clean)
    m_f = re.search(r'(\d+)\s+failed', clean)
    if m_p or m_f:
        passed = int(m_p.group(1)) if m_p else 0
        failed = int(m_f.group(1)) if m_f else 0
        total  = passed + failed
        if total:
            return passed, total, _extract_failure_names(clean) if failed else []

    # ── Vitest summary: "5 | 3 passed (8)" ──
    m = re.search(r'(\d+)\s+passed\s*\((\d+)\)', clean)
    if m:
        passed = int(m.group(1))
        total  = int(m.group(2))
        failed = total - passed
        return passed, total, _extract_failure_names(clean) if failed else []

    # ── Go test: count PASS/FAIL lines ──
    pass_count = len(re.findall(r'--- PASS:', clean))
    fail_count = len(re.findall(r'--- FAIL:', clean))
    if pass_count or fail_count:
        return pass_count, pass_count + fail_count, _extract_failure_names(clean)

    # ── RSpec: "5 examples, 2 failures" ──
    m = re.search(r'(\d+)\s+example[s]?,\s*(\d+)\s+failure', clean)
    if m:
        total  = int(m.group(1))
        failed = int(m.group(2))
        return total - failed, total, _extract_failure_names(clean)

    # ── Python unittest: "Ran 8 tests" + "OK" / "FAILED (failures=1, errors=9)" ──
    # Without this branch every failing unittest run collapsed to the (0, 1)
    # fallback below, so a suite with nine errors and one with a single
    # assertion failure scored identically. That made "did the last fix
    # help?" unanswerable for any unittest project — measured on a Pac-Man
    # run whose correct fix took the suite from 9 errors + 1 failure to 1
    # failure and was discarded as no progress.
    m_ran = re.search(r'Ran\s+(\d+)\s+tests?\b', clean)
    if m_ran:
        total = int(m_ran.group(1))
        failed = 0
        m_res = re.search(r'FAILED\s*\(([^)]*)\)', clean)
        if m_res:
            for kind in ('failures', 'errors'):
                mk = re.search(rf'\b{kind}\s*=\s*(\d+)', m_res.group(1))
                if mk:
                    failed += int(mk.group(1))
            # "FAILED (...)" with an unparsable detail still means failure.
            failed = failed or 1
        if total:
            # subTest can report more failures than there are test methods,
            # so clamp rather than return a negative pass count.
            return max(0, total - failed), total, _extract_failure_names(clean)

    # ── Fallback: overall pass/fail ──
    overall_pass = not re.search(r'\bFAIL\b|\bFAILED\b', clean, re.IGNORECASE)
    return (1, 1, []) if overall_pass else (0, 1, _extract_failure_names(clean))


def _extract_failure_names(clean: str) -> list[dict]:
    """Pull up to 5 individual failing test names from stripped test output."""
    names: list[dict] = []
    _PATTERNS = [
        re.compile(r'FAILED\s+([\w/:. -]+)'),          # pytest
        re.compile(r'×\s+([\w/:. -]+)'),               # Vitest ×
        re.compile(r'✕\s+([\w/:. -]+)'),               # Jest ✕
        re.compile(r'✗\s+([\w/:. -]+)'),               # generic
        re.compile(r'--- FAIL:\s+([\w/.]+)'),           # Go
        re.compile(r'it\s+[\'"](.+?)[\'"].*failed'),   # Mocha/Jest
        re.compile(r'^(?:FAIL|ERROR):\s+([\w.]+)', re.M),  # Python unittest
    ]
    seen: set[str] = set()
    for pat in _PATTERNS:
        for m in pat.finditer(clean):
            name = m.group(1).strip()[:60]
            if name and name not in seen:
                seen.add(name)
                names.append({"name": name, "message": ""})
                if len(names) >= 5:
                    return names
    return names


def _extract_test_error(output: str, max_chars: int = 1500) -> str:
    """Extract actionable error info from verbose test runner output.

    Strips ANSI escape codes, drops DOM/HTML dumps and deep stack traces,
    and keeps: error type + message, the failing source line, framework
    suggestions, and the test summary.  Works generically across Jest,
    Vitest, pytest, Go test, RSpec, etc.

    Returns at most *max_chars* of useful error context.
    """
    if not output:
        return ""

    # 1. Strip ANSI colour codes
    clean = _ANSI_RE.sub('', output)
    lines = clean.splitlines()

    kept: list[str] = []
    skip_block = False  # True while inside a DOM / HTML dump block

    # Patterns for lines we always want to keep
    _KEEP_PATTERNS = re.compile(
        r'(FAIL\b|FAILED\b|PASS\b|ERROR\b|Error[:\s]|error[:\s]'
        r'|TypeError|ReferenceError|SyntaxError|NameError'
        r'|ModuleNotFoundError|ImportError|AttributeError'
        r'|IndentationError|FileNotFoundError|AssertionError'
        r'|AssertError|KeyError|ValueError|TestingLibrary'
        r'|expect\(|Expected\b|Received\b|Difference:'
        r'|× |✕ |✗ |✓ |✔ '
        r'|Tests:|Test Files|Test Suites|Duration'
        r'|\d+ (failed|passed|skipped|pending)'
        r'|RUNS?\b)',
        re.IGNORECASE
    )

    # Patterns for the source pointer block (e.g. "  > 24 | const x = ...")
    _SOURCE_PTR = re.compile(r'^\s*(>\s*)?\d+\s*\|')

    # Patterns for suggestion / hint lines
    _SUGGESTION = re.compile(
        r'(\(If this is intentional|Hint:|Did you mean'
        r'|To fix|Possible fix|Consider using|use the .+variant)',
        re.IGNORECASE
    )

    # Patterns indicating a DOM / HTML dump to skip
    _DOM_LINE = re.compile(
        r'^\s*<(div|span|a|button|nav|header|footer|section|svg|path|ul|li|ol|img|form|input|p|h[1-6])'
        r'|^\s*\[\d+m<'  # Residual ANSI-prefixed HTML
        r'|^\s*Ignored nodes:',
        re.IGNORECASE
    )

    # Stack trace lines to skip (deep frames only, keep top 2)
    _STACK_FRAME = re.compile(r'^\s*(at |❯ |\.\.\.\s*\d+ more)')
    stack_frame_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            skip_block = False
            stack_frame_count = 0
            continue

        # Detect and skip DOM/HTML dump blocks
        if _DOM_LINE.match(stripped):
            if not skip_block:
                kept.append('    [... DOM/HTML output truncated ...]')
                skip_block = True
            continue

        if skip_block:
            # Still inside a DOM dump — skip unless we hit a keeper
            if _KEEP_PATTERNS.search(stripped) or _SUGGESTION.search(stripped):
                skip_block = False
            else:
                continue

        # Limit stack trace depth: keep first 2 frames, skip the rest
        if _STACK_FRAME.match(stripped):
            stack_frame_count += 1
            if stack_frame_count > 2:
                if stack_frame_count == 3:
                    kept.append('    [... stack frames truncated ...]')
                continue

        # Always keep: error messages, source pointers, suggestions, summaries
        if (_KEEP_PATTERNS.search(stripped)
                or _SOURCE_PTR.match(stripped)
                or _SUGGESTION.search(stripped)):
            kept.append(line)
            continue

        # Keep lines that are short context around errors (e.g. test names)
        if len(stripped) < 120:
            kept.append(line)

    result = '\n'.join(kept).strip()

    # Final cap
    if len(result) > max_chars:
        result = result[:max_chars] + '\n... [truncated]'

    return result if result else output[:max_chars]





# ---- Batch error summary for efficient test fixing ----

# Regex to match common error type lines across test runners
_ERROR_TYPE_RE = re.compile(
    r'(ModuleNotFoundError|ImportError|SyntaxError|NameError|'
    r'TypeError|AttributeError|IndentationError|FileNotFoundError|'
    r'AssertionError|AssertError|KeyError|ValueError|ReferenceError|'
    r'RangeError|RuntimeError|OSError|IOError|PermissionError|'
    r'expect\(received\))'
)

# Test name patterns for pytest, Jest/Vitest, Go, RSpec
_TEST_NAME_PATTERNS = [
    # pytest: "FAILED tests/test_foo.py::test_bar - Error..."
    re.compile(r'FAILED\s+([\w/.]+::\S+)'),
    # pytest short: "tests/test_foo.py::test_bar FAILED"
    re.compile(r'([\w/.]+::\S+)\s+FAILED'),
    # Jest/Vitest: "✕ test name" or "× test name" or "✗ test name"
    re.compile(r'[×✕✗]\s+(.+)'),
    # Jest: "FAIL path/to/test.js"
    re.compile(r'FAIL\s+([\w/.]+\.\w+)'),
    # Go: "--- FAIL: TestName"
    re.compile(r'---\s+FAIL:\s+(\S+)'),
    # RSpec: "rspec ./spec/foo_spec.rb:42"
    re.compile(r'rspec\s+([\w/.]+:\d+)'),
]


def _build_scoped_test_cmd(
    base_cmd: str,
    test_files: dict[str, str],
    subproject_cwd: str | None = None,
) -> str:
    """Build a test command scoped to only the files from this step.

    Appends file paths to the base test command so the runner only
    executes the tests that were written/modified in the current step,
    rather than the entire test suite.

    Supports pytest, vitest, jest, go test, and rspec.
    """
    if not test_files:
        return base_cmd

    # Get file paths relative to the test runner's working directory
    scoped_paths: list[str] = []
    for fpath in test_files:
        # If subproject_cwd is set, file paths in memory are already
        # prefixed with it (e.g. "react-app/src/__tests__/Foo.test.jsx").
        # The test command runs inside subproject_cwd, so strip the prefix.
        if subproject_cwd:
            prefix = subproject_cwd.rstrip("/\\") + "/"
            prefix_back = subproject_cwd.rstrip("/\\") + "\\"
            if fpath.startswith(prefix):
                fpath = fpath[len(prefix):]
            elif fpath.startswith(prefix_back):
                fpath = fpath[len(prefix_back):]
        # Use forward slashes for consistency
        scoped_paths.append(fpath.replace("\\", "/"))

    if not scoped_paths:
        return base_cmd

    # Build the scoped command based on the runner
    base_lower = base_cmd.lower()

    # pytest: `pytest file1.py file2.py`
    # vitest: `npx vitest run file1.test.jsx file2.test.jsx`
    # jest:   `npx jest file1.test.jsx file2.test.jsx`
    # go:     `go test ./path/to/...` (different pattern, skip scoping)
    # rspec:  `rspec file1_spec.rb file2_spec.rb`

    if "go test" in base_lower:
        # Go test uses package paths, not file paths — skip scoping
        return base_cmd

    def _dotted(paths: list[str]) -> list[str]:
        out: list[str] = []
        for p in paths:
            while p.startswith("./") or p.startswith(".\\"):
                p = p[2:]
            if p.endswith(".py"):
                p = p[:-3]
            out.append(p.replace("/", ".").replace("\\", "."))
        return out

    # Django's manage.py test requires dotted module names, not file paths.
    # Convert e.g. "tests/test_items_api.py" → "tests.test_items_api"
    if "manage.py test" in base_lower:
        return f"{base_cmd} {' '.join(_dotted(scoped_paths))}"

    # `python -m unittest` has the same requirement, and `discover` makes it
    # worse: its positional argument is the START DIRECTORY, so appending a
    # file path produces `unittest discover -v tests/test_x.py`, which fails
    # with "Start directory is not importable" — the tests never run at all.
    # Observed on a Pac-Man run where every TEST step reported 0/2 files
    # passed for this reason and nothing was wrong with the tests.
    #
    # Scoping to specific modules means discovery is not wanted here: drop
    # `discover` and its flags and name the modules directly.
    if "unittest" in base_lower:
        verbose = " -v" if re.search(r"(?:^|\s)-v\b", base_cmd) else ""
        interp = base_cmd.split("-m", 1)[0].strip() or "python"
        return (f"{interp} -m unittest{verbose} "
                f"{' '.join(_dotted(scoped_paths))}")

    # For all other runners, append file paths
    path_args = " ".join(scoped_paths)
    return f"{base_cmd} {path_args}"




# ---------------------------------------------------------------------------
# Per-file failure identification (for focused, per-file test fixing)
# ---------------------------------------------------------------------------



def _extract_file_specific_errors(
    output: str,
    test_file_path: str,
    max_chars: int = 3000,
) -> str:
    """Extract error output specific to a single test file.

    Scans test runner output for error blocks that mention the given
    file and captures only the relevant error context.  Falls back to
    the generic ``_extract_test_error`` if file-specific extraction
    yields nothing.
    """
    if not output:
        return ""

    clean = _ANSI_RE.sub('', output)
    basename = test_file_path.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]

    lines = clean.splitlines()
    relevant: list[str] = []
    in_file_block = False
    blank_count = 0

    # Pattern to detect another test file's summary block starting
    _other_file_re = re.compile(
        r'^\s*[✓❯]\s+\S+\.(?:test|spec)\.'
        r'|^\s*(?:FAIL|PASS)\s+\S+\.(?:test|spec)\.'
    )

    for line in lines:
        stripped = line.strip()

        # Start/continue capturing when we see the failing file referenced
        if basename in line:
            in_file_block = True
            blank_count = 0
            relevant.append(line)
            continue

        if in_file_block:
            if not stripped:
                blank_count += 1
                if blank_count >= 3:
                    in_file_block = False
                else:
                    relevant.append(line)
                continue

            # Another test file's block starting — stop capturing
            if _other_file_re.match(stripped) and basename not in stripped:
                in_file_block = False
                continue

            blank_count = 0
            relevant.append(line)

    result = '\n'.join(relevant).strip()

    if len(result) > max_chars:
        result = result[:max_chars] + '\n... [truncated]'

    return result if result else _extract_test_error(output, max_chars)


def _build_batch_error_summary(output: str, max_chars: int = 6000) -> str:
    """Build a structured, compact summary of ALL test failures.

    Groups failures by error type so the LLM can identify shared root causes
    (e.g. wrong import path affecting 10 tests → one fix). Works across
    pytest, Jest, Vitest, Go test, and RSpec output formats.

    Returns at most *max_chars* of structured error context.
    """
    if not output:
        return ""

    clean = _ANSI_RE.sub('', output)
    lines = clean.splitlines()

    # Collect individual failure blocks
    failures: list[dict] = []       # [{test, error_type, message, file, line}]
    error_groups: dict[str, list] = {}  # error_type -> [failure indices]

    # Extract test names that failed
    failed_tests: list[str] = []
    for line in lines:
        for pat in _TEST_NAME_PATTERNS:
            m = pat.search(line)
            if m:
                failed_tests.append(m.group(1).strip())
                break

    # Pattern for file:line references
    _file_line_re = re.compile(
        r'(?:File\s+["\'](.+?)["\'],\s+line\s+(\d+)'  # Python
        r'|(\S+\.\w+):(\d+))'                           # JS/Go/general
    )

    # ── Pre-pass: capture Vitest/Jest file-level failures ──────────
    # Vitest/Jest output "FAIL path/to/file.ext" for files that failed
    # to compile or import.  These often don't have a standard error
    # type; instead the error (e.g. "SyntaxError", "Cannot find module")
    # appears on subsequent lines.  We collect them into a dedicated
    # "FileLoadError" group so the LLM sees them.
    _FAIL_FILE_RE = re.compile(r'^(?:FAIL)\s+([\w/.@-]+\.\w+)')
    _seen_fail_files: set[str] = set()

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        m = _FAIL_FILE_RE.match(stripped)
        if m:
            fail_file = m.group(1)
            if fail_file not in _seen_fail_files:
                _seen_fail_files.add(fail_file)
                # Gather the next non-blank lines as the error message
                err_lines: list[str] = []
                j = i + 1
                while j < len(lines) and j < i + 8:
                    s = lines[j].strip()
                    if not s:
                        break
                    err_lines.append(s)
                    j += 1
                err_msg = ' '.join(err_lines).strip()
                if len(err_msg) > 200:
                    err_msg = err_msg[:200] + '...'

                # Try to identify the actual error type from the message
                et_m = _ERROR_TYPE_RE.search(err_msg)
                err_type = et_m.group(1) if et_m else 'FileLoadError'

                failure = {
                    'test': fail_file,
                    'error_type': err_type,
                    'message': err_msg or '(file failed to load/compile)',
                    'file': fail_file,
                    'line': None,
                }
                failures.append(failure)
                error_groups.setdefault(err_type, []).append(
                    len(failures) - 1)
        i += 1

    # ── Main pass: extract error details from assertion/runtime failures ──
    current_test = None
    current_error_type = None
    current_message_lines: list[str] = []
    current_file = None
    current_line_no = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Blank line — flush current failure if we have one
            if current_error_type and current_message_lines:
                msg = ' '.join(current_message_lines).strip()
                if len(msg) > 200:
                    msg = msg[:200] + '...'
                failure = {
                    'test': current_test or '(unknown test)',
                    'error_type': current_error_type,
                    'message': msg,
                    'file': current_file,
                    'line': current_line_no,
                }
                failures.append(failure)
                error_groups.setdefault(current_error_type, []).append(
                    len(failures) - 1)
                current_error_type = None
                current_message_lines = []
                current_file = None
                current_line_no = None
            continue

        # Detect test name context
        for pat in _TEST_NAME_PATTERNS:
            m = pat.search(stripped)
            if m:
                current_test = m.group(1).strip()
                break

        # Detect error type
        et_match = _ERROR_TYPE_RE.search(stripped)
        if et_match:
            # Flush previous if any
            if current_error_type and current_message_lines:
                msg = ' '.join(current_message_lines).strip()
                if len(msg) > 200:
                    msg = msg[:200] + '...'
                failure = {
                    'test': current_test or '(unknown test)',
                    'error_type': current_error_type,
                    'message': msg,
                    'file': current_file,
                    'line': current_line_no,
                }
                failures.append(failure)
                error_groups.setdefault(current_error_type, []).append(
                    len(failures) - 1)

            current_error_type = et_match.group(1)
            # Message is the rest of the line after the error type
            rest = stripped[et_match.end():].lstrip(': ')
            current_message_lines = [rest] if rest else []
            current_file = None
            current_line_no = None

        elif current_error_type:
            # Continuation of error message
            current_message_lines.append(stripped)

        # Detect file:line references
        fl_match = _file_line_re.search(stripped)
        if fl_match and current_error_type:
            current_file = fl_match.group(1) or fl_match.group(3)
            current_line_no = fl_match.group(2) or fl_match.group(4)

    # Flush last failure
    if current_error_type and current_message_lines:
        msg = ' '.join(current_message_lines).strip()
        if len(msg) > 200:
            msg = msg[:200] + '...'
        failure = {
            'test': current_test or '(unknown test)',
            'error_type': current_error_type,
            'message': msg,
            'file': current_file,
            'line': current_line_no,
        }
        failures.append(failure)
        error_groups.setdefault(current_error_type, []).append(
            len(failures) - 1)

    # If we couldn't parse structured failures, fall back to _extract_test_error
    if not failures:
        return _extract_test_error(output, max_chars=max_chars)

    # Build the structured summary grouped by error type
    total = len(failures)
    unique_files = set()
    for f in failures:
        if f.get('file'):
            unique_files.add(f['file'])

    parts: list[str] = []
    parts.append(f"FAILED TESTS SUMMARY ({total} failure(s)"
                 f"{f' across {len(unique_files)} file(s)' if unique_files else ''}):\n")

    # Sort error groups by count (most common first) so LLM fixes root causes
    sorted_groups = sorted(error_groups.items(),
                           key=lambda x: len(x[1]), reverse=True)

    failure_num = 0
    for error_type, indices in sorted_groups:
        count = len(indices)
        parts.append(f"--- {error_type} ({count} occurrence(s)) ---")

        for idx in indices:
            failure_num += 1
            f = failures[idx]
            loc = ""
            if f.get('file'):
                loc = f"  File: {f['file']}"
                if f.get('line'):
                    loc += f":{f['line']}"
            parts.append(f"  {failure_num}. {f['test']}")
            parts.append(f"     Error: {f['message']}")
            if loc:
                parts.append(f"    {loc}")

        parts.append("")  # blank line between groups

    # Add the test summary line if present
    for line in lines:
        stripped = line.strip()
        if re.search(r'\d+\s+(failed|passed|skipped)', stripped):
            parts.append(f"Summary: {stripped}")
            break

    result = '\n'.join(parts).strip()
    if len(result) > max_chars:
        result = result[:max_chars] + '\n... [truncated]'

    return result


_REVIEW_APPROVAL_PHRASES = (
    "code looks good", "looks good", "no issues", "no critical issues",
    "no bugs found", "code is correct", "functionally correct", "lgtm",
)


def _review_verdict(review: str, api_errs) -> bool:
    """Final accept/reject verdict for a code review.

    The runtime API probe is ground truth: a reviewer can add findings on
    top of it but can never subtract them (observed live: a reviewer
    approving code the probe had flagged, shipping dead APIs to disk).
    """
    if api_errs:
        return False
    review_lower = (review or "").lower()
    return any(p in review_lower for p in _REVIEW_APPROVAL_PHRASES)


def _api_grounding_context(
    written_files: dict[str, str],
    memory: FileMemory,
    executor: Executor,
    project_context=None,
    language: str | None = None,
    cwd: str | None = None,
) -> tuple[list[str], str]:
    """Ground *written_files* against the real project environment.

    Two checks feed the same gate: the API usage probe (module.attr
    existence with rename suggestions) and, when *language* has a
    registered per-step verifier, an execution-grade load check (import /
    compile) that catches integration failures the moment a file is
    written instead of at the end of the pipeline.

    Returns ``(errors, verified_ctx)``.  On a clean probe, *verified_ctx*
    is a reviewer prompt block listing the runtime-verified APIs — without
    it, the reviewer re-litigates API existence from training data and can
    FAIL correct code (observed: verified arcade 3.x names rejected as
    "misspelled" with the removed 2.x names recommended back).
    """
    load_errs: list[str] = []
    try:
        from .step_verify import verify_step_files
        load_errs = verify_step_files(
            written_files, language, executor, cwd=cwd)
    except Exception as exc:
        log.debug(f"[StepVerify] Skipped: {exc}")

    py_files = {p: c for p, c in written_files.items() if p.endswith(".py")}
    if not py_files:
        return load_errs, ""
    try:
        from .api_grounding import (collect_api_usages,
                                    local_top_levels_from_files,
                                    probe_api_usage)
        tops = local_top_levels_from_files(memory.all_files().keys())
        errs = probe_api_usage(py_files, executor, local_top_levels=tops)
        if errs or load_errs:
            return errs + load_errs, ""
        usages = collect_api_usages(py_files, tops)
    except Exception as exc:
        log.debug(f"[ApiGrounding] Probe skipped: {exc}")
        return load_errs, ""
    if not usages:
        return [], ""
    versions = getattr(project_context, "installed_versions", None) or {}
    tops_used = sorted({u.split(".")[0] for u in usages})
    pkg_note = ", ".join(
        f"{t}=={versions[t]}" if t in versions else t for t in tops_used)
    verified_ctx = (
        "\n\nVERIFIED APIs — a runtime probe against the installed "
        f"packages ({pkg_note}) confirmed that every one of these exists "
        "exactly as written: "
        + ", ".join(f"`{u}`" for u in usages[:40])
        + ". Do NOT flag any of them as missing, renamed, misspelled, or "
        "nonexistent — the probe result is ground truth and overrides "
        "your training data."
    )
    return [], verified_ctx


# Venv-activation segment inside a && chain — `call venv\Scripts\activate`,
# `source .venv/bin/activate`, `. venv/bin/activate` and bare-path variants.
_VENV_ACTIVATE_RE = re.compile(
    r'^(?:call\s+|source\s+|\.\s+)?\S*(?:venv|env)[\\/]+(?:Scripts|bin)'
    r'[\\/]+activate(?:\.bat|\.ps1)?$',
    re.IGNORECASE,
)


def _discover_django_settings(project_root: str = ".") -> str | None:
    """Return the project's DJANGO_SETTINGS_MODULE by reading manage.py, or None."""
    try:
        with open(os.path.join(project_root, "manage.py"),
                  "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return None
    m = re.search(
        r"DJANGO_SETTINGS_MODULE['\"]\s*,\s*['\"]([\w.]+)['\"]", content)
    return m.group(1) if m else None


def _djangoize_import_probe(cmd: str, project_root: str = ".") -> str:
    """Make a bare ``python -c "import app.module"`` gate Django-aware.

    A first-party import probe fails in a Django project unless the app
    registry is populated (``from django.contrib.auth.models import User``
    raises ``AppRegistryNotReady``). Planners emit these probes anyway, and
    to satisfy them the model injects ``django.setup()`` into the module
    itself — illegal in code Django imports during ``apps.populate()`` and
    the direct cause of a ``populate() isn't reentrant`` crash that broke
    ``manage.py check`` project-wide.

    Bootstrapping Django *inside the probe* makes the gate pass on correct
    code, so no source-level hack is needed. Non-probes and non-Django
    projects are returned unchanged.
    """
    m = re.match(r'^\s*(?:python3?|py)\s+-c\s+(["\'])(.*)\1\s*$', cmd, re.DOTALL)
    if not m:
        return cmd
    quote, body = m.group(1), m.group(2)
    if "import" not in body:
        return cmd
    if "django.setup" in body or "DJANGO_SETTINGS_MODULE" in body:
        return cmd  # already bootstraps Django itself
    settings = _discover_django_settings(project_root)
    if not settings:
        return cmd  # not a Django project — leave the probe alone
    prefix = (f"import os; os.environ.setdefault('DJANGO_SETTINGS_MODULE', "
              f"'{settings}'); import django; django.setup(); ")
    return f"python -c {quote}{prefix}{body}{quote}"


def _tests_are_discoverable(executor, language, subroot,
                            gate_cmd) -> tuple[bool, str, str]:
    """Can the project's OWN runner find the suite this step just wrote?

    A plan-declared gate may name a single file
    (``python -m unittest -v tests/test_game.py``). That proves the file
    runs; it says nothing about discovery. A run shipped exactly that: the
    gate was green and `python -m unittest -v` answered "Ran 0 tests"
    because nothing made ``tests/`` a package.

    The check re-runs the GATE with its file scoping removed, not the
    language default: the language default for Python is pytest, which
    collects ``tests/`` with no ``__init__.py`` at all and would have
    called the broken project green. The question is only ever "does the
    runner this step chose still find these tests without being pointed
    at them".

    Returns (ok, command, output). A gate that is already project-wide,
    or that names no runner we recognise, returns ok — inventing a
    verdict would fail steps blindly.
    """
    from ..agent_tools import _no_tests_collected, _TEST_RUNNER_TOKENS

    if not gate_cmd:
        return True, "", ""
    low = gate_cmd.lower()
    if not any(tok in low for tok in _TEST_RUNNER_TOKENS):
        return True, "", ""

    parts = gate_cmd.split()
    kept = [p for p in parts
            if not (("/" in p or "\\" in p
                     or p.endswith((".py", ".js", ".ts", ".tsx", ".go")))
                    and not p.startswith("-"))]
    if len(kept) == len(parts):
        return True, gate_cmd, ""      # already project-wide
    cmd = " ".join(kept)
    ok, out = executor.run_command(cmd, timeout=300)
    if _no_tests_collected(cmd, getattr(executor, "last_exit_code", None),
                           out or ""):
        return False, cmd, out or ""
    # A failing-but-collecting suite is a different problem, and the
    # step's own gate already had its say on correctness.
    return True, cmd, out or ""


_SET_BEFORE_AMP_RE = re.compile(
    r"(?i)\b((?:set|export)\s+\w+=[^&|\n]*?)[ \t]+(&&|\|\|)")


def _strip_space_before_amp(cmd: str) -> str:
    """Close the gap between a ``set VAR=value`` and a following ``&&``.

    ``cmd.exe`` assigns everything up to the separator, so
    ``set VAR=dummy && next`` sets VAR to ``"dummy "``. Harmless on POSIX,
    silently wrong on Windows — an SDL driver named ``"dummy "`` does not
    exist. Applied whenever this module reassembles a chain.
    """
    prev = None
    while prev != cmd:
        prev = cmd
        cmd = _SET_BEFORE_AMP_RE.sub(r"\1\2", cmd)
    return cmd


def _declared_verify_cmd(plan_step, memory: FileMemory,
                         task: str = "") -> str | None:
    """The plan-declared acceptance command for a step, or None.

    Sanitised before use — planners emit gates wrapped in scaffolding
    noise (observed: `cd <project_name> && call venv\\Scripts\\activate
    && python manage.py check` on every step):

      * ``<placeholder>`` tokens are resolved like CMD commands are.
      * Venv-activation segments are dropped — the gate runs with the
        pipeline's own Python, and the segment used to disqualify the
        whole command via the scaffold filter.
      * A leading ``cd`` into a directory that does not exist is dropped
        (the detected subproject prefix below re-adds the real one).

    One-shot scaffolding commands are still rejected — a verify gate
    must be safely re-runnable (same rule as the recovery verifier).
    """
    cmd = getattr(plan_step, "verify_cmd", None) if plan_step is not None else None
    if not cmd:
        return None

    cmd = resolve_cmd_placeholders(
        cmd, step_text=getattr(plan_step, "description", "") or "",
        task=task)

    # Unconditional, not just on the reassembly path below: the planner
    # writes `set SDL_VIDEODRIVER=dummy && python -c "..."` with the space
    # itself about as often as this module reintroduces one, and on
    # cmd.exe that assigns "dummy " either way. Repairing only what we
    # reassembled left the planner-authored form broken — observed failing
    # a gate three times in one step immediately after the narrower fix.
    cmd = _strip_space_before_amp(cmd)

    # A heredoc opener (`python - <<PY`) means the planner wrote a
    # multi-line script: the line parser kept only the first line, and
    # heredocs are bash-only anyway (cmd.exe runs the gate). A broken
    # gate is worse than none.
    if "<<" in cmd:
        return None

    _original = cmd
    segments = [s.strip() for s in cmd.split("&&")]
    _before = list(segments)
    segments = [s for s in segments
                if s and not _VENV_ACTIVATE_RE.match(s)]
    if segments and segments[0].lower().startswith("cd "):
        _dir = segments[0][3:].strip().strip('"\'')
        if _dir and not os.path.isdir(_dir):
            segments = segments[1:]
    if not segments:
        return None
    # Rejoining unconditionally rewrote whitespace even when nothing was
    # dropped, and on cmd.exe that is not cosmetic: `set VAR=dummy && ...`
    # assigns "dummy " — trailing space included. A planner had written
    # `set SDL_VIDEODRIVER=dummy&& ...` precisely to avoid that; the space
    # this put back made SDL look for a driver named "dummy ", failed the
    # gate, and cost a diagnosis round that "fixed" it by adding an
    # environment-scrubbing function to the generated project's main.py.
    if segments == _before:
        cmd = _original
    else:
        cmd = _strip_space_before_amp(" && ".join(segments))
    if not cmd:
        return None

    from .agent_loop import reverifiable_cmd
    if reverifiable_cmd(cmd) is None:
        return None
    sub = _detect_subproject_root(memory)
    # Bootstrap Django inside bare import-probe gates so they're satisfiable
    # by correct code (manage.py may live at the root or under a subproject).
    django_root = "."
    if sub and os.path.isfile(os.path.join(sub, "manage.py")):
        django_root = sub
    cmd = _djangoize_import_probe(cmd, django_root)
    if sub and not cmd.lstrip().startswith("cd "):
        # A `cd {sub}` prefix is only safe when the command is written to
        # run from inside the subproject. Three shapes are written for the
        # repo root and become self-contradictory once cwd changes:
        #   * a root-relative interpreter path (venv\Scripts\python.exe /
        #     venv/bin/python) no longer resolves from inside {sub};
        #   * `import {sub}.x` / `from {sub}.x` requires {sub}'s PARENT on
        #     sys.path — cd-ing into {sub} breaks the very import it probes;
        #   * any other path/module reference to {sub} itself (`-s {sub}`,
        #     `{sub}/tests`, `python -m {sub}.main`) points at {sub}/{sub}
        #     once cwd is {sub} — an unpassable gate. Observed: the loop
        #     burned its budget against `cd game && unittest discover -s
        #     game`, and the escalation model then "satisfied" it by
        #     creating a duplicate nested game/game/ package.
        _root_relative_venv = re.search(r"(?:^|[\s\"'])\.?[\\/]?venv[\\/]",
                                        cmd) is not None
        _sub_re = re.escape(sub.rstrip("/\\"))
        _imports_sub = re.search(rf"\b(?:import|from)\s+{_sub_re}\b",
                                 cmd) is not None
        # The leading class must admit path separators and `.`. The shape
        # that actually occurs is `require('./frontend/package.json')`,
        # where the character before the name is `/`, not a quote — so a
        # correct root-relative gate was read as making no reference to
        # `frontend` and became `cd frontend && ...'./frontend/...'`,
        # which nothing can satisfy. The recovery loop then satisfied it
        # by creating `frontend/frontend/package.json`, which is exactly
        # the duplicate nested package this guard exists to prevent.
        _references_sub = re.search(
            rf"(?:^|[\s\"'=./\\]){_sub_re}(?:[\\/.\s\"']|$)", cmd) is not None
        # A gate on a step the plan places under a SIBLING root is written
        # for the repo root, and `cd {sub}` silently redirects it to a
        # path that may well exist: `node -e "require('./backend/...')"`
        # prefixed with `cd frontend` read frontend/backend/ and passed,
        # which is how a whole misrouted backend went unnoticed.
        _sibling_root = _step_belongs_to_subproject(plan_step, sub, memory) is False
        if not (_root_relative_venv or _imports_sub or _references_sub
                or _sibling_root):
            cmd = f"cd {sub} && {cmd}"
    return cmd


def _gate_on_declared_verify(success: bool, error_info: str, plan_step,
                             executor: Executor, memory: FileMemory,
                             display: CLIDisplay,
                             step_idx: int, task: str = "") -> tuple[bool, str]:
    """Deterministic per-step acceptance gate for the classic path.

    A handler's own success claim is only accepted once the step's
    plan-declared ``verify:`` command (when present) exits 0. A failing
    gate turns the result into a failure whose error carries the command
    output, so the pipeline's normal diagnosis/recovery machinery gets
    the real evidence. No-op when the plan declared nothing.
    """
    if not success:
        return success, error_info
    cmd = _declared_verify_cmd(plan_step, memory, task=task)
    if not cmd:
        return success, error_info
    display.step_info(step_idx, f"Step gate: {cmd}")
    ok, out = executor.run_command(cmd, timeout=300)
    # A gate that collected NOTHING has proved nothing, whatever it exited
    # with. unittest only gained a non-zero status for a zero-test run in
    # CPython 3.12, so below that a step whose discovery quietly broke
    # passes its own acceptance command having executed no tests. Same
    # defect the agent loop's gate had; fixed on both paths so the two
    # cannot disagree about what "verified" means.
    from ..agent_tools import _no_tests_collected
    if ok and _no_tests_collected(
            cmd, getattr(executor, "last_exit_code", None), out or ""):
        display.step_info(step_idx, "Step gate collected no tests ✖")
        return False, (
            f"Step acceptance command COLLECTED NO TESTS: `{cmd}`\n"
            f"{(out or '(no output)')[-2000:]}\n"
            "It may have exited 0, but nothing ran, so it proves nothing. "
            "Make the tests discoverable.")
    if ok:
        display.step_info(step_idx, "Step gate passed ✔")
        # Stash the exact command that passed so _record_passed_gate logs
        # this form (not a re-derivation) into the monotonic ledger.
        if plan_step is not None:
            setattr(plan_step, "_verified_gate_cmd", cmd)
        return success, error_info
    return False, (
        f"Step acceptance command failed: `{cmd}`\n"
        f"{(out or '(no output)')[-2000:]}"
        + _broken_export_promise(plan_step, memory, out or ""))


def _broken_export_promise(plan_step, memory: FileMemory, out: str) -> str:
    """Name the declared exports the written files do not define.

    A plan step declares ``exports:`` and its ``verify:`` imports exactly
    those names, so a coder that invents its own naming produces a gate
    that CANNOT pass, however many times it is retried. Observed: the
    step promised ``tile_to_pixel_center`` / ``is_at_tile_center``, the
    file defined ``pixel_center_for_tile`` / ``is_aligned_to_tile_center``
    — same behaviour, different words — and two diagnosis rounds went by
    without either of them being told the contract was the problem.

    Only fires on an import/attribute error naming a declared export, so
    an unrelated gate failure is never mislabelled. Returns "" when there
    is nothing solid to say.
    """
    declared = [s for s in (getattr(plan_step, "exports", None) or []) if s]
    if not declared:
        return ""
    # The error must actually be about a name, not a logic failure.
    named = set(re.findall(r"cannot import name '([^']+)'", out))
    named |= set(re.findall(r"has no attribute '([^']+)'", out))
    named |= set(re.findall(r"name '([^']+)' is not defined", out))
    blocked = [s for s in declared if s in named]
    if not blocked:
        return ""

    try:
        from ..language_backend import get_backend
        backend = get_backend(getattr(plan_step, "language", None) or "python")
    except Exception:
        return ""

    defined: set[str] = set()
    files = {}
    try:
        files = memory.as_dict() or {}
    except Exception:
        pass
    for target in getattr(plan_step, "target_files", None) or []:
        content = (files.get(target)
                   or files.get(str(target).replace("\\", "/")))
        if not content:
            continue
        try:
            defined.update(backend.extract_exports(content) or [])
        except Exception:
            continue
    # No evidence beats bad evidence: an extractor that saw nothing must
    # not be read as "the file defines nothing".
    if not defined:
        return ""
    if not any(s not in defined for s in blocked):
        return ""
    # Once ONE declared export is provably absent the contract is broken,
    # so report every missing one. Python raises on the first bad name
    # only, and fixing them one per round trip wastes a diagnosis attempt
    # each time.
    missing = [s for s in declared if s not in defined]

    # Deliberately NOT guessing which existing name replaces which missing
    # one: difflib matched `tile_to_pixel_center` to
    # `is_aligned_to_tile_center` here, and a confident wrong hint would
    # rename the wrong function. The two lists are facts; the mapping is
    # the model's to make.
    return (
        "\n\nBROKEN EXPORT CONTRACT — this is why the command failed:\n"
        "The step declared these exports and the acceptance command "
        "imports them by those exact names, but the written file(s) do "
        "not define them:\n"
        + "".join(f"  missing: {s}\n" for s in missing)
        + "\nThe file(s) currently define:\n"
        + f"  {', '.join(sorted(defined)[:25])}\n"
        + "\nThe declared names are the contract. Rename the matching "
        "definitions to the declared names (do NOT change the acceptance "
        "command, and do NOT rewrite working logic — a rename, or an "
        "alias assignment, is usually the whole fix).")


def _django_lint_gate(success: bool, error_info: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int) -> tuple[bool, str]:
    """Fail a "successful" step whose files carry deterministic Django
    wiring bugs (namespace mismatches, missing {% load %} lines).

    These bugs pass ``manage.py check`` and only explode at request
    time; two consecutive benchmark runs died on them. The lint runs on
    both the loop and classic paths — it costs microseconds — and its
    error strings name the exact fix, so the recovery loop gets a
    mechanical instruction instead of a mystery traceback.

    Failed steps get the findings too, appended to their error: the
    recovery loop then starts from the deterministic diagnosis instead
    of rediscovering it (observed: 16 turns spent circling a
    NoReverseMatch whose one-line cause the lint states outright).
    """
    from .django_lint import check_django_project
    errors = check_django_project(memory.all_files())
    if not errors:
        return success, error_info
    findings = "\n".join(errors[:10])
    if success:
        display.step_info(
            step_idx,
            f"Django lint: {len(errors)} wiring error(s) — step gated")
        log.warning("[DjangoLint] step %d gated on %d wiring error(s)",
                    step_idx + 1, len(errors))
        return False, ("Django wiring lint failed — fix these exactly as "
                       "stated:\n" + findings)
    log.info("[DjangoLint] step %d already failed — %d lint finding(s) "
             "appended for the recovery loop", step_idx + 1, len(errors))
    return False, (f"{error_info}\n\n"
                   "Deterministic Django lint findings — fix these exactly "
                   "as stated:\n" + findings)


def _plan_step_brief(plan_step) -> str:
    """Structured step metadata for the agent loop's context.

    In intent mode (plan_mode: intent) the plan carries no file bodies,
    so these lines are everything the loop knows about WHAT to build —
    which files, which exports, which imports. Cheap and harmless in
    content mode too (steps with inline code never reach the loop).
    """
    if plan_step is None:
        return ""
    lines = []
    targets = getattr(plan_step, "target_files", None)
    if targets:
        lines.append("Target files: " + ", ".join(targets))
    exports = getattr(plan_step, "exports", None)
    if exports:
        lines.append("Must export: " + ", ".join(exports))
    imports_from = getattr(plan_step, "imports_from", None)
    if imports_from:
        pairs = [
            f"{fp}: {', '.join(syms)}" if syms else fp
            for fp, syms in imports_from.items()
        ]
        lines.append("Imports to use: " + "; ".join(pairs))
    return "\n".join(lines)


def _loop_preload_paths(plan_step) -> list[str]:
    """Files to hand the agent loop up front: targets AND declared imports.

    Only ``target_files`` used to be preloaded, so a step creating a NEW
    file had nothing to preload — its targets do not exist yet — and the
    loop spent its first turn calling read_file on the very dependencies
    the plan had already named. Observed repeatedly, including a step that
    burned turn 1 on three read_file calls plus list_files, and the
    "2 read-only turns — injecting act-now nudge" path that follows from
    it. Those reads then ride along in every later turn anyway, so the
    round-trip buys nothing but a turn out of a budget of eight.

    ``build_step_context`` has already resolved the step's declared
    imports (including ghost contracts for files a later step will
    create); reuse its keys rather than re-deriving them. Non-existent
    paths are skipped by the preloader, so ghosts cost nothing.

    Targets come first: for a step EDITING an existing file, that file is
    the most important thing in the budget.
    """
    paths: list[str] = list(getattr(plan_step, "target_files", None) or [])
    try:
        from .memory import get_plan_context_files
        paths.extend(get_plan_context_files() or {})
    except Exception:
        pass
    seen: set[str] = set()
    ordered: list[str] = []
    for p in paths:
        key = str(p).replace("\\", "/")
        if key and key not in seen:
            seen.add(key)
            ordered.append(p)
    return ordered


def _record_passed_gate(success: bool, plan_step, memory: FileMemory,
                        task: str = "") -> None:
    """Add a step's declared verify to the monotonic gate ledger.

    Only called with the step's final verdict: by then the gate has been
    enforced either by the agent loop (as its exit criterion) or by
    :func:`_gate_on_declared_verify`, so a successful step means the
    command passed. Later fix rounds are rechecked against the ledger
    and rolled back if they break it.

    The ledger's invariant is "this exact string previously exited 0", so
    we record the command the gate ACTUALLY executed (stashed on the step
    as ``_verified_gate_cmd``), never a freshly re-derived one.
    ``_declared_verify_cmd`` calls ``_detect_subproject_root(memory)``,
    whose result flips once the step writes files — the loop evaluated the
    gate before those writes (no subproject → repo-root form) while a
    re-derivation here runs after (subproject → ``cd sub &&`` form). The
    re-derived command was never run, can be self-contradictory, and then
    "regresses" on the monotonic recheck, rolling back working code.
    """
    if not success:
        return
    cmd = getattr(plan_step, "_verified_gate_cmd", None) \
        if plan_step is not None else None
    if not cmd:
        cmd = _declared_verify_cmd(plan_step, memory, task=task)
    if not cmd:
        return
    # Same invariant, one step further: when the gate was satisfied by a
    # proven-equivalent form (a platform re-reading, or the flag-variant
    # the loop itself ran), THAT is the string which exited 0. Recording
    # the original instead hands the ledger a command already shown to be
    # unpassable, and its next recheck reports a regression on code that
    # never changed — the rollback this function's own history warns of.
    from .gate_integrity import effective_gate
    cmd = effective_gate(cmd)
    from .wave_snapshots import get_gate_ledger
    get_gate_ledger().record(
        cmd, getattr(plan_step, "id", "") or "")


def _handle_code_step(step_text: str, coder: CoderAgent,
                      reviewer: ReviewerAgent, executor: Executor,
                      task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int,
                      **kwargs) -> tuple[bool, str]:
    """Public CODE-step entry: runs the handler, then the acceptance gate.

    The agent-loop path gates itself (the declared verify is its exit
    criterion, with feedback turns to fix failures), so the outer gate
    only fires for the classic generate→review path, where nothing else
    executes the plan's ``verify:`` command.
    """
    success, error_info = _handle_code_step_impl(
        step_text, coder, reviewer, executor, task, memory, display,
        step_idx, **kwargs)
    from .agent_loop import agent_loop_enabled
    if not agent_loop_enabled(kwargs.get("cfg"),
                              getattr(coder, "llm_client", None)):
        success, error_info = _gate_on_declared_verify(
            success, error_info, kwargs.get("plan_step"), executor, memory,
            display, step_idx, task=task)
    success, error_info = _django_lint_gate(
        success, error_info, memory, display, step_idx)
    _record_passed_gate(success, kwargs.get("plan_step"), memory, task=task)
    return success, error_info


def _handle_code_step_impl(step_text: str, coder: CoderAgent, reviewer: ReviewerAgent,
                      executor: Executor, task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int,
                      language: str | None = None,
                      cfg: Config | None = None,
                      auto: bool = False,
                      code_graph=None,
                      project_profile=None,
                      skip_review: bool = False,
                      project_context=None,
                      plan_step=None,
                      all_plan_steps=None,
                      kb_context_builder=None,
                      partial_inline_code: dict[str, str] | None = None,
                      initial_error: str = "",
                      intent_spec=None) -> tuple[bool, str]:
    # ── Agent loop (opt-in): run the step as a bounded tool-calling loop ──
    from .agent_loop import agent_loop_enabled
    if agent_loop_enabled(cfg, getattr(coder, "llm_client", None)):
        from .agent_loop import (
            build_step_tools, run_agent_loop_with_escalation,
        )
        display.step_info(step_idx, "Agent loop: executing step with tools")
        tools = build_step_tools(executor, memory,
                                 kb_context_builder=kb_context_builder)
        loop_context = memory.summary()
        _brief = _plan_step_brief(plan_step)
        if _brief:
            loop_context = f"{_brief}\n\n{loop_context}"
        if initial_error:
            loop_context = f"{loop_context}\n\nKnown problem to fix:\n{initial_error}"
        # Plan-declared acceptance command becomes the loop's exit gate:
        # the model's "done" claim is only accepted once it exits 0, and
        # failures feed back as fix turns.
        verify_cmd = _declared_verify_cmd(plan_step, memory, task=task)
        if verify_cmd:
            loop_context += (
                f"\n\nThis step is complete ONLY when `{verify_cmd}` exits "
                "successfully — it will be run to verify your work.")
        loop_result = run_agent_loop_with_escalation(
            coder.llm_client, tools, step_text, task,
            escalation_client=getattr(coder, "escalation_client", None),
            display=display, step_idx=step_idx, language=language,
            max_turns=getattr(cfg, "AGENT_LOOP_MAX_TURNS", 8),
            verify_cmd=verify_cmd,
            context=loop_context,
            preload_files=_loop_preload_paths(plan_step),
            preload_full_paths=set(
                getattr(plan_step, 'target_files', None) or ()),
            required_files=set(
                getattr(plan_step, 'target_files', None) or ()),
        )
        # Record the gate exactly as the loop enforced it, so the monotonic
        # ledger rechecks the command that actually passed (see
        # _record_passed_gate) — not a re-derivation whose subproject
        # prefix can diverge after this step's writes. Resolving a gate
        # that was satisfied by an equivalent form is deliberately NOT
        # done here — `_record_passed_gate` is the single chokepoint for
        # every path (loop, recovery, classic), so the transform lives
        # there rather than in each caller.
        if loop_result[0] and verify_cmd and plan_step is not None:
            setattr(plan_step, "_verified_gate_cmd", verify_cmd)
        return loop_result

    # --- Proactive pre-install: ensure all required packages are installed ---
    # CMD steps scaffold the project first (e.g. npm create vite@latest).
    # By the first CODE step the manifest exists, so we bulk-install any
    # packages from the plan summary that are still missing.
    if project_context is not None:
        subproject_cwd = _detect_subproject_root(memory)
        _ensure_packages_installed(
            project_context, executor, memory, display, step_idx,
            subproject_cwd=subproject_cwd, language=language,
        )

    # Pre-fetch KB docs for code generation — use planner-declared titles
    # (plan_step.kb_docs) when available for an exact lookup so that only
    # the docs the planner actually referenced are injected.  Fall back to
    # a broad semantic search only when the planner declared no titles.
    _code_behavioral_ctx = ""
    if kb_context_builder is not None:
        try:
            _gstore = getattr(kb_context_builder, '_global_store', None)
            if _gstore is not None:
                _declared_kb = getattr(plan_step, 'kb_docs', None) if plan_step else None
                if _declared_kb:
                    _beh_results = _gstore.get_by_titles(_declared_kb)
                else:
                    _step_query = (
                        (plan_step.description if plan_step and getattr(plan_step, 'description', None) else None)
                        or step_text.split("\n")[0].strip()
                    )
                    _beh_results = _gstore.search(
                        query=_step_query,
                        categories=["behavioral", "doc"],
                        top_k=4,
                        api_client=getattr(kb_context_builder, '_api_client', None),
                        language=language,
                    )
                if _beh_results:
                    _beh_parts = []
                    for item in _beh_results:
                        content = getattr(item, "content", "") or getattr(item, "title", "")
                        if content:
                            _beh_parts.append(content)
                    if _beh_parts:
                        _code_behavioral_ctx = (
                            "\n[KB INSTRUCTIONS]\n"
                            + "\n".join(_beh_parts) + "\n"
                        )
        except Exception:
            pass

    # Enrich step_text with plan-declared target files so _detect_target_files
    # can locate them even when the step description omits the filename
    # (e.g. "Modify header component" doesn't mention "Header.jsx").
    _edit_step_text = step_text
    if plan_step and getattr(plan_step, 'target_files', None):
        _targets_hint = " ".join(plan_step.target_files)
        _edit_step_text = f"[targets: {_targets_hint}]\n{step_text}"

    # When falling back from inline static-check failure, prepend the specific
    # errors so all edit tiers (chunk, diff, full-file) know what to fix.
    if initial_error:
        _edit_step_text = f"{_edit_step_text}\n\n[ERRORS TO FIX]\n{initial_error}"

    # Detect primary target file to see if we should force DiffEdit for non-AST
    target_for_route = _detect_target_file(_edit_step_text, memory)
    is_non_ast = False
    if target_for_route:
        ext = os.path.splitext(target_for_route)[1].lower()
        if ext in {".html", ".css", ".scss", ".json", ".yaml", ".yml", ".md", ".txt"}:
            is_non_ast = True

    # --- Tier 1: Diff-aware editing ---
    try_diff = False
    if cfg and getattr(cfg, "EDITING_DIFF_MODE", False) and code_graph is not None:
        try_diff = True
    if is_non_ast:
        try_diff = True

    if try_diff:
        diff_result = _try_diff_edit(
            step_text=_edit_step_text, coder=coder, task=task,
            memory=memory, display=display, step_idx=step_idx,
            language=language, cfg=cfg, code_graph=code_graph,
            project_profile=project_profile,
            kb_context_builder=kb_context_builder,
        )
        if diff_result is not None:
            return diff_result

    # --- Tier 2: Chunk edit (regex-based, no KB graph needed) ---
    if cfg and getattr(cfg, "EDITING_CHUNK_MODE", True) and not is_non_ast:
        chunk_result = _try_chunk_edit(
            step_text=_edit_step_text, coder=coder, reviewer=reviewer,
            executor=executor, task=task, memory=memory,
            display=display, step_idx=step_idx,
            language=language, cfg=cfg, auto=auto,
            project_profile=project_profile,
            project_context=project_context,
            kb_context_builder=kb_context_builder,
            behavioral_ctx=_code_behavioral_ctx,
        )
        if chunk_result is not None:
            return chunk_result

    # --- Tier 3: Full-file flow (fallback) ---
    feedback = initial_error  # Pre-seed with inline static errors when falling back
    _chunk_review_fb = getattr(memory, "_chunk_review_feedback", "")
    if _chunk_review_fb:
        memory._chunk_review_feedback = ""  # consume once
        feedback = ((feedback + "\n\n") if feedback else "") + (
            "A chunk-level edit for this step was already attempted and "
            "reverted after review. Address this feedback in your rewrite:\n"
            + _chunk_review_fb
        )
    context_window = cfg.CONTEXT_WINDOW if cfg else 8192
    ctx_budget = int(context_window * 0.8)
    prev_files: dict[str, str] = {}  # Track files from previous attempt

    # Pre-compute CSS conflicts once (not per retry) so styling steps also
    # update the global CSS files that would otherwise override the change.
    _t3_plan_targets = (
        list(plan_step.target_files)
        if (plan_step and getattr(plan_step, 'target_files', None))
        else []
    )
    _t3_css_conflicts = _find_css_conflicts(
        step_text,
        _t3_plan_targets or _detect_target_files(step_text, memory),
        memory,
    )

    for attempt in range(1, MAX_STEP_RETRIES + 1):
        # Prepend project orientation + knowledge context
        context_prefix = ""
        if project_context is not None:
            coder_analysis = project_context.format_for_coder()
            if coder_analysis:
                context_prefix = coder_analysis + "\n\n"
        kb_ctx = getattr(memory, '_kb_context', '')
        if kb_ctx:
            context_prefix += kb_ctx + "\n\n"
        elif project_profile is not None:
            # Only inject profile directly when it wasn't already included in
            # _kb_context (pipeline.py always puts the profile there first).
            try:
                context_prefix += project_profile.format_for_prompt() + "\n\n"
            except Exception:
                pass
        # Inject explicitly-fetched behavioral instructions for JS/TS
        # ONLY when batch_search didn't already include them (trimmed or
        # missed by vector search).  This avoids bloating the prompt and
        # ensures framework/library docs keep their higher priority.
        if (_code_behavioral_ctx
                and "[BEHAVIORAL INSTRUCTIONS]" not in context_prefix
                and "[KB INSTRUCTIONS]" not in context_prefix):
            context_prefix += _code_behavioral_ctx + "\n\n"

        # Inject task briefing (Preserve / Key constraint / Agent directive)
        # so the coder never accidentally removes features or breaks contracts.
        _task_briefing = getattr(memory, '_task_briefing', '')
        if _task_briefing:
            context_prefix += (
                "TASK BRIEFING — follow the Preserve and Key constraint lines exactly:\n"
                f"{_task_briefing}\n\n"
            )

        context = context_prefix + f"Task: {task}"

        # ── Partial inline code hint (truncation recovery) ──
        # When the planner's inline code was cut off by a token limit,
        # inject the partial content so the coder completes rather than
        # regenerates the entire file from scratch.
        if partial_inline_code:
            _partial_hint = "\n\n[PARTIAL CODE — the planner started writing this; " \
                            "complete it without restarting from scratch]\n"
            for _ph_path, _ph_content in partial_inline_code.items():
                _partial_hint += (
                    f"\n#### [FILE]: {_ph_path}\n"
                    f"```\n{_ph_content}\n```\n"
                    f"^ This file is INCOMPLETE — output the full completed version.\n"
                )
            context += _partial_hint

        # ── Target file enforcement (full-file tier) ──
        # Explicitly tell the LLM which file to modify so it doesn't
        # accidentally generate code for a different file from context.
        if plan_step and getattr(plan_step, 'target_files', None):
            _tf = ", ".join(plan_step.target_files)
            context += (
                f"\n\nTARGET FILE(S): {_tf}"
                f"\nONLY output `#### [FILE]: ...` blocks for the target file(s) above."
                f"\nAll other files in context are READ-ONLY reference — do NOT output `#### [FILE]:` blocks for them."
            )

        # ── Inject search context from prior SEARCH steps ──
        # When a SEARCH step ran before this CODE step, its results contain
        # setup guides, API docs, etc. that the coder needs.
        _all_mem = memory.all_files()
        for _sk, _sv in _all_mem.items():
            if _sk.startswith('_search_context/') and _sv.strip():
                context += (
                    f"\n\nReference Documentation (from prior search step):\n"
                    f"{_sv}\n"
                )
                break  # inject once — multiple search steps are rare

        # ── Plan-aware context injection ──
        # When a structured plan step is available, use plan-declared
        # imports/targets for precise context (fewer tokens, better relevance).
        # Falls back to legacy related_context / slim context otherwise.
        from .memory import get_plan_context_files as _get_plan_ctx
        plan_ctx = _get_plan_ctx()
        if plan_ctx and plan_step is not None:
            # Full content for plan-declared files (imports + targets)
            for fpath, content in plan_ctx.items():
                context += f"\n\n#### [FILE]: {fpath}\n```\n{content}\n```"
            # Slim skeletons for other files in memory (so LLM knows what exists)
            # Uses cached skeletons — avoids re-running regex on every retry.
            slim_parts: list[str] = []
            for fpath, content in memory.all_files().items():
                if fpath in plan_ctx:
                    continue
                if fpath.startswith(('_cmd_output/', '_fix_output/', '_search_context/')):
                    continue
                skeleton = memory.get_skeleton(fpath, content)
                if skeleton:
                    slim_parts.append(skeleton)
                else:
                    slim_parts.append(f"- {fpath}")
            if slim_parts:
                context += "\n\nOther project files (signatures only):\n" + "\n".join(slim_parts)
            log.info(
                f"Step {step_idx+1}: Plan-aware context: {len(plan_ctx)} full "
                f"+ {len(slim_parts)} skeleton(s)"
            )
        else:
            # Legacy context: semantic search or slim + targets
            use_slim = cfg and getattr(cfg, "EDITING_SLIM_CONTEXT", True)
            targets = _detect_target_files(step_text, memory) if use_slim else []
            if use_slim and targets:
                slim = memory.related_context_slim(step_text, max_tokens=ctx_budget)
                if slim:
                    context += f"\nProject file structures:\n{slim}"
                for tf in targets:
                    tf_content = memory.get(tf)
                    if tf_content:
                        context += f"\n\n#### [FILE]: {tf}\n```\n{tf_content}\n```"
            else:
                related = memory.related_context(step_text, max_tokens=ctx_budget)
                if related:
                    context += f"\nExisting files (overwrite as needed):\n{related}"

        # ── CSS conflict injection ──
        # When the step changes a visual style, inject any global CSS files whose
        # selectors match the target component so the LLM can also remove the
        # conflicting rule that would otherwise override the change.
        if _t3_css_conflicts:
            _css_injected: list[str] = []
            for css_path, css_content in _t3_css_conflicts.items():
                if f"#### [FILE]: {css_path}" not in context:
                    context += f"\n\n#### [FILE]: {css_path}\n```css\n{css_content}\n```"
                _css_injected.append(css_path)
            context += (
                "\n\nCSS OVERRIDE WARNING: The CSS file(s) shown above contain global "
                "rules that may override your component-level style change. For example, "
                "`.header, .footer { background-color: var(--bg); }` will cascade over "
                "a component's own `background-color` declaration. "
                "After updating the target component, ALSO update those CSS file(s): "
                "remove the conflicting property from the shared rule, scope it more "
                "narrowly, or replace it — so the user's intended change is actually visible."
            )
            # Expand TARGET FILE(S) hint to include the conflicting CSS files
            # so the LLM knows it is allowed to output [FILE] blocks for them.
            if plan_step and getattr(plan_step, 'target_files', None):
                _tf_base = ", ".join(plan_step.target_files)
                _tf_extra = [p for p in _css_injected if p not in plan_step.target_files]
                if _tf_extra and f"TARGET FILE(S): {_tf_base}" in context:
                    context = context.replace(
                        f"TARGET FILE(S): {_tf_base}",
                        f"TARGET FILE(S): {_tf_base}, " + ", ".join(_tf_extra),
                    )

        if memory.summary() != "(no files yet)":
            context += f"\nAll project files: {memory.summary()}"

        # Show the coder the command this step will be GRADED by.
        #
        # _gate_on_declared_verify runs plan_step.verify_cmd as the
        # pass/fail gate right after this call, but nothing was telling the
        # coder what it says — so the step was judged against a contract it
        # never saw. That gate is usually an exact API specification
        # (`p=Player(m); p.pixel_pos(); p.set_direction((1,0))`), and a
        # coder that has not read it invents a different signature and
        # fails. Observed on a Pac-Man run: the coder wrote
        # `Player(spawn_tile, game_map)` with no pixel_pos(), and three
        # diagnosis rounds were spent reverse-engineering the contract
        # out of error messages before the step halted.
        #
        # Cheap in tokens (one line) and it removes whole diagnosis rounds,
        # so it pays for itself many times over.
        try:
            _gate_cmd = _declared_verify_cmd(plan_step, memory, task=task)
        except Exception:
            _gate_cmd = None
        if _gate_cmd:
            context += (
                f"\n\nACCEPTANCE CHECK — this step is judged by running:\n"
                f"    {_gate_cmd}\n"
                "Your code MUST satisfy it exactly: the names, call "
                "signatures and return types it uses are the required API, "
                "not suggestions. Match them even if you would have chosen "
                "differently.")

        if feedback:
            context += f"\nFeedback: {feedback}"
            # On retry, tell the coder to ONLY fix the flagged issues
            context += (
                "\n\nCRITICAL: Only fix the specific issues mentioned in the "
                "feedback above. Do NOT modify any code that is unrelated to "
                "the feedback. Preserve ALL existing content, formatting, and "
                "special characters exactly as they are. Only output the "
                "file(s) that need changes."
            )

        display.step_info(step_idx, f"Coding (attempt {attempt}/{MAX_STEP_RETRIES})...")
        sent_before, recv_before = token_tracker.snapshot()

        response = coder.process(step_text, context=context, language=language)

        sent_after, recv_after = token_tracker.snapshot()
        sent_delta = sent_after - sent_before
        recv_delta = recv_after - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        explanation = CLIDisplay.extract_explanation(response)
        if explanation:
            display.add_llm_log(explanation, source="Coder")

        display.step_info(step_idx, "Processing LLM response...")
        files = executor.parse_code_blocks(response)
        if not files:
            files = executor.parse_code_blocks_fuzzy(response)
        if not files:
            # Fallback: LLM response might contain echo commands
            from .plan_step import _parse_echo_commands as _parse_echo_resp
            _echo_lines = [l.strip().lstrip('> ').lstrip('$ ')
                           for l in response.splitlines()
                           if l.strip().startswith(('>', '$', 'echo '))]
            if _echo_lines:
                files = _parse_echo_resp(_echo_lines)
        if not files:
            # Last resort before giving up: when the step declares exactly
            # ONE target there is nothing to guess, so an unlabelled code
            # block belongs to that file. Every earlier extractor needs the
            # model to name the file, and Pattern 5 needs a KB symbol index
            # a blank project does not have yet — so a model that answers
            # in prose with a bare fence produced nothing at all and halted
            # the run. Observed on Gemini: correct code, no filename, two
            # failed attempts plus two diagnosis rounds, 12 minutes and
            # 129k tokens for zero files written.
            _targets = list(getattr(plan_step, "target_files", None) or [])
            if len(_targets) == 1:
                files = executor.parse_blocks_for_single_target(
                    response, _targets[0])
                if files:
                    log.info(
                        "Step %d: response had no file markers — attributing "
                        "its code block to the step's only target (%s)",
                        step_idx + 1, _targets[0])
        if not files:
            feedback = "No file markers found. Use #### [FILE]: path/to/file.py format."
            display.step_info(step_idx, "No files parsed, retrying...")
            log.warning(f"Step {step_idx+1}: No files parsed from coder response.")
            continue

        # Normalize paths: fix LLM-generated paths that are suffixes of
        # known project files (e.g. src/App.js → my-app/src/App.js)
        files = _normalize_fix_paths(files, memory)

        # Prefix files with sub-project root if detected
        # (e.g. components/Header.tsx → my-app/components/Header.tsx)
        subproject = _detect_subproject_root(memory)
        if subproject:
            files = _prefix_subproject_paths(files, subproject, memory)

        # Strip protected manifest files (package.json, etc.) to prevent
        # LLM from overwriting them with corrupted versions
        files = _strip_protected_files(files, intent_spec=intent_spec)

        # Apply KB-driven content fixes (e.g. Tailwind v3→v4 directives)
        content_fixes = getattr(memory, '_content_fixes', None)
        files = _apply_content_fixes(files, content_fixes)

        # Auto-normalise duplicate imports (zero LLM cost, deterministic)
        files = _auto_dedup_imports(files, display, step_idx)

        # On retry, merge: keep previously approved files that weren't
        # re-generated, so the coder doesn't need to regenerate everything
        if attempt > 1 and prev_files:
            merged = dict(prev_files)
            merged.update(files)  # new files override previous
            files = merged

        # Auto-fix hazardous diffs before showing to user
        files = _auto_fix_hazards(files, coder, executor, display, step_idx,
                                  step_text, language=language, memory=memory)

        # Show diffs and wait for approval before writing
        display.stop_spinner()
        approved = prompt_diff_approval(files, auto=auto, display=display,
                                        base_dir=getattr(memory, 'base_dir', "."))
        display.step_info(step_idx, "Processing...")
        if not approved:
            feedback = "User rejected the changes. Try a different approach."
            display.step_info(step_idx, "Changes rejected by user, retrying...")
            log.info(f"Step {step_idx+1}: User rejected diff, retrying.")
            continue

        display.step_info(step_idx, "Processing approved changes...")

        # Build review context before updating memory/disk so diffs are captured correctly
        use_diff_review = cfg and getattr(cfg, "EDITING_REVIEWER_DIFF_MODE", True)
        if use_diff_review:
            review_ctx = _build_review_context(files, memory, step_text)

        prev_files = dict(files)  # Save for potential merge on retry
        # Captured BEFORE the write: FileMemory's protected-basename guard
        # tests os.path.isfile(), which is true the moment write_files
        # creates the file, so a manifest this step created looked
        # pre-existing and was dropped from memory. Same fix as the
        # plan-step inline writer; this is the path a plain CODE step for
        # requirements.txt actually takes.
        _existing_before = {p for p in files if os.path.exists(p)}
        written = executor.write_files(files)
        memory.update(files,
                      allow_protected={p for p in files
                                       if p not in _existing_before})
        display.step_info(step_idx, f"Written: {', '.join(written)}")

        # Auto-install any new package imports found in the generated code
        # (e.g. @heroicons/react, lucide-react, etc. that the LLM used
        # but didn't list in required_packages)
        _auto_install_code_imports(files, executor, memory, display, step_idx)

        # Skip review for non-code files (README, LICENSE, configs, etc.)
        if _all_non_code_files(list(files.keys())):
            display.step_info(step_idx, "Non-code files, skipping review ✔")
            log.info(f"Step {step_idx+1}: Skipped review (non-code files: {list(files.keys())})")
            return True, ""

        # ── Review gate ──────────────────────────────────────────
        # Always run free offline lint + import checks first.
        lint_errors = _quick_offline_lint(files)
        import_errors = _validate_import_paths(files, memory)
        lint_errors = lint_errors + import_errors

        # API grounding probe: verify module.attr usages exist in the
        # installed package versions (project venv).  Runs on EVERY retry —
        # a fix that swaps one removed API for another hallucinated one must
        # fail here with the exact error, not survive to runtime.
        _api_errs, _verified_api_ctx = _api_grounding_context(
            files, memory, executor, project_context, language=language)
        if _api_errs:
            log.warning(
                f"Step {step_idx+1}: API grounding found "
                f"{len(_api_errs)} issue(s)")
            lint_errors = ((lint_errors + "\n") if lint_errors else "") \
                + "\n".join(_api_errs)

        # Determine review mode: "static" (default) or "full"
        review_mode = "static"
        if cfg:
            review_mode = getattr(cfg, "REVIEW_MODE", "static")

        if skip_review:
            # Test step follows — only lint, skip all review
            if lint_errors:
                feedback = f"Lint/syntax errors found:\n{lint_errors}\nFix these issues."
                display.step_info(step_idx, "Lint errors found, retrying...")
                log.warning(f"Step {step_idx+1}: Code lint errors: {lint_errors[:300]}")
                continue
            display.step_info(step_idx, "Review skipped (test step follows) ✔")
            log.info(f"Step {step_idx+1}: Skipped LLM review (test step follows)")
            return True, ""

        # Static review mode: accept if lint + import checks pass,
        # only retry on static failures (no LLM reviewer call).
        if review_mode == "static":
            if lint_errors:
                feedback = f"Static checks found issues:\n{lint_errors}\nFix these issues."
                display.step_info(step_idx, "Static check errors, retrying...")
                log.warning(f"Step {step_idx+1}: Static errors: {lint_errors[:300]}")
                continue
            display.step_info(step_idx, "Static checks passed ✔")
            log.info(f"Step {step_idx+1}: Static review passed (lint + imports OK)")
            return True, ""

        # Skip LLM review if all files were already correct (coder produced no
        # actual diffs). The reviewer would incorrectly FAIL on "no changes"
        # even when the file already satisfies the task requirements.
        if use_diff_review and "```diff" not in review_ctx:
            display.step_info(step_idx, "File(s) already correct, no review needed ✔")
            log.info(f"Step {step_idx+1}: Skipped LLM review — file(s) already correct")
            return True, ""

        # Full LLM review path (review_mode == "full")
        display.step_info(step_idx, "Reviewing code...")
        sent_before, recv_before = token_tracker.snapshot()

        lint_context = f"\n\n{lint_errors}Please fix these errors in your review." if lint_errors else ""

        # Inject KB context so the reviewer has up-to-date framework docs
        # and doesn't reject valid code based on outdated training data.
        reviewer_kb = ""
        if kb_ctx:
            reviewer_kb = (
                f"\n\n[KB Documentation — trust this over your training data]\n"
                f"{kb_ctx}\n"
            )

        # Inject success criteria so the reviewer validates completeness
        criteria_ctx = ""
        if project_context is not None:
            criteria = getattr(project_context, 'success_criteria', [])
            if criteria:
                criteria_ctx = (
                    "\n\nSUCCESS CRITERIA — reject if any are NOT met by the changes:\n"
                    + "\n".join(f"- {c}" for c in criteria)
                )

        # Inject task briefing so the reviewer can catch regressions against
        # the Preserve list (e.g. a feature accidentally removed by the coder).
        _reviewer_briefing = getattr(memory, '_task_briefing', '')
        if _reviewer_briefing:
            criteria_ctx += (
                "\n\nTASK BRIEFING — flag as FAIL if any Preserve item was broken:\n"
                f"{_reviewer_briefing}"
            )

        use_diff_review = cfg and getattr(cfg, "EDITING_REVIEWER_DIFF_MODE", True)
        if use_diff_review:
            review = reviewer.process(
                f"Review this code change for the step: {step_text}\n\n{review_ctx}",
                context=f"Step: {step_text}\nReview ONLY the changes shown.{lint_context}{reviewer_kb}{criteria_ctx}{_verified_api_ctx}",
                language=language,
                review_mode="diff",
            )
        else:
            review = reviewer.process(
                f"Review this code for the step: {step_text}\n\n{response}",
                context=f"Step: {step_text}\nOnly review changes relevant to this step.{lint_context}{reviewer_kb}{criteria_ctx}{_verified_api_ctx}",
                language=language,
            )

        sent_after, recv_after = token_tracker.snapshot()
        sent_delta = sent_after - sent_before
        recv_delta = recv_after - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        if review:
            display.add_llm_log(review, source="Reviewer")

        log.info(f"Step {step_idx+1}: Review:\n{review}")

        review_lower = review.lower()
        # Accept if the reviewer explicitly approves — but never while the
        # API probe reports errors (probe is ground truth).
        approved = _review_verdict(review, _api_errs)
        if not approved and _api_errs and _review_verdict(review, None):
            log.warning(
                f"Step {step_idx+1}: Reviewer approved but the API probe "
                f"found {len(_api_errs)} issue(s) — probe is ground truth, "
                f"rejecting")

        if approved:
            display.step_info(step_idx, "Review passed ✔")
            return True, ""

        # On the last attempt, accept the code if the review only has
        # minor/style suggestions (no keywords indicating actual bugs)
        if attempt == MAX_STEP_RETRIES:
            has_critical = any(kw in review_lower for kw in (
                "error", "bug", "crash", "undefined", "missing import",
                "will fail", "won't work", "does not work", "broken",
                "incorrect", "wrong", "typeerror", "nameerror",
                "syntaxerror", "attributeerror", "keyerror",
                "referenceerror",
            ))
            if not has_critical and not _api_errs:
                display.step_info(step_idx, "Review has only minor suggestions, accepting ✔")
                log.info(f"Step {step_idx+1}: Accepted on last attempt "
                         f"(review had no critical keywords)")
                return True, ""

        feedback = review
        if _api_errs:
            # The reviewer's prose can hallucinate replacement API names
            # (observed: recommending the removed `draw_lrtb_*` as the fix).
            # The probe's suggestions are runtime-verified — make them
            # outrank anything the reviewer suggested.
            feedback += (
                "\n\nAUTHORITATIVE API CHECK (runtime probe against the "
                "installed packages — if the review above suggests different "
                "API names, use THESE instead):\n" + "\n".join(_api_errs)
            )
        display.step_info(step_idx, "Review found issues, retrying...")
        log.warning(f"Step {step_idx+1}: Review issues: {review[:200]}")

    log.error(f"Step {step_idx+1}: Failed after {MAX_STEP_RETRIES} attempts.")
    return False, f"Code step failed after {MAX_STEP_RETRIES} attempts.\nLast review feedback:\n{feedback}"



def _normalize_fix_paths(fix_files: dict[str, str],
                        memory: FileMemory) -> dict[str, str]:
    """Correct LLM-generated paths that are suffixes of known project paths.

    Example: if memory has ``my-app/src/index.js`` and the LLM outputs
    ``src/index.js``, remap to the full path.
    """
    known_paths = set(memory.all_files().keys())
    if not known_paths:
        return fix_files

    corrected: dict[str, str] = {}
    for fpath, content in fix_files.items():
        if fpath in known_paths:
            corrected[fpath] = content
            continue

        # Check if fpath is a suffix of an existing known path
        matched = None
        for known in known_paths:
            if known.endswith('/' + fpath) or known.endswith('\\' + fpath):
                matched = known
                break

        if matched:
            log.warning(f"[PathFix] Remapped '{fpath}' → '{matched}' "
                        f"(matched existing project file)")
            corrected[matched] = content
        else:
            corrected[fpath] = content

    return corrected


def _remap_test_to_existing(test_files: dict[str, str],
                            memory: FileMemory,
                            base_dir: str = ".") -> dict[str, str]:
    """Remap generated test file paths to their existing locations on disk.

    When the LLM generates a test at ``__tests__/App.test.jsx`` but the
    project already has ``src/App.test.jsx``, the generated path is
    remapped to the existing location.  New test files (no existing match)
    are left unchanged — they use the default ``test_root``.

    When multiple test files share the same basename (e.g. both
    ``src/App.test.jsx`` and ``src/__tests__/App.test.jsx`` exist),
    the co-located path (not inside a dedicated test directory) is
    preferred.
    """
    import glob as _glob
    import re as _re

    _TEST_FILE_RE = _re.compile(r'\.(test|spec)\.\w+$')
    _EXCLUDE_DIRS = {'node_modules', '.git', '__pycache__', 'dist', 'build'}
    _DEDICATED_TEST_DIRS = {'__tests__', 'tests', 'test', 'spec'}

    def _is_test_name(name: str) -> bool:
        return bool(_TEST_FILE_RE.search(name)) or name.startswith("test_")

    def _in_dedicated_test_dir(path: str) -> bool:
        """Return True if *path* contains a dedicated test directory segment."""
        parts = path.replace("\\", "/").split("/")
        return any(p in _DEDICATED_TEST_DIRS for p in parts)

    # Build basename → [paths] map (multiple paths possible)
    all_paths: dict[str, list[str]] = {}

    # 1) Existing test files in memory
    for fpath in memory.all_files():
        norm = fpath.replace("\\", "/")
        basename = os.path.basename(norm)
        if _is_test_name(basename):
            all_paths.setdefault(basename, []).append(norm)

    # 2) Scan disk for test files not yet in memory
    for pattern in ("**/*.test.*", "**/*.spec.*", "**/test_*.py"):
        try:
            for match in _glob.glob(pattern, root_dir=base_dir,
                                    recursive=True):
                norm = match.replace("\\", "/")
                parts = norm.split("/")
                if any(p in _EXCLUDE_DIRS for p in parts):
                    continue
                basename = os.path.basename(norm)
                known = all_paths.get(basename, [])
                if norm not in known:
                    all_paths.setdefault(basename, []).append(norm)
        except Exception:
            pass

    if not all_paths:
        return test_files

    # Resolve duplicates: prefer co-located over dedicated test dir
    existing: dict[str, str] = {}
    for basename, paths in all_paths.items():
        if len(paths) == 1:
            existing[basename] = paths[0]
        else:
            # Multiple paths — prefer co-located (not in __tests__/ etc.)
            colocated = [p for p in paths
                         if not _in_dedicated_test_dir(p)]
            existing[basename] = colocated[0] if colocated else paths[0]

    remapped: dict[str, str] = {}
    for fpath, content in test_files.items():
        norm = fpath.replace("\\", "/")
        basename = os.path.basename(norm)
        if basename in existing and existing[basename] != norm:
            target = existing[basename]
            log.warning(f"[TestPathRemap] '{norm}' -> '{target}' "
                        f"(matched existing test file)")
            # Rewrite relative import paths to account for the new
            # directory location.  Without this, imports like
            #   import X from '../components/Foo'
            # that were correct for the OLD path become broken at
            # the NEW path (e.g. src/__tests__/ → src/components/__tests__/).
            content = _rewrite_imports_for_remap(content, norm, target)
            remapped[target] = content
        else:
            remapped[fpath] = content

    return remapped


def _rewrite_imports_for_remap(content: str, old_path: str,
                               new_path: str) -> str:
    """Rewrite relative import paths in *content* when a file moves.

    Given that a file was originally at *old_path* but is being written
    to *new_path*, adjust all relative imports (``./`` or ``../``) so
    they resolve to the same target from the new location.

    Only processes JS/TS-style imports (``import … from '…'`` and
    ``require('…')``).  Python relative imports are not handled here
    because Python test files rarely get remapped.
    """
    import posixpath

    old_dir = posixpath.dirname(old_path.replace("\\", "/"))
    new_dir = posixpath.dirname(new_path.replace("\\", "/"))

    if old_dir == new_dir:
        return content  # same directory — nothing to rewrite

    # Pattern matches:  from '...'  or  from "..."  or  require('...')
    _IMPORT_PATH_RE = re.compile(
        r"""((?:from\s+|require\s*\(\s*)['"])"""   # prefix: from ' or require('
        r"""(\.\.?/.+?)"""                          # the relative path
        r"""(['")]\s*\)?)""",                       # closing quote (and optional paren)
    )

    def _rebase(match: re.Match) -> str:
        prefix = match.group(1)
        rel_path = match.group(2)
        suffix = match.group(3)

        # Resolve the import target as an absolute posix path relative
        # to the OLD directory, then compute a new relative path from
        # the NEW directory.
        abs_target = posixpath.normpath(
            posixpath.join(old_dir, rel_path))
        new_rel = posixpath.relpath(abs_target, new_dir)

        # Ensure it starts with ./ or ../ (bare names are treated as
        # package imports by bundlers).
        if not new_rel.startswith("."):
            new_rel = "./" + new_rel

        if new_rel != rel_path:
            log.debug("[TestPathRemap] Rewrote import: '%s' → '%s' "
                      "(file moved %s → %s)",
                      rel_path, new_rel, old_path, new_path)

        return prefix + new_rel + suffix

    return _IMPORT_PATH_RE.sub(_rebase, content)


def _fix_subproject_test_paths(test_files: dict[str, str],
                                subproject: str | None) -> dict[str, str]:
    """Remap test paths that the LLM placed outside the subproject directory.

    When source files live at ``{subproject}/src/...``, LLMs often mirror
    the full workspace-relative path under ``__tests__/``, producing paths
    like ``__tests__/{subproject}/src/Foo.test.jsx``.  These files land
    outside the subproject and are invisible to the test runner that
    executes inside ``{subproject}/``.

    This function rewrites such paths to
    ``{subproject}/__tests__/src/Foo.test.jsx`` so the test runner can
    find and execute them.
    """
    if not subproject:
        return test_files

    sub_norm = subproject.replace("\\", "/").rstrip("/")
    prefix = f"__tests__/{sub_norm}/"

    remapped: dict[str, str] = {}
    for fpath, content in test_files.items():
        norm = fpath.replace("\\", "/")
        if norm.startswith(prefix):
            rest = norm[len(prefix):]
            new_path = f"{sub_norm}/__tests__/{rest}"
            log.warning(
                f"[SubprojectTestPath] '{norm}' -> '{new_path}' "
                f"(moved inside subproject dir)"
            )
            remapped[new_path] = content
        else:
            remapped[fpath] = content

    return remapped


def _filter_test_only_files(fix_files: dict[str, str],
                            test_files: dict[str, str],
                            memory: FileMemory) -> dict[str, str]:
    """Filter fix files to only allow test files during test fix loop.

    Blocks writes to:
    - Protected manifest files (package.json, etc.)
    - Source files that already exist in memory (prevents overwrite)

    Allows writes to:
    - Files that were part of the original test_files
    - Files in test directories (__tests__/, tests/, spec/, test/)
    - Files with test naming patterns (test_*, *.test.*, *_test.*, *_spec.*)
    """
    import re
    import os

    allowed: dict[str, str] = {}
    known_source_files = set(memory.all_files().keys())
    test_paths = set(test_files.keys())

    # Patterns for test files
    _TEST_DIR_PATTERNS = {'__tests__', 'tests', 'test', 'spec'}
    _TEST_NAME_RE = re.compile(
        r'(^test_|[./]test[./]|\.test\.|_test\.|_spec\.|spec[./])',
        re.IGNORECASE
    )

    for fpath, content in fix_files.items():
        basename = os.path.basename(fpath)

        # Block: protected manifest files
        if basename in Executor._PROTECTED_FILENAMES:
            log.warning(f"[TestFix] Blocked write to protected file: {fpath}")
            continue

        # Allow: file was in original test_files
        if fpath in test_paths:
            allowed[fpath] = content
            continue

        # Allow: file is in a test directory
        path_parts = set(fpath.replace('\\', '/').split('/'))
        if path_parts & _TEST_DIR_PATTERNS:
            allowed[fpath] = content
            continue

        # Allow: file matches test naming pattern
        if _TEST_NAME_RE.search(fpath):
            allowed[fpath] = content
            continue

        # Block: file is a known source file in memory
        if fpath in known_source_files:
            log.warning(f"[TestFix] Blocked write to source file during "
                        f"test fix: {fpath}")
            continue

        # Default: allow unknown new files (might be test helpers)
        allowed[fpath] = content

    blocked_count = len(fix_files) - len(allowed)
    if blocked_count > 0:
        log.info(f"[TestFix] Blocked {blocked_count} non-test file(s) "
                 f"from test fix write")

    return allowed


def _cleanup_stale_js_test_files(
    test_files: dict[str, str],
    memory: FileMemory,
    display: CLIDisplay,
    step_idx: int,
    subproject_cwd: str | None,
) -> None:
    """Remove stale .test.js files when a .test.jsx version was just written.

    Vite/Rollup cannot parse JSX in .js files.  When the TesterAgent generates
    .test.jsx files, any leftover .test.js with the same base name will still
    be picked up by the test runner and cause parse errors.
    """
    jsx_files = [f for f in test_files if f.endswith(('.test.jsx', '.test.tsx'))]
    if not jsx_files:
        return

    removed: list[str] = []
    for jsx_path in jsx_files:
        # Derive the .js / .ts counterpart
        if jsx_path.endswith('.test.jsx'):
            stale = jsx_path.replace('.test.jsx', '.test.js')
        else:
            stale = jsx_path.replace('.test.tsx', '.test.ts')

        abs_stale = (os.path.join(subproject_cwd, stale) if subproject_cwd
                     else stale)
        if os.path.isfile(abs_stale):
            try:
                os.remove(abs_stale)
                removed.append(stale)
                log.info(f"Step {step_idx+1}: Removed stale test file "
                         f"{stale} (replaced by {jsx_path})")
            except OSError as e:
                log.warning(f"Step {step_idx+1}: Failed to remove "
                            f"stale {stale}: {e}")

    if removed:
        display.step_info(step_idx,
                          f"Cleaned up stale .js test files: "
                          f"{', '.join(removed)}")


def _cleanup_ghost_test_files(
    test_files: dict[str, str],
    prev_step_test_files: set[str],
    display: CLIDisplay,
    step_idx: int,
    subproject_cwd: str | None,
) -> None:
    """Remove ghost test files left by previous gen attempts of the SAME step.

    When the TesterAgent retries generation (gen_attempt loop), the new attempt
    may produce a different set of files than the previous attempt.  Files from
    the old attempt that aren't in the new ``test_files`` become "ghosts" —
    the test runner picks them up and they fail with stale import paths.

    ``prev_step_test_files`` should contain the set of file paths written by
    earlier gen attempts **of the same step** (NOT files from other steps).
    This prevents accidentally deleting test files produced by earlier steps
    in the pipeline.
    """
    if not prev_step_test_files:
        return

    current_paths = set(test_files.keys())
    ghosts = prev_step_test_files - current_paths
    if not ghosts:
        return

    removed: list[str] = []
    base = subproject_cwd or "."

    for ghost_path in ghosts:
        abs_path = os.path.join(base, ghost_path) if not os.path.isabs(ghost_path) else ghost_path
        if os.path.isfile(abs_path):
            try:
                os.remove(abs_path)
                removed.append(ghost_path)
                log.info(f"Step {step_idx+1}: Removed ghost test file "
                         f"{ghost_path}")
            except OSError as e:
                log.warning(f"Step {step_idx+1}: Failed to remove "
                            f"ghost {ghost_path}: {e}")

    if removed:
        display.step_info(step_idx,
                          f"Cleaned up {len(removed)} ghost test file(s): "
                          f"{', '.join(removed)}")


# ── Enhancement #6: Import-trace context injection ──────────────
def _extract_imported_sources(test_files: dict[str, str],
                              memory: FileMemory,
                              resolve_from_disk: bool = False) -> dict[str, str]:
    """Parse import statements in test files to identify tested source files.

    Returns a dict of {filepath: content} for source files that are
    directly imported by the test files.

    When *resolve_from_disk* is True, imports that are not found in memory
    are resolved relative to the test file's directory on disk and read
    directly.  This covers checkpoint-resume runs where source files written
    in a previous session are no longer tracked in FileMemory.
    """
    import re as _re
    import os as _os

    # Python: from src.snake_game import ... / import snake_game
    _PY_IMPORT_RE = _re.compile(
        r'(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))')
    # JS/TS: import ... from '../src/app' / require('../src/app')
    _JS_IMPORT_RE = _re.compile(
        r'''(?:from\s+['"](.+?)['"]|require\s*\(\s*['"](.+?)['"]\s*\))''')

    _JS_EXTS = ('.jsx', '.js', '.tsx', '.ts', '.mjs', '.cjs')
    _PY_EXTS = ('.py',)

    all_files = memory.all_files()
    imported_sources: dict[str, str] = {}

    for _tpath, tcontent in test_files.items():
        # Collect candidate module paths from imports
        candidates: set[str] = set()
        # Also keep the raw relative paths for disk resolution
        raw_rel_imports: list[str] = []

        for m in _PY_IMPORT_RE.finditer(tcontent):
            mod = m.group(1) or m.group(2)
            if mod:
                candidates.add(mod.replace('.', '/'))
        for m in _JS_IMPORT_RE.finditer(tcontent):
            rel = m.group(1) or m.group(2)
            if rel and not rel.startswith(('@', 'react', 'vitest', 'jest', 'node:')):
                raw_rel_imports.append(rel)
                clean = _re.sub(r'^\.\.?/', '', rel)
                candidates.add(clean)

        # Match candidates against known memory files
        for fpath, content in all_files.items():
            if fpath in imported_sources:
                continue
            # Skip test files themselves
            if 'test' in fpath.lower():
                continue
            for cand in candidates:
                if cand in fpath or fpath.endswith(cand) or fpath.endswith(cand + '.py'):
                    imported_sources[fpath] = content
                    break

        # Disk fallback: resolve relative imports from the test file's directory
        if resolve_from_disk and raw_rel_imports:
            test_dir = _os.path.dirname(_tpath)
            for rel in raw_rel_imports:
                # Only resolve relative paths (start with . or ..)
                if not rel.startswith('.'):
                    continue
                resolved_base = _os.path.normpath(
                    _os.path.join(test_dir, rel)).replace('\\', '/')
                # Skip if already found via memory
                if any(resolved_base in fp or fp.startswith(resolved_base)
                       for fp in imported_sources):
                    continue
                # Try common source extensions
                exts = _JS_EXTS if _tpath.endswith(
                    ('.jsx', '.js', '.tsx', '.ts', '.mjs')) else _PY_EXTS
                for ext in ('', *exts):
                    candidate_path = resolved_base + ext
                    if candidate_path in imported_sources:
                        break
                    try:
                        with open(candidate_path, 'r', encoding='utf-8',
                                  errors='replace') as _f:
                            imported_sources[candidate_path] = _f.read()
                        break
                    except OSError:
                        continue

    return imported_sources


# ── Enhancement #9: Spec-driven test generation ─────────────────
def _extract_source_specs(code_summary: str) -> str:
    """Extract function/class signatures and docstrings from source code.

    Returns a compact spec block the tester can use to understand
    what behaviors to test.
    """
    import re as _re

    specs: list[str] = []
    # Parse #### [FILE]: ... ```...``` blocks
    for match in _re.finditer(
            r'####\s+\[FILE\]:\s*(.+?)\n```\w*\n(.*?)```',
            code_summary, _re.DOTALL):
        filepath, content = match.groups()
        if 'test' in filepath.lower():
            continue
        file_specs: list[str] = []
        # Extract class/function definitions + docstrings
        for func_match in _re.finditer(
                r'((?:class|def)\s+\w+[^:]*:)\s*\n(\s+(?:""".*?"""|\'\'\'.*?\'\'\'))?',
                content, _re.DOTALL):
            sig = func_match.group(1).strip()
            doc = func_match.group(2)
            if doc:
                file_specs.append(f"  {sig}\n    {doc.strip()}")
            else:
                file_specs.append(f"  {sig}")
        if file_specs:
            specs.append(f"# {filepath}\n" + "\n".join(file_specs))

    return "\n\n".join(specs) if specs else ""


# ── Enhancement #5: Bidirectional bug detection ──────────────────
_TEST_BUG_PATTERNS = re.compile(
    r"Unable to find an element with the text:"
    r"|You cannot render a <Router> inside another <Router>"
    r"|TestingLibraryElementError:"
    r"|Unable to find role"
    r"|Expected.*to have class"
    r"|Expected.*to have attribute"
    r"|Expected.*to be in the document"
    # Import resolution failures in test files — almost always a wrong
    # relative path in the test, not a missing source file.
    r"|Failed to resolve import .* from .*\.(?:test|spec)\."
    r"|Cannot find module .* from .*\.(?:test|spec)\."
    r"|Module not found:.*\.(?:test|spec)\.",
    re.DOTALL,
)


# ── Error signature extraction for oscillation detection ──────────
_ERROR_SIG_PATTERNS = re.compile(
    r"(?:Failed to resolve import .+? from .+)"
    r"|(?:Cannot find module .+? from .+)"
    r"|(?:Module not found:.+)"
    r"|(?:TestingLibraryElementError:.+)"
    r"|(?:Unable to find (?:an element|role).+)"
    r"|(?:Error: .+? is not (?:a function|defined|a constructor))"
    r"|(?:(?:TypeError|ReferenceError|SyntaxError):.+)"
    r"|(?:Expected .+? (?:to (?:be|have|equal|match)|not to).+)",
)


def _extract_error_signature(output: str) -> str | None:
    """Extract a short, stable signature from test failure output.

    Returns a normalised string that identifies the *class* of error
    (e.g. the failing import path, the missing element query, the
    TypeError message) without being sensitive to line numbers or
    whitespace differences.  Returns ``None`` if no recognisable
    error pattern is found.
    """
    # Try structured patterns first
    m = _ERROR_SIG_PATTERNS.search(output)
    if m:
        sig = m.group(0).strip()
        # Normalise whitespace and truncate to keep signatures comparable
        sig = re.sub(r'\s+', ' ', sig)[:200]
        return sig

    # Fallback: grab the first FAIL line (Vitest / Jest style)
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAIL") or "FAILED" in stripped:
            return re.sub(r'\s+', ' ', stripped)[:200]

    return None


def effective_bug_origin(bug_origin: str, config_fix_was_empty: bool) -> str:
    """Route a repeat CONFIG_BUG to the source path once config is ruled out.

    A CONFIG_BUG fix that produced no in-scope change has said something
    useful: the model does not think the problem is in the config either —
    asked to fix the test environment, it answered with source files, which
    the scope guard then dropped. Believing the verdict a second time makes
    the step loop (triage CONFIG_BUG -> everything dropped -> attempt spent
    -> triage CONFIG_BUG again): safe, because nothing is written, but it
    consumes the retry budget without ever touching the defect. Observed
    three times consecutively on a step whose real failure was a single
    missing method, ``Ghost.get_tile_position``.

    Only the repeat is redirected; the first CONFIG_BUG verdict is honoured
    normally, because genuinely missing config does exist and fixing it is
    what this path is for.
    """
    if bug_origin == "CONFIG_BUG" and config_fix_was_empty:
        return "SOURCE_BUG"
    return bug_origin


def _triage_test_failure(error_detail: str, source_summary: str,
                         test_summary: str, llm_client,
                         display: CLIDisplay, step_idx: int,
                         language: str | None = None) -> str:
    """Determine whether a test failure is a TEST_BUG or SOURCE_BUG.

    Returns 'TEST_BUG', 'SOURCE_BUG', or 'CONFIG_BUG'.
    """
    # Fast-path: known JS/RTL patterns that are unambiguously test bugs
    if _TEST_BUG_PATTERNS.search(error_detail):
        log.info(f"Step {step_idx+1}: Triage result: TEST_BUG (pattern match)")
        return "TEST_BUG"

    # Fast-path: SyntaxError / IndentationError / circular import in a source
    # file is always a SOURCE_BUG — never a config issue.  The LLM frequently
    # misclassifies these as CONFIG_BUG because the test "cannot even run",
    # but the real fix is always in the source file that has the problem.
    if re.search(r'(?:IndentationError|SyntaxError)\b', error_detail):
        log.info(f"Step {step_idx+1}: Triage result: SOURCE_BUG "
                 f"(SyntaxError/IndentationError pattern match)")
        return "SOURCE_BUG"

    if re.search(r'(?:most likely due to a circular import|'
                 r'cannot import name .* from partially initialized module)',
                 error_detail):
        log.info(f"Step {step_idx+1}: Triage result: SOURCE_BUG "
                 f"(circular import pattern match)")
        return "SOURCE_BUG"

    lang_hint = f"Project language: {language}\n" if language else ""
    triage_prompt = (
        "A test has failed. Analyze the error and determine the root cause.\n"
        f"{lang_hint}"
        "Answer with ONLY one word: TEST_BUG, SOURCE_BUG, or CONFIG_BUG\n\n"
        "- TEST_BUG    = the test assertion, selector, or import is wrong "
        "(e.g. looking for text/classes that do not match what the source renders)\n"
        "- SOURCE_BUG  = the source code under test has a logic, syntax, or "
        "implementation error that the test correctly exposes\n"
        "- CONFIG_BUG  = the test framework or environment is misconfigured — "
        "the test cannot even run (e.g. missing jsdom environment so `document` "
        "is undefined, missing pytest fixture, missing test setup file, wrong "
        "config key, missing required package for the test runner)\n\n"
        "CRITICAL FOR UI COMPONENTS: If a test fails because it cannot find an "
        "element (text, role, test-id, etc.) or asserts specific CSS classes that "
        "do not exist in the source, it is ALMOST ALWAYS a TEST_BUG — do NOT label "
        "SOURCE_BUG just to force the source to match brittle assertions.\n\n"
        f"Test output:\n{error_detail[:3000]}\n\n"
    )
    if source_summary:
        triage_prompt += f"Source files:\n{source_summary[:4000]}\n\n"
    if test_summary:
        triage_prompt += f"Test files:\n{test_summary[:4000]}\n"

    display.step_info(step_idx, "Analyzing failure origin (test / source / config)...")

    sent_before, recv_before = token_tracker.snapshot()
    response = llm_client.generate_response(triage_prompt).strip().upper()
    sent_after, recv_after = token_tracker.snapshot()
    display.step_tokens(step_idx, sent_after - sent_before,
                        recv_after - recv_before)

    if "CONFIG_BUG" in response:
        log.info(f"Step {step_idx+1}: Triage result: CONFIG_BUG")
        return "CONFIG_BUG"
    if "SOURCE_BUG" in response:
        log.info(f"Step {step_idx+1}: Triage result: SOURCE_BUG")
        return "SOURCE_BUG"
    log.info(f"Step {step_idx+1}: Triage result: TEST_BUG")
    return "TEST_BUG"

def _handle_test_step(step_text: str, tester: TesterAgent, coder: CoderAgent,
                      reviewer: ReviewerAgent, executor: Executor,
                      task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int,
                      **kwargs) -> tuple[bool, str]:
    """Public TEST-step entry: runs the handler, then the acceptance gate.

    Mirrors :func:`_handle_code_step` — the agent-loop path already gates
    on the declared verify (or the language default), so the outer gate
    only fires for the classic generate→run→fix path.
    """
    success, error_info = _handle_test_step_impl(
        step_text, tester, coder, reviewer, executor, task, memory,
        display, step_idx, **kwargs)
    from .agent_loop import agent_loop_enabled
    if not agent_loop_enabled(kwargs.get("cfg"),
                              getattr(tester, "llm_client", None)):
        success, error_info = _gate_on_declared_verify(
            success, error_info, kwargs.get("plan_step"), executor, memory,
            display, step_idx, task=task)
    success, error_info = _django_lint_gate(
        success, error_info, memory, display, step_idx)
    _record_passed_gate(success, kwargs.get("plan_step"), memory, task=task)
    return success, error_info


def _handle_test_step_impl(step_text: str, tester: TesterAgent, coder: CoderAgent,
                      reviewer: ReviewerAgent, executor: Executor,
                      task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int,
                      language: str | None = None,
                      auto: bool = False,
                      search_agent=None,
                      project_context=None,
                      kb_context_builder=None,
                      plan_step=None,
                      all_plan_steps=None,
                      project_profile=None,
                      intent_spec=None,
                      cfg: Config | None = None) -> tuple[bool, str]:
    # ── Agent loop (opt-in): run the step as a bounded tool-calling loop ──
    from .agent_loop import agent_loop_enabled
    if agent_loop_enabled(cfg, getattr(tester, "llm_client", None)):
        from .agent_loop import (
            build_step_tools, run_agent_loop_with_escalation,
        )
        display.step_info(step_idx, "Agent loop: executing test step with tools")
        tools = build_step_tools(executor, memory,
                                 kb_context_builder=kb_context_builder)
        # Deterministic exit: the model's "done" claim is only accepted
        # once the test suite actually passes (when a trustworthy test
        # command exists for the language). The project may live in a
        # scaffolded subdirectory — detect it so manage.py/package.json
        # are found and the command runs in the right cwd.
        from .agent_loop import verify_cmd_for_language
        _vsub = _detect_subproject_root(memory)
        # Plan-declared acceptance command wins (it names the exact
        # suite/app this step must satisfy); the language-level default
        # is the fallback when the plan declared nothing.
        verify_cmd = _declared_verify_cmd(plan_step, memory, task=task)
        if not verify_cmd:
            verify_cmd = verify_cmd_for_language(language, _vsub or ".")
            if verify_cmd and _vsub:
                verify_cmd = f"cd {_vsub} && {verify_cmd}"

        # Ground the loop in the CURRENT test status: without this the
        # model tends to spend its whole turn budget reading files
        # (observed twice: 8 turns of read_file, zero commands run).
        # Seeing the failing output — and the exact exit criterion —
        # up front turns the loop into a fix loop instead of a tour.
        loop_context = memory.summary()
        _brief = _plan_step_brief(plan_step)
        if _brief:
            loop_context = f"{_brief}\n\n{loop_context}"
        if verify_cmd:
            _pre_ok, _pre_out = executor.run_command(verify_cmd, timeout=300)
            # Missing-dependency failures are environment problems the
            # loop's file edits can never fix — heal them before the loop
            # so its turns go to real test work (mirrors BulkTest).
            if not _pre_ok:
                from .agent_loop import attempt_env_self_heal
                _healed: set = set()
                while not _pre_ok and attempt_env_self_heal(
                        tools, _pre_out or "", language, _healed, verify_cmd,
                        planned_files=_loop_preload_paths(plan_step)):
                    _pre_ok, _pre_out = executor.run_command(
                        verify_cmd, timeout=300)
            _status = "PASSING" if _pre_ok else "FAILING"
            loop_context += (
                f"\n\nCurrent test status ({_status}) from `{verify_cmd}`:\n"
                f"{(_pre_out or '(no output)')[-1500:]}\n\n"
                f"This step is complete ONLY when `{verify_cmd}` exits "
                f"successfully — it will be run to verify your work.")

        loop_result = run_agent_loop_with_escalation(
            tester.llm_client, tools, step_text, task,
            escalation_client=getattr(tester, "escalation_client", None),
            display=display, step_idx=step_idx, language=language,
            max_turns=getattr(cfg, "AGENT_LOOP_MAX_TURNS", 8),
            verify_cmd=verify_cmd,
            context=loop_context,
            preload_files=_loop_preload_paths(plan_step),
            preload_full_paths=set(
                getattr(plan_step, 'target_files', None) or ()),
            required_files=set(
                getattr(plan_step, 'target_files', None) or ()),
        )
        # A green gate that named ONE test file proves that file runs — not
        # that the project's own runner can find it. Observed: the plan
        # declared `python -m unittest -v tests/test_game.py`, the step
        # passed, and the delivered project answered `python -m unittest -v`
        # with "Ran 0 tests" because nothing made tests/ a package. The
        # whole point of a TEST step is a suite the project can run.
        if loop_result[0]:
            from .agent_loop import truncate_middle
            _disc_ok, _disc_cmd, _disc_out = _tests_are_discoverable(
                executor, language, _vsub, verify_cmd)
            if not _disc_ok:
                log.error(
                    "[TestStep] gate %r passed but %r collects no tests — "
                    "the suite is not discoverable by the project's own "
                    "runner", verify_cmd, _disc_cmd)
                return False, (
                    f"The step's gate ({verify_cmd}) passed, but the "
                    f"project's own test runner finds nothing:\n\n"
                    f"$ {_disc_cmd}\n{truncate_middle(_disc_out, 800)}\n\n"
                    "Tests that only run when named by path are not a "
                    "suite. Make them discoverable (for unittest, that "
                    "usually means an __init__.py in the tests package).")

        # Record the gate exactly as the loop enforced it (see
        # _record_passed_gate) so the monotonic recheck can never diverge
        # from what actually ran green.
        if loop_result[0] and verify_cmd and plan_step is not None:
            setattr(plan_step, "_verified_gate_cmd", verify_cmd)
        return loop_result

    # Detect sub-project (if the test targets a nested folder)
    subproject_cwd = _detect_subproject_root(memory)

    # ── Proactive pre-install: ensure packages are installed ──
    # Runs once, just before the first test suite command,
    # when the project scaffold already exists.
    if project_context is not None:
        _ensure_packages_installed(
            project_context, executor, memory, display, step_idx,
            subproject_cwd=subproject_cwd, language=language,
        )

    # ── Scan ALL memory files for missing package imports ──
    # Code steps only scan their own generated files, but tests
    # transitively import other components that may reference
    # packages not yet installed (e.g. @heroicons/react).
    # Scan the full memory to catch these before the test runs.
    _auto_install_code_imports(
        memory.all_files(), executor, memory, display, step_idx,
    )

    # Infer language from memory file paths when not explicitly provided
    if language is None:
        mem_files = list(memory.all_files().keys())
        if mem_files:
            language = detect_language_from_files(mem_files)
            if language:
                log.info(f"Step {step_idx+1}: Inferred language '{language}' "
                         f"from memory files")

    # Detect JS/TS project environment for ESM-aware test generation
    js_env: dict | None = None
    test_runner: str | None = None
    if language in ("javascript", "typescript"):
        js_env = _read_js_project_env(subproject_cwd)
        test_runner = js_env.get("test_runner")
        log.info(f"Step {step_idx+1}: JS project env: {js_env}")

        # ── Auto-create vitest config when missing ──
        # If the project uses Vitest (installed or Vite-based) but has no
        # vitest.config.js, tests fail with "describe is not defined"
        # because globals:true isn't set.  Create the config from KB
        # templates (zero LLM calls) to prevent cascading failures.
        if (test_runner == "vitest"
                and not js_env.get("has_vitest_config")
                and js_env.get("has_vitest")):
            _cwd = subproject_cwd or "."
            _vitest_cfg_path = os.path.join(_cwd, "vitest.config.js")
            _vitest_setup_path = os.path.join(_cwd, "vitest.setup.js")

            # Check if jsx/tsx files exist to decide on React plugin
            _has_jsx = js_env.get("has_jsx") or js_env.get("has_tsx")

            if not os.path.isfile(_vitest_cfg_path):
                _cfg_content = (
                    "import { defineConfig } from 'vitest/config'\n"
                )
                if _has_jsx:
                    _cfg_content += (
                        "import react from '@vitejs/plugin-react'\n"
                    )
                _cfg_content += (
                    "\nexport default defineConfig({\n"
                )
                if _has_jsx:
                    _cfg_content += "  plugins: [react()],\n"
                _cfg_content += (
                    "  test: {\n"
                    "    environment: 'jsdom',\n"
                    "    globals: true,\n"
                    "    setupFiles: './vitest.setup.js',\n"
                    "  },\n"
                    "})\n"
                )
                _cfg_rel = os.path.relpath(_vitest_cfg_path, ".")
                executor.write_files({_cfg_rel: _cfg_content})
                memory.update({_cfg_rel: _cfg_content})
                log.info(f"Step {step_idx+1}: Auto-created {_cfg_rel} "
                         f"(vitest installed but no config found)")

            if not os.path.isfile(_vitest_setup_path):
                _setup_content = "import '@testing-library/jest-dom/vitest'\n"
                _setup_rel = os.path.relpath(_vitest_setup_path, ".")
                executor.write_files({_setup_rel: _setup_content})
                memory.update({_setup_rel: _setup_content})
                log.info(f"Step {step_idx+1}: Auto-created {_setup_rel}")

            # Re-read env now that config exists
            js_env = _read_js_project_env(subproject_cwd)
            test_runner = js_env.get("test_runner")

    # Use language-aware defaults (fall back to Python only as last resort)
    lang_tag = get_code_block_lang(language) if language else "python"
    fw = get_test_framework(language, test_runner=test_runner) if language else get_test_framework("python")
    test_cmd = fw["command"]

    # ── Plan-declared runner ──
    # The language default is a guess; the plan's own acceptance command is
    # evidence. A task whose acceptance is `python -m unittest -v`, planned
    # with a `verify: python -m unittest ...` gate, was still handed the
    # Python default of pytest — so the run installed pytest it did not
    # need, took its baseline through a runner the project never uses, and
    # wrote pytest-flavoured tests against a unittest acceptance gate.
    # unittest is stdlib, so switching to it costs nothing and can never
    # fail to be present.
    _planned_runner = _plan_declared_test_runner(plan_step, language)
    if _planned_runner and _planned_runner != test_cmd:
        log.info(f"Step {step_idx+1}: Plan's verify runs {_planned_runner!r} "
                 f"— using it instead of the language default {test_cmd!r}")
        test_cmd = _planned_runner

    # Django project detection: manage.py test is the canonical runner for Django.
    # It handles test DB setup/teardown and settings without requiring pytest.
    if (language == "python" or not language) and os.path.isfile(
        os.path.join(subproject_cwd, "manage.py") if subproject_cwd else "manage.py"
    ):
        test_cmd = "python manage.py test"
        log.info(
            f"Step {step_idx+1}: Django project detected (manage.py present) "
            f"— overriding test command to 'python manage.py test'"
        )

    # Ensure the test runner binary is installed before attempting to run tests
    parts = test_cmd.split()
    runner = parts[0]
    # For "python -m <module>" (e.g. "python -m pytest"), check if the module
    # is importable rather than checking the binary — python is always on PATH.
    if runner in ("python", "python3") and "-m" in parts:
        try:
            m_idx = parts.index("-m")
            actual_tool = parts[m_idx + 1] if m_idx + 1 < len(parts) else None
        except (ValueError, IndexError):
            actual_tool = None
        if actual_tool:
            # Probe the interpreter that will actually run the tests (the
            # executor resolves `python` to the project venv when one exists)
            # — an in-process find_spec would check agentchanti's own
            # environment instead and give a false positive/negative.
            if re.fullmatch(r'[\w.]+', actual_tool):
                module_found, _ = executor.run_command(
                    f'python -c "import {actual_tool}"', cwd=subproject_cwd,
                )
            else:
                module_found = False
            if not module_found:
                install_cmd = _get_runner_install_cmd(actual_tool, cwd=subproject_cwd)
                if install_cmd:
                    display.step_info(step_idx, f"`{actual_tool}` not found, installing...")
                    log.info(f"Step {step_idx+1}: Auto-installing: {install_cmd}")
                    ok, out = executor.run_command(install_cmd, cwd=subproject_cwd)
                    if ok:
                        display.step_info(step_idx, f"Installed `{actual_tool}`")
                    else:
                        log.warning(f"Step {step_idx+1}: Failed to install "
                                    f"{actual_tool}: {out[:200]}")
    # For "npx <tool>", the binary to check is "npx" itself
    elif not shutil.which(runner):
        actual_tool = parts[1] if runner == "npx" and len(parts) > 1 else runner
        install_cmd = _get_runner_install_cmd(actual_tool, cwd=subproject_cwd)
        if install_cmd is None:
            # System-level tool (go, cargo, etc.) — can't auto-install
            msg = (f"`{runner}` is not installed. It must be installed manually "
                   f"(it cannot be installed via pip/npm).")
            display.step_info(step_idx, msg)
            log.error(f"Step {step_idx+1}: {msg}")
            return False, msg
        display.step_info(step_idx, f"`{runner}` not found, installing...")
        log.info(f"Step {step_idx+1}: Auto-installing: {install_cmd}")
        ok, out = executor.run_command(install_cmd, cwd=subproject_cwd)
        if ok:
            display.step_info(step_idx, f"Installed `{actual_tool}`")
        else:
            log.warning(f"Step {step_idx+1}: Failed to install "
                        f"{actual_tool}: {out[:200]}")

    # Auto-setup for Jest-based JS/TS projects (skip when Vitest is the runner)
    if js_env and test_runner != "vitest":
        # Auto-setup for ESM projects: install @jest/globals if needed
        if js_env.get("is_esm") and not js_env.get("has_jest_globals"):
            display.step_info(step_idx, "ESM project detected, installing @jest/globals...")
            ok, out = executor.run_command("npm install --save-dev @jest/globals", cwd=subproject_cwd)
            if ok:
                js_env["has_jest_globals"] = True
                display.step_info(step_idx, "Installed @jest/globals")
            else:
                log.warning(f"Step {step_idx+1}: Failed to install @jest/globals: {out[:200]}")

        # Create minimal jest.config for ESM if missing
        if js_env.get("is_esm") and not js_env.get("has_jest_config"):
            jest_config_content = (
                "// Auto-generated for ESM compatibility\n"
                "export default {\n"
                "  transform: {},\n"
                "};\n"
            )
            config_path = os.path.join(subproject_cwd, "jest.config.js") if subproject_cwd else "jest.config.js"
            if not os.path.isfile(config_path):
                try:
                    with open(config_path, "w", encoding="utf-8") as f:
                        f.write(jest_config_content)
                    js_env["has_jest_config"] = True
                    display.step_info(step_idx, "Created jest.config.js for ESM")
                    log.info(f"Step {step_idx+1}: Auto-created jest.config.js for ESM")
                except OSError as e:
                    log.warning(f"Step {step_idx+1}: Failed to create jest.config.js: {e}")

    # ── Scoped context: only include files relevant to this test step ──
    # Priority order:
    #   A. Plan-aware context (structured plan imports/targets — most precise)
    #   B. Scoped context (target test file → parse imports → setup files)
    #   C. Semantic search fallback
    all_files = memory.all_files()
    # Plan context moved to a thread-local store when parallel wave steps
    # started racing on the shared FileMemory instance; this reader kept
    # the old attribute name and so was always None — branch A below has
    # been dead ever since. Cost of the silent fallback: a TEST step that
    # declares `target: test_pacman.py` and four imports was still handed
    # a 12k-token semantic-search blob of every file in the project, and
    # logged "No target test files detected" while the plan named one.
    from .memory import get_plan_context_files as _get_test_plan_ctx
    plan_ctx = _get_test_plan_ctx()

    if plan_ctx and plan_step is not None:
        # ── Plan-aware context: use plan-declared imports + targets ──
        # Merge plan context with setup/config files the test runner needs
        _SETUP_FILE_NAMES = frozenset({
            'vitest.config.ts', 'vitest.config.js', 'vitest.config.mts',
            'vitest.setup.js', 'vitest.setup.ts',
            'jest.config.js', 'jest.config.ts', 'jest.config.mjs', 'jest.config.cjs',
            'setupTests.js', 'setupTests.ts', 'setup.js', 'setup.ts',
            'conftest.py', 'pytest.ini', 'setup.cfg',
            'vite.config.js', 'vite.config.ts',
        })
        full_content_files: dict[str, str] = dict(plan_ctx)
        # Also include setup files from memory that plan may not declare
        for fpath, content in all_files.items():
            if fpath not in full_content_files and os.path.basename(fpath) in _SETUP_FILE_NAMES:
                full_content_files[fpath] = content

        code_summary = ""
        included_full: set[str] = set()
        for fname, content in full_content_files.items():
            code_summary += f"#### [FILE]: {fname}\n```{lang_tag}\n{content}\n```\n\n"
            included_full.add(fname)

        # Slim skeletons for remaining files
        from .memory import _extract_file_skeleton
        slim_parts: list[str] = []
        for fname, content in all_files.items():
            if fname in included_full:
                continue
            if fname.startswith(('_cmd_output/', '_fix_output/', '_search_context/')):
                continue
            skeleton = _extract_file_skeleton(content, fname)
            if skeleton:
                slim_parts.append(skeleton)
            else:
                slim_parts.append(f"- {fname}")
        if slim_parts:
            code_summary += "Other project files (signatures only):\n" + "\n".join(slim_parts) + "\n\n"

        log.info(f"Step {step_idx+1}: Plan-aware test context: {len(full_content_files)} full "
                 f"+ {len(slim_parts)} skeleton(s)")
    else:
        # ── Legacy scoped context ──
        target_test_files = _detect_target_files(step_text, memory, max_files=5)

        # Identify source files imported by the target test(s)
        target_contents: dict[str, str] = {}
        for tf in target_test_files:
            if tf in all_files:
                target_contents[tf] = all_files[tf]
        imported_sources = _extract_imported_sources(target_contents, memory) if target_contents else {}

        if target_test_files:
            _SETUP_FILE_NAMES = frozenset({
                'vitest.config.ts', 'vitest.config.js', 'vitest.config.mts',
                'vitest.setup.js', 'vitest.setup.ts',
                'jest.config.js', 'jest.config.ts', 'jest.config.mjs', 'jest.config.cjs',
                'setupTests.js', 'setupTests.ts', 'setup.js', 'setup.ts',
                'conftest.py', 'pytest.ini', 'setup.cfg',
                'vite.config.js', 'vite.config.ts',
            })

            full_content_files: dict[str, str] = {}
            for fpath, content in all_files.items():
                basename = os.path.basename(fpath)
                if fpath in target_contents:
                    full_content_files[fpath] = content
                elif fpath in imported_sources:
                    full_content_files[fpath] = content
                elif basename in _SETUP_FILE_NAMES:
                    full_content_files[fpath] = content

            code_summary = ""
            included_full: set[str] = set()
            for fname, content in full_content_files.items():
                code_summary += f"#### [FILE]: {fname}\n```{lang_tag}\n{content}\n```\n\n"
                included_full.add(fname)

            from .memory import _extract_file_skeleton
            slim_parts: list[str] = []
            for fname, content in all_files.items():
                if fname in included_full:
                    continue
                if fname.startswith(('_cmd_output/', '_fix_output/', '_search_context/')):
                    continue
                skeleton = _extract_file_skeleton(content, fname)
                if skeleton:
                    slim_parts.append(skeleton)
                else:
                    slim_parts.append(f"- {fname}")

            if slim_parts:
                code_summary += "Other project files (signatures only):\n" + "\n".join(slim_parts) + "\n\n"

            log.info(f"Step {step_idx+1}: Test context scoped to {len(full_content_files)} "
                     f"full file(s) + {len(slim_parts)} skeleton(s) "
                     f"(from {len(all_files)} total in memory)")
        else:
            code_summary = memory.related_context(step_text)
            if not code_summary:
                code_summary = ""
                for fname, content in all_files.items():
                    code_summary += f"#### [FILE]: {fname}\n```{lang_tag}\n{content}\n```\n\n"
            log.info(f"Step {step_idx+1}: No target test files detected, using "
                     f"semantic search context (from {len(all_files)} files)")

    log.info(f"Step {step_idx+1}: JS project env: {js_env}")

    # ── Pre-execution Analysis ──
    pre_analysis_results = perform_baseline_test_analysis(
        memory, executor, language,
        project_profile=project_profile,
        display=display,
        step_idx=step_idx,
    )

    # ── Skip test generation when all tests already pass ──
    # If the baseline run succeeded (all tests green) and there are no
    # new source files tracked in memory (i.e. this is a pure verification
    # step, not a code-then-test step), skip the TesterAgent entirely and
    # just run the test suite directly.
    if getattr(memory, '_tester_baseline_success', False) and not memory.all_files():
        display.step_info(step_idx, "All tests already passing — running suite directly (skipping generation).")
        log.info(f"Step {step_idx+1}: Baseline passed and no new source files — skipping test generation.")
        ok, out = executor.run_tests(test_cmd, cwd=subproject_cwd)
        return ok, out

    feedback = ""
    last_test_output = ""
    prev_gen_error = None  # Track errors across gen attempts for early exit
    prev_step_test_files: set[str] = set()  # Files from earlier gen attempts of THIS step

    # Pre-fetch KB docs for test generation — use planner-declared titles
    # (plan_step.kb_docs) when available for an exact lookup so that only
    # the docs the planner actually referenced are injected.  Fall back to
    # a broad semantic search only when the planner declared no titles.
    _behavioral_ctx = ""
    if kb_context_builder is not None:
        try:
            _gstore = getattr(kb_context_builder, '_global_store', None)
            if _gstore is not None:
                _declared_kb = getattr(plan_step, 'kb_docs', None) if plan_step else None
                if _declared_kb:
                    _beh_results = _gstore.get_by_titles(_declared_kb)
                else:
                    _step_query = (
                        (plan_step.description if plan_step and getattr(plan_step, 'description', None) else None)
                        or step_text.split("\n")[0].strip()
                    )
                    _beh_results = _gstore.search(
                        query=_step_query,
                        categories=["behavioral", "doc"],
                        top_k=4,
                        api_client=getattr(kb_context_builder, '_api_client', None),
                        language=language,
                    )
                if _beh_results:
                    _beh_parts = []
                    for item in _beh_results:
                        content = getattr(item, "content", "") or getattr(item, "title", "")
                        if content:
                            _beh_parts.append(content)
                    if _beh_parts:
                        _behavioral_ctx = (
                            "\n[KB INSTRUCTIONS]\n"
                            + "\n".join(_beh_parts) + "\n"
                        )
        except Exception:
            pass


    for gen_attempt in range(1, MAX_TEST_GEN_RETRIES + 1):
        display.step_info(step_idx, f"Generating tests (attempt {gen_attempt}/{MAX_TEST_GEN_RETRIES})...")
        kb_ctx = getattr(memory, '_kb_context', '')
        gen_context = ""
        # Inject structured project analysis (gives tester awareness of
        # end-to-end goal, installed packages, import patterns, and
        # assertion guidance — prevents vacuum-based test generation)
        if project_context is not None:
            analysis_block = project_context.format_for_tester()
            if analysis_block:
                gen_context += analysis_block + "\n\n"
        if kb_ctx:
            gen_context += kb_ctx + "\n\n"
        # Inject behavioral instructions for JS/TS test generation
        # only when batch_search didn't already include them (avoids
        # bloating prompt; framework/library docs keep higher priority).
        if (_behavioral_ctx
                and "[BEHAVIORAL INSTRUCTIONS]" not in gen_context
                and "[KB INSTRUCTIONS]" not in gen_context):
            gen_context += _behavioral_ctx + "\n\n"
        gen_context += f"Code:\n{code_summary}"

        # ── Enhancement #9: spec-driven test generation ──
        source_specs = _extract_source_specs(code_summary)
        if source_specs:
            gen_context += (
                "\nSOURCE CODE SPECIFICATIONS (test these behaviors):\n"
                f"{source_specs}\n"
            )

        if feedback:
            gen_context += f"\nFeedback: {feedback}"
        # Add JS/TS environment info to context
        if js_env:
            env_note = f"\nProject environment: {js_env}"
            if js_env.get('is_esm') and test_runner != "vitest":
                env_note += (
                    "\nCRITICAL: This is an ES Module project. "
                    "Tests MUST import from '@jest/globals'.\n"
                )
            gen_context += env_note

        sent_before, recv_before = token_tracker.snapshot()

        # Resolve test_root from project_context so TesterAgent writes
        # tests to the correct directory (e.g. src/__tests__ vs __tests__).
        _test_root = None
        if project_context is not None:
            _test_root = getattr(project_context, "test_root", None) or None

        test_response = tester.process(
            step_text, context=gen_context, language=language,
            env_info=js_env, test_root=_test_root,
            pre_analysis_results=pre_analysis_results,
            test_command=test_cmd,
            # The gate was written against the plan's target, so the tester
            # has to write that exact path rather than a conventional one.
            target_files=list(
                getattr(plan_step, "target_files", None) or []) or None)

        sent_after, recv_after = token_tracker.snapshot()
        sent_delta = sent_after - sent_before
        recv_delta = recv_after - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        explanation = CLIDisplay.extract_explanation(test_response)
        if explanation:
            display.add_llm_log(explanation, source="Tester")

        test_files = executor.parse_code_blocks(test_response)
        if not test_files:
            test_files = executor.parse_code_blocks_fuzzy(test_response)
        if not test_files:
            # Fallback: LLM response might contain echo commands
            from .plan_step import _parse_echo_commands as _parse_echo_resp
            _echo_lines = [l.strip().lstrip('> ').lstrip('$ ')
                           for l in test_response.splitlines()
                           if l.strip().startswith(('>', '$', 'echo '))]
            if _echo_lines:
                test_files = _parse_echo_resp(_echo_lines)
        if not test_files:
            feedback = "No test files found. Use #### [FILE]: format."
            display.step_info(step_idx, "No test files parsed, retrying...")
            continue

        # Strip protected manifest files before they reach memory
        test_files = _strip_protected_files(test_files, intent_spec=intent_spec)

        # Apply KB-driven content fixes (e.g. jest-dom → jest-dom/vitest)
        content_fixes = getattr(memory, '_content_fixes', None)
        test_files = _apply_content_fixes(test_files, content_fixes)

        # Auto-normalise duplicate imports (zero LLM cost, deterministic)
        test_files = _auto_dedup_imports(test_files, display, step_idx)

        # Normalize paths: fix LLM-generated paths that are suffixes of known files
        test_files = _normalize_fix_paths(test_files, memory)

        # Remap test paths to existing locations on disk — if a test file
        # already exists (e.g. src/App.test.jsx), save there instead of
        # the framework default dir (e.g. __tests__/App.test.jsx).
        test_files = _remap_test_to_existing(
            test_files, memory,
            base_dir=getattr(memory, 'base_dir', "."))

        # Fix paths the LLM placed outside the subproject.
        # e.g. __tests__/{subproject}/src/Foo.test.jsx
        #   -> {subproject}/__tests__/src/Foo.test.jsx
        test_files = _fix_subproject_test_paths(test_files, subproject_cwd)

        # Filter: only allow test files (block any source/config files)
        test_files = _filter_test_only_files(test_files, test_files, memory)
        if not test_files:
            feedback = "Generated files were all non-test files. Generate ONLY test files."
            display.step_info(step_idx, "No valid test files after filtering, retrying...")
            continue

        # Strip scaffold/setup files that have no actual test blocks.
        # LLMs sometimes generate "vitestSetup.test.js" or similar files
        # that contain only config/setup but no describe/it/test blocks,
        # causing "No test suite found" errors from the runner.
        test_files = _strip_empty_test_files(test_files)
        if not test_files:
            feedback = ("All generated files were setup/scaffold files with no test blocks. "
                        "Generate ONLY test files containing describe/it/test blocks.")
            display.step_info(step_idx, "No test suites found in generated files, retrying...")
            log.warning(f"Step {step_idx+1}: All generated test files stripped — no test blocks")
            continue

        # Quick offline lint check (free, no LLM cost) — catches syntax
        # errors before writing to disk.  Real test failures are caught by
        # the execution loop below, which provides concrete error output
        # that the coder can act on.
        lint_errors = _quick_offline_lint(test_files)
        if lint_errors:
            feedback = f"Lint/syntax errors found:\n{lint_errors}\nFix these issues."
            display.step_info(step_idx, "Lint errors found, regenerating...")
            log.warning(f"Step {step_idx+1}: Test lint errors: {lint_errors[:300]}")
            continue

        # Show diffs and wait for approval before writing test files
        display.stop_spinner()
        approved = prompt_diff_approval(test_files, auto=auto, display=display,
                                        base_dir=getattr(memory, 'base_dir', "."))
        display.step_info(step_idx, "Processing...")
        if not approved:
            feedback = "User rejected the test changes. Try a different approach."
            display.step_info(step_idx, "Test changes rejected by user, retrying...")
            log.info(f"Step {step_idx+1}: User rejected test diff, retrying.")
            continue

        display.step_info(step_idx, "Processing approved test files...")

        # Prefix test file paths with sub-project root (same as CODE steps)
        if subproject_cwd:
            test_files = _prefix_subproject_paths(
                test_files, subproject_cwd, memory)

        written = executor.write_files(test_files)
        memory.update(test_files)
        display.step_info(step_idx, f"Tests written: {', '.join(written)}")

        # ── Cleanup stale .js test files when .jsx versions exist ──
        # Vite/Rollup cannot parse JSX in .js files.  When the LLM generates
        # .test.jsx files, any leftover .test.js file with the same base name
        # causes persistent "Parse failure: Expression expected" errors.
        _cleanup_stale_js_test_files(test_files, memory, display, step_idx,
                                     subproject_cwd)

        # ── Cleanup ghost test files from earlier gen attempts of THIS step ──
        # If this is a retry (gen_attempt > 1), the previous attempt may have
        # written different files.  Those ghosts get picked up by the runner
        # and fail with stale imports.  Only cleans files from THIS step's
        # earlier attempts — never touches files from other pipeline steps.
        _cleanup_ghost_test_files(test_files, prev_step_test_files,
                                  display, step_idx, subproject_cwd)

        # Track files from this gen attempt so later attempts can clean them
        prev_step_test_files = set(test_files.keys())

        # Safety check: if generated test files import from 'vitest' but
        # the detected test command is Jest, override to vitest.
        # This catches cases where the LLM generates Vitest tests but the
        # framework detection still defaults to Jest.
        if "jest" in test_cmd.lower():
            uses_vitest = any(
                "from 'vitest'" in content or 'from "vitest"' in content
                for content in test_files.values()
            )
            if uses_vitest:
                test_cmd = "npx vitest run"
                test_runner = "vitest"
                log.info(f"Step {step_idx+1}: Overriding test command to "
                         f"'{test_cmd}' — test files import from 'vitest'")
                display.step_info(step_idx, "Detected vitest imports → using vitest runner")

        # ── Pre-install: scan imports and install missing packages ──
        # Avoids wasting a test run + LLM fix cycle on missing packages.
        _preinstall_missing_packages(
            test_files, memory, executor, display, step_idx,
            language, subproject_cwd)

        # ── Per-file test execution and fix loop ──
        # Run each test file individually so failures in one file don't
        # block or pollute the fix loop of another.  The coder sees only
        # the errors and source imports for the single file it's fixing.

        jsx_hint = ""
        if test_runner == "vitest" and language in ("javascript", "typescript"):
            jsx_ext = ".jsx" if language == "javascript" else ".tsx"
            js_ext = ".js" if language == "javascript" else ".ts"
            jsx_hint = (
                f"IMPORTANT: Vite/Rollup CANNOT parse JSX in {js_ext} files. "
                f"If a test file contains JSX (e.g. <Component />, render(<App />)), "
                f"it MUST use the .test{jsx_ext} extension, NOT .test{js_ext}. "
                f"If you need to rename a file from .test{js_ext} to .test{jsx_ext}, "
                f"output the corrected content under the new .test{jsx_ext} path. "
            )

        failed_files: list[str] = []

        # Filter out non-test files before the per-file loop.
        # The LLM auto-fix can include config files (pytest.ini,
        # conftest.py, __init__.py) alongside test files.  Running
        # `python -m pytest pytest.ini` fails with "not found".
        _TEST_FILE_RE = re.compile(r'(?:^|/)test[_.].*\.py$|\.test\.[jt]sx?$')
        _runnable_test_files = {
            fp: content for fp, content in test_files.items()
            if _TEST_FILE_RE.search(os.path.basename(fp))
        }
        if not _runnable_test_files and test_files:
            # Fallback: if no files match the pattern, run all .py files
            _runnable_test_files = {
                fp: content for fp, content in test_files.items()
                if fp.endswith('.py') and not fp.endswith('__init__.py')
            }
        file_count = len(_runnable_test_files)

        for file_idx, (test_path, test_content) in enumerate(
                list(_runnable_test_files.items()), 1):
            f_basename = os.path.basename(test_path)
            display.step_info(
                step_idx,
                f"Testing {f_basename} ({file_idx}/{file_count})...")

            # Build a single-file test command
            single_cmd = _build_scoped_test_cmd(
                test_cmd, {test_path: test_content}, subproject_cwd)
            log.info(f"Step {step_idx+1}: Running: {single_cmd}"
                     + (f" in {subproject_cwd}" if subproject_cwd else ""))

            success, output = executor.run_tests(
                single_cmd, cwd=subproject_cwd)
            log.info(f"Step {step_idx+1}: [{f_basename}] output:\n"
                     f"{output or '(no output)'}")
            last_test_output = output

            if success:
                _t_passed, _t_total, _ = _parse_test_counts(output)
                display.step_info(
                    step_idx,
                    f"{f_basename} passed ✔  {_t_passed}/{_t_total}"
                    f" ({file_idx}/{file_count})")
                display.record_test_result(test_path, passed=_t_passed,
                                           total=_t_total, failures=[])
                continue

            # Record initial failure immediately so TEST RESULTS panel appears
            _t_passed0, _t_total0, _t_fails0 = _parse_test_counts(output)
            display.record_test_result(test_path, passed=_t_passed0,
                                       total=_t_total0, failures=_t_fails0)

            # ── System / env checks (shared across files, run once) ──
            from .pipeline import _detect_system_level_failure
            sys_issue = _detect_system_level_failure(output)
            if sys_issue:
                msg = (f"System dependency missing: {sys_issue}. "
                       f"Cannot fix by editing code.")
                display.step_info(step_idx, msg)
                log.error(f"Step {step_idx+1}: {msg}")
                return False, msg

            # Auto-install missing packages (applies to all subsequent files)
            missing_pkgs = executor.detect_missing_packages(output)
            if missing_pkgs:
                install_tool = ("npm install --save-dev"
                                if language in ("javascript", "typescript")
                                else "pip install")
                display.step_info(
                    step_idx,
                    f"Installing missing packages: {', '.join(missing_pkgs)}")
                install_ok, _ = executor.install_packages(
                    missing_pkgs, tool=install_tool, cwd=subproject_cwd)
                if install_ok:
                    # Re-run this single file after install
                    success, output = executor.run_tests(
                        single_cmd, cwd=subproject_cwd)
                    last_test_output = output
                    if success:
                        display.step_info(
                            step_idx,
                            f"{f_basename} passed after install ✔ "
                            f"({file_idx}/{file_count})")
                        continue

            # ── Per-file fix loop ──
            file_fixed = False
            prev_output = output
            # Track error signatures across attempts for oscillation
            # detection.  An "error signature" is the first meaningful
            # error line (import failure, assertion, etc.) — if it
            # recurs across two non-consecutive attempts, we're stuck.
            _seen_error_sigs: list[str] = []
            # True once a CONFIG_BUG attempt produced no in-scope change,
            # which routes a repeat CONFIG_BUG verdict to the source path.
            _config_fix_was_empty = False
            # Snapshot before attempting fixes so we can restore if a
            # bad fix corrupts the source files across attempts.
            _test_fix_snap = memory.snapshot()
            for fix_attempt in range(1, MAX_STEP_RETRIES + 1):
                # Restore on retry to prevent compounding bad fixes
                if fix_attempt > 1:
                    memory.restore(_test_fix_snap, executor=executor)
                    log.info(
                        "Step %d: Restored file snapshot before fix "
                        "attempt %d for %s",
                        step_idx + 1, fix_attempt, f_basename)

                display.step_info(
                    step_idx,
                    f"Fixing {f_basename} (attempt {fix_attempt}/{MAX_STEP_RETRIES})...")

                # Get the latest content from memory (may have been updated)
                current_content = (memory.all_files().get(test_path)
                                   or test_files.get(test_path, ""))

                # Focused source context: prefer plan-declared imports,
                # fall back to parsing test file imports from memory
                from .memory import get_plan_context_files as _get_fix_plan_ctx
                _fix_plan_ctx = _get_fix_plan_ctx()
                if _fix_plan_ctx and plan_step is not None:
                    single_imports = {
                        fp: cnt for fp, cnt in _fix_plan_ctx.items()
                        if fp != test_path
                    }
                else:
                    single_imports = _extract_imported_sources(
                        {test_path: current_content}, memory)
                source_ctx = ""
                for fp, cnt in single_imports.items():
                    source_ctx += (
                        f"#### [FILE]: {fp}\n"
                        f"```{lang_tag}\n{cnt}\n```\n\n")

                # Build error detail from this file's output only
                error_detail = _build_batch_error_summary(output) if output else ""
                if not error_detail:
                    error_detail = (
                        f"(command `{single_cmd}` produced no output)")

                # Triage: test bug or source bug?
                bug_origin = _triage_test_failure(
                    error_detail, source_ctx,
                    f"{test_path}:\n{current_content[:3000]}\n",
                    coder.llm_client, display, step_idx,
                    language=language)
                _routed = effective_bug_origin(bug_origin,
                                               _config_fix_was_empty)
                if _routed != bug_origin:
                    log.info(
                        "Step %d: config fix changed nothing last attempt "
                        "— treating this failure as a %s instead",
                        step_idx + 1, _routed)
                    bug_origin = _routed
                is_source_bug = (bug_origin == "SOURCE_BUG")
                is_config_bug = (bug_origin == "CONFIG_BUG")

                # KB error-fix lookup
                kb_fix_context = ""
                if kb_context_builder is not None:
                    try:
                        kb_ctx = kb_context_builder.build_context(
                            task_description=step_text,
                            error_output=output,
                            max_tokens=2000,
                            step_type="TEST",
                        )
                        if kb_ctx.error_fixes:
                            kb_fix_context = (
                                kb_context_builder
                                .format_context_for_prompt(kb_ctx))
                    except Exception:
                        pass

                # Build focused fix context
                # Prepend task context so the coder never "fixes" a test by
                # removing the component under test or deleting assertions.
                _task_context_header = f"TASK CONTEXT:\n{task}\n\n"
                if pre_analysis_results:
                    _task_context_header += f"{pre_analysis_results}\n\n"

                if is_config_bug:
                    # ── CONFIG_BUG: fix the test environment config ──────────
                    subproject = _detect_subproject_root(memory)

                    # Use pluggable LanguageBackend for CONFIG_BUG handling
                    from ..language_backend import get_backend
                    _lang_backend = get_backend(language)
                    display.step_info(
                        step_idx,
                        f"Config bug — fixing {_lang_backend.display_name} "
                        f"test environment config...")
                    _cfg_candidates = _lang_backend.get_config_candidates()
                    _cfg_lang_tag = _lang_backend.get_config_lang_tag()
                    _no_cfg_msg = _lang_backend.get_config_no_found_msg()
                    _fix_detail = _lang_backend.get_config_fix_detail()
                    _fix_runner = f"Test runner: {test_cmd}\n"
                    _fix_prompt_body = _lang_backend.get_config_fix_prompt(test_cmd)

                    _cfg_paths = []
                    for _cn in _cfg_candidates:
                        for _pfx in ([f"{subproject}/{_cn}"] if subproject else []) + [_cn]:
                            _cfg_paths.append(_pfx)

                    _cfg_file = ""
                    _cfg_path = ""
                    for _cp in _cfg_paths:
                        _cfg_file = memory.get(_cp) or ""
                        if not _cfg_file:
                            try:
                                with open(_cp, "r", encoding="utf-8", errors="replace") as _cf:
                                    _cfg_file = _cf.read()
                            except OSError:
                                pass
                        if _cfg_file:
                            _cfg_path = _cp
                            break

                    _cfg_ctx = (
                        f"#### [FILE]: {_cfg_path}\n```{_cfg_lang_tag}\n{_cfg_file}\n```\n\n"
                        if _cfg_file and _cfg_path
                        else _no_cfg_msg
                    )
                    fix_ctx = (
                        f"{_task_context_header}"
                        f"ALL tests are failing with:\n{error_detail}\n\n"
                        f"{_fix_detail}"
                        f"Current config:\n{_cfg_ctx}"
                        f"{_fix_runner}"
                    )
                    fix_prompt = _fix_prompt_body
                    sent_before, recv_before = token_tracker.snapshot()
                    fix_response = coder.process(
                        fix_prompt, context=fix_ctx, language=language)
                    sent_after, recv_after = token_tracker.snapshot()
                    display.step_tokens(step_idx,
                                        sent_after - sent_before,
                                        recv_after - recv_before)
                    fix_files = executor.parse_code_blocks(fix_response)
                    if not fix_files:
                        fix_files = executor.parse_code_blocks_fuzzy(fix_response)
                    if fix_files:
                        if subproject:
                            fix_files = _prefix_subproject_paths(fix_files, subproject, memory)
                        # A CONFIG_BUG is by definition "the test framework or
                        # environment is misconfigured", so its fix belongs in
                        # config files. Nothing stopped the model returning
                        # source modules instead, and nothing stopped this from
                        # writing them: observed five times in one step, each
                        # "CONFIG_BUG fix applied" rewriting map.py, player.py,
                        # ghost.py, game.py and main.py at once. The suite went
                        # from 1 failure in 61 tests to 4 failures and 3 errors
                        # in 64 — a misfiled triage rewrote a nearly-passing
                        # game. Keep the fix inside its own scope; if that
                        # leaves nothing, the triage was wrong and the next
                        # attempt re-triages rather than doing damage.
                        _allowed = {c.lower() for c in _cfg_candidates}
                        _kept = {p: c for p, c in fix_files.items()
                                 if os.path.basename(p).lower() in _allowed}
                        _dropped = [p for p in fix_files if p not in _kept]
                        if _dropped:
                            log.warning(
                                "Step %d: CONFIG_BUG fix tried to rewrite "
                                "non-config file(s) %s — dropped; a config "
                                "fix may only touch %s",
                                step_idx + 1, _dropped,
                                sorted(_allowed)[:6])
                        fix_files = _kept
                    # Remember whether this attempt actually changed any
                    # config, so a repeat CONFIG_BUG verdict can be routed
                    # to the source path instead of looping.
                    _config_fix_was_empty = not fix_files
                    if fix_files:
                        executor.write_files(fix_files)
                        memory.update(fix_files)
                        log.info("Step %d: CONFIG_BUG fix applied: %s",
                                 step_idx + 1, list(fix_files.keys()))
                    continue  # re-run the test file after config fix

                elif is_source_bug:
                    display.step_info(
                        step_idx,
                        f"Bug in SOURCE for {f_basename}, fixing source...")
                    fix_ctx = (
                        f"{_task_context_header}"
                        f"Test command: `{single_cmd}`\n\n"
                        f"{error_detail}\n\n"
                        f"Source files under test:\n{source_ctx}\n\n"
                        f"Test file (for reference):\n"
                        f"#### [FILE]: {test_path}\n"
                        f"```{lang_tag}\n{current_content}\n```\n\n")
                    fix_prompt = (
                        "The test has revealed a BUG IN THE SOURCE CODE.\n"
                        "Fix the SOURCE files to make the test pass.\n"
                        "Do NOT modify the test file.\n"
                        "CRITICAL: Do NOT remove, comment out, or skip the tested component or feature. "
                        "The task requires this functionality to exist and work correctly.\n"
                        f"{jsx_hint}"
                        "Return corrected source file(s) using "
                        "#### [FILE]: format.")
                else:
                    fix_ctx = (
                        f"{_task_context_header}"
                        f"Test command: `{single_cmd}`\n\n"
                        f"ERRORS for {test_path}:\n{error_detail}\n\n"
                        f"Test file to fix:\n"
                        f"#### [FILE]: {test_path}\n"
                        f"```{lang_tag}\n{current_content}\n```\n\n")
                    if source_ctx:
                        fix_ctx += (
                            f"Source files imported by this test:\n"
                            f"{source_ctx}\n")
                    fix_prompt = (
                        f"Fix ONLY the test file '{test_path}'.\n"
                        "Do NOT output any other test files or source files.\n"
                        "Focus on fixing ONLY the failing tests.\n"
                        "CRITICAL: Do NOT remove, comment out, or skip tests for the component/feature "
                        "mentioned in the task. If a test fails due to an integration issue (e.g. anime.js, "
                        "a missing mock, or a changed API), FIX the test to correctly test the feature — "
                        "do NOT delete the test or the component import.\n"
                        f"{jsx_hint}"
                        "Do NOT modify source files, package.json, or "
                        "config files. Do NOT add new dependencies.\n"
                        "Return the corrected file using #### [FILE]: format.")

                if kb_fix_context:
                    fix_ctx += f"\n\n{kb_fix_context}"

                # ── Try surgical chunk-level fix first ──
                # For both TEST_BUG and SOURCE_BUG, attempt to edit only the
                # failing chunk(s) rather than rewriting the whole file.
                # Falls back to full-file if ChunkEditor can't parse/apply.
                _chunk_step_desc = fix_prompt + "\n\n" + error_detail
                if is_source_bug:
                    # Try chunk fix on each source file individually, each
                    # seeing the others' signatures so it can tell whether
                    # the fix is even its to make.
                    _chunk_fix_results: dict[str, str] = {}
                    for _sfp, _sfc in single_imports.items():
                        _res = _chunk_fix_file(
                            _sfp, _sfc, _chunk_step_desc,
                            coder.llm_client, language, memory,
                            display, step_idx,
                            sibling_sources=single_imports)
                        if _res is CHUNK_FIX_WANTS_FULL_FILE:
                            # The model wants to work at file level. Every
                            # remaining per-file call would be answered the
                            # same way, and the fallback below already sends
                            # all the sources together — stop paying for
                            # them and go there now.
                            _chunk_fix_results.clear()
                            break
                        if _res:
                            _chunk_fix_results.update(_res)
                    fix_files = _chunk_fix_results or None
                else:
                    fix_files = _chunk_fix_file(
                        test_path, current_content, _chunk_step_desc,
                        coder.llm_client, language, memory,
                        display, step_idx)

                # Fall back to full-file if chunk fix didn't produce results
                if not fix_files:
                    # Ensure _task_briefing is in the fallback context
                    _briefing = getattr(memory, '_task_briefing', '')
                    if _briefing and _briefing not in fix_ctx:
                        fix_ctx = (
                            "TASK BRIEFING (respect Preserve and Key constraint):\n"
                            f"{_briefing}\n\n"
                        ) + fix_ctx
                    sent_before, recv_before = token_tracker.snapshot()
                    fix_response = coder.process(
                        fix_prompt, context=fix_ctx, language=language)
                    sent_after, recv_after = token_tracker.snapshot()
                    display.step_tokens(
                        step_idx,
                        sent_after - sent_before,
                        recv_after - recv_before)

                    explanation = CLIDisplay.extract_explanation(fix_response)
                    if explanation:
                        display.add_llm_log(explanation, source="Coder")

                    fix_files = executor.parse_code_blocks(fix_response)
                    if not fix_files:
                        fix_files = executor.parse_code_blocks_fuzzy(fix_response)
                if fix_files:
                    fix_files = _strip_protected_files(fix_files, intent_spec=intent_spec)
                    fix_files = _normalize_fix_paths(fix_files, memory)
                    fix_files = _remap_test_to_existing(
                        fix_files, memory,
                        base_dir=getattr(memory, 'base_dir', "."))
                    fix_files = _fix_subproject_test_paths(fix_files, subproject_cwd)
                    if not is_source_bug:
                        fix_files = _filter_test_only_files(
                            fix_files, test_files, memory)
                    else:
                        # ── Source-file protection during test diagnosis ──
                        # When triage says SOURCE_BUG, the LLM is allowed to
                        # edit source files.  But we must guard against two
                        # failure modes:
                        #   1) Full-file replacement — the LLM rewrites the
                        #      entire source instead of a surgical fix.
                        #   2) Excessive diff — the fix changes >40% of lines,
                        #      which almost always means the LLM is rewriting
                        #      the component to match brittle test assertions.
                        _guarded: dict[str, str] = {}
                        for _fp, _fc in fix_files.items():
                            _basename = os.path.basename(_fp)
                            _is_test = any(p in _fp for p in (
                                '__tests__', '/tests/', '/test/', '.test.', '.spec.', 'test_'))
                            if _is_test:
                                _guarded[_fp] = _fc
                                continue
                            # Check 1: block full-file replacement of source files
                            _orig = memory.get(_fp)
                            if _orig is None:
                                try:
                                    with open(_fp, 'r', encoding='utf-8',
                                              errors='replace') as _f:
                                        _orig = _f.read()
                                except OSError:
                                    _orig = None
                            if _orig is not None:
                                _orig_lines = _orig.splitlines()
                                _new_lines = _fc.splitlines()
                                _orig_len = len(_orig_lines)
                                _new_len = len(_new_lines)
                                # Full-file replacement: new content shares <30%
                                # of original lines — reject it.
                                if _orig_len >= 10:
                                    _shared = sum(1 for ln in _new_lines
                                                  if ln in set(_orig_lines))
                                    _shared_ratio = _shared / max(_orig_len, 1)
                                    if _shared_ratio < 0.30:
                                        log.warning(
                                            "Step %d: Blocked full-file "
                                            "replacement of source %s during "
                                            "test diagnosis (shared %.0f%% of "
                                            "%d lines) — skipping",
                                            step_idx + 1, _basename,
                                            _shared_ratio * 100, _orig_len)
                                        continue
                                # Check 2: diff-size circuit breaker (>40% changed)
                                if _orig_len >= 5:
                                    import difflib as _dl
                                    _sm = _dl.SequenceMatcher(
                                        None, _orig_lines, _new_lines)
                                    _changed = sum(
                                        max(j2 - j1, i2 - i1)
                                        for tag, i1, i2, j1, j2
                                        in _sm.get_opcodes()
                                        if tag != 'equal')
                                    _change_ratio = _changed / max(_orig_len, 1)
                                    if _change_ratio > 0.40:
                                        log.warning(
                                            "Step %d: Rejected source fix for "
                                            "%s — changes %.0f%% of lines "
                                            "(threshold 40%%) during test "
                                            "diagnosis",
                                            step_idx + 1, _basename,
                                            _change_ratio * 100)
                                        continue
                            _guarded[_fp] = _fc
                        fix_files = _guarded
                    if fix_files:
                        # Syntax guard: reject Python files with syntax errors
                        _syntax_ok: dict[str, str] = {}
                        for _fp, _fc in fix_files.items():
                            _ext = os.path.splitext(_fp)[1].lower()
                            if _ext in ('.py', '.pyw'):
                                from ..py_syntax import check_python_syntax
                                _se = check_python_syntax(_fc, _fp)
                                if _se:
                                    log.warning(
                                        "Step %d: Syntax error in fix for "
                                        "%s: %s — skipping",
                                        step_idx + 1, _fp, _se)
                                else:
                                    _syntax_ok[_fp] = _fc
                            else:
                                _syntax_ok[_fp] = _fc
                        fix_files = _syntax_ok
                    if fix_files:
                        show_diffs(fix_files, log_only=True)
                        executor.write_files(fix_files)
                        memory.update(fix_files)
                        _cleanup_stale_js_test_files(
                            fix_files, memory, display,
                            step_idx, subproject_cwd)
                        # Update test_files if the fix changed the test
                        test_files.update(fix_files)
                        # Rebuild single command in case path changed
                        if test_path in fix_files:
                            single_cmd = _build_scoped_test_cmd(
                                test_cmd,
                                {test_path: fix_files[test_path]},
                                subproject_cwd)

                # Re-run the single file after fix
                success, output = executor.run_tests(
                    single_cmd, cwd=subproject_cwd)
                last_test_output = output

                if success:
                    _t_passed, _t_total, _ = _parse_test_counts(output)
                    display.step_info(
                        step_idx,
                        f"{f_basename} passed after fix ✔  {_t_passed}/{_t_total}"
                        f" ({file_idx}/{file_count})")
                    display.record_test_result(test_path, passed=_t_passed,
                                               total=_t_total, failures=[])
                    file_fixed = True
                    break

                # ── Stuck / oscillation detection ──
                # 1) Identical output: bail immediately (no point retrying).
                if output == prev_output:
                    log.warning(
                        f"Step {step_idx+1}: [{f_basename}] identical "
                        f"output after fix attempt {fix_attempt}, stopping.")
                    display.step_info(
                        step_idx,
                        f"{f_basename}: same error after fix — skipping.")
                    break

                # 2) Error-class oscillation: extract a short signature
                #    from the error output and check if we've seen it
                #    before.  Two occurrences of the same signature means
                #    the fix loop is going in circles.
                _err_sig = _extract_error_signature(output)
                if _err_sig and _err_sig in _seen_error_sigs:
                    log.warning(
                        f"Step {step_idx+1}: [{f_basename}] error "
                        f"signature recurred after fix attempt "
                        f"{fix_attempt}: {_err_sig!r} — stopping.")
                    display.step_info(
                        step_idx,
                        f"{f_basename}: recurring error pattern — "
                        f"skipping.")
                    break
                if _err_sig:
                    _seen_error_sigs.append(_err_sig)

                prev_output = output

            if not file_fixed:
                failed_files.append(test_path)
                log.warning(
                    f"Step {step_idx+1}: [{f_basename}] still failing "
                    f"after {MAX_STEP_RETRIES} fixes.")
                _t_passed, _t_total, _fail_names = _parse_test_counts(output)
                display.record_test_result(test_path, passed=_t_passed,
                                           total=_t_total, failures=_fail_names)

        # ── Summary ──
        if not failed_files:
            display.step_info(
                step_idx, f"All {file_count} test files passed ✔")
            return True, ""
        else:
            passed = file_count - len(failed_files)
            failed_names = [os.path.basename(f) for f in failed_files]
            msg = (f"{passed}/{file_count} test files passed. "
                   f"Failed: {', '.join(failed_names)}")
            display.step_info(step_idx, msg)
            log.error(f"Step {step_idx+1}: {msg}")
            return False, (
                f"Tests partially failing: {msg}\n"
                f"Last output:\n{last_test_output}")

    log.error(f"Step {step_idx+1}: Could not generate valid tests after {MAX_TEST_GEN_RETRIES} attempts.")
    return False, f"Could not generate valid tests after {MAX_TEST_GEN_RETRIES} attempts.\nLast feedback:\n{feedback}"


# ---------------------------------------------------------------------------
# Diff-aware editing (Phase 5)
# ---------------------------------------------------------------------------

# Returned by _chunk_fix_file when the model answered in full-file format
# — distinct from None ("nothing to change here"), because it means every
# further per-file chunk call would be answered the same way.
CHUNK_FIX_WANTS_FULL_FILE: dict[str, str] = {}


def _chunk_fix_file(
    file_path: str,
    content: str,
    step_description: str,
    llm_client,
    language: str | None,
    memory: "FileMemory",
    display: "CLIDisplay",
    step_idx: int,
    sibling_sources: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """Try to fix a single file using chunk-level editing.

    Sends only the chunks relevant to the error/fix description to the LLM,
    then splices the result back — avoiding unintended changes to unrelated code.

    *sibling_sources* are the other source files under test. They are
    included as signatures only — not editable here, but visible. Without
    them the call is blind: shown player.py alone and asked to fix
    "Ghost 2 spawn at (17, 13) is not walkable", a model spent its whole
    response explaining that it could not, because the spawn table lives
    in a file it was not shown. Every source file got its own such call,
    so a single failing assertion cost one blind call per file before the
    full-file fallback did the real work.

    Returns ``{file_path: new_content}`` on success, or ``None`` to signal
    the caller should fall back to full-file editing.
    """
    try:
        from ..editing.chunk_editor import ChunkEditor
    except ImportError:
        return None

    chunk_editor = ChunkEditor()
    chunks = chunk_editor.chunk_file(file_path, content)
    if not chunks:
        return None

    targets = chunk_editor.identify_target_chunks(chunks, step_description)
    if not targets:
        # No targeted chunks found — send all non-import chunks but warn
        targets = [c.chunk_id for c in chunks if c.chunk_type != "imports"]
        if not targets:
            return None

    formatted = chunk_editor.format_chunks_for_prompt(chunks, targets)

    _task_briefing = getattr(memory, '_task_briefing', '')
    briefing_prefix = (
        "TASK BRIEFING (respect Preserve and Key constraint — fix ONLY what the error requires):\n"
        f"{_task_briefing}\n\n"
    ) if _task_briefing else ""

    slim_context = ""
    if sibling_sources:
        from .memory import _extract_file_skeleton
        _skeletons = []
        for _sp, _sc in sibling_sources.items():
            if _sp == file_path:
                continue
            _skeletons.append(_extract_file_skeleton(_sc, _sp) or f"- {_sp}")
        if _skeletons:
            slim_context = (
                "Other source files under test (signatures only — you "
                "CANNOT edit them in this response; if the fix belongs in "
                "one of them, say so briefly and change nothing):\n"
                + "\n".join(_skeletons) + "\n")

    prompt = briefing_prefix + _build_chunk_prompt(
        step_description, formatted, slim_context, language=language)

    display.step_info(
        step_idx,
        f"[ChunkFix] {os.path.basename(file_path)}: {len(targets)} target chunk(s)...")

    sent_before, recv_before = token_tracker.snapshot()
    llm_response = llm_client.generate_response(prompt)
    sent_after, recv_after = token_tracker.snapshot()
    display.step_tokens(step_idx, sent_after - sent_before, recv_after - recv_before)

    edits = chunk_editor.parse_chunk_response(llm_response)
    if edits is None:
        # parse_chunk_response collapses two different answers into None:
        # "here is the whole file" and "I parsed no edits at all" (which
        # includes prose refusals). Only the former means the model wants
        # to work at file level; asking the marker directly keeps the log
        # honest and stops a file with genuinely nothing to change from
        # cancelling the sweep over its siblings.
        from ..editing.chunk_editor import _FULL_FILE_MARKER
        if _FULL_FILE_MARKER.search(llm_response):
            log.info("[ChunkFix] LLM used full-file format for %s, "
                     "falling back", file_path)
            return CHUNK_FIX_WANTS_FULL_FILE
        log.info("[ChunkFix] No chunk edits parsed for %s", file_path)
        return None
    if not edits:
        return None

    file_chunks = [c for c in chunks if c.file_path == file_path]
    try:
        new_content = chunk_editor.apply_chunk_edits(
            content, edits, known_chunks=file_chunks)
    except Exception as exc:
        log.warning("[ChunkFix] Failed to apply edits to %s: %s", file_path, exc)
        return None

    # Destructive edit guard: if output shrinks >75%, fall back
    if content and len(content) > 200:
        ratio = len(new_content) / len(content)
        if ratio < 0.25:
            log.warning(
                "[ChunkFix] Destructive edit for %s (%.0f%% reduction), falling back",
                file_path, (1 - ratio) * 100)
            return None

    return {file_path: new_content}


def _try_diff_edit(
    *,
    step_text: str,
    coder: CoderAgent,
    task: str,
    memory: FileMemory,
    display: CLIDisplay,
    step_idx: int,
    language: str | None,
    cfg: Config,
    code_graph,
    project_profile=None,
    kb_context_builder=None,
) -> tuple[bool, str] | None:
    """Attempt a diff-aware edit.  Returns ``(success, error_info)`` on
    success, or ``None`` to signal the caller should fall back to the
    full-file flow.
    """
    import re as _re

    try:
        from ..editing.scope_resolver import ScopeResolver
        from ..editing.context_slicer import ContextSlicer
        from ..editing.diff_parser import DiffParser
        from ..editing.patch_applier import PatchApplier
        from ..editing.metrics import log_edit_metric
    except ImportError as exc:
        log.debug("[DiffEdit] editing module not available: %s", exc)
        return None

    # Identify target file from step text or memory (KB fallback if needed)
    target_file = _detect_target_file(step_text, memory,
                                      kb_context_builder=kb_context_builder)
    if not target_file:
        log.debug("[DiffEdit] No target file identified from step text")
        return None

    # Check the file actually exists on disk
    if not os.path.isfile(target_file):
        log.debug("[DiffEdit] Target file does not exist: %s", target_file)
        return None

    display.step_info(step_idx, f"[DiffEdit] Resolving scope for {os.path.basename(target_file)}...")

    # 1. Resolve scope
    resolver = ScopeResolver(code_graph)
    scope = resolver.resolve(step_text, target_file, code_graph)

    # Bypass slicing and confidence check for non-AST files
    non_ast_exts = {".html", ".css", ".scss", ".json", ".yaml", ".yml", ".md", ".txt"}
    ext = os.path.splitext(target_file)[1].lower()
    is_non_ast = ext in non_ast_exts

    if is_non_ast:
        scope.confidence = 1.0
        try:
            with open(target_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            content = memory.get(target_file, "")

        line_count = len(content.splitlines())
        formatted = f"=== FILE: {target_file} ({line_count} lines total) ===\n"
        formatted += f"Language: {ext.lstrip('.')}\n\n"
        formatted += content
        if not content.endswith("\n") and content:
            formatted += "\n"
        formatted += f"=== END FILE ===\n"
        
        full_file_lines = line_count
        sliced_lines = line_count
    else:
        min_conf = getattr(cfg, "EDITING_MIN_CONFIDENCE", 0.60)
        if scope.confidence < min_conf:
            log.warning(
                "[DiffEdit] Confidence %.2f < %.2f for %s, falling back",
                scope.confidence, min_conf, target_file,
            )
            _log_fallback_metric(cfg, target_file, step_text, scope, "low_confidence")
            return None

        # 2. Slice files
        ctx_lines = getattr(cfg, "EDITING_CONTEXT_LINES", 5)
        slicer = ContextSlicer()

        scopes_map: dict = {}
        for af in scope.affected_files:
            scopes_map[af] = scope

        slices = slicer.slice_files(scopes_map)
        formatted = slicer.format_for_prompt(slices)

        # Compute token stats
        full_file_lines = 0
        sliced_lines = 0
        for fslice in slices.values():
            full_file_lines += fslice.total_lines
            sliced_lines += sum(b.line_end - b.line_start + 1 for b in fslice.slices)
            if fslice.imports_block:
                sliced_lines += fslice.imports_block.count("\n") + 1

    # Prepend project orientation + knowledge context to sliced context
    prefix_parts = []
    if project_profile is not None:
        try:
            prefix_parts.append(project_profile.format_for_prompt())
        except Exception:
            pass
    kb_ctx = getattr(memory, '_kb_context', '')
    if kb_ctx:
        prefix_parts.append(kb_ctx)
    _diff_briefing = getattr(memory, '_task_briefing', '')
    if _diff_briefing:
        prefix_parts.append(
            "TASK BRIEFING (respect Preserve and Key constraint):\n"
            + _diff_briefing
        )
    if prefix_parts:
        formatted = "\n\n".join(prefix_parts) + "\n\n" + formatted

    display.step_info(step_idx, f"[DiffEdit] Sending {sliced_lines}/{full_file_lines} lines to LLM...")

    # 3. Build diff prompt and call LLM
    diff_prompt = _build_diff_prompt(step_text, formatted)
    sent_before, recv_before = token_tracker.snapshot()

    llm_response = coder.llm_client.generate_response(diff_prompt)

    sent_after, recv_after = token_tracker.snapshot()
    sent_delta = sent_after - sent_before
    recv_delta = recv_after - recv_before
    display.step_tokens(step_idx, sent_delta, recv_delta)

    # 4. Parse diff
    parser = DiffParser()
    parsed = parser.parse(llm_response)

    if parsed is None:
        log.warning("[DiffEdit] LLM did not return valid diff format, falling back")
        _log_fallback_metric(cfg, target_file, step_text, scope, "parse_failed")
        return None

    # Validate hunks against actual file content
    file_contents: dict[str, list[str]] = {}
    for patch in parsed.file_patches:
        try:
            with open(patch.file_path, "r", encoding="utf-8", errors="replace") as f:
                file_contents[patch.file_path] = f.readlines()
        except OSError:
            pass

    parsed = parser.validate(parsed, file_contents)
    if parsed is None or not parsed.parse_successful:
        log.warning("[DiffEdit] >50%% hunks invalid, falling back")
        _log_fallback_metric(cfg, target_file, step_text, scope, "validation_failed")
        return None

    # 5. Apply patches
    fuzzy_window = getattr(cfg, "EDITING_FUZZY_MATCH_WINDOW", 3)
    validate_syntax = getattr(cfg, "EDITING_VALIDATE_SYNTAX", True)
    fallback_syntax = getattr(cfg, "EDITING_FALLBACK_ON_SYNTAX_ERROR", True)

    applier = PatchApplier(
        fuzzy_match_window=fuzzy_window,
        validate_syntax=validate_syntax,
        fallback_on_syntax_error=fallback_syntax,
    )
    result = applier.apply(parsed)

    if not result.success:
        log.warning("[DiffEdit] Patch apply failed: %s", result.error)
        _log_fallback_metric(cfg, target_file, step_text, scope, f"apply_failed: {result.error}")
        return None

    # Update memory with modified files
    for fpath in result.files_modified:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                memory.update({fpath: f.read()})
        except OSError:
            pass

    display.step_info(
        step_idx,
        f"[DiffEdit] Applied {result.hunks_applied} hunk(s) to "
        f"{len(result.files_modified)} file(s)"
    )

    # Log metrics
    if getattr(cfg, "EDITING_TRACK_METRICS", True):
        reduction = (
            round((1 - sliced_lines / full_file_lines) * 100, 1)
            if full_file_lines > 0 else 0
        )
        log_edit_metric({
            "file": target_file,
            "task_length_chars": len(step_text),
            "resolution_method": scope.resolution_method,
            "confidence": round(scope.confidence, 2),
            "full_file_lines": full_file_lines,
            "sliced_lines_sent": sliced_lines,
            "token_reduction_pct": reduction,
            "hunks_applied": result.hunks_applied,
            "hunks_failed": result.hunks_failed,
            "fallback_used": False,
            "syntax_valid": result.syntax_valid,
            "affected_files_count": len(scope.affected_files),
        })

    return True, ""


def _resolve_from_kb(step_text: str, memory: FileMemory, kb_context_builder) -> list[str]:
    """Use KB semantic search to find project files not yet tracked in memory.

    Extracts filename hints from *step_text*, asks the KB for the most
    relevant project files, cross-references by basename, reads matching
    files from disk, and loads them into *memory* so subsequent context
    builders can include them.

    Returns the list of file paths that were resolved and loaded.
    """
    import re as _re

    # Extract bare filenames mentioned in the step text (e.g. "main.jsx")
    filename_hints: set[str] = set()
    for m in _re.finditer(r'[\w/\\]+\.\w{1,5}', step_text):
        filename_hints.add(os.path.basename(m.group().replace("\\", "/")))

    if not filename_hints:
        return []

    try:
        kb_files = kb_context_builder.get_relevant_files(step_text, max_files=10)
    except Exception as exc:
        log.debug("[TargetResolve] KB get_relevant_files failed: %s", exc)
        return []

    to_load: dict[str, str] = {}
    existing = memory.all_files()
    for fpath in kb_files:
        if os.path.basename(fpath) in filename_hints and fpath not in existing:
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as _f:
                        to_load[fpath] = _f.read()
                except Exception:
                    pass

    if to_load:
        memory.update(to_load)
        log.debug("[TargetResolve] KB resolved %d file(s) into memory: %s",
                  len(to_load), list(to_load.keys()))

    return list(to_load.keys())


def _detect_target_file(step_text: str, memory: FileMemory,
                        kb_context_builder=None) -> str | None:
    """Try to identify the target file for editing from the step text."""
    import re as _re

    # Look for explicit file path mentions in the step text
    known_files = list(memory.all_files().keys())

    # Direct mention of a known file path
    for fpath in known_files:
        if fpath in step_text:
            return fpath

    # Check for basename mention
    for fpath in known_files:
        basename = os.path.basename(fpath)
        if basename and basename in step_text and basename.count(".") > 0:
            return fpath

    # Look for file-path-like patterns in the step text
    path_pattern = _re.compile(r'[\w/\\]+\.\w{1,5}')
    for m in path_pattern.finditer(step_text):
        candidate = m.group().replace("\\", "/")
        for fpath in known_files:
            if fpath.endswith(candidate) or candidate.endswith(fpath):
                return fpath

    # If only one file in memory, use it
    if len(known_files) == 1:
        return known_files[0]

    # Fallback: ask KB to find the file on disk and load it into memory
    if kb_context_builder is not None:
        resolved = _resolve_from_kb(step_text, memory, kb_context_builder)
        if resolved:
            return resolved[0]

    return None


_CSS_STYLE_STEP_RE = re.compile(
    r'\b(background|bg.?color|color|colour|theme|dark.?mode|light.?mode|'
    r'primary|secondary|accent|brand|fill|stroke|opacity|gradient|'
    r'foreground|surface|tint|shade|hue|palette|style)\b',
    re.IGNORECASE,
)


def _find_css_conflicts(
    step_text: str,
    target_files: list[str],
    memory: FileMemory,
) -> dict[str, str]:
    """Return CSS/SCSS files whose selectors may override styles in target components.

    Only triggered when the step text involves colour/style changes.
    Returns ``{css_file_path: full_content}`` for every CSS file that
    contains a selector matching a class name used in the target components.
    The caller should inject these files into context and instruct the LLM
    to also update any conflicting rules.
    """
    if not _CSS_STYLE_STEP_RE.search(step_text):
        return {}

    all_files = memory.all_files()
    css_candidates = {
        fpath: content
        for fpath, content in all_files.items()
        if fpath.endswith(('.css', '.scss', '.sass', '.less'))
        and fpath not in target_files
    }
    if not css_candidates:
        return {}

    # Collect CSS class-name hints from the target component files.
    component_classes: set[str] = set()
    for fpath in target_files:
        content = all_files.get(fpath, '') or ''
        # JSX: className="foo bar" or className='foo bar'
        for m in re.finditer(r'className=["\']([^"\']+)["\']', content):
            component_classes.update(m.group(1).split())
        # JSX: className={`foo bar ${dynamic}`} — template literals (very common in React)
        for m in re.finditer(r'className=\{`([^`]+)`\}', content):
            static_part = re.sub(r'\$\{[^}]*\}', ' ', m.group(1))
            component_classes.update(static_part.split())
        # CSS selectors inside CSS files (.foo, .foo-bar)
        for m in re.finditer(r'\.([\w-]+)', content):
            component_classes.add(m.group(1))
        # File stem: Header.jsx → "header"
        raw_stem = os.path.basename(fpath).split('.')[0]
        component_classes.add(raw_stem.lower())
        # Convert PascalCase/camelCase stem to kebab-case: HeroBanner → hero-banner
        kebab_stem = re.sub(r'([a-z])([A-Z])', r'\1-\2', raw_stem).lower()
        if kebab_stem != raw_stem.lower():
            component_classes.add(kebab_stem)

    # Common component keywords from the step text itself.
    for word in re.findall(
        r'\b(header|footer|hero|banner|sidebar|navbar|nav|main|app|body)\b',
        step_text, re.IGNORECASE,
    ):
        component_classes.add(word.lower())
    # Also extract hyphenated CSS-class-like tokens from step text
    # (e.g. "hero-banner", "hero-section" written in the task description).
    for m in re.findall(r'\b([a-z][a-z0-9]*(?:-[a-z0-9]+)+)\b', step_text, re.IGNORECASE):
        component_classes.add(m.lower())

    if not component_classes:
        return {}

    # Find CSS files whose rules target any of those classes.
    conflicts: dict[str, str] = {}
    for css_path, css_content in css_candidates.items():
        for cls in component_classes:
            # Match ".header" but not ".my-header-thing" (word-boundary on both sides)
            if re.search(
                rf'(?<![.\w-])\.{re.escape(cls)}(?![.\w-])',
                css_content, re.IGNORECASE
            ):
                conflicts[css_path] = css_content
                break

    return conflicts


def _detect_target_files(step_text: str, memory: FileMemory,
                         max_files: int = 3,
                         kb_context_builder=None) -> list[str]:
    """Identify ALL target files for editing from the step text."""
    import re as _re

    known_files = list(memory.all_files().keys())
    found: list[str] = []
    found_set: set[str] = set()

    def _add(fpath: str) -> None:
        if fpath not in found_set and len(found) < max_files:
            if os.path.isfile(fpath):
                found.append(fpath)
                found_set.add(fpath)

    for fpath in known_files:
        if fpath in step_text:
            _add(fpath)

    for fpath in known_files:
        basename = os.path.basename(fpath)
        if basename and basename in step_text and basename.count(".") > 0:
            _add(fpath)

    path_pattern = _re.compile(r'[\w/\\]+\.\w{1,5}')
    for m in path_pattern.finditer(step_text):
        candidate = m.group().replace("\\", "/")
        for fpath in known_files:
            if fpath.endswith(candidate) or candidate.endswith(fpath):
                _add(fpath)

    if not found and len(known_files) == 1:
        _add(known_files[0])

    # Fallback: ask KB to find files on disk and load them into memory
    if not found and kb_context_builder is not None:
        for fpath in _resolve_from_kb(step_text, memory, kb_context_builder):
            _add(fpath)

    return found


def _build_chunk_prompt(
    task_description: str,
    formatted_chunks: str,
    slim_context: str,
    language: str | None = None,
) -> str:
    """Build the LLM prompt for chunk-level editing."""
    lang_tag = language or "python"
    return f"""You are editing existing code. You receive CHUNKS of files, not full files.
You MUST respond ONLY with the chunks you want to change.
Do NOT output unchanged chunks. Do NOT output full files.
Do NOT use #### [FILE]: markers. Use #### [EDIT]: markers instead.

Task: {task_description}

{slim_context}

{formatted_chunks}

For EACH chunk you want to change, use EXACTLY this format:

#### [EDIT]: {{file_path}}:{{function_or_class_name}} (lines {{start}}-{{end}})
```{lang_tag}
// complete replacement for this chunk only
```

For adding a NEW function/class, use:

#### [NEW]: {{file_path}} (after line {{line_number}})
```{lang_tag}
// new code to insert
```

Rules:
1. Only output chunks that ACTUALLY CHANGE
2. Include the COMPLETE replacement chunk (not a partial diff)
3. Preserve the function/class signature unless the task requires changing it
4. Match the existing indentation style exactly
5. Use the exact file paths shown in the context above
6. The line numbers MUST match the line ranges shown in EDITABLE markers
7. TAILWIND DEDUPLICATION — when adding a Tailwind utility class to a className:
   a. Do NOT add the same class token more than once (e.g. do not write `bg-orange-100 ... bg-orange-100`).
   b. If the task is ADDING a utility (e.g. `bg-orange-100`), append it ONCE and keep all unrelated classes.
   c. If the new utility conflicts with an existing one for the same CSS property (same prefix, e.g. `bg-indigo-600` vs `bg-orange-100`), REMOVE the old conflicting utility and keep only the new one — unless the task explicitly says to keep both.
8. CHUNK COMPLETENESS — your replacement must include EVERY line shown in the original EDITABLE chunk.
   Do NOT stop outputting after the function/class closing brace if the original chunk continues
   beyond it (e.g. `Foo.propTypes = ...`, `Foo.defaultProps = ...`, `export default Foo`).
   If those lines appear in the original chunk, reproduce them verbatim in your replacement.
9. NEVER abbreviate code inside your chunk with comments like `// existing code` or `/* unchanged */`. Write out every line.
"""


def _find_importers(fpath: str, memory: FileMemory) -> list[tuple[str, str]]:
    """Return (path, content) pairs for files in memory that import *fpath*.

    Used to give the reviewer visibility into how the changed file is consumed,
    catching issues like duplicate Router/Provider wrapping.
    """
    basename = os.path.splitext(os.path.basename(fpath))[0]  # e.g. "App"
    results: list[tuple[str, str]] = []
    for mpath, mcontent in memory.all_files().items():
        if mpath == fpath:
            continue
        # Quick check: does this file reference the changed file's basename?
        if basename not in mcontent:
            continue
        # Confirm it's an actual import/require, not just a comment mention
        if not re.search(
            r'(import\b|require\s*\()',
            mcontent,
        ):
            continue
        results.append((mpath, mcontent))
        if len(results) >= 3:  # cap to avoid bloating the prompt
            break
    return results


def _build_review_context(
    new_files: dict[str, str],
    memory: FileMemory,
    step_text: str,
) -> str:
    """Build a compact review context showing what changed plus importer files."""
    import difflib

    parts: list[str] = []

    for fpath, new_content in new_files.items():
        old_content = memory.get(fpath)
        if old_content is None and os.path.isfile(fpath):
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    old_content = f.read()
            except OSError:
                pass

        if old_content is not None:
            old_lines = old_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            diff = difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{fpath}", tofile=f"b/{fpath}",
                n=3,
            )
            diff_text = "".join(diff)
            if diff_text.strip():
                parts.append(f"#### Changes in {fpath}:\n```diff\n{diff_text}```")
            else:
                parts.append(f"#### {fpath}: no changes")
        else:
            parts.append(f"#### [NEW FILE]: {fpath}\n```\n{new_content}\n```")

        # Inject files that import this file so the reviewer can catch
        # duplicate wrappers (e.g. BrowserRouter in both App.jsx and main.jsx)
        importers = _find_importers(fpath, memory)
        for imp_path, imp_content in importers:
            parts.append(
                f"#### [IMPORTER — read-only context]: {imp_path}\n"
                f"```\n{imp_content[:2000]}\n```"
            )

    return "\n\n".join(parts)


def _try_chunk_edit(
    *,
    step_text: str,
    coder: CoderAgent,
    reviewer: ReviewerAgent,
    executor: Executor,
    task: str,
    memory: FileMemory,
    display: CLIDisplay,
    step_idx: int,
    language: str | None,
    cfg: Config,
    auto: bool = False,
    project_profile=None,
    project_context=None,
    kb_context_builder=None,
    behavioral_ctx: str = "",
) -> tuple[bool, str] | None:
    """Attempt chunk-level editing. Returns (success, error) or None for fallback."""
    try:
        from ..editing.chunk_editor import ChunkEditor
    except ImportError:
        return None

    max_files = getattr(cfg, "EDITING_MAX_CHUNK_FILES", 3)
    target_files = _detect_target_files(step_text, memory, max_files=max_files,
                                        kb_context_builder=kb_context_builder)
    if not target_files:
        log.debug("[ChunkEdit] No target files identified")
        return None

    # Styling steps that conflict with global CSS need full-file context so
    # both the component AND the overriding CSS file can be updated together.
    # Chunk edit can only emit [EDIT] markers for the component chunks, so
    # fall through to Tier 3 which handles the joint edit.
    if _find_css_conflicts(step_text, target_files, memory):
        log.debug("[ChunkEdit] CSS override conflicts detected — deferring to Tier 3")
        return None

    chunk_editor = ChunkEditor()
    all_chunks: list = []
    all_target_ids: list[str] = []

    for fpath in target_files:
        content = memory.get(fpath)
        if content is None:
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                continue

        chunks = chunk_editor.chunk_file(fpath, content)
        if not chunks:
            continue

        targets = chunk_editor.identify_target_chunks(chunks, step_text)
        if not targets:
            targets = [c.chunk_id for c in chunks if c.chunk_type != "imports"]

        all_chunks.extend(chunks)
        all_target_ids.extend(targets)

    if not all_chunks:
        log.debug("[ChunkEdit] No chunks extracted")
        return None

    formatted = chunk_editor.format_chunks_for_prompt(all_chunks, all_target_ids)

    slim_ctx = ""
    from .memory import get_plan_context_files as _get_chunk_plan_ctx
    plan_ctx = _get_chunk_plan_ctx()
    if plan_ctx:
        # Plan-aware: include plan-declared files that aren't chunk targets
        from .memory import _extract_file_skeleton
        parts_slim: list[str] = []
        for fpath, content in plan_ctx.items():
            if fpath not in target_files:
                parts_slim.append(f"#### [FILE]: {fpath}\n```\n{content}\n```")
        for fpath, content in memory.all_files().items():
            if fpath in plan_ctx or fpath in target_files:
                continue
            if fpath.startswith(('_cmd_output/', '_fix_output/', '_search_context/')):
                continue
            skeleton = _extract_file_skeleton(content, fpath)
            if skeleton:
                parts_slim.append(skeleton)
        slim_ctx = "\n\n".join(parts_slim) if parts_slim else ""
    elif getattr(cfg, "EDITING_SLIM_CONTEXT", True):
        slim_ctx = memory.related_context_slim(
            step_text,
            max_tokens=int((cfg.CONTEXT_WINDOW if cfg else 8192) * 0.3),
        )
        for tf in target_files:
            pattern = f"#### [FILE_STRUCTURE]: {tf}"
            if pattern in slim_ctx:
                start = slim_ctx.find(pattern)
                next_entry = slim_ctx.find("\n\n#### [FILE_STRUCTURE]:", start + 1)
                if next_entry != -1:
                    slim_ctx = slim_ctx[:start] + slim_ctx[next_entry + 2:]
                else:
                    slim_ctx = slim_ctx[:start].rstrip()

    prompt_prefix = ""
    if project_profile is not None:
        try:
            prompt_prefix = project_profile.format_for_prompt() + "\n\n"
        except Exception:
            pass
    kb_ctx = getattr(memory, '_kb_context', '')
    if kb_ctx:
        prompt_prefix += kb_ctx + "\n\n"
    # Reuse behavioral instructions already fetched by _handle_code_step
    # (avoids a duplicate KB search call for the same step).
    if (behavioral_ctx
            and "[BEHAVIORAL INSTRUCTIONS]" not in prompt_prefix
            and "[KB INSTRUCTIONS]" not in prompt_prefix):
        prompt_prefix += behavioral_ctx + "\n\n"
    _chunk_briefing = getattr(memory, '_task_briefing', '')
    if _chunk_briefing:
        prompt_prefix += (
            "TASK BRIEFING (respect Preserve and Key constraint):\n"
            f"{_chunk_briefing}\n\n"
        )

    chunk_prompt = prompt_prefix + _build_chunk_prompt(
        step_text, formatted, slim_ctx, language=language,
    )

    display.step_info(step_idx, f"[ChunkEdit] Sending {len(all_target_ids)} target chunks...")
    sent_before, recv_before = token_tracker.snapshot()

    llm_response = coder.llm_client.generate_response(chunk_prompt)

    sent_after, recv_after = token_tracker.snapshot()
    sent_delta = sent_after - sent_before
    recv_delta = recv_after - recv_before
    display.step_tokens(step_idx, sent_delta, recv_delta)

    edits = chunk_editor.parse_chunk_response(llm_response)
    if edits is None:
        log.info("[ChunkEdit] LLM used full-file format or no edits parsed, falling back")
        return None

    # Detect dropped edits: count [EDIT] markers in the raw response and
    # compare against parsed edits.  If the LLM produced EDIT markers that
    # failed to parse (e.g. multi-word chunk names before the regex fix),
    # fall back to full-file mode rather than applying partial edits.
    raw_edit_count = len(re.findall(
        r"####\s*\[EDIT\]:", llm_response,
    ))
    if raw_edit_count > len(edits):
        log.warning(
            "[ChunkEdit] Parsed %d edits but found %d [EDIT] markers in "
            "response — some edits were dropped, falling back to full-file",
            len(edits), raw_edit_count,
        )
        return None

    edits_by_file: dict[str, list] = {}
    for edit in edits:
        edits_by_file.setdefault(edit.file_path, []).append(edit)

    result_files: dict[str, str] = {}
    original_files: dict[str, str | None] = {}
    for fpath, file_edits in edits_by_file.items():
        original = memory.get(fpath)
        if original is None:
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    original = f.read()
            except OSError:
                log.info("[ChunkEdit] Treating %s as a new file", fpath)
                original = None

        original_files[fpath] = original

        try:
            # Filter chunks for this file so the editor can resolve
            # mismatched line numbers from the LLM.
            file_chunks = [c for c in all_chunks if c.file_path == fpath]
            result_files[fpath] = chunk_editor.apply_chunk_edits(
                original or "", file_edits, known_chunks=file_chunks,
            )
        except Exception as exc:
            log.warning("[ChunkEdit] Failed to apply edits to %s: %s", fpath, exc)
            return None

    if not result_files:
        return None

    # Detect destructive chunk edits: if the result is dramatically
    # smaller than the original, the LLM likely wiped the file instead
    # of making a targeted edit.  Fall back to full-file mode so the
    # LLM is re-prompted with the full context.
    for fpath, new_content in result_files.items():
        original = original_files.get(fpath)
        if original and len(original) > 200:
            ratio = len(new_content) / len(original)
            if ratio < 0.25:
                log.warning(
                    "[ChunkEdit] Destructive edit for %s: "
                    "%.0f%% size reduction (%d → %d chars), "
                    "falling back to full-file mode",
                    fpath, (1 - ratio) * 100,
                    len(original), len(new_content),
                )
                return None

    # Build review context before updating memory/disk so diffs are captured correctly
    review_ctx = _build_review_context(result_files, memory, step_text)

    display.stop_spinner()
    approved = prompt_diff_approval(result_files, auto=auto, display=display,
                                    base_dir=getattr(memory, 'base_dir', "."))
    display.step_info(step_idx, "Processing...")
    if not approved:
        display.step_info(step_idx, "Changes rejected by user")
        return False, "User rejected chunk edits."

    display.step_info(step_idx, "Processing approved changes...")
    written = executor.write_files(result_files)
    memory.update(result_files)
    display.step_info(step_idx, f"[ChunkEdit] Written: {', '.join(written)}")

    if _all_non_code_files(list(result_files.keys())):
        display.step_info(step_idx, "Non-code files, skipping review")
        return True, ""

    # Static checks (free — no LLM cost)
    lint_errors = _quick_offline_lint(result_files)
    import_errors = _validate_import_paths(result_files, memory)
    lint_errors = lint_errors + import_errors

    # API grounding: probe module.attr usages against installed packages.
    # Errors join the lint feedback; a clean probe arms the reviewer with
    # the verified-API list so it cannot FAIL correct code from stale
    # training data and talk the coder into reintroducing removed APIs.
    _api_errs, _verified_api_ctx = _api_grounding_context(
        result_files, memory, executor, project_context, language=language)
    if _api_errs:
        log.warning("[ChunkEdit] API grounding found %d issue(s)",
                    len(_api_errs))
        lint_errors = ((lint_errors + "\n") if lint_errors else "") \
            + "\n".join(_api_errs)

    # Determine review mode: "static" (default) or "full"
    review_mode_cfg = getattr(cfg, "REVIEW_MODE", "static") if cfg else "static"

    if review_mode_cfg == "static":
        if lint_errors:
            log.warning("[ChunkEdit] Static check errors, reverting: %s", lint_errors[:300])
            # Revert and fall back to full-file for retry with error feedback
            for fpath, orig_content in original_files.items():
                if orig_content is not None:
                    memory.update({fpath: orig_content})
                    try:
                        with open(fpath, "w", encoding="utf-8") as f:
                            f.write(orig_content)
                    except OSError:
                        pass
                else:
                    try:
                        if fpath in memory.all_files():
                            del memory.all_files()[fpath]
                        os.remove(fpath)
                    except Exception:
                        pass
            return None  # Fall back to full-file with static errors as feedback
        display.step_info(step_idx, "Static checks passed ✔")
        log.info("[ChunkEdit] Static review passed (lint + imports OK)")
        return True, ""

    # Skip LLM review if all files were already correct (no actual diffs).
    if "```diff" not in review_ctx:
        display.step_info(step_idx, "File(s) already correct, no review needed ✔")
        log.info("[ChunkEdit] Skipped LLM review — file(s) already correct")
        return True, ""

    # Full LLM review path (review_mode == "full")
    reviewer_mode = "diff" if getattr(cfg, "EDITING_REVIEWER_DIFF_MODE", True) else "full"

    display.step_info(step_idx, "Reviewing changes...")
    sent_before, recv_before = token_tracker.snapshot()

    lint_context = f"\n\n{lint_errors}Please fix these errors in your review." if lint_errors else ""

    # Inject KB context so the reviewer has up-to-date framework docs
    reviewer_kb = ""
    if kb_ctx:
        reviewer_kb = (
            f"\n\n[KB Documentation — trust this over your training data]\n"
            f"{kb_ctx}\n"
        )

    # Inject success criteria so the reviewer validates completeness
    criteria_ctx = ""
    if project_context is not None:
        criteria = getattr(project_context, 'success_criteria', [])
        if criteria:
            criteria_ctx = (
                "\n\nSUCCESS CRITERIA — reject if any are NOT met by the changes:\n"
                + "\n".join(f"- {c}" for c in criteria)
            )
    _chunk_reviewer_briefing = getattr(memory, '_task_briefing', '')
    if _chunk_reviewer_briefing:
        criteria_ctx += (
            "\n\nTASK BRIEFING — flag as FAIL if any Preserve item was broken:\n"
            f"{_chunk_reviewer_briefing}"
        )

    review = reviewer.process(
        f"Review this code change for the step: {step_text}\n\n{review_ctx}",
        context=f"Step: {step_text}\nReview ONLY the changes shown.{lint_context}{reviewer_kb}{criteria_ctx}{_verified_api_ctx}",
        language=language,
        review_mode=reviewer_mode,
    )

    sent_after, recv_after = token_tracker.snapshot()
    sent_delta = sent_after - sent_before
    recv_delta = recv_after - recv_before
    display.step_tokens(step_idx, sent_delta, recv_delta)

    if review:
        display.add_llm_log(review, source="Reviewer")

    # Probe errors are a hard gate — reviewer approval cannot waive them.
    approved = _review_verdict(review, _api_errs)
    if not approved and _api_errs and _review_verdict(review, None):
        log.warning(
            "[ChunkEdit] Reviewer approved but the API probe found %d "
            "issue(s) — probe is ground truth, rejecting", len(_api_errs))

    if approved:
        display.step_info(step_idx, "Review passed")
        return True, ""

    log.info("[ChunkEdit] Review found issues, reverting and falling back to full-file for retry")
    log.info("[ChunkEdit] Review feedback:\n%s", review[:2000])

    # Carry the review into the Tier-3 full-file retry — a blind
    # regeneration rediscovers the same mistakes from scratch (observed:
    # a reverted chunk fix regenerated with removed 2.x draw APIs).
    _chunk_fb = review
    if _api_errs:
        _chunk_fb += (
            "\n\nAUTHORITATIVE API CHECK (runtime probe against the "
            "installed packages — trust these over any review suggestion "
            "above):\n" + "\n".join(_api_errs)
        )
    try:
        memory._chunk_review_feedback = _chunk_fb
    except Exception:
        pass

    # Revert memory and disk since the review failed
    for fpath, orig_content in original_files.items():
        if orig_content is not None:
            memory.update({fpath: orig_content})
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(orig_content)
            except OSError:
                pass
        else:
            # It was a new file, remove it
            try:
                if fpath in memory.all_files():
                    del memory.all_files()[fpath]  # Depending on FileMemory implementation
                os.remove(fpath)
            except Exception:
                pass

    return None


def _build_diff_prompt(task_description: str, formatted_slices: str) -> str:
    """Build the LLM prompt requesting a structured diff response."""
    return f"""You are editing existing code. You will receive minimal file slices, not full files.
You MUST respond with ONLY a diff in the exact format specified.
NEVER rewrite the full file. NEVER include unchanged lines in your response.
ALWAYS use the exact line numbers shown in the slice annotations.

Task: {task_description}

{formatted_slices}

Respond with ONLY a diff in this exact format — nothing else:

@@DIFF_START@@
FILE: {{file_path}}
<<<<<<< ORIGINAL (line {{start_line}})
{{exactly as lines appear in the slice}}
=======
{{your replacement lines}}
>>>>>>> UPDATED
@@DIFF_END@@

Rules:
- Use line numbers exactly as they appear in the ORIGINAL slice block annotations. First line of ORIGINAL must match {{start_line}}.
- Only include blocks that actually change.
- Preserve indentation exactly.
- If no changes are needed for a file, omit it entirely.
- The ORIGINAL block must EXACTLY MATCH the slice content you want to replace, including all whitespace.
- For multiple changes in the same file, include multiple ORIGINAL/UPDATED blocks under the same FILE header.
- For changes across multiple files, include multiple FILE sections."""


def _log_fallback_metric(
    cfg: Config,
    target_file: str,
    step_text: str,
    scope,
    reason: str,
) -> None:
    """Log a fallback metric entry."""
    if not getattr(cfg, "EDITING_TRACK_METRICS", True):
        return
    try:
        from ..editing.metrics import log_edit_metric
        log_edit_metric({
            "file": target_file,
            "task_length_chars": len(step_text),
            "resolution_method": scope.resolution_method,
            "confidence": round(scope.confidence, 2),
            "full_file_lines": 0,
            "sliced_lines_sent": 0,
            "token_reduction_pct": 0,
            "hunks_applied": 0,
            "hunks_failed": 0,
            "fallback_used": True,
            "fallback_reason": reason,
            "syntax_valid": True,
            "affected_files_count": len(scope.affected_files),
        })
    except Exception:
        pass
