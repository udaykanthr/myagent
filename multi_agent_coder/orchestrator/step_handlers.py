"""
Step handlers — CMD, CODE, and TEST step execution logic.
"""
# Separate retry limit for test generation (lower than code to avoid pipeline halts)
MAX_TEST_GEN_RETRIES = 2

import json
import os
import shutil

from ..config import Config
from ..agents.coder import CoderAgent
from ..agents.reviewer import ReviewerAgent
from ..agents.tester import TesterAgent
from ..executor import Executor
from ..cli_display import CLIDisplay, token_tracker, log
from ..language import get_code_block_lang, get_test_framework

from .memory import FileMemory
from .classification import _extract_command_from_step

from ..diff_display import show_diffs, prompt_diff_approval, _detect_hazards


MAX_STEP_RETRIES = 3  # Used for code steps and test run/fix attempts

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


def _get_runner_install_cmd(runner: str) -> str | None:
    """Return the install command for a test runner binary.

    Returns ``None`` for tools that must be installed manually (e.g. ``go``,
    ``cargo``) — the caller must handle this gracefully.
    """
    return _RUNNER_INSTALL.get(runner, f"pip install {runner}")


def _read_js_project_env(cwd: str | None = None) -> dict:
    """Read package.json and project config to detect JS/TS environment.

    Returns a dict with:
        is_esm: bool       — True if package.json has "type": "module"
        has_jest: bool      — True if jest in dependencies/devDependencies
        has_jest_globals: bool — True if @jest/globals is installed
        has_jest_config: bool — True if jest.config.* exists
        module_type: str    — "module" or "commonjs"
    """
    env = {
        "is_esm": False,
        "has_jest": False,
        "has_jest_globals": False,
        "has_jest_config": False,
        "module_type": "commonjs",
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

        # Jest in dependencies
        all_deps = {}
        all_deps.update(pkg.get("dependencies", {}))
        all_deps.update(pkg.get("devDependencies", {}))
        env["has_jest"] = "jest" in all_deps
        env["has_jest_globals"] = "@jest/globals" in all_deps

    # Jest config detection
    for config_name in ("jest.config.js", "jest.config.ts", "jest.config.mjs",
                        "jest.config.cjs", "jest.config.json"):
        cfg_path = os.path.join(cwd, config_name) if cwd else config_name
        if os.path.isfile(cfg_path):
            env["has_jest_config"] = True
            break

    return env


import platform


def _strip_protected_files(files: dict[str, str]) -> dict[str, str]:
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

        # Not a protected file — pass through
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

    Merges new keys in ``dependencies``, ``devDependencies``, and ``scripts``.
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
            "Do NOT use PowerShell cmdlets like Get-ChildItem, Select-Object, etc.\n"
        )
    else:
        base = (
            f"Use standard shell commands for {platform.system()}.\n"
            "For listing files use: find . -type f\n"
            "For reading a file use: cat <path>\n"
            "For creating a directory use: mkdir -p <path>\n"
            "For installing Python packages use: pip install <package>\n"
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
    """Collect outputs of prior steps from memory for context."""
    parts: list[str] = []
    all_files = memory.all_files()
    for i in range(step_idx):
        key = f"_cmd_output/step_{i+1}.txt"
        if key in all_files:
            parts.append(f"Step {i+1} output:\n{all_files[key]}")
    if not parts:
        return ""
    return "Previously executed steps:\n" + "\n\n".join(parts) + "\n\n"


def _detect_subproject_root(memory: FileMemory) -> str | None:
    """Detect if all project files share a common subdirectory prefix.

    When an earlier CMD step created a project in a subdirectory (e.g.
    ``npx create-react-app my-app``), subsequent files in memory will all
    live under ``my-app/``.  This function finds that common root.

    Returns the subdirectory name (e.g. ``my-app``) or ``None``.
    """
    all_files = memory.all_files()
    # Only consider real source files, not internal tracking paths.
    # Internal paths use underscore-prefixed directories (_cmd_output/,
    # _fix_output/, etc.) and must be excluded from sub-project detection.
    source_paths = [
        p for p in all_files
        if not p.startswith('_') and '/' in p
    ]
    if not source_paths:
        return None

    # Extract first path component from each file
    first_components: set[str] = set()
    for p in source_paths:
        parts = p.replace('\\', '/').split('/')
        if len(parts) >= 2:  # must have at least dir/file
            first_components.add(parts[0])

    # If all files share the same single first directory component,
    # that's our sub-project root
    if len(first_components) == 1:
        subproject = first_components.pop()
        # Sanity check: the directory should exist on disk
        if os.path.isdir(subproject):
            log.info(f"[SubProject] Detected sub-project root: {subproject}/")
            return subproject

    # Fallback 1: if memory contains files from multiple top-level directories
    # (e.g. search provider added files), look for a known project manifest
    from ..executor import Executor
    manifest_dirs = set()
    for p in source_paths:
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
            for p in source_paths
            if len(p.replace('\\', '/').split('/')) >= 2
        )
        if counts:
            best, best_count = counts.most_common(1)[0]
            total = sum(counts.values())
            # Only use majority if it covers >70% of files
            if best_count > total * 0.7 and os.path.isdir(best):
                log.info(f"[SubProject] Detected sub-project root via "
                         f"majority ({best_count}/{total} files): {best}/")
                return best

    return None


def _handle_cmd_step(step_text: str, executor: Executor,
                     llm_client, memory: FileMemory,
                     display: CLIDisplay, step_idx: int,
                     language: str | None = None) -> tuple[bool, str]:
    cmd = _extract_command_from_step(step_text)

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
        if prior_context:
            gen_prompt += (
                f"{prior_context}"
                "IMPORTANT: Use the exact names, paths, and values from the "
                "previous steps above. Do NOT guess or use defaults.\n\n"
            )
        if file_summary != "(no files yet)":
            gen_prompt += f"Project files: {file_summary}\n\n"
        gen_prompt += f"Step: {step_text}\n\nCommand:"
        sent_before = token_tracker.total_prompt_tokens
        recv_before = token_tracker.total_completion_tokens

        cmd_response = llm_client.generate_response(gen_prompt).strip()

        sent_delta = token_tracker.total_prompt_tokens - sent_before
        recv_delta = token_tracker.total_completion_tokens - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        if cmd_response:
            display.add_llm_log(cmd_response, source="Coder")

        cmd = cmd_response
        cmd = cmd.strip('`').strip()
        if cmd.startswith('```'):
            cmd = cmd.split('\n', 1)[-1].rsplit('```', 1)[0].strip()

        if not cmd:
            display.step_info(step_idx, "Could not generate command, skipping.")
            log.warning(f"Step {step_idx+1}: LLM returned empty command.")
            return True, ""

    # Detect sub-project root so commands like `npm install` run in the
    # correct directory instead of the repo root.
    subproject_cwd = None
    subproject = _detect_subproject_root(memory)
    if subproject:
        import re as _re_sp
        # Commands that should run inside the sub-project directory
        _subproject_cmd_patterns = (
            r'\bnpm\s+(install|start|run|test|build|ci)\b',
            r'\bnpx\s+',
            r'\byarn\s+(install|add|start|dev|build|test)\b',
            r'\bpnpm\s+(install|add|start|dev|build|test)\b',
            r'\bnode\s+',
            r'\bng\s+(serve|build|test)\b',
        )
        needs_subproject = any(
            _re_sp.search(p, cmd, _re_sp.IGNORECASE)
            for p in _subproject_cmd_patterns
        )
        # Don't set cwd if the command already includes a `cd` to the subproject
        already_has_cd = f'cd {subproject}' in cmd or f'cd ./{subproject}' in cmd
        if needs_subproject and not already_has_cd:
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
    
    cwd_note = f" (in {subproject_cwd}/)" if subproject_cwd else ""
    if is_background:
        display.step_info(step_idx, f"Running background: {cmd}{cwd_note}")
        log.info(f"Step {step_idx+1}: Running background command: {cmd}")
    else:
        display.step_info(step_idx, f"Running: {cmd}{cwd_note}")
        log.info(f"Step {step_idx+1}: Running command: {cmd}")

    success, output = executor.run_command(
        cmd, background=is_background, cwd=subproject_cwd)
    log.info(f"Step {step_idx+1}: Command output:\n{output}")

    if output:
        truncated = output[:4000] if len(output) > 4000 else output
        memory.update({
            f"_cmd_output/step_{step_idx+1}.txt": f"$ {cmd}\n\n{truncated}"
        })

    if success:
        display.step_info(step_idx, "Command succeeded.")
        return True, ""
    else:
        display.step_info(step_idx, "Command failed. See log.")
        log.warning(f"Step {step_idx+1}: Command failed.")
        return False, f"Command `{cmd}` failed.\nOutput:\n{output}"


def _auto_fix_hazards(files: dict[str, str], coder: CoderAgent,
                      executor: Executor, display: CLIDisplay,
                      step_idx: int, step_text: str,
                      language: str | None = None,
                      base_dir: str = ".") -> dict[str, str]:
    """Scan generated files for hazardous diffs and auto-fix them.

    For each file where ``_detect_hazards`` flags problems (e.g. significant
    size reduction, dependency removal), the coder LLM is asked to produce
    a corrected version that preserves the existing content while applying
    only the intended changes.

    Returns the (potentially corrected) file dict.
    """
    fixed_files = dict(files)

    for filepath, new_content in list(files.items()):
        full_path = os.path.join(base_dir, filepath)
        if not os.path.isfile(full_path):
            continue  # new file, nothing to compare

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
            f"Produce a CORRECTED version of `{filepath}` that:\n"
            f"1. Keeps ALL existing content (dependencies, imports, configs, etc.)\n"
            f"2. Only adds/changes what the step requires\n"
            f"3. Does NOT remove anything that was in the original file\n\n"
            f"#### [FILE]: {filepath}\n"
            f"```\n"
            f"(write the complete corrected file here)\n"
            f"```"
        )

        sent_before = token_tracker.total_prompt_tokens
        recv_before = token_tracker.total_completion_tokens

        fix_response = coder.process(fix_prompt, context="", language=language)

        sent_delta = token_tracker.total_prompt_tokens - sent_before
        recv_delta = token_tracker.total_completion_tokens - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

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


def _handle_code_step(step_text: str, coder: CoderAgent, reviewer: ReviewerAgent,
                      executor: Executor, task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int,
                      language: str | None = None,
                      cfg: Config | None = None,
                      auto: bool = False) -> tuple[bool, str]:
    feedback = ""
    context_window = cfg.CONTEXT_WINDOW if cfg else 8192
    ctx_budget = int(context_window * 0.8)
    prev_files: dict[str, str] = {}  # Track files from previous attempt

    for attempt in range(1, MAX_STEP_RETRIES + 1):
        context = f"Task: {task}"
        related = memory.related_context(step_text, max_tokens=ctx_budget)
        if related:
            context += f"\nExisting files (overwrite as needed):\n{related}"
        if memory.summary() != "(no files yet)":
            context += f"\nAll project files: {memory.summary()}"
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
        sent_before = token_tracker.total_prompt_tokens
        recv_before = token_tracker.total_completion_tokens

        response = coder.process(step_text, context=context, language=language)

        sent_delta = token_tracker.total_prompt_tokens - sent_before
        recv_delta = token_tracker.total_completion_tokens - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        explanation = CLIDisplay.extract_explanation(response)
        if explanation:
            display.add_llm_log(explanation, source="Coder")

        files = executor.parse_code_blocks(response)
        if not files:
            files = executor.parse_code_blocks_fuzzy(response)
        if not files:
            feedback = "No file markers found. Use #### [FILE]: path/to/file.py format."
            display.step_info(step_idx, "No files parsed, retrying...")
            log.warning(f"Step {step_idx+1}: No files parsed from coder response.")
            continue

        # Normalize paths: fix LLM-generated paths that are suffixes of
        # known project files (e.g. src/App.js → my-app/src/App.js)
        files = _normalize_fix_paths(files, memory)

        # Strip protected manifest files (package.json, etc.) to prevent
        # LLM from overwriting them with corrupted versions
        files = _strip_protected_files(files)

        # On retry, merge: keep previously approved files that weren't
        # re-generated, so the coder doesn't need to regenerate everything
        if attempt > 1 and prev_files:
            merged = dict(prev_files)
            merged.update(files)  # new files override previous
            files = merged

        # Auto-fix hazardous diffs before showing to user
        files = _auto_fix_hazards(files, coder, executor, display, step_idx,
                                  step_text, language=language)

        # Show diffs and wait for approval before writing
        approved = prompt_diff_approval(files, auto=auto)
        if not approved:
            feedback = "User rejected the changes. Try a different approach."
            display.step_info(step_idx, "Changes rejected by user, retrying...")
            log.info(f"Step {step_idx+1}: User rejected diff, retrying.")
            continue

        prev_files = dict(files)  # Save for potential merge on retry
        written = executor.write_files(files)
        memory.update(files)
        display.step_info(step_idx, f"Written: {', '.join(written)}")

        # Skip review for non-code files (README, LICENSE, configs, etc.)
        if _all_non_code_files(list(files.keys())):
            display.step_info(step_idx, "Non-code files, skipping review ✔")
            log.info(f"Step {step_idx+1}: Skipped review (non-code files: {list(files.keys())})")
            return True, ""

        # Review — pass step description so reviewer scopes to this step
        display.step_info(step_idx, "Reviewing code...")
        sent_before = token_tracker.total_prompt_tokens
        recv_before = token_tracker.total_completion_tokens

        review = reviewer.process(
            f"Review this code for the step: {step_text}\n\n{response}",
            context=f"Step: {step_text}\nOnly review changes relevant to this step.",
            language=language,
        )

        sent_delta = token_tracker.total_prompt_tokens - sent_before
        recv_delta = token_tracker.total_completion_tokens - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        if review:
            display.add_llm_log(review, source="Reviewer")

        log.info(f"Step {step_idx+1}: Review:\n{review}")

        review_lower = review.lower()
        # Accept if the reviewer explicitly approves
        approved = any(phrase in review_lower for phrase in (
            "code looks good",
            "looks good",
            "no issues",
            "no critical issues",
            "no bugs found",
            "code is correct",
            "functionally correct",
            "lgtm",
        ))

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
            if not has_critical:
                display.step_info(step_idx, "Review has only minor suggestions, accepting ✔")
                log.info(f"Step {step_idx+1}: Accepted on last attempt "
                         f"(review had no critical keywords)")
                return True, ""

        feedback = review
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


def _handle_test_step(step_text: str, tester: TesterAgent, coder: CoderAgent,
                      reviewer: ReviewerAgent, executor: Executor,
                      task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int,
                      language: str | None = None,
                      auto: bool = False,
                      search_agent=None) -> tuple[bool, str]:
    lang_tag = get_code_block_lang(language) if language else "python"
    test_cmd = get_test_framework(language)["command"] if language else "pytest"

    # Detect sub-project (if the test targets a nested folder)
    subproject_cwd = _detect_subproject_root(memory)

    # Ensure the test runner binary is installed before attempting to run tests
    parts = test_cmd.split()
    runner = parts[0]
    # For "npx <tool>", the binary to check is "npx" itself
    if not shutil.which(runner):
        actual_tool = parts[1] if runner == "npx" and len(parts) > 1 else runner
        install_cmd = _get_runner_install_cmd(actual_tool)
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

    # Detect JS/TS project environment for ESM-aware test generation
    js_env: dict | None = None
    if language in ("javascript", "typescript"):
        js_env = _read_js_project_env(subproject_cwd)
        log.info(f"Step {step_idx+1}: JS project env: {js_env}")

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

    code_summary = ""
    for fname, content in memory.all_files().items():
        code_summary += f"#### [FILE]: {fname}\n```{lang_tag}\n{content}\n```\n\n"

    feedback = ""
    last_test_output = ""
    prev_gen_error = None  # Track errors across gen attempts for early exit

    for gen_attempt in range(1, MAX_TEST_GEN_RETRIES + 1):
        display.step_info(step_idx, f"Generating tests (attempt {gen_attempt}/{MAX_TEST_GEN_RETRIES})...")
        gen_context = f"Code:\n{code_summary}"
        if feedback:
            gen_context += f"\nFeedback: {feedback}"
        # Add JS/TS environment info to context
        if js_env:
            env_note = f"\nProject environment: {js_env}"
            if js_env.get('is_esm'):
                env_note += (
                    "\nCRITICAL: This is an ES Module project. "
                    "Tests MUST import from '@jest/globals'.\n"
                )
            gen_context += env_note

        sent_before = token_tracker.total_prompt_tokens
        recv_before = token_tracker.total_completion_tokens

        test_response = tester.process(
            step_text, context=gen_context, language=language,
            env_info=js_env)

        sent_delta = token_tracker.total_prompt_tokens - sent_before
        recv_delta = token_tracker.total_completion_tokens - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        explanation = CLIDisplay.extract_explanation(test_response)
        if explanation:
            display.add_llm_log(explanation, source="Tester")

        test_files = executor.parse_code_blocks(test_response)
        if not test_files:
            test_files = executor.parse_code_blocks_fuzzy(test_response)
        if not test_files:
            feedback = "No test files found. Use #### [FILE]: format."
            display.step_info(step_idx, "No test files parsed, retrying...")
            continue

        # Strip protected manifest files before they reach memory
        test_files = _strip_protected_files(test_files)

        # Normalize paths: fix LLM-generated paths that are suffixes of known files
        test_files = _normalize_fix_paths(test_files, memory)

        # Filter: only allow test files (block any source/config files)
        test_files = _filter_test_only_files(test_files, test_files, memory)
        if not test_files:
            feedback = "Generated files were all non-test files. Generate ONLY test files."
            display.step_info(step_idx, "No valid test files after filtering, retrying...")
            continue

        # Review tests
        display.step_info(step_idx, "Reviewing tests...")
        sent_before = token_tracker.total_prompt_tokens
        recv_before = token_tracker.total_completion_tokens

        review = reviewer.process(
            f"Review these tests for correctness, especially import paths:\n{test_response}",
            context=f"Project files: {memory.summary()}\n{code_summary}",
            language=language,
        )

        sent_delta = token_tracker.total_prompt_tokens - sent_before
        recv_delta = token_tracker.total_completion_tokens - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        if review:
            display.add_llm_log(review, source="Reviewer")

        log.info(f"Step {step_idx+1}: Test review:\n{review}")

        review_lower = review.lower()
        test_approved = any(phrase in review_lower for phrase in (
            "code looks good", "looks good", "no issues",
            "no critical issues", "no bugs found", "code is correct",
            "functionally correct", "lgtm", "tests look good",
        ))

        # On last attempt, accept if no critical issues found
        if not test_approved and gen_attempt == MAX_TEST_GEN_RETRIES:
            has_critical = any(kw in review_lower for kw in (
                "error", "bug", "crash", "undefined", "missing import",
                "will fail", "won't work", "incorrect", "wrong import",
            ))
            if not has_critical:
                test_approved = True
                log.info(f"Step {step_idx+1}: Test accepted on last attempt (minor issues only)")

        if not test_approved:
            feedback = review
            display.step_info(step_idx, "Test review found issues, regenerating...")
            continue

        # Show diffs and wait for approval before writing test files
        approved = prompt_diff_approval(test_files, auto=auto)
        if not approved:
            feedback = "User rejected the test changes. Try a different approach."
            display.step_info(step_idx, "Test changes rejected by user, retrying...")
            log.info(f"Step {step_idx+1}: User rejected test diff, retrying.")
            continue

        written = executor.write_files(test_files)
        memory.update(test_files)
        display.step_info(step_idx, f"Tests written: {', '.join(written)}")

        prev_output = None
        for run_attempt in range(1, MAX_STEP_RETRIES + 1):
            display.step_info(step_idx, f"Running: {test_cmd} (attempt {run_attempt})...")
            log.info(f"Step {step_idx+1}: Running test command: {test_cmd}" + (f" in {subproject_cwd}" if subproject_cwd else ""))
            success, output = executor.run_tests(test_cmd, cwd=subproject_cwd)
            log.info(f"Step {step_idx+1}: Test run output:\n{output or '(no output)'}")

            last_test_output = output

            if success:
                display.step_info(step_idx, "Tests passed ✔")
                return True, ""

            # Extract error classes for smarter deduplication
            import re as _re
            _error_classes = set(_re.findall(
                r'(ModuleNotFoundError|ImportError|SyntaxError|NameError|'
                r'TypeError|AttributeError|IndentationError|FileNotFoundError|'
                r'AssertionError|KeyError|ValueError|ReferenceError)',
                output or ""
            ))

            # Detect stuck loop: compare error classes, not just exact output
            if prev_output and run_attempt > 1:
                prev_error_classes = set(_re.findall(
                    r'(ModuleNotFoundError|ImportError|SyntaxError|NameError|'
                    r'TypeError|AttributeError|IndentationError|FileNotFoundError|'
                    r'AssertionError|KeyError|ValueError|ReferenceError)',
                    prev_output or ""
                ))
                if _error_classes and _error_classes == prev_error_classes:
                    display.step_info(step_idx,
                                      "Same error types repeating — fix not working, stopping.")
                    log.warning(f"Step {step_idx+1}: Same error classes on attempt "
                                f"{run_attempt}: {_error_classes}, breaking retry loop.")
                    break
                if output == prev_output:
                    display.step_info(step_idx,
                                      "Same error repeating — not a code issue, stopping retry loop.")
                    log.warning(f"Step {step_idx+1}: Identical test output on attempt "
                                f"{run_attempt}, breaking retry loop.")
                    break
            prev_output = output

            # Early exit for unfixable error types
            _unfixable = _error_classes & {'SyntaxError', 'IndentationError'}
            if _unfixable and run_attempt > 1:
                display.step_info(step_idx,
                                  f"Persistent {', '.join(_unfixable)} — likely unfixable, stopping.")
                log.warning(f"Step {step_idx+1}: Unfixable errors after "
                            f"{run_attempt} attempts: {_unfixable}")
                break

            # Detect system-level / environment failures early — these
            # can't be fixed by editing code (e.g. missing Gemfile,
            # missing runtime, missing package manager).
            from .pipeline import _detect_system_level_failure
            sys_issue = _detect_system_level_failure(output)
            if sys_issue:
                msg = (f"System dependency missing: {sys_issue}. "
                       f"Cannot fix by editing code.")
                display.step_info(step_idx, msg)
                log.error(f"Step {step_idx+1}: {msg}")
                return False, msg

            # If test runner itself is not installed, try to install it
            if "not installed" in output or "not on PATH" in output:
                runner_parts = test_cmd.split()
                actual_tool = runner_parts[1] if runner_parts[0] == "npx" and len(runner_parts) > 1 else runner_parts[0]
                install_cmd = _get_runner_install_cmd(actual_tool)
                if install_cmd is None:
                    # System-level tool — can't auto-install, stop retrying
                    display.step_info(step_idx,
                                      f"`{actual_tool}` must be installed manually.")
                    log.error(f"Step {step_idx+1}: `{actual_tool}` is not installed "
                              f"and cannot be auto-installed.")
                    break
                display.step_info(step_idx, f"Installing `{actual_tool}`...")
                log.info(f"Step {step_idx+1}: Installing test runner: {install_cmd}")
                ok, out = executor.run_command(install_cmd, cwd=subproject_cwd)
                if ok:
                    display.step_info(step_idx, f"Installed `{actual_tool}`, re-running...")
                    success, output = executor.run_tests(test_cmd, cwd=subproject_cwd)
                    last_test_output = output
                    if success:
                        display.step_info(step_idx, "Tests passed after runner install ✔")
                        return True, ""
                continue  # retry with coder fix if runner install + rerun still failed

            # Auto-install missing packages before asking coder to fix
            missing_pkgs = executor.detect_missing_packages(output)
            if missing_pkgs:
                install_tool = "npm install --save-dev" if language in ("javascript", "typescript") else ("pip install")
                
                display.step_info(step_idx, f"Installing missing packages: {', '.join(missing_pkgs)}")
                log.info(f"Step {step_idx+1}: Auto-installing: {missing_pkgs} with {install_tool}")
                install_ok, install_out = executor.install_packages(missing_pkgs, tool=install_tool, cwd=subproject_cwd)
                if install_ok:
                    display.step_info(step_idx, "Packages installed, re-running tests...")
                    log.info(f"Step {step_idx+1}: Re-running test command: {test_cmd}")
                    success, output = executor.run_tests(test_cmd, cwd=subproject_cwd)
                    log.info(f"Step {step_idx+1}: Test re-run after install:\n{output or '(no output)'}")
                    last_test_output = output
                    if success:
                        display.step_info(step_idx, "Tests passed after package install ✔")
                        return True, ""
                else:
                    log.warning(f"Step {step_idx+1}: Package install failed: {install_out}")

            # Early exit: if the same error type repeats across gen attempts,
            # the fix isn't working — break to avoid wasting LLM calls
            if prev_gen_error:
                prev_gen_classes = set(_re.findall(
                    r'(ModuleNotFoundError|ImportError|SyntaxError|NameError|'
                    r'TypeError|AttributeError|IndentationError|FileNotFoundError|'
                    r'AssertionError|KeyError|ValueError)',
                    prev_gen_error or ""
                ))
                if (output == prev_gen_error or
                        (_error_classes and _error_classes == prev_gen_classes)):
                    display.step_info(step_idx,
                                      "Same test error repeating across attempts — stopping.")
                    log.warning(f"Step {step_idx+1}: Same error types across gen "
                                f"attempts: {_error_classes}, breaking retry loop.")
                    break
            prev_gen_error = output

            display.step_info(step_idx, "Tests failed, asking coder to fix...")
            error_detail = output[:1000] if output else f"(command `{test_cmd}` produced no output — it may have crashed or the test framework may not be installed)"

            # Search the web for error documentation to help the coder fix
            search_context = ""
            if search_agent is not None:
                display.step_info(step_idx, "Searching web for test error fix...")
                try:
                    search_context = search_agent.search_for_error(
                        error_detail, step_text, language=language)
                    if search_context:
                        log.info(f"Step {step_idx+1}: Search agent found "
                                 f"test error documentation")
                except Exception as exc:
                    log.warning(f"Step {step_idx+1}: Search agent error "
                                f"during test fix: {exc}")

            # Build a more specific fix prompt that restricts changes to test files
            test_file_list = ", ".join(test_files.keys())
            fix_context = (
                f"Test command: `{test_cmd}`\n"
                f"Test errors:\n{error_detail}\n"
                f"Test files that need fixing: {test_file_list}\n"
                f"Project files:\n{code_summary}"
            )
            if search_context:
                fix_context += (
                    f"\n\nThe following web search results contain relevant "
                    f"documentation and solutions for this error. Use them "
                    f"to inform your fix:\n\n{search_context}"
                )
            fix_prompt = (
                "Fix ONLY the test files so tests pass. "
                "Do NOT modify source files, package.json, or any config files. "
                "Do NOT add new dependencies. "
                "Only output the corrected test file(s) using #### [FILE]: format."
            )

            sent_before = token_tracker.total_prompt_tokens
            recv_before = token_tracker.total_completion_tokens

            fix_response = coder.process(
                fix_prompt, context=fix_context, language=language)

            sent_delta = token_tracker.total_prompt_tokens - sent_before
            recv_delta = token_tracker.total_completion_tokens - recv_before
            display.step_tokens(step_idx, sent_delta, recv_delta)

            explanation = CLIDisplay.extract_explanation(fix_response)
            if explanation:
                display.add_llm_log(explanation, source="Coder")

            fix_files = executor.parse_code_blocks(fix_response)
            if not fix_files:
                fix_files = executor.parse_code_blocks_fuzzy(fix_response)
            if fix_files:
                # Strip protected manifest files
                fix_files = _strip_protected_files(fix_files)
                # Normalize paths and filter to test-only files
                fix_files = _normalize_fix_paths(fix_files, memory)
                fix_files = _filter_test_only_files(
                    fix_files, test_files, memory)
                if fix_files:
                    show_diffs(fix_files, log_only=True)
                    executor.write_files(fix_files)
                    memory.update(fix_files)
                else:
                    log.warning(f"Step {step_idx+1}: All fix files were "
                                f"blocked by test-only filter")
                code_summary = ""
                for fname, content in memory.all_files().items():
                    code_summary += f"#### [FILE]: {fname}\n```{lang_tag}\n{content}\n```\n\n"

        log.error(f"Step {step_idx+1}: Tests still failing after {MAX_STEP_RETRIES} fixes.")
        return False, f"Tests still failing after {MAX_STEP_RETRIES} fix attempts.\nLast test output:\n{last_test_output}"

    log.error(f"Step {step_idx+1}: Could not generate valid tests after {MAX_TEST_GEN_RETRIES} attempts.")
    return False, f"Could not generate valid tests after {MAX_TEST_GEN_RETRIES} attempts.\nLast feedback:\n{feedback}"
