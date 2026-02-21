import os
import re
import subprocess
from typing import Dict, List, Tuple
from .cli_display import log


class Executor:
    def __init__(self):
        self._background_processes: List[subprocess.Popen] = []

    # Phrases that indicate the model is producing generic filler instead
    # of an actual plan.  Matched case-insensitively against each step.
    _VAGUE_STEP_PATTERNS = [
        r"^implement\b.*\b(core|main|basic|provided|the)\b.*\b(functionality|solution|feature|logic)\b",
        r"^(begin|start)\b.*\b(simple|basic|clear)\b.*\b(abstraction|understanding|overview)\b",
        r"^(review|analyze|understand|study)\b.*\b(problem|statement|requirements?|codebase)\b",
        r"^(set up|setup|configure)\b.*\b(environment|workspace|tooling)\b",
        r"^(ensure|verify|validate)\b.*\b(everything|all|code)\b.*\b(works?|correct|proper)\b",
        r"^(finalize|complete|finish)\b.*\b(implementation|solution|project)\b",
        r"^(test|debug)\b.*\b(thoroughly|completely|everything)\b",
        r"^(deploy|deliver|submit)\b.*\b(final|completed?|finished)\b",
        r"^(read|gather|collect)\b.*\b(information|data|input)\b",
        r"^(write|create)\b.*\b(documentation|docs|readme)\b.*\b(for|about)\b",
    ]

    @classmethod
    def validate_plan_quality(cls, steps: List[str]) -> tuple[bool, str]:
        """Check if parsed plan steps are actionable or generic filler.

        Returns ``(is_valid, reason)``.  A plan is invalid when:
        - Too few steps (< 1) or too many (> 25)
        - Majority of steps match vague/generic patterns
        - Steps are extremely short (avg < 8 chars) suggesting fragments
        """
        if not steps:
            return False, "no steps parsed"
        if len(steps) > 25:
            return False, f"too many steps ({len(steps)})"

        avg_len = sum(len(s) for s in steps) / len(steps)
        if avg_len < 8:
            return False, "steps are too short / fragmented"

        vague_count = 0
        for step in steps:
            for pat in cls._VAGUE_STEP_PATTERNS:
                if re.search(pat, step, re.IGNORECASE):
                    vague_count += 1
                    break

        if len(steps) <= 3 and vague_count >= len(steps):
            return False, "all steps are generic filler"
        if vague_count > len(steps) * 0.5:
            return False, f"{vague_count}/{len(steps)} steps are vague/generic"

        return True, ""

    @staticmethod
    def parse_plan_steps(plan_text: str) -> List[str]:
        """
        Splits a numbered plan into individual step strings using
        simple string manipulation.
        Input:
            1. Check python env
            2. Create calculator.py
            3. Write tests
        Output: ["Check python env", "Create calculator.py", "Write tests"]
        """
        steps = []
        # Match lines starting with a number followed by a dot
        pattern = r"^\s*\d+\.\s*(.*)"
        for line in plan_text.splitlines():
            match = re.match(pattern, line)
            if match:
                step_text = match.group(1).strip()
                if step_text:
                    steps.append(step_text)
        return steps

    # Generic placeholder path segments that local models hallucinate
    _PLACEHOLDER_SEGMENTS = {
        'path', 'to', 'your', 'my', 'the', 'project', 'folder',
        'directory', 'example', 'sample', 'some', 'filename',
        'yourproject', 'myproject', 'your_project', 'my_project',
    }

    @staticmethod
    def _sanitize_filename(raw: str) -> str:
        """Clean up LLM-generated filenames that may contain junk.

        Also blocks path traversal (``../``) so that LLM output can
        never write files outside the project directory.
        Returns empty string for clearly invalid/placeholder paths.
        """
        # Reject anything with newlines (multi-line capture mistake)
        name = raw.split('\n')[0].strip()
        # Strip trailing parenthetical descriptions: "file.py (main file)"
        name = re.sub(r'\s*\(.*?\)\s*$', '', name)
        # Strip trailing comments: "file.py # main module"
        name = re.sub(r'\s*#.*$', '', name)
        # Remove backticks
        name = name.strip('`').strip()
        # Remove template-style brackets: [path/to]/[filename].[ext]
        name = re.sub(r'\[([^\]]*)\]', r'\1', name)
        # Normalize backslashes to forward slashes
        name = name.replace('\\', '/')
        # Remove leading ./ if present
        name = re.sub(r'^\./', '', name)
        # Block path traversal: remove all ".." segments
        parts = [p for p in name.split('/') if p and p != '..']
        name = '/'.join(parts)
        # Remove leading slashes (absolute paths → relative)
        name = name.lstrip('/')
        name = name.strip()

        # Strip duplicate directory prefixes: my-app/my-app/src → my-app/src
        if '/' in name:
            segments = name.split('/')
            if len(segments) >= 2 and segments[0] == segments[1]:
                name = '/'.join(segments[1:])

        # Reject if too long (real filenames rarely exceed 200 chars)
        if len(name) > 200:
            return ""
        # Reject if it contains spaces (almost never valid in code paths)
        if ' ' in name:
            return ""
        # Reject placeholder paths like "path/to/file.py", "your/project/app.js"
        if parts:
            dir_parts = {p.lower() for p in parts[:-1]}  # all except filename
            if dir_parts & Executor._PLACEHOLDER_SEGMENTS:
                return ""

        return name

    @staticmethod
    def _looks_like_code(content: str) -> bool:
        """Return True if *content* looks like actual code rather than prose.

        Local models sometimes return natural-language paragraphs instead of
        code.  This heuristic catches the most obvious cases so we don't
        write garbage files to disk.
        """
        lines = content.strip().splitlines()
        if not lines:
            return False
        # Prose indicator: average line length > 120 chars (code is usually shorter)
        avg_len = sum(len(l) for l in lines) / len(lines)
        if avg_len > 120:
            return False
        # Prose indicator: majority of lines start with uppercase letter
        # (sentences) rather than code-like characters (import, def, {, <, etc.)
        if len(lines) >= 3:
            prose_starts = sum(
                1 for l in lines
                if l.strip() and l.strip()[0].isupper()
                and not l.strip().startswith(('I', 'If', 'In'))  # allow some keywords
                or l.strip().startswith(('The ', 'This ', 'It ', 'Please ', 'Here ',
                                         'A ', 'An ', 'I am ', 'I can '))
            )
            if prose_starts > len(lines) * 0.5:
                return False
        return True

    @staticmethod
    def parse_code_blocks(text: str) -> Dict[str, str]:
        """
        Parses Markdown code blocks with file markers.
        Expected: #### [FILE]: path/to/file.py followed by ```lang ... ```
        """
        files = {}
        pattern = r"####\s*\[FILE\]:\s*(.*?)\n```(?:\w+)?\n(.*?)\n```"
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            raw_filename = match.group(1).strip()
            filename = Executor._sanitize_filename(raw_filename)
            content = match.group(2)
            # Skip if filename still looks invalid
            if not filename or '/' not in filename and '.' not in filename:
                continue
            # Skip if content looks like prose rather than code
            if not Executor._looks_like_code(content):
                log.warning(f"[Executor] Skipping '{filename}': content looks like prose, not code")
                continue
            files[filename] = content
        return files

    @staticmethod
    def _try_add_file(files: Dict[str, str], filename: str, content: str):
        """Add file to *files* dict only if the content looks like real code."""
        if not Executor._looks_like_code(content):
            log.warning(f"[Executor] Skipping '{filename}': content looks like prose, not code")
            return
        files[filename] = content

    @staticmethod
    def parse_code_blocks_fuzzy(text: str) -> Dict[str, str]:
        """Fallback parser for LLM responses that don't follow the strict format.

        Handles common patterns:
        1. ``#### [FILE]:`` on the first line inside ANY code block (python, diff, etc.)
        2. Diff blocks with ``+`` prefixed ``#### [FILE]:`` lines
        3. Code blocks preceded by a line mentioning a file path
        4. Code blocks whose first line is a ``# filepath`` comment
        """
        files: Dict[str, str] = {}

        # ── Pattern 1: #### [FILE]: as first line inside any code block ──
        # The LLM sometimes wraps everything in ```python ... ``` but puts
        # the marker inside.  The content may be plain code or diff-style.
        for m in re.finditer(r"```(?:\w*)\n(.*?)```", text, re.DOTALL):
            block = m.group(1)
            first_line = block.split("\n", 1)[0].strip()
            fmatch = re.match(r"^(?:\+\s*)?####\s*\[FILE\]:\s*(.+)", first_line)
            if not fmatch:
                continue
            raw = fmatch.group(1).strip()
            filename = Executor._sanitize_filename(raw)
            if not filename or ('/' not in filename and '.' not in filename):
                continue
            rest = block.split("\n", 1)[1] if "\n" in block else ""
            # Check if the content uses diff markers (+/-/@@)
            has_diff = any(
                l.startswith(('+', '-', '@@'))
                for l in rest.splitlines()[:10] if l.strip()
            )
            if has_diff:
                content_lines = []
                for line in rest.splitlines():
                    if line.startswith('@@'):
                        continue
                    elif line.startswith('-'):
                        continue
                    elif line.startswith('+'):
                        content_lines.append(line[1:])
                    else:
                        content_lines.append(line)
                if content_lines:
                    Executor._try_add_file(files, filename, "\n".join(content_lines))
            else:
                Executor._try_add_file(files, filename, rest.rstrip("\n"))

        if files:
            return files

        # ── Pattern 2: diff blocks with +#### [FILE]: or +# filepath ──
        for m in re.finditer(r"```diff\n(.*?)```", text, re.DOTALL):
            block = m.group(1)
            fname_match = (
                re.search(r"^\+\s*####\s*\[FILE\]:\s*(.+)", block, re.MULTILINE)
                or re.search(r"^\+\s*#\s*(\S+\.\w{1,5})\s*$", block, re.MULTILINE)
            )
            if not fname_match:
                continue
            raw = fname_match.group(1).strip()
            filename = Executor._sanitize_filename(raw)
            if not filename or ('/' not in filename and '.' not in filename):
                continue
            content_lines = []
            past_header = False
            for line in block.splitlines():
                if not past_header:
                    if fname_match.group(0).strip() in line:
                        past_header = True
                    continue
                if line.startswith('+'):
                    content_lines.append(line[1:])
                elif not line.startswith('-') and not line.startswith('@@'):
                    content_lines.append(line)
            if content_lines:
                Executor._try_add_file(files, filename, "\n".join(content_lines))

        if files:
            return files

        # ── Pattern 3: text before code block mentions a file path ──
        for m in re.finditer(
            r"(?:^|\n)[^\n]*?(?:`([^`\n]+\.\w{1,5})`|(\b\S+\.\w{1,5}))\s*:?\s*\n"
            r"```(?:\w+)?\n(.*?)```",
            text, re.DOTALL,
        ):
            raw = (m.group(1) or m.group(2) or "").strip()
            filename = Executor._sanitize_filename(raw)
            if not filename or ('/' not in filename and '.' not in filename):
                continue
            Executor._try_add_file(files, filename, m.group(3).rstrip("\n"))

        if files:
            return files

        # ── Pattern 4: first line of code block is a # filepath comment ──
        for m in re.finditer(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL):
            block = m.group(1)
            first_line = block.split("\n", 1)[0].strip()
            fname_match = re.match(r"^#\s*(\S+\.\w{1,5})\s*$", first_line)
            if fname_match:
                filename = Executor._sanitize_filename(fname_match.group(1))
                if filename and ('/' in filename or '.' in filename):
                    rest = block.split("\n", 1)[1] if "\n" in block else ""
                    Executor._try_add_file(files, filename, rest.rstrip("\n"))

        return files

    # Dependency manifests and lock files that should NEVER be overwritten
    # by LLM-generated content.  These files are managed by package managers
    # and an LLM rewrite almost always drops dependencies, corrupting the
    # project.  They can still be *created* if they don't exist yet.
    _PROTECTED_FILENAMES: set[str] = {
        'package.json', 'package-lock.json',
        'yarn.lock', 'pnpm-lock.yaml',
        'go.mod', 'go.sum',
        'Cargo.toml', 'Cargo.lock',
        'Gemfile', 'Gemfile.lock',
        'composer.json', 'composer.lock',
        'Pipfile', 'Pipfile.lock', 'poetry.lock',
        'requirements.txt',
    }

    @staticmethod
    def write_files(files: Dict[str, str], base_dir: str = ".") -> List[str]:
        """
        Writes files to disk. Returns list of written file paths.

        For Python files, automatically creates ``__init__.py`` in every
        parent directory so that imports like ``from src.module import X``
        work out of the box.

        Protected manifest files (package.json, go.mod, etc.) are never
        overwritten if they already exist — LLM-generated replacements
        almost always drop dependencies and corrupt the project.
        """
        written = []
        init_dirs: set[str] = set()

        # Track basenames we've already written to detect path conflicts
        written_basenames: dict[str, str] = {}  # basename → full relative path

        for filename, content in files.items():
            filepath = os.path.join(base_dir, filename)
            dirpath = os.path.dirname(filepath)

            # Guard: never overwrite dependency manifests / lock files
            basename = os.path.basename(filename)
            if basename in Executor._PROTECTED_FILENAMES and os.path.isfile(filepath):
                log.warning(f"[Executor] Skipping protected file: {filepath} "
                            f"(already exists — overwriting could corrupt dependencies)")
                continue

            # Warn about potential path conflicts (same basename, different dir)
            if basename in written_basenames:
                prev_path = written_basenames[basename]
                if prev_path != filename:
                    log.warning(f"[Executor] Path conflict: '{filename}' has same "
                                f"basename as already-written '{prev_path}'")
            written_basenames[basename] = filename
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            log.info(f"Written: {filepath}")
            written.append(filepath)

            # Track directories that contain .py files
            if filename.endswith(".py") and dirpath and dirpath != base_dir:
                # Walk up to base_dir creating __init__.py at each level
                d = dirpath
                while d and d != base_dir and d != os.path.dirname(d):
                    init_dirs.add(d)
                    d = os.path.dirname(d)

        # Auto-create missing __init__.py so directories are importable packages
        for dirpath in sorted(init_dirs):
            init_path = os.path.join(dirpath, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, "w", encoding="utf-8") as f:
                    f.write("")
                log.info(f"Auto-created: {init_path}")
                written.append(init_path)

        return written

    # PowerShell cmdlets that cmd.exe cannot run directly
    _PS_CMDLETS = (
        'Get-ChildItem', 'Set-Location', 'Get-Content', 'Select-Object',
        'Where-Object', 'ForEach-Object', 'New-Item', 'Remove-Item',
        'Copy-Item', 'Move-Item', 'Test-Path', 'Invoke-WebRequest',
        'Write-Output', 'Out-File', 'Set-Content', 'Get-Command',
        'Get-Process', 'Stop-Process', 'Get-Service', 'Resolve-Path',
    )

    @staticmethod
    def _needs_powershell(cmd: str) -> bool:
        """Return True if *cmd* contains PowerShell-specific cmdlets."""
        for cmdlet in Executor._PS_CMDLETS:
            if cmdlet in cmd:
                return True
        return False

    # Known interactive commands and their non-interactive flags.
    # Each entry: (regex_pattern, flags_that_mean_already_handled, flag_to_append)
    _INTERACTIVE_REWRITES: list[tuple[str, tuple[str, ...], str]] = [
        (r'\bnpx\s+create-next-app\b', ('--yes',), ' --yes'),
        (r'\bnpm\s+init\b', ('--yes', '-y'), ' --yes'),
        (r'\byarn\s+init\b', ('--yes', '-y'), ' --yes'),
        (r'\bng\s+new\b', ('--defaults',), ' --defaults'),
        (r'\bcomposer\s+create-project\b', ('--no-interaction',), ' --no-interaction'),
    ]

    @staticmethod
    def _rewrite_interactive_cmd(cmd: str) -> str:
        """Rewrite known interactive commands to add non-interactive flags.

        Acts as a safety net so that even if the LLM forgets ``--yes``,
        the command won't hang waiting for stdin.
        """
        for pattern, existing_flags, add_flag in Executor._INTERACTIVE_REWRITES:
            if not re.search(pattern, cmd):
                continue
            if any(flag in cmd for flag in existing_flags):
                break  # already has a non-interactive flag
            cmd = cmd.rstrip() + add_flag
            log.info(f"[Executor] Auto-added non-interactive flag: {add_flag.strip()}")
            break
        return cmd

    @staticmethod
    def _is_likely_interactive(cmd: str) -> bool:
        """Return True if *cmd* matches patterns of commonly interactive CLI tools."""
        patterns = (
            r'\bcreate-next-app\b', r'\bcreate-react-app\b', r'\bcreate-vue\b',
            r'\bcreate-vite\b', r'\bnpm\s+init\b', r'\byarn\s+init\b',
            r'\bng\s+new\b', r'\bexpo\s+init\b', r'\bcomposer\s+create-project\b',
        )
        return any(re.search(p, cmd) for p in patterns)

    def run_command(self, cmd: str, env: dict | None = None,
                    timeout: int = 120, background: bool = False) -> Tuple[bool, str]:
        """
        Runs an arbitrary shell command. Returns (success, output).
        On Windows, auto-wraps PowerShell cmdlets so they don't fail
        in the default cmd.exe shell.

        If *background* is True, the process is started and tracked. The 
        method waits briefly (3s) to see if it crashes; if not, it returns success.
        """
        try:
            log.info(f"[Executor] Running {'background ' if background else ''}command: {cmd}")
            if os.name == 'nt' and Executor._needs_powershell(cmd):
                # Escape double quotes inside the command for PowerShell
                escaped = cmd.replace('"', '\\"')
                cmd = f'powershell -NoProfile -Command "{escaped}"'

            # Safety net: add non-interactive flags to known interactive commands
            cmd = Executor._rewrite_interactive_cmd(cmd)

            # Build environment — disable color codes and interactive prompts
            run_env = env if env else os.environ.copy()
            run_env.setdefault("NO_COLOR", "1")
            run_env.setdefault("FORCE_COLOR", "0")
            # Non-interactive: prevent CLI tools from prompting for input
            run_env.setdefault("CI", "true")
            run_env.setdefault("DEBIAN_FRONTEND", "noninteractive")
            run_env.setdefault("PIP_NO_INPUT", "1")
            run_env.setdefault("NPM_CONFIG_YES", "true")

            # Read as raw bytes and decode manually. On Windows, text=True
            # uses cp1252 by default, but most tools (Node.js, Jest, Go)
            # output UTF-8.  This mismatch causes empty/garbled output.
            proc = subprocess.Popen(
                cmd, shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=run_env,
            )

            if background:
                self._background_processes.append(proc)
                # Wait briefly to see if it dies instantly (e.g. port already in use)
                try:
                    stdout_bytes, _ = proc.communicate(timeout=3)
                    output = Executor._decode_output(stdout_bytes)
                    return proc.returncode == 0, output.strip()
                except subprocess.TimeoutExpired:
                    # Still running after 3s — assume background success for now
                    return True, "[Background process started]"

            stdout_bytes, _ = proc.communicate(timeout=timeout)
            output = Executor._decode_output(stdout_bytes)
            log.info(f"[Executor] Exit code: {proc.returncode}, "
                     f"output={len(output)} chars")

            if not output.strip() and proc.returncode != 0:
                # Command failed silently — provide useful context
                interactive_hint = ""
                if Executor._is_likely_interactive(cmd):
                    interactive_hint = (
                        "- The command may require interactive input (prompts) "
                        "which is not available. Try adding --yes, -y, or "
                        "--defaults flag.\n"
                    )
                output = (
                    f"Command `{cmd}` exited with code {proc.returncode} "
                    f"but produced no output.\n"
                    f"Possible causes:\n"
                    f"{interactive_hint}"
                    f"- The tool/binary is not installed or not on PATH\n"
                    f"- A required config file is missing\n"
                    f"- The command crashed before it could produce output"
                )
                log.warning(f"[Executor] {output}")

            return proc.returncode == 0, output.strip()
        except subprocess.TimeoutExpired:
            log.warning(f"[Executor] Command timed out after {timeout}s: {cmd}")
            proc.kill()
            stdout_bytes, _ = proc.communicate()
            output = Executor._decode_output(stdout_bytes)
            return False, f"Command timed out after {timeout} seconds.\n{output}".strip()
        except Exception as e:
            log.error(f"[Executor] Exception running command: {e}")
            return False, str(e)

    def cleanup(self):
        """Terminate all background processes."""
        if not self._background_processes:
            return
        log.info(f"[Executor] Cleaning up {len(self._background_processes)} background processes")
        for proc in self._background_processes:
            try:
                if proc.poll() is None:  # still running
                    # On Windows, taskkill is often more reliable for tree cleanup
                    if os.name == 'nt':
                         subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                                        capture_output=True)
                    else:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            proc.kill()
            except Exception as e:
                log.warning(f"[Executor] Failed to cleanup process {proc.pid}: {e}")
        self._background_processes.clear()

    @staticmethod
    def _decode_output(raw: bytes | None) -> str:
        """Decode subprocess output, trying UTF-8 first then system default."""
        if not raw:
            return ""
        try:
            return raw.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            pass
        try:
            import locale
            return raw.decode(locale.getpreferredencoding(False), errors="replace")
        except (UnicodeDecodeError, ValueError, LookupError):
            return raw.decode("ascii", errors="replace")

    def run_tests(self, test_command: str = "pytest") -> Tuple[bool, str]:
        """Run tests with the project root on PYTHONPATH.

        This ensures imports like ``from src.my_module import X`` resolve
        correctly regardless of how pytest discovers the tests.

        If the test runner binary is not found, returns a clear error
        message instead of a silent failure.
        """
        env = os.environ.copy()
        cwd = os.getcwd()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = cwd + (os.pathsep + existing if existing else "")

        # Quick check: does the test runner binary exist?
        runner = test_command.split()[0]  # e.g. "pytest", "npx", "go"
        import shutil
        if not shutil.which(runner, path=env.get("PATH")):
            # Provide install hints appropriate to the tool
            _system_tools = {"go", "cargo", "rustc", "javac", "java", "dotnet", "gcc", "g++"}
            if runner in _system_tools:
                hint = (f"`{runner}` must be installed manually from its "
                        f"official website (not available via pip/npm).")
            else:
                hint = (f"Install it first (e.g. `pip install {runner}` or "
                        f"`npm install --save-dev {runner}`).")
            msg = f"Test runner `{runner}` is not installed or not on PATH.\n{hint}"
            log.warning(f"[Executor] {msg}")
            return False, msg

        return self.run_command(test_command, env=env)

    # ── Missing-package auto-install ──

    # Well-known module → pip-package mappings where the names differ
    _MODULE_TO_PACKAGE = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "yaml": "pyyaml",
        "bs4": "beautifulsoup4",
        "dotenv": "python-dotenv",
        "gi": "PyGObject",
        "serial": "pyserial",
        "usb": "pyusb",
        "attr": "attrs",
        "dateutil": "python-dateutil",
        "jose": "python-jose",
        "jwt": "PyJWT",
        "magic": "python-magic",
        "lxml": "lxml",
    }

    # pytest fixtures that come from well-known plugins
    _FIXTURE_TO_PACKAGE = {
        "benchmark": "pytest-benchmark",
        "httpserver": "pytest-localserver",
        "mocker": "pytest-mock",
        "faker": "faker",
        "freezer": "pytest-freezegun",
        "celery_app": "pytest-celery",
        "async_client": "httpx",
        "anyio_backend": "anyio",
        "respx_mock": "respx",
    }

    @staticmethod
    def detect_missing_packages(test_output: str) -> List[str]:
        """Parse test output and return a list of pip packages to install.

        Detects:
        - ``ModuleNotFoundError: No module named 'xyz'``
        - ``ImportError: No module named 'xyz'``
        - ``fixture 'xyz' not found`` (pytest plugin fixtures)
        """
        packages: list[str] = []
        seen: set[str] = set()

        # Missing modules
        for m in re.finditer(
            r"(?:ModuleNotFoundError|ImportError):\s*No module named ['\"]([^'\"]+)['\"]",
            test_output,
        ):
            module = m.group(1).split(".")[0]  # top-level package
            pkg = Executor._MODULE_TO_PACKAGE.get(module, module)
            if pkg not in seen:
                packages.append(pkg)
                seen.add(pkg)

        # Missing pytest fixtures (plugin packages)
        for m in re.finditer(r"fixture ['\"](\w+)['\"] not found", test_output):
            fixture = m.group(1)
            pkg = Executor._FIXTURE_TO_PACKAGE.get(fixture)
            if pkg and pkg not in seen:
                packages.append(pkg)
                seen.add(pkg)

        return packages

    def install_packages(self, packages: List[str]) -> Tuple[bool, str]:
        """Install packages via pip. Returns (all_succeeded, combined_output)."""
        if not packages:
            return True, ""
        cmd = f"pip install {' '.join(packages)}"
        log.info(f"[Executor] Auto-installing: {cmd}")
        return self.run_command(cmd)

    @staticmethod
    def parse_step_dependencies(steps: List[str]) -> Tuple[List[str], Dict[int, set]]:
        """Parse ``(depends: N, M)`` markers from step text.

        Returns ``(cleaned_steps, dependencies)`` where *cleaned_steps*
        has dependency markers removed and *dependencies* maps each
        step index to a set of dependency indices (0-based).

        If **no** dependency markers are found at all, falls back to
        strict sequential ordering (each step depends on the previous)
        so that steps never run out of order.
        """
        cleaned: List[str] = []
        deps: Dict[int, set] = {}
        dep_pattern = re.compile(r"\s*\(depends?:\s*([\d,\s]+)\)\s*$", re.IGNORECASE)
        found_any_marker = False

        for idx, step in enumerate(steps):
            match = dep_pattern.search(step)
            if match:
                found_any_marker = True
                raw = match.group(1)
                # Parse comma-separated step numbers (1-based → 0-based)
                dep_indices = set()
                for num_str in raw.split(","):
                    num_str = num_str.strip()
                    if num_str.isdigit():
                        dep_indices.add(int(num_str) - 1)  # 1-based → 0-based
                deps[idx] = dep_indices
                cleaned.append(step[:match.start()].rstrip())
            else:
                cleaned.append(step)
                deps[idx] = set()

        # No markers at all → sequential: each step depends on its predecessor
        if not found_any_marker:
            for idx in range(1, len(cleaned)):
                deps[idx] = {idx - 1}

        return cleaned, deps
