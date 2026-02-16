import os
import re
import subprocess
from typing import Dict, List, Tuple
from .cli_display import log


class Executor:

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

    @staticmethod
    def _sanitize_filename(raw: str) -> str:
        """Clean up LLM-generated filenames that may contain junk."""
        name = raw.strip()
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
        return name.strip()

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
            files[filename] = content
        return files

    @staticmethod
    def write_files(files: Dict[str, str], base_dir: str = ".") -> List[str]:
        """
        Writes files to disk. Returns list of written file paths.

        For Python files, automatically creates ``__init__.py`` in every
        parent directory so that imports like ``from src.module import X``
        work out of the box.
        """
        written = []
        init_dirs: set[str] = set()

        for filename, content in files.items():
            filepath = os.path.join(base_dir, filename)
            dirpath = os.path.dirname(filepath)
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

    @staticmethod
    def run_command(cmd: str, env: dict | None = None) -> Tuple[bool, str]:
        """
        Runs an arbitrary shell command. Returns (success, output).
        On Windows, auto-wraps PowerShell cmdlets so they don't fail
        in the default cmd.exe shell.
        """
        try:
            if os.name == 'nt' and Executor._needs_powershell(cmd):
                # Escape double quotes inside the command for PowerShell
                escaped = cmd.replace('"', '\\"')
                cmd = f'powershell -NoProfile -Command "{escaped}"'

            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                check=False, env=env,
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output.strip()
        except Exception as e:
            return False, str(e)

    @staticmethod
    def run_tests(test_command: str = "pytest") -> Tuple[bool, str]:
        """Run tests with the project root on PYTHONPATH.

        This ensures imports like ``from src.my_module import X`` resolve
        correctly regardless of how pytest discovers the tests.
        """
        env = os.environ.copy()
        cwd = os.getcwd()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = cwd + (os.pathsep + existing if existing else "")
        return Executor.run_command(test_command, env=env)

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
