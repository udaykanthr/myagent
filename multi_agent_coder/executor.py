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
        """
        written = []
        for filename, content in files.items():
            filepath = os.path.join(base_dir, filename)
            dirpath = os.path.dirname(filepath)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            log.info(f"Written: {filepath}")
            written.append(filepath)
        return written

    @staticmethod
    def run_command(cmd: str) -> Tuple[bool, str]:
        """
        Runs an arbitrary shell command. Returns (success, output).
        """
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=False
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output.strip()
        except Exception as e:
            return False, str(e)

    @staticmethod
    def run_tests(test_command: str = "pytest") -> Tuple[bool, str]:
        """Shortcut to run tests."""
        return Executor.run_command(test_command)
