"""
Step classification and command extraction utilities.
"""

import re

from ..cli_display import CLIDisplay, token_tracker


def _classify_step(step_text: str, llm_client, display: CLIDisplay, step_idx: int) -> str:
    display.step_info(step_idx, "Classifying step...")
    prompt = (
        "Classify the following task step into exactly one category.\n"
        "Reply with ONLY one word: CMD, CODE, TEST, or IGNORE\n\n"
        "  CMD    = anything that can be done by running shell commands, including:\n"
        "           - scanning or listing files/directories (ls, tree, find, dir)\n"
        "           - reading or inspecting file contents (cat, type, head)\n"
        "           - checking project structure or dependencies\n"
        "           - installing packages (pip install, npm install)\n"
        "           - running scripts, builds, or any CLI tool\n"
        "           - navigating or exploring a codebase\n\n"
        "  CODE   = create or modify source code files (writing new code or editing existing files)\n\n"
        "  TEST   = write or run unit/integration tests\n\n"
        "  IGNORE = not actionable by a program (e.g. open an IDE, review code visually,\n"
        "           think about architecture, make a decision)\n\n"
        f"Step: {step_text}\n\n"
        "Category:"
    )
    sent_before = token_tracker.total_prompt_tokens
    recv_before = token_tracker.total_completion_tokens

    response = llm_client.generate_response(prompt).strip().upper()

    sent_delta = token_tracker.total_prompt_tokens - sent_before
    recv_delta = token_tracker.total_completion_tokens - recv_before
    display.step_tokens(step_idx, sent_delta, recv_delta)

    for keyword in ("IGNORE", "CMD", "CODE", "TEST"):
        if keyword in response:
            return keyword
    return "CODE"


def _is_file_path(text: str) -> bool:
    """Return True if *text* looks like a bare file/directory path, not a command."""
    text = text.strip()
    # No spaces usually means it's a path, not a command with arguments
    # Exception: single-word commands like "pytest" are handled by _looks_like_command
    if ' ' in text:
        return False
    # Looks like a file path: contains slashes and/or has a file extension
    has_sep = '/' in text or '\\' in text
    has_ext = bool(re.search(r'\.\w{1,5}$', text))
    return has_sep or has_ext


def _looks_like_command(text: str) -> bool:
    """Return True if *text* looks like an executable shell command."""
    text = text.strip()
    if not text:
        return False
    # Reject bare file paths
    if _is_file_path(text):
        return False
    # Extract the first token, splitting on whitespace AND shell operators
    # so that "echo.>file" splits to "echo." and "type nul > file" splits to "type"
    first_token = re.split(r'[\s>|&;<]', text)[0].lower()
    # Strip trailing .exe suffix (not rstrip which eats individual chars)
    if first_token.endswith('.exe'):
        first_token = first_token[:-4]
    # Strip trailing dots (CMD echo. syntax)
    first_token = first_token.rstrip('.')

    known_commands = {
        'pip', 'pip3', 'python', 'python3', 'py',
        'npm', 'npx', 'node', 'yarn', 'pnpm',
        'go', 'cargo', 'rustc', 'mvn', 'gradle', 'javac', 'java',
        'ruby', 'bundle', 'gem', 'rspec',
        'git', 'docker', 'make', 'cmake',
        'mkdir', 'rmdir', 'del', 'copy', 'move', 'ren', 'type', 'dir',
        'ls', 'cat', 'cp', 'mv', 'rm', 'find', 'grep', 'chmod', 'chown',
        'cd', 'echo', 'set', 'export', 'source', 'touch',
        'curl', 'wget', 'ssh', 'scp',
        'apt', 'apt-get', 'brew', 'choco', 'yum', 'dnf', 'pacman',
        'powershell', 'pwsh', 'cmd',
        'pytest', 'jest', 'tox', 'mypy', 'flake8', 'black', 'ruff',
    }
    return first_token in known_commands


def _extract_commands_from_text(text: str) -> list[str]:
    """Extract shell commands from *text*, handling both triple- and single-backtick blocks.

    Prefers triple-backtick code blocks (```cmd, ```bash, ```shell, ```)
    over single-backtick inline code.  Filters out file paths and non-commands.
    """
    commands: list[str] = []
    seen: set[str] = set()

    # 1. Triple-backtick code blocks (```lang\n...\n```)
    for m in re.finditer(r"```(?:\w*)\n(.*?)```", text, re.DOTALL):
        block = m.group(1).strip()
        for line in block.splitlines():
            line = line.strip()
            if line and _looks_like_command(line) and line not in seen:
                commands.append(line)
                seen.add(line)

    # 2. Single-backtick inline commands (`...`)
    for m in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", text):
        cmd = m.group(1).strip()
        if cmd and _looks_like_command(cmd) and cmd not in seen:
            commands.append(cmd)
            seen.add(cmd)

    return commands


def _extract_command_from_step(step_text: str) -> str | None:
    """Extract an inline command from a step description.

    Only matches backtick content that looks like a real command,
    skipping bare file paths like ``tests/test_main.py``.
    """
    for m in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", step_text):
        candidate = m.group(1).strip()
        if _looks_like_command(candidate):
            return candidate
    return None
