"""
Diff display — compute and show colored unified diffs before writing files.
"""

import os
import difflib


def compute_diff(filepath: str, new_content: str, base_dir: str = ".") -> str | None:
    """Return unified diff string if file exists and content differs.

    Returns None if the file doesn't exist (new file) or content is unchanged.
    """
    full_path = os.path.join(base_dir, filepath)
    if not os.path.isfile(full_path):
        return None  # new file, no diff

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            old_content = f.read()
    except OSError:
        return None

    if old_content == new_content:
        return None  # unchanged

    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
        lineterm="",
    )
    diff_text = "\n".join(diff)
    return diff_text if diff_text.strip() else None


def format_colored_diff(diff_text: str) -> str:
    """Add ANSI colors to a unified diff string.

    Green for additions (+), red for deletions (-), cyan for @@ hunks.
    """
    lines = diff_text.splitlines()
    colored: list[str] = []
    for line in lines:
        if line.startswith("+++") or line.startswith("---"):
            colored.append(f"\033[1m{line}\033[0m")  # bold
        elif line.startswith("@@"):
            colored.append(f"\033[36m{line}\033[0m")  # cyan
        elif line.startswith("+"):
            colored.append(f"\033[32m{line}\033[0m")  # green
        elif line.startswith("-"):
            colored.append(f"\033[31m{line}\033[0m")  # red
        else:
            colored.append(line)
    return "\n".join(colored)


def compute_diffs(files: dict[str, str], base_dir: str = ".") -> list[tuple[str, str]]:
    """Compute diffs for all files. Returns list of (filepath, diff_text)."""
    diffs: list[tuple[str, str]] = []
    for filepath, content in files.items():
        diff = compute_diff(filepath, content, base_dir)
        if diff:
            diffs.append((filepath, diff))
    return diffs


def show_diffs(files: dict[str, str], base_dir: str = ".",
               log_only: bool = False) -> list[str]:
    """Compute and display diffs for all files.

    When *log_only* is True, diffs are returned but not printed (for --auto mode).
    Returns list of diff strings.
    """
    from .cli_display import log

    diffs = compute_diffs(files, base_dir)
    diff_strings: list[str] = []

    for filepath, diff_text in diffs:
        colored = format_colored_diff(diff_text)
        diff_strings.append(diff_text)

        if log_only:
            log.info(f"Diff for {filepath}:\n{diff_text}")
        else:
            print(f"\n{'─' * 60}")
            print(colored)

    new_files = [f for f in files if not os.path.isfile(os.path.join(base_dir, f))]
    if new_files and not log_only:
        print(f"\n  New files: {', '.join(new_files)}")

    return diff_strings
