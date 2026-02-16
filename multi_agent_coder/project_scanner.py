"""
Project scanner — reads existing project structure and key files
so the planner agent has awareness of the current codebase.
"""

import os

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "venv", ".venv", "env",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    "target", "bin", "obj", ".idea", ".vscode", ".eggs",
    "site-packages", ".next", ".nuxt", "coverage", "htmlcov",
}

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".exe", ".dll", ".so", ".dylib", ".o", ".obj",
    ".class", ".jar", ".war", ".zip", ".tar", ".gz", ".bz2",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".bmp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".woff", ".woff2", ".ttf", ".eot",
    ".lock", ".db", ".sqlite", ".sqlite3",
}

KEY_FILENAMES = {
    "requirements.txt", "setup.py", "setup.cfg", "pyproject.toml",
    "package.json", "tsconfig.json",
    "Cargo.toml", "go.mod", "go.sum",
    "Gemfile", "pom.xml", "build.gradle",
    "Makefile", "Dockerfile", "docker-compose.yml",
    "main.py", "app.py", "index.js", "index.ts",
    "manage.py", "settings.py",
    ".env.example", "README.md",
}

_MAX_KEY_FILES = 15
_MAX_LINES_PER_FILE = 100


def scan_project(directory: str = ".") -> dict:
    """Walk *directory* and return project structure info.

    Returns::

        {
            "tree": str,          # indented directory tree
            "key_files": dict,    # filename → first N lines of content
            "file_count": int,
            "languages": dict,    # extension → count
        }
    """
    tree_lines: list[str] = []
    key_files: dict[str, str] = {}
    file_count = 0
    lang_counts: dict[str, int] = {}

    abs_dir = os.path.abspath(directory)

    for root, dirs, files in os.walk(abs_dir):
        # Filter out skipped directories (in-place so os.walk respects it)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)

        rel_root = os.path.relpath(root, abs_dir)
        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
        indent = "  " * depth
        dir_name = os.path.basename(root) if rel_root != "." else "."
        tree_lines.append(f"{indent}{dir_name}/")

        for fname in sorted(files):
            _, ext = os.path.splitext(fname)
            if ext in SKIP_EXTENSIONS:
                continue

            file_count += 1
            tree_lines.append(f"{indent}  {fname}")

            # Count language extensions
            if ext:
                lang_counts[ext] = lang_counts.get(ext, 0) + 1

            # Read key files (up to limit)
            if fname in KEY_FILENAMES and len(key_files) < _MAX_KEY_FILES:
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, abs_dir)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        lines = []
                        for line_no, line in enumerate(f):
                            if line_no >= _MAX_LINES_PER_FILE:
                                lines.append(f"... (truncated at {_MAX_LINES_PER_FILE} lines)")
                                break
                            lines.append(line.rstrip())
                        key_files[rel_path] = "\n".join(lines)
                except OSError:
                    pass

    return {
        "tree": "\n".join(tree_lines),
        "key_files": key_files,
        "file_count": file_count,
        "languages": lang_counts,
    }


def format_scan_for_planner(scan_result: dict, max_chars: int = 6000) -> str:
    """Format scan results into a compact context string for the planner prompt.

    Truncates to *max_chars* to avoid overwhelming the context window.
    """
    parts: list[str] = []

    # Project tree (cap at ~40% of budget)
    tree = scan_result.get("tree", "")
    tree_budget = max_chars * 2 // 5
    if len(tree) > tree_budget:
        tree = tree[:tree_budget] + "\n... (tree truncated)"
    parts.append(f"## Project Structure\n```\n{tree}\n```")

    # Key files
    parts.append("\n## Key Files")
    remaining = max_chars - len("\n".join(parts))
    for fpath, content in scan_result.get("key_files", {}).items():
        entry = f"\n### {fpath}\n```\n{content}\n```"
        if len(entry) > remaining:
            break
        parts.append(entry)
        remaining -= len(entry)

    # Stats
    fc = scan_result.get("file_count", 0)
    parts.append(f"\n**{fc} files detected.**")

    result = "\n".join(parts)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... (truncated)"
    return result
