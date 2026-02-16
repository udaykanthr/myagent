"""
Git integration — checkpoint branches, commit, and rollback utilities.
"""

import re
import subprocess
import time


def _run_git(cmd: str) -> tuple[bool, str]:
    """Run a git command and return ``(success, output)``."""
    try:
        result = subprocess.run(
            f"git {cmd}",
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output
    except Exception as e:
        return False, str(e)


def is_git_repo() -> bool:
    """Return ``True`` if the CWD is inside a git repository."""
    ok, _ = _run_git("rev-parse --is-inside-work-tree")
    return ok


def has_changes() -> bool:
    """Return ``True`` if there are uncommitted changes (staged or unstaged)."""
    ok, output = _run_git("status --porcelain")
    return ok and bool(output.strip())


def get_current_branch() -> str:
    """Return the name of the current branch, or ``"HEAD"`` on detached HEAD."""
    ok, output = _run_git("rev-parse --abbrev-ref HEAD")
    return output.strip() if ok else "HEAD"


def create_checkpoint_branch(task_summary: str) -> str | None:
    """Create a checkpoint branch and auto-commit any dirty state.

    Returns the branch name on success, or ``None`` on failure.
    """
    # Sanitize task into a branch-safe slug
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", task_summary)[:40].strip("-").lower()
    timestamp = int(time.time())
    branch_name = f"agentchanti/pre-{slug}-{timestamp}"

    # Auto-commit dirty state first
    if has_changes():
        _run_git("add -A")
        _run_git('commit -m "AgentChanti: auto-save before task"')

    ok, _ = _run_git(f"branch {branch_name}")
    if not ok:
        return None
    return branch_name


def commit_changes(message: str) -> tuple[bool, str]:
    """Stage all changes and commit with *message*."""
    _run_git("add -A")
    return _run_git(f'commit -m "{message}"')


def rollback_to_branch(branch_name: str) -> tuple[bool, str]:
    """Restore working tree to the state of *branch_name*."""
    ok1, out1 = _run_git(f"checkout {branch_name} -- .")
    ok2, out2 = _run_git("clean -fd")
    return ok1 and ok2, f"{out1}\n{out2}"


def delete_checkpoint_branch(branch_name: str) -> tuple[bool, str]:
    """Delete a checkpoint branch after successful completion."""
    return _run_git(f"branch -D {branch_name}")
