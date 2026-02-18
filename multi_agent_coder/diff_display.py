"""
Diff display — compute and show colored unified diffs before writing files.

Includes a Textual-based interactive diff viewer that pauses execution so the
user can review changes and approve/reject before files are written to disk.
"""

from __future__ import annotations

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


# ══════════════════════════════════════════════════════════════════
#  Interactive Diff Approval — Textual TUI
# ══════════════════════════════════════════════════════════════════

def prompt_diff_approval(files: dict[str, str], base_dir: str = ".",
                         auto: bool = False) -> bool:
    """Show diffs in an interactive Textual viewer and wait for approval.

    Returns ``True`` if the user approves (or if running in auto mode).
    Returns ``False`` if the user rejects.
    """
    from .cli_display import log

    diffs = compute_diffs(files, base_dir)
    new_files = [f for f in files if not os.path.isfile(os.path.join(base_dir, f))]

    # Nothing to review
    if not diffs and not new_files:
        return True

    # Auto mode — log diffs and approve
    if auto:
        for filepath, diff_text in diffs:
            log.info(f"[auto] Diff for {filepath}:\n{diff_text}")
        if new_files:
            log.info(f"[auto] New files: {', '.join(new_files)}")
        return True

    # Try Textual TUI
    try:
        return _textual_diff_approval(diffs, new_files, files)
    except ImportError:
        log.warning("Textual not installed — falling back to console diff approval.")
    except Exception as e:
        log.warning(f"Textual diff viewer failed: {e}")

    # Fallback: console-based approval
    return _console_diff_approval(diffs, new_files)


def _format_rich_diff(diff_text: str) -> str:
    """Convert unified diff text to Rich markup for Textual display."""
    lines = diff_text.splitlines()
    markup_lines: list[str] = []
    for line in lines:
        # Escape Rich markup characters in the line content
        escaped = line.replace("[", "\\[")
        if line.startswith("+++") or line.startswith("---"):
            markup_lines.append(f"[bold white]{escaped}[/bold white]")
        elif line.startswith("@@"):
            markup_lines.append(f"[cyan]{escaped}[/cyan]")
        elif line.startswith("+"):
            markup_lines.append(f"[green]{escaped}[/green]")
        elif line.startswith("-"):
            markup_lines.append(f"[red]{escaped}[/red]")
        else:
            markup_lines.append(escaped)
    return "\n".join(markup_lines)


def _textual_diff_approval(diffs: list[tuple[str, str]],
                           new_files: list[str],
                           files: dict[str, str]) -> bool:
    """Launch a Textual app to display diffs and get approval."""
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, VerticalScroll
    from textual.widgets import Button, Footer, Static
    from textual.binding import Binding

    class DiffApprovalApp(App):
        """Interactive diff viewer with approve/reject."""

        CSS = """
        Screen {
            background: $surface;
        }
        #title-bar {
            dock: top;
            height: 3;
            background: #1a1a2e;
            color: #e94560;
            text-align: center;
            padding: 1;
            text-style: bold;
        }
        #diff-scroll {
            height: 1fr;
            margin: 1 2;
            border: round #444;
            padding: 1;
        }
        .file-header {
            color: #e9c46a;
            text-style: bold;
            margin: 1 0 0 0;
        }
        .diff-content {
            margin: 0 0 1 0;
        }
        .new-file-section {
            color: #2a9d8f;
            text-style: bold;
            margin: 1 0;
        }
        #action-buttons {
            dock: bottom;
            height: 3;
            align: center middle;
            padding: 0 2;
        }
        #action-buttons Button {
            margin: 0 2;
            min-width: 20;
        }
        #summary {
            dock: bottom;
            height: 1;
            text-align: center;
            color: #888;
        }
        """

        BINDINGS = [
            Binding("a", "approve", "Approve"),
            Binding("ctrl+s", "approve", "Approve"),
            Binding("escape", "reject", "Reject"),
            Binding("r", "reject", "Reject"),
        ]

        def __init__(self, diffs: list[tuple[str, str]],
                     new_files: list[str],
                     files: dict[str, str]) -> None:
            super().__init__()
            self._diffs = diffs
            self._new_files = new_files
            self._files = files
            self._approved: bool = False

        def compose(self) -> ComposeResult:
            file_count = len(self._diffs) + len(self._new_files)
            yield Static(
                f" ━━  Diff Review — {file_count} file(s) changed  ━━ ",
                id="title-bar",
            )
            with VerticalScroll(id="diff-scroll"):
                for filepath, diff_text in self._diffs:
                    yield Static(
                        f"[bold yellow]{'─' * 58}[/bold yellow]\n"
                        f"[bold yellow]  {filepath}[/bold yellow]",
                        classes="file-header",
                    )
                    yield Static(
                        _format_rich_diff(diff_text),
                        classes="diff-content",
                    )
                if self._new_files:
                    yield Static(
                        f"[bold #2a9d8f]{'─' * 58}[/bold #2a9d8f]\n"
                        f"[bold #2a9d8f]  New files:[/bold #2a9d8f]",
                        classes="new-file-section",
                    )
                    for nf in self._new_files:
                        lines = self._files.get(nf, "")
                        line_count = len(lines.splitlines()) if lines else 0
                        yield Static(
                            f"  [green]+ {nf}[/green]  ({line_count} lines)",
                        )
            summary_parts = []
            if self._diffs:
                summary_parts.append(f"{len(self._diffs)} modified")
            if self._new_files:
                summary_parts.append(f"{len(self._new_files)} new")
            yield Static(
                f"  {' | '.join(summary_parts)}  —  "
                f"Press [bold]A[/bold] to approve, [bold]R[/bold] or Esc to reject",
                id="summary",
            )
            with Horizontal(id="action-buttons"):
                yield Button(
                    "✔ Approve", id="approve-btn", variant="success",
                )
                yield Button(
                    "✕ Reject", id="reject-btn", variant="error",
                )
            yield Footer()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "approve-btn":
                self._approved = True
                self.exit()
            elif event.button.id == "reject-btn":
                self._approved = False
                self.exit()

        def action_approve(self) -> None:
            self._approved = True
            self.exit()

        def action_reject(self) -> None:
            self._approved = False
            self.exit()

    app = DiffApprovalApp(diffs, new_files, files)
    app.run()
    return app._approved


def _console_diff_approval(diffs: list[tuple[str, str]],
                           new_files: list[str]) -> bool:
    """Fallback console-based diff approval when Textual is unavailable."""
    print("\n" + "=" * 60)
    print("  DIFF REVIEW")
    print("=" * 60)

    for filepath, diff_text in diffs:
        print(f"\n{'─' * 60}")
        print(format_colored_diff(diff_text))

    if new_files:
        print(f"\n  New files: {', '.join(new_files)}")

    print("\n" + "=" * 60)
    print("  [A]pprove  |  [R]eject")
    print()

    while True:
        try:
            choice = input("  Your choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if choice in ("a", "approve"):
            return True
        elif choice in ("r", "reject"):
            return False
        else:
            print("  Invalid choice. Use A or R.")
