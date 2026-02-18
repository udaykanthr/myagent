"""
TUI Plan Editor — Textual-based interactive plan editing.

Provides a modern, robust GUI-like experience in the terminal with
clickable buttons, inline editing, and step reordering.

Falls back to a lightweight ANSI menu editor if Textual is not installed.
"""

from __future__ import annotations
import sys
import os


def launch_tui_editor(steps: list[str]) -> list[str] | None:
    """Launch the TUI plan editor.

    Returns the edited steps list, or None if the user cancelled.
    """
    # Try Textual first (modern, robust, works everywhere)
    try:
        return _textual_plan_editor(steps)
    except ImportError:
        _log_warning("Textual not installed — falling back to ANSI editor. "
                     "Install with: pip install textual")
    except Exception as e:
        _log_warning(f"Textual TUI failed: {e}")

    # Fallback: ANSI-based editor (works everywhere, no deps)
    try:
        return _ansi_plan_editor(steps)
    except Exception as e:
        _log_warning(f"ANSI editor failed: {e}")
        return None


def _log_warning(msg: str):
    """Try to log; don't fail if logger isn't available."""
    try:
        from .cli_display import log
        log.warning(f"[TUI] {msg}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
#  Textual Plan Editor — modern GUI-like TUI
# ══════════════════════════════════════════════════════════════════

def _textual_plan_editor(steps: list[str]) -> list[str] | None:
    """Launch the Textual-based plan editor."""
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import (
        Button, Footer, Header, Input, Label, ListItem, ListView, Static,
    )
    from textual.binding import Binding
    from textual import on

    class StepItem(ListItem):
        """A single step in the plan list."""

        def __init__(self, step_text: str, step_num: int) -> None:
            super().__init__()
            self.step_text = step_text
            self.step_num = step_num

        def compose(self) -> ComposeResult:
            with Horizontal(classes="step-row"):
                yield Label(f" {self.step_num:2d}. ", classes="step-num")
                yield Label(self.step_text, classes="step-text")
                yield Button("✎", id="edit", variant="default", classes="step-btn")
                yield Button("▲", id="up", variant="default", classes="step-btn")
                yield Button("▼", id="down", variant="default", classes="step-btn")
                yield Button("✕", id="delete", variant="error", classes="step-btn")

    class PlanEditorApp(App):
        """Textual app for editing the execution plan."""

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
        #step-list {
            height: 1fr;
            margin: 1 2;
            border: round #444;
            padding: 1;
        }
        .step-row {
            height: 3;
            align: left middle;
            padding: 0 1;
        }
        .step-num {
            width: 6;
            color: #e9c46a;
            text-style: bold;
        }
        .step-text {
            width: 1fr;
            color: #f8f8f2;
        }
        .step-btn {
            min-width: 4;
            width: 4;
            height: 3;
            margin: 0 0 0 1;
        }
        ListItem {
            background: #16213e;
            margin: 0 0 1 0;
        }
        ListItem:hover {
            background: #0f3460;
        }
        ListView > ListItem.--highlight {
            background: #0f3460;
        }
        #bottom-bar {
            dock: bottom;
            height: auto;
            padding: 1 2;
        }
        #action-buttons {
            height: 3;
            align: center middle;
        }
        #action-buttons Button {
            margin: 0 2;
            min-width: 16;
        }
        #add-bar {
            height: 3;
            margin: 0 2 1 2;
            align: left middle;
        }
        #add-input {
            width: 1fr;
        }
        #add-btn {
            width: 12;
            margin: 0 0 0 1;
        }
        #edit-bar {
            height: 3;
            margin: 0 2 1 2;
            align: left middle;
            display: none;
        }
        #edit-input {
            width: 1fr;
        }
        #edit-save-btn {
            width: 12;
            margin: 0 0 0 1;
        }
        #edit-cancel-btn {
            width: 12;
            margin: 0 0 0 1;
        }
        #step-count {
            text-align: center;
            color: #888;
            margin: 0 0 1 0;
        }
        """

        BINDINGS = [
            Binding("escape", "cancel", "Cancel"),
            Binding("ctrl+s", "approve", "Approve Plan"),
        ]

        def __init__(self, initial_steps: list[str]) -> None:
            super().__init__()
            self._steps = list(initial_steps)
            self._result: list[str] | None = None
            self._editing_index: int = -1

        def compose(self) -> ComposeResult:
            yield Static(
                " ━━  AgentChanti — Plan Editor  ━━ ",
                id="title-bar",
            )
            yield ListView(
                *[StepItem(s, i + 1) for i, s in enumerate(self._steps)],
                id="step-list",
            )
            yield Static("", id="step-count")
            with Horizontal(id="edit-bar"):
                yield Input(placeholder="Edit step text...", id="edit-input")
                yield Button("Save", id="edit-save-btn", variant="success")
                yield Button("Cancel", id="edit-cancel-btn", variant="default")
            with Horizontal(id="add-bar"):
                yield Input(
                    placeholder="Type a new step and press Enter or click Add...",
                    id="add-input",
                )
                yield Button("+ Add", id="add-btn", variant="success")
            with Horizontal(id="action-buttons"):
                yield Button(
                    "✔ Approve Plan", id="approve-btn", variant="success"
                )
                yield Button(
                    "✕ Cancel", id="cancel-btn", variant="error"
                )
            yield Footer()

        def on_mount(self) -> None:
            self._update_count()

        def _rebuild_list(self) -> None:
            """Rebuild the ListView from self._steps."""
            lv = self.query_one("#step-list", ListView)
            lv.clear()
            for i, s in enumerate(self._steps):
                lv.append(StepItem(s, i + 1))
            self._update_count()

        def _update_count(self) -> None:
            self.query_one("#step-count", Static).update(
                f"  {len(self._steps)} step(s)"
            )

        def _get_step_index(self, button: Button) -> int:
            """Get the step index from a button inside a StepItem."""
            item = button.ancestors_with_self
            for ancestor in item:
                if isinstance(ancestor, StepItem):
                    lv = self.query_one("#step-list", ListView)
                    children = list(lv.children)
                    for idx, child in enumerate(children):
                        if child is ancestor:
                            return idx
                    break
            return -1

        @on(Button.Pressed, "#edit")
        def on_edit(self, event: Button.Pressed) -> None:
            idx = self._get_step_index(event.button)
            if 0 <= idx < len(self._steps):
                self._editing_index = idx
                edit_input = self.query_one("#edit-input", Input)
                edit_input.value = self._steps[idx]
                # Show edit bar, hide add bar
                self.query_one("#edit-bar").styles.display = "block"
                self.query_one("#add-bar").styles.display = "none"
                edit_input.focus()

        @on(Button.Pressed, "#edit-save-btn")
        def on_edit_save(self, event: Button.Pressed) -> None:
            text = self.query_one("#edit-input", Input).value.strip()
            if text and 0 <= self._editing_index < len(self._steps):
                self._steps[self._editing_index] = text
                self._rebuild_list()
            self._editing_index = -1
            self.query_one("#edit-bar").styles.display = "none"
            self.query_one("#add-bar").styles.display = "block"

        @on(Button.Pressed, "#edit-cancel-btn")
        def on_edit_cancel(self, event: Button.Pressed) -> None:
            self._editing_index = -1
            self.query_one("#edit-bar").styles.display = "none"
            self.query_one("#add-bar").styles.display = "block"

        @on(Button.Pressed, "#up")
        def on_move_up(self, event: Button.Pressed) -> None:
            idx = self._get_step_index(event.button)
            if idx > 0:
                self._steps[idx], self._steps[idx - 1] = (
                    self._steps[idx - 1], self._steps[idx]
                )
                self._rebuild_list()

        @on(Button.Pressed, "#down")
        def on_move_down(self, event: Button.Pressed) -> None:
            idx = self._get_step_index(event.button)
            if 0 <= idx < len(self._steps) - 1:
                self._steps[idx], self._steps[idx + 1] = (
                    self._steps[idx + 1], self._steps[idx]
                )
                self._rebuild_list()

        @on(Button.Pressed, "#delete")
        def on_delete(self, event: Button.Pressed) -> None:
            idx = self._get_step_index(event.button)
            if 0 <= idx < len(self._steps):
                self._steps.pop(idx)
                self._rebuild_list()

        @on(Button.Pressed, "#add-btn")
        def on_add(self, event: Button.Pressed) -> None:
            self._do_add()

        @on(Input.Submitted, "#add-input")
        def on_add_submit(self, event: Input.Submitted) -> None:
            self._do_add()

        def _do_add(self) -> None:
            add_input = self.query_one("#add-input", Input)
            text = add_input.value.strip()
            if text:
                self._steps.append(text)
                add_input.value = ""
                self._rebuild_list()

        @on(Input.Submitted, "#edit-input")
        def on_edit_submit(self, event: Input.Submitted) -> None:
            text = event.value.strip()
            if text and 0 <= self._editing_index < len(self._steps):
                self._steps[self._editing_index] = text
                self._rebuild_list()
            self._editing_index = -1
            self.query_one("#edit-bar").styles.display = "none"
            self.query_one("#add-bar").styles.display = "block"

        @on(Button.Pressed, "#approve-btn")
        def on_approve(self, event: Button.Pressed) -> None:
            self._result = list(self._steps) if self._steps else None
            self.exit()

        @on(Button.Pressed, "#cancel-btn")
        def on_cancel_btn(self, event: Button.Pressed) -> None:
            self._result = None
            self.exit()

        def action_cancel(self) -> None:
            self._result = None
            self.exit()

        def action_approve(self) -> None:
            self._result = list(self._steps) if self._steps else None
            self.exit()

    app = PlanEditorApp(steps)
    app.run()
    return app._result


# ══════════════════════════════════════════════════════════════════
#  ANSI Plan Editor — no dependencies, works everywhere (fallback)
# ══════════════════════════════════════════════════════════════════

def _clear_screen():
    """Clear terminal screen cross-platform."""
    os.system('cls' if os.name == 'nt' else 'clear')


def _ansi_plan_editor(steps: list[str]) -> list[str] | None:
    """Lightweight plan editor using plain print + input.

    Supports: view, edit, add, delete, reorder, approve, cancel.
    """
    edited = list(steps)

    while True:
        _clear_screen()
        # Header
        print("\033[1;36m" + "═" * 60 + "\033[0m")
        print("\033[1;36m  AgentChanti — Plan Editor\033[0m")
        print("\033[1;36m" + "═" * 60 + "\033[0m")
        print()

        # Steps
        for i, step in enumerate(edited):
            num = f"  \033[33m{i+1:2d}.\033[0m "
            print(f"{num}{step}")
        print()
        print(f"\033[90m  {len(edited)} steps\033[0m")
        print()

        # Menu
        print("\033[7m Commands: \033[0m")
        print("  \033[1ma\033[0m  Add step      "
              "\033[1me\033[0m  Edit step     "
              "\033[1md\033[0m  Delete step")
        print("  \033[1mu\033[0m  Move up       "
              "\033[1mn\033[0m  Move down     "
              "\033[1mq\033[0m  Cancel")
        print("  \033[1;32mEnter\033[0m  Approve plan")
        print()

        try:
            choice = input("  \033[1m>\033[0m ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if choice == "" or choice == "approve":
            return edited if edited else None

        elif choice == "q" or choice == "cancel":
            return None

        elif choice == "a":
            try:
                after = input("  Add after step # (0 for beginning): ").strip()
                pos = int(after)
                text = input("  Step text: ").strip()
                if text:
                    edited.insert(pos, text)
            except (ValueError, EOFError, KeyboardInterrupt):
                pass

        elif choice == "e":
            try:
                num = input(f"  Edit step # (1-{len(edited)}): ").strip()
                idx = int(num) - 1
                if 0 <= idx < len(edited):
                    print(f"  Current: {edited[idx]}")
                    new_text = input("  New text (Enter to keep): ").strip()
                    if new_text:
                        edited[idx] = new_text
            except (ValueError, EOFError, KeyboardInterrupt):
                pass

        elif choice == "d":
            try:
                num = input(f"  Delete step # (1-{len(edited)}): ").strip()
                idx = int(num) - 1
                if 0 <= idx < len(edited):
                    removed = edited.pop(idx)
                    print(f"  \033[31mRemoved:\033[0m {removed}")
            except (ValueError, EOFError, KeyboardInterrupt):
                pass

        elif choice == "u":
            try:
                num = input(f"  Move step # up (1-{len(edited)}): ").strip()
                idx = int(num) - 1
                if 0 < idx < len(edited):
                    edited[idx], edited[idx-1] = edited[idx-1], edited[idx]
            except (ValueError, EOFError, KeyboardInterrupt):
                pass

        elif choice == "n":
            try:
                num = input(f"  Move step # down (1-{len(edited)}): ").strip()
                idx = int(num) - 1
                if 0 <= idx < len(edited) - 1:
                    edited[idx], edited[idx+1] = edited[idx+1], edited[idx]
            except (ValueError, EOFError, KeyboardInterrupt):
                pass
