"""
TUI Plan Editor — curses-based interactive plan editing.

Provides arrow-key navigation, inline editing, reordering, and
step deletion. Falls back to the text editor if curses is unavailable.
"""

from __future__ import annotations


def launch_tui_editor(steps: list[str]) -> list[str] | None:
    """Launch the TUI plan editor.

    Returns the edited steps list, or None if:
    - The user cancelled (pressed 'q')
    - curses is not available (caller should fall back to text editor)
    """
    try:
        import curses
        editor = PlanEditorTUI(steps)
        return curses.wrapper(editor._main_loop)
    except (ImportError, Exception):
        return None  # fallback to text editor


class PlanEditorTUI:
    """Curses-based interactive plan editor.

    Key bindings:
        ↑ / k      Move cursor up
        ↓ / j      Move cursor down
        e          Edit step text
        d          Delete step
        a          Add new step after cursor
        K (shift)  Move step up
        J (shift)  Move step down
        Enter      Approve plan
        q          Cancel (discard changes)
    """

    def __init__(self, steps: list[str]):
        self.steps = list(steps)
        self.cursor = 0
        self.scroll_offset = 0

    def _main_loop(self, stdscr) -> list[str] | None:
        import curses
        curses.curs_set(0)
        curses.use_default_colors()

        # Define colors if terminal supports it
        if curses.has_colors():
            curses.init_pair(1, curses.COLOR_CYAN, -1)     # header
            curses.init_pair(2, curses.COLOR_GREEN, -1)     # selected
            curses.init_pair(3, curses.COLOR_YELLOW, -1)    # step number
            curses.init_pair(4, curses.COLOR_WHITE, -1)     # normal
            curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_WHITE)  # help bar

        while True:
            self._draw(stdscr)
            key = stdscr.getch()

            if key in (ord('q'), 27):  # q or Escape
                return None

            elif key in (curses.KEY_ENTER, 10, 13):  # Enter
                return self.steps if self.steps else None

            elif key in (curses.KEY_UP, ord('k')):
                self.cursor = max(0, self.cursor - 1)

            elif key in (curses.KEY_DOWN, ord('j')):
                self.cursor = min(len(self.steps) - 1, self.cursor + 1)

            elif key == ord('K'):  # Move step up
                if self.cursor > 0:
                    self.steps[self.cursor], self.steps[self.cursor - 1] = \
                        self.steps[self.cursor - 1], self.steps[self.cursor]
                    self.cursor -= 1

            elif key == ord('J'):  # Move step down
                if self.cursor < len(self.steps) - 1:
                    self.steps[self.cursor], self.steps[self.cursor + 1] = \
                        self.steps[self.cursor + 1], self.steps[self.cursor]
                    self.cursor += 1

            elif key == ord('d'):  # Delete step
                if self.steps:
                    self.steps.pop(self.cursor)
                    self.cursor = min(self.cursor, len(self.steps) - 1)

            elif key == ord('a'):  # Add new step
                new_text = self._edit_text(stdscr, "")
                if new_text:
                    self.steps.insert(self.cursor + 1, new_text)
                    self.cursor += 1

            elif key == ord('e'):  # Edit step
                if self.steps:
                    new_text = self._edit_text(stdscr, self.steps[self.cursor])
                    if new_text is not None:
                        self.steps[self.cursor] = new_text

    def _draw(self, stdscr):
        import curses
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Header
        header = " AgentChanti — Plan Editor "
        try:
            stdscr.addstr(0, 0, header.center(width), curses.color_pair(1) | curses.A_BOLD)
        except curses.error:
            pass

        # Calculate visible range
        visible_lines = height - 5  # reserve for header + footer
        if self.cursor < self.scroll_offset:
            self.scroll_offset = self.cursor
        elif self.cursor >= self.scroll_offset + visible_lines:
            self.scroll_offset = self.cursor - visible_lines + 1

        # Steps
        for i, step in enumerate(self.steps):
            if i < self.scroll_offset or i >= self.scroll_offset + visible_lines:
                continue

            row = 2 + (i - self.scroll_offset)
            if row >= height - 2:
                break

            is_selected = (i == self.cursor)
            num_str = f" {i + 1:2d}. "
            step_str = step[:width - 8]  # truncate to fit

            try:
                if is_selected:
                    stdscr.addstr(row, 0, "→", curses.color_pair(2) | curses.A_BOLD)
                    stdscr.addstr(row, 1, num_str, curses.color_pair(3) | curses.A_BOLD)
                    stdscr.addstr(row, len(num_str) + 1, step_str,
                                 curses.color_pair(2) | curses.A_BOLD)
                else:
                    stdscr.addstr(row, 0, " ")
                    stdscr.addstr(row, 1, num_str, curses.color_pair(3))
                    stdscr.addstr(row, len(num_str) + 1, step_str, curses.color_pair(4))
            except curses.error:
                pass

        # Help bar
        help_text = " ↑↓:Nav  e:Edit  d:Del  a:Add  K/J:Reorder  Enter:Approve  q:Cancel "
        try:
            stdscr.addstr(height - 1, 0, help_text[:width].ljust(width),
                         curses.color_pair(5))
        except curses.error:
            pass

        # Step count
        count_str = f" {len(self.steps)} steps "
        try:
            stdscr.addstr(height - 2, 0, count_str, curses.color_pair(1))
        except curses.error:
            pass

        stdscr.refresh()

    def _edit_text(self, stdscr, current_text: str) -> str | None:
        """Simple inline text editor. Returns edited text or None if cancelled."""
        import curses
        curses.curs_set(1)
        height, width = stdscr.getmaxyx()

        # Show edit prompt at bottom
        prompt = "Edit step (Enter=save, Esc=cancel): "
        try:
            stdscr.addstr(height - 3, 0, prompt, curses.color_pair(1))
        except curses.error:
            pass

        # Create text buffer
        text = list(current_text)
        pos = len(text)
        max_len = width - len(prompt) - 2

        while True:
            # Display current text
            display_text = "".join(text)[:max_len]
            try:
                stdscr.addstr(height - 3, len(prompt), display_text.ljust(max_len))
                stdscr.move(height - 3, len(prompt) + min(pos, max_len))
            except curses.error:
                pass
            stdscr.refresh()

            key = stdscr.getch()

            if key == 27:  # Escape
                curses.curs_set(0)
                return None

            elif key in (curses.KEY_ENTER, 10, 13):
                curses.curs_set(0)
                result = "".join(text).strip()
                return result if result else None

            elif key in (curses.KEY_BACKSPACE, 127, 8):
                if pos > 0:
                    text.pop(pos - 1)
                    pos -= 1

            elif key == curses.KEY_LEFT:
                pos = max(0, pos - 1)

            elif key == curses.KEY_RIGHT:
                pos = min(len(text), pos + 1)

            elif key == curses.KEY_HOME:
                pos = 0

            elif key == curses.KEY_END:
                pos = len(text)

            elif 32 <= key <= 126:  # printable ASCII
                text.insert(pos, chr(key))
                pos += 1
