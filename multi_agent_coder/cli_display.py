import logging
import os
import sys
import shutil
from datetime import datetime


class TokenTracker:
    """Global tracker for token usage across all LLM calls."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0

    def record(self, prompt_tokens: int, completion_tokens: int):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.call_count += 1

    @property
    def total_tokens(self):
        return self.total_prompt_tokens + self.total_completion_tokens


# Global singleton
token_tracker = TokenTracker()


def setup_logger(log_dir: str = "logs") -> logging.Logger:
    """Creates a file logger. All verbose output goes here."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"agent_{timestamp}.log")

    logger = logging.getLogger("multi_agent_coder")
    logger.setLevel(logging.DEBUG)

    # File handler — captures everything
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)

    return logger


# Global logger instance
log = setup_logger()


class CLIDisplay:
    """Manages the terminal CLI progress display."""

    ICONS = {
        "pending":  "○",
        "active":   "◉",
        "done":     "✔",
        "failed":   "✘",
        "skipped":  "–",
    }

    def __init__(self, task_description: str):
        self.task = task_description
        self.steps: list[dict] = []
        self.current_step = -1
        self._refresh_size()

    def _refresh_size(self):
        size = shutil.get_terminal_size((80, 24))
        self.term_width = size.columns
        self.term_height = size.lines

    def _center(self, text: str) -> str:
        """Center text within terminal width."""
        return text.center(self.term_width)

    def _move_to(self, row: int):
        """Move cursor to a specific row (1-indexed)."""
        sys.stdout.write(f"\033[{row};1H")

    def set_steps(self, step_texts: list[str]):
        self.steps = [
            {"text": t, "status": "pending", "type": "?"}
            for t in step_texts
        ]

    def _progress_bar(self) -> str:
        total = len(self.steps)
        done = sum(1 for s in self.steps if s["status"] in ("done", "skipped"))
        pct = int((done / total) * 100) if total else 0
        bar_len = 30
        filled = int(bar_len * done / total) if total else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        return f"[{bar}] {pct}% ({done}/{total})"

    def _token_summary(self) -> str:
        t = token_tracker
        return (
            f"Tokens  ↑ sent: {t.total_prompt_tokens}  "
            f"↓ recv: {t.total_completion_tokens}  "
            f"Σ total: {t.total_tokens}  "
            f"({t.call_count} calls)"
        )

    def _build_step_lines(self) -> list[str]:
        """Build the step list lines for bottom section."""
        lines = []
        for i, step in enumerate(self.steps):
            icon = self.ICONS.get(step["status"], "?")
            type_tag = f"[{step['type']}]" if step["type"] != "?" else ""
            prefix = " → " if i == self.current_step else "   "

            if i == self.current_step:
                lines.append(f"{prefix}{icon} Step {i+1}: {step['text']}")
                if type_tag:
                    lines.append(f"        Type: {type_tag}")
                if "info" in step and step["info"]:
                    for info_line in step["info"]:
                        lines.append(f"        {info_line}")
                if "tokens" in step:
                    t = step["tokens"]
                    lines.append(f"        Tokens: ↑{t['sent']} ↓{t['recv']}")
            else:
                status_label = ""
                if step["status"] == "done":
                    status_label = " ✔ done"
                elif step["status"] == "failed":
                    status_label = " ✘ failed"
                elif step["status"] == "skipped":
                    status_label = " – skipped"

                line = f"{prefix}{icon} Step {i+1}: {step['text'][:55]}"
                if type_tag:
                    line += f" {type_tag}"
                line += status_label
                lines.append(line)
        return lines

    def render(self):
        """Redraw the full CLI display with positioned sections."""
        self._refresh_size()
        w = self.term_width
        h = self.term_height

        # Clear screen
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

        # ── TOP: Title centered ──
        self._move_to(1)
        print("═" * w)
        print(self._center("Agent Chanti - Local coder"))
        print(self._center(f"Task: {self.task}"))
        print("═" * w)

        # ── MIDDLE: Progress + Tokens centered ──
        step_lines = self._build_step_lines()
        steps_height = len(step_lines) + 2  # +2 for separator + padding

        # Middle zone: between header (row 5) and steps section
        middle_start = 5
        bottom_start = max(h - steps_height, middle_start + 6)
        mid_row = (middle_start + bottom_start) // 2 - 2

        self._move_to(mid_row)
        progress_text = f"Progress: {self._progress_bar()}"
        print(self._center(progress_text))
        print()
        print(self._center(self._token_summary()))

        # Step-level tokens for active step
        if 0 <= self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            if "tokens" in step:
                t = step["tokens"]
                step_tok = f"Step {self.current_step+1} tokens: ↑{t['sent']}  ↓{t['recv']}"
                print()
                print(self._center(step_tok))

        # ── BOTTOM: Steps list ──
        self._move_to(bottom_start)
        print("─" * w)
        for line in step_lines:
            print(line)

        sys.stdout.flush()

    def start_step(self, index: int, step_type: str = "?"):
        self.current_step = index
        self.steps[index]["status"] = "active"
        self.steps[index]["type"] = step_type
        self.steps[index]["info"] = []
        self.steps[index]["tokens"] = {"sent": 0, "recv": 0}
        self.render()

    def step_info(self, index: int, message: str):
        """Add a log line to the current step's display."""
        if 0 <= index < len(self.steps):
            info_list = self.steps[index].get("info", [])
            if len(info_list) >= 5:
                info_list.pop(0)
            info_list.append(message)
            self.steps[index]["info"] = info_list
        self.render()

    def step_tokens(self, index: int, sent: int, recv: int):
        """Update token counts for the active step."""
        if 0 <= index < len(self.steps):
            t = self.steps[index].get("tokens", {"sent": 0, "recv": 0})
            t["sent"] += sent
            t["recv"] += recv
            self.steps[index]["tokens"] = t
        self.render()

    def complete_step(self, index: int, status: str = "done"):
        """Mark step as done/failed/skipped."""
        self.steps[index]["status"] = status
        self.render()

    def finish(self, success: bool = True):
        self._move_to(self.term_height - 1)
        if success:
            print(self._center("✔  All steps processed successfully!"))
        else:
            print(self._center("✘  Some steps failed. Check logs for details."))
        print()
