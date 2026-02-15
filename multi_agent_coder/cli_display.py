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
        self.term_width = shutil.get_terminal_size((80, 24)).columns

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

    def render(self):
        """Redraw the full CLI display."""
        # Clear screen
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

        # Header
        print(f"{'═' * self.term_width}")
        print(f"  Multi-Agent Coder")
        print(f"  Task: {self.task}")
        print(f"{'═' * self.term_width}")
        print()

        # Progress bar
        print(f"  Progress: {self._progress_bar()}")
        print(f"  {self._token_summary()}")
        print(f"{'─' * self.term_width}")
        print()

        # Step list
        for i, step in enumerate(self.steps):
            icon = self.ICONS.get(step["status"], "?")
            type_tag = f"[{step['type']}]" if step["type"] != "?" else ""
            prefix = "→ " if i == self.current_step else "  "

            if i == self.current_step:
                # Active step — show full info
                print(f"  {prefix}{icon} Step {i+1}: {step['text']}")
                if type_tag:
                    print(f"       Type: {type_tag}")
                if "info" in step and step["info"]:
                    for line in step["info"]:
                        print(f"       {line}")
                if "tokens" in step:
                    t = step["tokens"]
                    print(f"       Tokens: ↑{t['sent']} ↓{t['recv']}")
            else:
                # Compact view for other steps
                status_label = ""
                if step["status"] == "done":
                    status_label = " ✔ done"
                elif step["status"] == "failed":
                    status_label = " ✘ failed"
                elif step["status"] == "skipped":
                    status_label = " – skipped"

                line = f"  {prefix}{icon} Step {i+1}: {step['text'][:60]}"
                if type_tag:
                    line += f" {type_tag}"
                line += status_label
                print(line)

        print()
        print(f"{'─' * self.term_width}")
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
            # Keep only last 5 info lines on screen
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
        print()
        if success:
            print("  ✔  All steps processed successfully!")
        else:
            print("  ✘  Some steps failed. Check logs for details.")
        print()
