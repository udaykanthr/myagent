import logging
import os
import re
import subprocess
import sys
import shutil
import tempfile
import threading
import time as _time
from datetime import datetime


class TokenTracker:
    """Global tracker for token usage and cost across all LLM calls."""

    def __init__(self, pricing: dict | None = None):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self.pricing = pricing or {}

    def record(self, prompt_tokens: int, completion_tokens: int, model_name: str | None = None):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.call_count += 1
        
        if model_name:
            self._calculate_cost(model_name, prompt_tokens, completion_tokens)

    def _calculate_cost(self, model_name: str, prompt: int, completion: int):
        # Simple match or regex match for pricing
        price_entry = None
        for pattern, prices in self.pricing.items():
            if pattern in model_name.lower():
                price_entry = prices
                break
        
        if price_entry:
            # Pricing is per 1M tokens
            cost = (prompt * price_entry["input"] / 1_000_000) + \
                   (completion * price_entry["output"] / 1_000_000)
            self.total_cost += cost

    @property
    def total_tokens(self):
        return self.total_prompt_tokens + self.total_completion_tokens


# Global singleton (pricing will be injected during CLI init)
token_tracker = TokenTracker()


def setup_logger(log_dir: str = ".agentchanti/logs") -> logging.Logger:
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

    # Spinner frames for waiting animation (ASCII-safe for Windows cp1252)
    _SPINNER_FRAMES = ["|", "/", "-", "\\"]
    _WAITING_PHRASES = [
        "Waiting for response",
        "Still thinking",
        "Processing",
        "Working on it",
    ]

    def __init__(self, task_description: str):
        self.task = task_description
        self.steps: list[dict] = []
        self.current_step = -1
        self.status_message = ""
        self._refresh_size()
        self._render_lock = threading.Lock()
        self._last_stream_render: float = 0.0
        self._header_end = 4  # compact header rows
        self._left_pane_width = 24
        self._llm_log: list[str] = []
        # Spinner state
        self._spinner_thread: threading.Thread | None = None
        self._spinner_stop = threading.Event()
        self._spinner_message: str = ""

    def _refresh_size(self):
        size = shutil.get_terminal_size((80, 24))
        self.term_width = size.columns
        self.term_height = size.lines

    def _center(self, text: str) -> str:
        """Center text within terminal width."""
        return text.center(self.term_width)

    def _wrap_task(self, text: str, width: int, max_lines: int = 2) -> list[str]:
        """Wrap and truncate task description to fit within given width.

        Large prompts are pre-truncated so only the first ``max_lines``
        worth of characters are processed.
        """
        if width <= 0 or not text:
            return []
        # Collapse to a single line and pre-truncate to avoid processing
        # extremely long prompt text (e.g. loaded from a file).
        text = " ".join(text.split())
        max_chars = max_lines * width + 20  # small slack for word-break
        if len(text) > max_chars:
            text = text[:max_chars]
        if not text:
            return []
        lines = []
        remaining = text
        for i in range(max_lines):
            if not remaining:
                break
            if len(remaining) <= width:
                lines.append(remaining)
                break
            if i == max_lines - 1:
                lines.append(remaining[:max(0, width - 3)] + "...")
            else:
                cut = remaining.rfind(' ', 0, width)
                if cut <= 0:
                    cut = width
                lines.append(remaining[:cut])
                remaining = remaining[cut:].lstrip()
        return lines

    # Regex to strip LLM special tokens: <|...|>, <|...|, <<...>>, [|...|] etc.
    _GIBBERISH_RE = re.compile(
        r'<\|[^|>]*\|?>|'       # <|token|> or <|token
        r'<<[^>]*>>|'            # <<token>>
        r'\[\|[^|\]]*\|?\]|'    # [|token|] or [|token]
        r'<\/?s>|'               # <s> </s> (sentence tokens)
        r'\[INST\]|\[\/INST\]|'  # [INST] [/INST] (Llama chat tokens)
        r'\[UNUSED_TOKEN_\d+\]'  # [UNUSED_TOKEN_145] etc.
    )
    # Characters that count as "readable" for the gibberish ratio check
    _READABLE_RE = re.compile(r'[a-zA-Z0-9\s]')

    @classmethod
    def _sanitize_line(cls, text: str) -> str:
        """Strip LLM special tokens and gibberish from a single line."""
        if not text:
            return ""
        original_len = len(text)
        # Remove known special token patterns
        cleaned = cls._GIBBERISH_RE.sub('', text).strip()
        if not cleaned:
            return ""
        # If most of the original was special tokens, the leftover is likely junk
        if original_len > 10 and len(cleaned) / original_len < 0.4:
            return ""
        # Strip orphaned quotes/brackets left after token removal
        cleaned = cleaned.strip("'\"[](){}<>,;:|`")
        cleaned = cleaned.strip()
        if not cleaned:
            return ""
        # Reject lines that are mostly non-readable characters
        readable = len(cls._READABLE_RE.findall(cleaned))
        if len(cleaned) > 3 and readable / len(cleaned) < 0.4:
            return ""
        return cleaned

    @staticmethod
    def extract_explanation(response: str) -> str:
        """Extract non-code explanation text from an LLM response."""
        lines = response.splitlines()
        result = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            if stripped.startswith("#### [FILE]:"):
                continue
            cleaned = CLIDisplay._sanitize_line(stripped)
            if cleaned:
                result.append(cleaned)
        return "\n".join(result)

    def add_llm_log(self, text: str, source: str = ""):
        """Add LLM thinking text to the log pane.

        *source*: agent label e.g. 'Coder', 'Reviewer', 'Tester', 'Diagnosis'
        """
        added = False
        if source:
            self._llm_log.append(f"[{source}]")
        for line in text.splitlines():
            cleaned = self._sanitize_line(line.strip())
            if cleaned:
                self._llm_log.append(f"  {cleaned}" if source else cleaned)
                added = True
        if added:
            self._llm_log.append("")  # blank separator
            self.render()

    def _build_log_lines(self, width: int, max_lines: int) -> list[str]:
        """Build wrapped log lines for the right pane (auto-scroll to latest)."""
        if width <= 0 or not self._llm_log:
            return []
        C = self.C_CYAN; G = self.C_GREEN; Y = self.C_YELLOW
        RED = self.C_RED; O = self.C_ORANGE; D = self.C_DIM; R = self.C_RESET
        _SRC_COLORS = {
            "Coder": C, "Reviewer": G, "Tester": Y, "Diagnosis": RED,
        }
        wrapped: list[str] = []
        for entry in self._llm_log:
            if not entry:
                wrapped.append("")
                continue
            # Color source header lines like "[Coder]"
            if entry.startswith("[") and "]" in entry:
                tag = entry[1:entry.index("]")]
                color = _SRC_COLORS.get(tag, O)
                wrapped.append(f"{color}▸ {entry}{R}")
                continue
            # Word-wrap regular lines
            remaining = entry
            while remaining and len(remaining) > width:
                cut = remaining.rfind(' ', 0, width)
                if cut <= 0:
                    cut = width
                wrapped.append(f"{D}{remaining[:cut]}{R}")
                remaining = remaining[cut:].lstrip()
            if remaining:
                wrapped.append(f"{D}{remaining}{R}")
        # Auto-scroll: show the last N lines
        return wrapped[-max_lines:] if len(wrapped) > max_lines else wrapped

    def _move_to(self, row: int):
        """Move cursor to a specific row (1-indexed)."""
        sys.stdout.write(f"\033[{row};1H")

    def set_steps(self, step_texts: list[str]):
        self._stop_spinner()
        self.steps = [
            {"text": t, "status": "pending", "type": "?"}
            for t in step_texts
        ]

    # ── Color palette ──
    C_ORANGE = "\033[38;5;208m"
    C_CYAN   = "\033[38;5;81m"
    C_GREEN  = "\033[38;5;114m"
    C_RED    = "\033[38;5;203m"
    C_YELLOW = "\033[38;5;221m"
    C_DIM    = "\033[38;5;243m"
    C_WHITE  = "\033[38;5;255m"
    C_BOLD   = "\033[1m"
    C_RESET  = "\033[0m"

    def _ansi_center(self, text: str) -> str:
        """Center text that contains ANSI codes within terminal width."""
        vis_len = len(re.sub(r'\033\[[0-9;]*m', '', text))
        pad = self.term_width - vis_len
        if pad <= 0:
            return text
        lpad = pad // 2
        return " " * lpad + text

    # ── Spinner animation ──

    def _start_spinner(self, message: str = ""):
        """Start a background spinner animation for the current waiting state."""
        self._stop_spinner()  # stop any existing one
        self._spinner_stop.clear()
        self._spinner_message = message
        self._spinner_thread = threading.Thread(
            target=self._spinner_loop, daemon=True)
        self._spinner_thread.start()

    def _stop_spinner(self):
        """Stop the background spinner if running."""
        if self._spinner_thread and self._spinner_thread.is_alive():
            self._spinner_stop.set()
            self._spinner_thread.join(timeout=1.0)
        self._spinner_thread = None

    def stop_spinner(self):
        """Public: stop the spinner before interactive prompts."""
        self._stop_spinner()

    def _spinner_loop(self):
        """Background loop that animates a spinner on the display."""
        C = self.C_CYAN; D = self.C_DIM; Y = self.C_YELLOW; R = self.C_RESET
        frame_idx = 0
        start_time = _time.monotonic()

        while not self._spinner_stop.is_set():
            elapsed = _time.monotonic() - start_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}:{secs:02d}" if mins else f"{secs}s"

            phrase_idx = int(elapsed // 8) % len(self._WAITING_PHRASES)
            phrase = self._WAITING_PHRASES[phrase_idx]

            spinner = self._SPINNER_FRAMES[frame_idx % len(self._SPINNER_FRAMES)]
            dots = "." * ((frame_idx % 3) + 1)
            base_msg = self._spinner_message or phrase
            base_clean = base_msg.rstrip(". ")

            anim_text = f"{Y}{spinner}{R} {C}{base_clean}{dots:<3}{R} {D}({time_str}){R}"

            try:
                with self._render_lock:
                    self._refresh_size()
                    header_end = self._header_end
                    sep_row = self.term_height - 2

                    if self.steps:
                        # Two-pane mode: show spinner in right pane
                        left_w = self._left_pane_width
                        right_col = left_w + 3
                        content_start = header_end + 3
                        content_height = max(0, sep_row - content_start)
                        spinner_row = content_start + content_height - 1

                        if content_start < spinner_row < sep_row:
                            sys.stdout.write(
                                f"\033[{spinner_row};{right_col}H\033[K"
                                f"{anim_text}")
                            sys.stdout.flush()

                    elif self.status_message:
                        avail_height = sep_row - header_end
                        mid_row = header_end + max(avail_height // 2 - 1, 1)
                        spinner_row = mid_row + 1
                        if spinner_row < sep_row:
                            self._move_to(spinner_row)
                            sys.stdout.write("\033[2K")
                            sys.stdout.write(self._ansi_center(
                                f"        {anim_text}"))
                            sys.stdout.flush()
            except (OSError, ValueError):
                break

            frame_idx += 1
            self._spinner_stop.wait(0.15)

    def _progress_bar_compact(self) -> str:
        """Short progress bar for status line."""
        total = len(self.steps)
        done = sum(1 for s in self.steps if s["status"] in ("done", "skipped"))
        pct = int((done / total) * 100) if total else 0
        bar_len = 15
        filled = int(bar_len * done / total) if total else 0
        G = self.C_GREEN; D = self.C_DIM; R = self.C_RESET
        bar = f"{G}{'█' * filled}{R}{D}{'░' * (bar_len - filled)}{R}"
        return f"{bar} {pct}% ({done}/{total})"

    def _vis_len(self, text: str) -> int:
        """Visible length of text after stripping ANSI codes."""
        return len(re.sub(r'\033\[[0-9;]*m', '', text))

    def _render_status_bar(self):
        """Render a status bar: progress centered, tokens+cost right-aligned."""
        w = self.term_width
        t = token_tracker
        D = self.C_DIM; W = self.C_WHITE; C = self.C_CYAN
        G = self.C_GREEN; R = self.C_RESET
        BG = "\033[48;5;236m"

        # Build the two parts
        progress = self._progress_bar_compact()

        right = (f"{D}↑{R}{W}{t.total_prompt_tokens:,}{R} "
                 f"{D}↓{R}{W}{t.total_completion_tokens:,}{R} "
                 f"{D}Σ{R}{C}{t.total_tokens:,}{R} "
                 f"{D}{t.call_count} calls{R}")
        if t.total_cost > 0:
            right += f"  {G}${t.total_cost:.4f}{R}"

        prog_vis = self._vis_len(progress)
        right_vis = self._vis_len(right)

        # Center the progress bar
        prog_lpad = max(0, (w - prog_vis) // 2)
        # Right-align token details (1 char margin)
        right_start = max(prog_lpad + prog_vis + 1, w - right_vis - 1)
        gap = max(1, right_start - prog_lpad - prog_vis)

        line = " " * prog_lpad + progress + " " * gap + right
        line_vis = self._vis_len(line)
        # Pad to fill full width for background
        line += " " * max(0, w - line_vis)

        print(f"{BG}{line}{R}", end="")

    def _build_step_lines(self) -> list[str]:
        """Build compact step list: icon Task N  status."""
        lines = []
        D = self.C_DIM; W = self.C_WHITE
        G = self.C_GREEN; RED = self.C_RED; Y = self.C_YELLOW
        B = self.C_BOLD; R = self.C_RESET

        for i, step in enumerate(self.steps):
            icon_raw = self.ICONS.get(step["status"], "?")
            icon_color = {"pending": D, "active": Y, "done": G,
                          "failed": RED, "skipped": D}.get(step["status"], D)
            icon = f"{icon_color}{icon_raw}{R}"

            if i == self.current_step:
                prefix = f" {Y}▸{R}"
            else:
                prefix = "  "

            label = f"Task {i + 1}"
            status = step["status"]
            name_color = W if status != "pending" else D

            if status == "pending":
                status_text = ""
            else:
                sc = {"active": Y, "done": G, "failed": RED,
                      "skipped": D}.get(status, D)
                status_text = f" {sc}{status}{R}"

            lines.append(f"{prefix} {icon} {name_color}{label}{R}{status_text}")
        return lines

    def render(self):
        """Redraw the full CLI display with positioned sections."""
        with self._render_lock:
            self._render_unlocked()

    def _render_unlocked(self):
        """Internal render (caller must hold _render_lock)."""
        self._refresh_size()
        w = self.term_width
        h = self.term_height
        O = self.C_ORANGE
        D = self.C_DIM
        W = self.C_WHITE
        Y = self.C_YELLOW
        R = self.C_RESET

        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')

        # ── TOP: Compact left-aligned brand + task description ──
        brand_text = "Agent Chanti"
        sub_text = "\u2501\u2501 Local Coder \u2501\u2501"
        brand_col = max(len(brand_text), len(sub_text)) + 4

        task_start = brand_col + 3
        task_width = max(0, w - task_start - 1)
        task_lines = self._wrap_task(self.task, task_width, max_lines=2)
        t1 = task_lines[0] if len(task_lines) > 0 else ""
        t2 = task_lines[1] if len(task_lines) > 1 else ""

        self._move_to(1)
        print(f"{O}{'═' * w}{R}")
        gap1 = " " * max(1, brand_col - len(brand_text) - 2)
        print(f"  {O}{self.C_BOLD}{brand_text}{R}{gap1}{D}\u2502{R} {W}{t1}{R}")
        gap2 = " " * max(1, brand_col - len(sub_text) - 2)
        print(f"  {D}{sub_text}{R}{gap2}{D}\u2502{R} {D}{t2}{R}")
        print(f"{O}{'═' * w}{R}")

        header_end = 4
        self._header_end = header_end

        # Reserve bottom: 1 line status bar + 1 separator
        status_row = h - 1
        sep_row = h - 2

        left_w = self._left_pane_width
        right_w = max(0, w - left_w - 3)  # 3 for " │ "

        if not self.steps and self.status_message:
            # No steps yet — show planning status centered
            avail = sep_row - header_end
            mid_row = header_end + max(avail // 2 - 1, 1)
            self._move_to(mid_row)
        else:
            # ── CENTER: Two-pane layout ──
            B = self.C_BOLD
            pane_row = header_end + 1
            self._move_to(pane_row)

            # Pane headers
            lh = f"  {B}{W}Steps{R}"
            # Show active task description beside LLM Thinking header
            active_desc = ""
            if 0 <= self.current_step < len(self.steps):
                active_desc = self.steps[self.current_step].get("text", "")
            rh_label = "LLM Thinking"
            if active_desc:
                # Truncate to 1 line: reserve space for label + separator
                sep = " \u2500 "
                max_desc = right_w - len(rh_label) - len(sep)
                if max_desc > 10:
                    desc = " ".join(active_desc.split())
                    if len(desc) > max_desc:
                        desc = desc[:max(0, max_desc - 3)] + "..."
                    rh = f"{B}{W}{rh_label}{R}{D}{sep}{Y}{desc}{R}"
                else:
                    rh = f"{B}{W}{rh_label}{R}"
            else:
                rh = f"{B}{W}{rh_label}{R}"
            lh_pad = " " * max(0, left_w - 7)  # 7 = len("  Steps")
            print(f"{lh}{lh_pad}{D}\u2502{R} {rh}")

            # Pane separator line
            hl = "\u2500"  # ─
            print(f"{D}{hl * left_w}\u253c{hl * (w - left_w - 1)}{R}")

            content_start = pane_row + 2
            content_height = max(0, sep_row - content_start)

            step_lines = self._build_step_lines()
            log_lines = self._build_log_lines(right_w, content_height)

            # Show scroll indicator if log overflows
            has_more = len(self._llm_log) > 0 and len(log_lines) == content_height

            for row_i in range(content_height):
                self._move_to(content_start + row_i)

                # Left pane
                if row_i < len(step_lines):
                    left = step_lines[row_i]
                else:
                    left = ""
                lv = self._vis_len(left)
                lpad = " " * max(0, left_w - lv)

                # Right pane
                if row_i < len(log_lines):
                    right = log_lines[row_i]
                else:
                    right = ""

                sys.stdout.write(f"{left}{lpad}{D}\u2502{R} {right}\033[K\n")

            # Scroll indicator at bottom of right pane
            if has_more:
                ind_row = sep_row - 1
                if ind_row > content_start:
                    ind_text = f"{D}\u2500\u2500\u25bc\u2500\u2500{R}"
                    ind_col = left_w + 3 + max(0, (right_w - 5) // 2)
                    sys.stdout.write(f"\033[{ind_row};{ind_col}H{ind_text}")

        # ── BOTTOM: Status bar (pinned) ──
        self._move_to(sep_row)
        print(f"{D}{'─' * w}{R}", end="")
        self._move_to(status_row)
        self._render_status_bar()

        sys.stdout.flush()

    def show_status(self, message: str):
        """Show a status message in the center (before steps are loaded)."""
        self.status_message = message
        self.render()
        self._start_spinner(message)

    def start_step(self, index: int, step_type: str = "?"):
        self.current_step = index
        self.steps[index]["status"] = "active"
        self.steps[index]["type"] = step_type
        self.steps[index]["info"] = []
        self.steps[index]["tokens"] = {"sent": 0, "recv": 0}
        self.render()

    def step_info(self, index: int, message: str):
        """Add a log line to the current step's display."""
        self._stop_spinner()
        if 0 <= index < len(self.steps):
            info_list = self.steps[index].get("info", [])
            if len(info_list) >= 5:
                info_list.pop(0)
            info_list.append(message)
            self.steps[index]["info"] = info_list
        self.render()
        # Restart spinner for messages that indicate waiting
        if any(kw in message.lower() for kw in (
            "generating", "coding", "classifying", "reviewing",
            "analyzing", "requesting", "running", "installing",
            "re-planning", "retrying",
        )):
            self._start_spinner(message)

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
        self._stop_spinner()
        self.steps[index]["status"] = status
        self.render()

    def finish(self, success: bool = True):
        """Render a full completion screen with header and centred report."""
        self._stop_spinner()
        self._refresh_size()
        w = self.term_width
        h = self.term_height
        O = self.C_ORANGE; D = self.C_DIM; W = self.C_WHITE
        G = self.C_GREEN; RED = self.C_RED; C = self.C_CYAN
        Y = self.C_YELLOW; B = self.C_BOLD; R = self.C_RESET

        # ── Clear and redraw header ──
        os.system('cls' if os.name == 'nt' else 'clear')

        brand_text = "Agent Chanti"
        sub_text = "\u2501\u2501 Local Coder \u2501\u2501"

        self._move_to(1)
        print(f"{O}{'═' * w}{R}")
        print(self._ansi_center(f"{O}{B}{brand_text}{R}"))
        print(self._ansi_center(f"{D}{sub_text}{R}"))
        print(f"{O}{'═' * w}{R}")

        header_end = 4

        # ── Build report lines ──
        t = token_tracker
        report_lines: list[str] = []

        # Line 1: success / fail status
        if success:
            status_line = f"{G}✔  All tasks completed successfully!{R}"
        else:
            status_line = f"{RED}✘  Some tasks failed. Check logs for details.{R}"
        report_lines.append(status_line)

        # Blank spacer
        report_lines.append("")

        # Line 2+: token & cost summary
        token_line = (
            f"{D}Total Tokens:{R} {C}{t.total_tokens:,}{R}    "
            f"{D}Input Tokens:{R} {W}{t.total_prompt_tokens:,}{R}    "
            f"{D}Output Tokens:{R} {W}{t.total_completion_tokens:,}{R}"
        )
        report_lines.append(token_line)

        if t.total_cost > 0:
            cost_line = f"{D}Estimated Cost:{R} {G}${t.total_cost:.4f}{R}"
            report_lines.append(cost_line)

        # ── Centre the block vertically in the remaining space ──
        avail = h - header_end - 2  # leave 2 rows margin at bottom
        block_height = len(report_lines)
        start_row = header_end #+ max((avail - block_height) // 2, 1)

        for i, line in enumerate(report_lines):
            self._move_to(start_row + i)
            sys.stdout.write("\033[2K")  # clear line
            sys.stdout.write(self._ansi_center(line))

        # Park cursor below the block
        self._move_to(start_row + block_height + 1)
        sys.stdout.flush()

    def budget_check(self, limit: float) -> bool:
        """Check if total cost exceeds limit. Returns True if over budget."""
        if limit > 0 and token_tracker.total_cost >= limit:
            with self._render_lock:
                self._move_to(self.term_height - 3)
                RED = self.C_RED; R = self.C_RESET
                msg = f"{RED}⚠  BUDGET EXCEEDED (${token_tracker.total_cost:.4f} >= ${limit:.2f})  ⚠{R}"
                print(self._ansi_center(msg))
            return True
        return False

    # ── Interactive prompts (temporarily exit full-screen mode) ──

    def update_streaming_progress(self, step_idx: int, tokens: int):
        """Throttled progress update during streaming (max every 0.5s)."""
        now = _time.monotonic()
        if now - self._last_stream_render < 0.5:
            return
        self._last_stream_render = now
        self.step_info(step_idx, f"Generating... ({tokens} tokens)")

    @staticmethod
    def prompt_plan_approval(steps: list[str],
                             use_tui: bool = False) -> tuple[str, list[int], list[str] | None]:
        """Show numbered steps and ask user to approve, replan, or edit.

        Returns ``(action, removed_indices, edited_steps)`` where *action*
        is ``"approve"``, ``"replan"``, or ``"edit"``.

        The TUI editor (Textual-based) is always available via [E]dit.
        A system text editor fallback is also available via [T]ext.
        """
        print("\n" + "=" * 60)
        print("  PROPOSED PLAN")
        print("=" * 60)
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        print("=" * 60)
        print("  [A]pprove  |  [R]eplan  |  [E]dit (TUI)  |  [T]ext editor")
        print()

        while True:
            choice = input("  Your choice: ").strip().lower()
            if choice in ("a", "approve"):
                return "approve", [], None
            elif choice in ("r", "replan"):
                return "replan", [], None
            elif choice in ("e", "edit"):
                # Launch the Textual TUI editor
                try:
                    from .tui_editor import launch_tui_editor
                    edited = launch_tui_editor(steps)
                    if edited:
                        return "edit", [], edited
                    # TUI cancelled — fall through
                    print("  Edit cancelled or no changes.")
                except Exception as e:
                    print(f"  TUI editor failed ({e}). Try [T] for text editor.")
                    log.warning(f"TUI editor exception: {e}")
                # Re-show the menu
                print()
                print("  [A]pprove  |  [R]eplan  |  [E]dit (TUI)  |  [T]ext editor")
                print()
            elif choice in ("t", "text"):
                # System text editor (vi/nano/notepad)
                edited = CLIDisplay._edit_plan_in_editor(steps)
                if edited:
                    return "edit", [], edited
                else:
                    print("  No changes detected or empty plan.")
            else:
                print("  Invalid choice. Use A, R, E, or T.")

    @staticmethod
    def _edit_plan_in_editor(steps: list[str]) -> list[str] | None:
        """Write *steps* to a temp file, open a system editor, and return
        the modified steps after the user saves and closes the editor.

        Uses ``notepad`` on Windows and ``vi`` on other operating systems.
        Returns ``None`` if the resulting file is empty.
        """
        # Build file content with numbered steps
        content = "# Edit the plan below. One step per line.\n"
        content += "# Lines starting with '#' are ignored.\n"
        content += "# You may add, remove, or reorder steps.\n\n"
        for i, step in enumerate(steps, 1):
            content += f"{i}. {step}\n"

        # Write to a temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="plan_", delete=False, encoding="utf-8"
        )
        try:
            tmp.write(content)
            tmp.close()

            # Choose editor based on OS
            if os.name == "nt":
                editor = "notepad"
            else:
                editor = os.environ.get("EDITOR", "vi")

            print(f"\n  Opening plan in {editor}...")
            print("  Save and close the editor when done.\n")

            subprocess.call([editor, tmp.name])

            # Read back the edited file
            with open(tmp.name, "r", encoding="utf-8") as f:
                edited_content = f.read()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        # Parse edited content into step list
        edited_steps: list[str] = []
        for line in edited_content.splitlines():
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Strip leading number + dot  (e.g. "1. Do something" -> "Do something")
            line = re.sub(r"^\d+\.\s*", "", line)
            if line:
                edited_steps.append(line)

        return edited_steps if edited_steps else None

    @staticmethod
    def prompt_resume(checkpoint_state: dict) -> bool:
        """Show checkpoint info and ask whether to resume.

        Returns ``True`` to resume, ``False`` to start fresh.
        """
        print("\n" + "=" * 60)
        print("  CHECKPOINT FOUND")
        print("=" * 60)
        print(f"  Task: {checkpoint_state.get('task', '?')}")
        completed = checkpoint_state.get("completed_step", -1)
        total = len(checkpoint_state.get("steps", []))
        print(f"  Progress: {completed + 1}/{total} steps completed")
        print(f"  Language: {checkpoint_state.get('language', '?')}")
        print("=" * 60)
        print("  [R]esume  |  [S]tart fresh")
        print()

        while True:
            choice = input("  Your choice: ").strip().lower()
            if choice in ("r", "resume"):
                return True
            elif choice in ("s", "start", "fresh"):
                return False
            else:
                print("  Invalid choice. Use R or S.")

    @staticmethod
    def prompt_git_action(action: str) -> str:
        """Ask user about a git action.

        *action* is ``"complete"`` (task succeeded) or ``"failed"`` (task failed).
        Returns ``"commit"``, ``"rollback"``, or ``"skip"``.
        """
        print("\n" + "=" * 60)
        if action == "complete":
            print("  TASK COMPLETED — Git Options")
            print("=" * 60)
            print("  [C]ommit changes  |  [S]kip (leave uncommitted)")
        else:
            print("  TASK FAILED — Git Options")
            print("=" * 60)
            print("  [R]ollback to checkpoint  |  [C]ommit as-is  |  [S]kip")
        print()

        while True:
            choice = input("  Your choice: ").strip().lower()
            if choice in ("c", "commit"):
                return "commit"
            elif choice in ("r", "rollback") and action != "complete":
                return "rollback"
            elif choice in ("s", "skip"):
                return "skip"
            else:
                print("  Invalid choice. Try again.")
