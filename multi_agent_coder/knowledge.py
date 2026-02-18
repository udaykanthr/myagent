"""
Knowledge Base — persists learnings across runs for smarter planning.

Stores patterns, fixes, conventions, and dependencies discovered during
pipeline execution. Loaded into the planner prompt for context.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .cli_display import log


@dataclass
class KnowledgeEntry:
    """A single piece of project knowledge."""
    category: str        # "pattern", "fix", "convention", "dependency"
    content: str
    source_task: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class KnowledgeBase:
    """Persistent project knowledge store.

    Saves to ``.agentchanti/knowledge.json`` by default.
    """

    MAX_ENTRIES = 100  # keep knowledge base manageable

    def __init__(self, path: str = ".agentchanti/knowledge.json"):
        self._path = path
        self._entries: list[KnowledgeEntry] = []
        self.load()

    def load(self) -> list[KnowledgeEntry]:
        """Load knowledge from disk."""
        if not os.path.isfile(self._path):
            return self._entries

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._entries = [
                    KnowledgeEntry(**entry) for entry in data
                    if isinstance(entry, dict)
                ]
            log.debug(f"[KnowledgeBase] Loaded {len(self._entries)} entries")
        except (json.JSONDecodeError, OSError, TypeError) as e:
            log.warning(f"[KnowledgeBase] Load error: {e}")

        return self._entries

    def save(self):
        """Persist knowledge to disk."""
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        try:
            data = [asdict(e) for e in self._entries[-self.MAX_ENTRIES:]]
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            log.debug(f"[KnowledgeBase] Saved {len(data)} entries")
        except OSError as e:
            log.warning(f"[KnowledgeBase] Save error: {e}")

    def add(self, category: str, content: str, source_task: str):
        """Add a knowledge entry."""
        entry = KnowledgeEntry(
            category=category,
            content=content,
            source_task=source_task,
        )
        self._entries.append(entry)
        # Trim to max entries
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[-self.MAX_ENTRIES:]

    def extract_from_run(self, task: str, steps: list[str],
                         file_memory_dict: dict[str, str], llm_client) -> list[KnowledgeEntry]:
        """Use LLM to extract key learnings from a completed run.

        Extracts patterns, error fixes, and conventions discovered.
        """
        # Build a concise summary of what was done
        files_list = [f for f in file_memory_dict.keys() if not f.startswith("_")]
        files_summary = ", ".join(files_list[:20])

        prompt = (
            "Analyze this completed coding task and extract 2-5 key learnings.\n"
            "For each learning, output ONE line in this format:\n"
            "  CATEGORY: description\n\n"
            "Categories: pattern, fix, convention, dependency\n\n"
            f"Task: {task}\n"
            f"Steps completed: {len(steps)}\n"
            f"Files created/modified: {files_summary}\n\n"
            "Steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps[:10])) + "\n\n"
            "Learnings:"
        )

        try:
            response = llm_client.generate_response(prompt)
            new_entries: list[KnowledgeEntry] = []

            for line in response.strip().splitlines():
                line = line.strip().lstrip("- •0123456789.")
                if ":" in line:
                    parts = line.split(":", 1)
                    category = parts[0].strip().lower()
                    content = parts[1].strip()
                    if category in ("pattern", "fix", "convention", "dependency") and content:
                        entry = KnowledgeEntry(
                            category=category,
                            content=content,
                            source_task=task[:100],
                        )
                        self._entries.append(entry)
                        new_entries.append(entry)

            if new_entries:
                self.save()
                log.info(f"[KnowledgeBase] Extracted {len(new_entries)} learnings")

            return new_entries

        except Exception as e:
            log.warning(f"[KnowledgeBase] Extraction failed: {e}")
            return []

    def format_for_planner(self, max_entries: int = 20) -> str:
        """Format stored knowledge as context for the planner prompt."""
        if not self._entries:
            return ""

        recent = self._entries[-max_entries:]
        lines: list[str] = ["Project knowledge from previous runs:"]

        for entry in recent:
            lines.append(f"  - [{entry.category}] {entry.content}")

        return "\n".join(lines)

    @property
    def size(self) -> int:
        return len(self._entries)
