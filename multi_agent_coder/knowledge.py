"""
Knowledge Base — persists structured project knowledge across runs.

Stores project summary, tech stack, installed packages, file purposes,
patterns, and fixes discovered during pipeline execution. Loaded into
agent prompts for context awareness.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .cli_display import log


# ── Legacy dataclass (kept for backward compat migration) ────────
@dataclass
class KnowledgeEntry:
    """A single piece of project knowledge (legacy format)."""
    category: str        # "pattern", "fix", "convention", "dependency"
    content: str
    source_task: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# ── New structured knowledge ─────────────────────────────────────
@dataclass
class TechStack:
    language: str = ""
    framework: str = ""
    test_framework: str = ""
    package_manager: str = ""


@dataclass
class ProjectKnowledge:
    project_summary: str = ""
    tech_stack: TechStack = field(default_factory=TechStack)
    installed_packages: list[str] = field(default_factory=list)
    file_purposes: dict[str, str] = field(default_factory=dict)
    patterns: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    last_updated: str = ""

    MAX_PATTERNS = 20
    MAX_FIXES = 20
    MAX_FILE_PURPOSES = 30
    MAX_PACKAGES = 100


class KnowledgeBase:
    """Persistent structured project knowledge store.

    Saves to ``.agentchanti/knowledge.json`` by default.
    Automatically migrates from legacy flat-list format on load.
    """

    def __init__(self, path: str = ".agentchanti/knowledge.json"):
        self._path = path
        self._knowledge = ProjectKnowledge()
        self.load()

    # ── Persistence ──────────────────────────────────────────────

    def load(self) -> ProjectKnowledge:
        """Load knowledge from disk. Auto-migrates legacy format."""
        if not os.path.isfile(self._path):
            return self._knowledge

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                # Legacy format: list of KnowledgeEntry dicts
                self._migrate_legacy(data)
                log.debug("[KnowledgeBase] Migrated legacy format")
            elif isinstance(data, dict):
                self._load_structured(data)

            log.debug(f"[KnowledgeBase] Loaded: {self.size} total entries")
        except (json.JSONDecodeError, OSError, TypeError) as e:
            log.warning(f"[KnowledgeBase] Load error: {e}")

        return self._knowledge

    def _migrate_legacy(self, entries: list[dict]):
        """Convert old flat list format to structured ProjectKnowledge."""
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cat = entry.get("category", "")
            content = entry.get("content", "")
            if not content:
                continue

            if cat == "pattern" or cat == "convention":
                if content not in self._knowledge.patterns:
                    self._knowledge.patterns.append(content[:80])
            elif cat == "fix":
                if content not in self._knowledge.fixes:
                    self._knowledge.fixes.append(content[:80])
            elif cat == "dependency":
                # Try to extract package name from dependency entries
                pkg = _extract_package_name(content)
                if pkg and pkg not in self._knowledge.installed_packages:
                    self._knowledge.installed_packages.append(pkg)

        self._knowledge.last_updated = datetime.now().isoformat()
        self.save()

    def _load_structured(self, data: dict):
        """Load new structured format."""
        ts_data = data.get("tech_stack", {})
        self._knowledge = ProjectKnowledge(
            project_summary=data.get("project_summary", ""),
            tech_stack=TechStack(
                language=ts_data.get("language", ""),
                framework=ts_data.get("framework", ""),
                test_framework=ts_data.get("test_framework", ""),
                package_manager=ts_data.get("package_manager", ""),
            ),
            installed_packages=data.get("installed_packages", []),
            file_purposes=data.get("file_purposes", {}),
            patterns=data.get("patterns", []),
            fixes=data.get("fixes", []),
            last_updated=data.get("last_updated", ""),
        )

    def save(self):
        """Persist knowledge to disk in structured format."""
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        try:
            # Trim before saving
            k = self._knowledge
            k.patterns = k.patterns[-k.MAX_PATTERNS:]
            k.fixes = k.fixes[-k.MAX_FIXES:]
            k.installed_packages = k.installed_packages[-k.MAX_PACKAGES:]
            # Trim file_purposes to most recent
            if len(k.file_purposes) > k.MAX_FILE_PURPOSES:
                keys = list(k.file_purposes.keys())
                for key in keys[:-k.MAX_FILE_PURPOSES]:
                    del k.file_purposes[key]

            k.last_updated = datetime.now().isoformat()

            data = {
                "project_summary": k.project_summary,
                "tech_stack": asdict(k.tech_stack),
                "installed_packages": k.installed_packages,
                "file_purposes": k.file_purposes,
                "patterns": k.patterns,
                "fixes": k.fixes,
                "last_updated": k.last_updated,
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            log.debug(f"[KnowledgeBase] Saved ({self.size} entries)")
        except OSError as e:
            log.warning(f"[KnowledgeBase] Save error: {e}")

    # ── Recording methods (called during pipeline execution) ─────

    def record_install(self, package_name: str):
        """Record a successfully installed package."""
        pkg = package_name.strip().lower()
        if pkg and pkg not in self._knowledge.installed_packages:
            self._knowledge.installed_packages.append(pkg)

    def record_file_purpose(self, path: str, purpose: str):
        """Record the purpose of a file (max 60 chars)."""
        if path and purpose:
            self._knowledge.file_purposes[path] = purpose[:60]

    def update_tech_stack(self, project_profile=None):
        """Update tech stack from a ProjectProfile (from project_orientation)."""
        if project_profile is None:
            return
        ts = self._knowledge.tech_stack
        if hasattr(project_profile, "language") and project_profile.language:
            ts.language = project_profile.language
        if hasattr(project_profile, "framework") and project_profile.framework:
            ts.framework = project_profile.framework
        if hasattr(project_profile, "test_frameworks") and project_profile.test_frameworks:
            ts.test_framework = ", ".join(project_profile.test_frameworks)
        if hasattr(project_profile, "package_manager") and project_profile.package_manager:
            ts.package_manager = project_profile.package_manager

    def update_project_summary(self, summary: str):
        """Set the project summary (1-2 sentences)."""
        if summary:
            self._knowledge.project_summary = summary[:200]

    def is_package_installed(self, package_name: str) -> bool:
        """Check if a package is recorded as installed."""
        return package_name.strip().lower() in self._knowledge.installed_packages

    # ── Legacy compat: add() still works ─────────────────────────

    def add(self, category: str, content: str, source_task: str):
        """Add a knowledge entry (routes to appropriate structured field)."""
        content = content.strip()[:80]
        if not content:
            return

        if category in ("pattern", "convention"):
            if content not in self._knowledge.patterns:
                self._knowledge.patterns.append(content)
        elif category == "fix":
            if content not in self._knowledge.fixes:
                self._knowledge.fixes.append(content)
        elif category == "dependency":
            pkg = _extract_package_name(content)
            if pkg:
                self.record_install(pkg)

    # ── Knowledge extraction from completed runs ─────────────────

    def extract_from_run(self, task: str, steps: list[str],
                         file_memory_dict: dict[str, str], llm_client) -> int:
        """Use LLM to extract key learnings from a completed run.

        Returns the number of new entries added.
        """
        files_list = [f for f in file_memory_dict.keys() if not f.startswith("_")]
        files_summary = ", ".join(files_list[:20])

        prompt = (
            "Analyze this completed coding task and extract concise learnings.\n"
            "Output ONLY lines in this exact format (no numbering, no bullets):\n"
            "  CATEGORY: short description (max 60 chars)\n\n"
            "Categories:\n"
            "  pattern — coding pattern or convention used\n"
            "  fix — error fix or workaround discovered\n"
            "  summary — one-sentence project description (only if not obvious)\n\n"
            f"Task: {task}\n"
            f"Files: {files_summary}\n\n"
            "Steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps[:10])) + "\n\n"
            "Learnings:"
        )

        try:
            response = llm_client.generate_response(prompt)
            added = 0

            for line in response.strip().splitlines():
                line = line.strip().lstrip("- \u2022 0123456789.")
                if ":" not in line:
                    continue
                parts = line.split(":", 1)
                category = parts[0].strip().lower()
                content = parts[1].strip()[:80]
                if not content:
                    continue

                if category == "summary" and not self._knowledge.project_summary:
                    self._knowledge.project_summary = content[:200]
                    added += 1
                elif category == "pattern":
                    if content not in self._knowledge.patterns:
                        self._knowledge.patterns.append(content)
                        added += 1
                elif category == "fix":
                    if content not in self._knowledge.fixes:
                        self._knowledge.fixes.append(content)
                        added += 1

            # Auto-extract installed packages from install steps
            for step in steps:
                pkgs = _extract_packages_from_step(step)
                for pkg in pkgs:
                    self.record_install(pkg)
                    added += 1

            # Auto-extract file purposes from file list
            for fpath in files_list[:20]:
                if fpath not in self._knowledge.file_purposes:
                    purpose = _infer_file_purpose(fpath)
                    if purpose:
                        self._knowledge.file_purposes[fpath] = purpose

            if added > 0:
                self.save()
                log.info(f"[KnowledgeBase] Extracted {added} learnings")

            return added

        except Exception as e:
            log.warning(f"[KnowledgeBase] Extraction failed: {e}")
            return 0

    # ── Formatting for prompts ───────────────────────────────────

    def format_for_planner(self, max_entries: int = 20) -> str:
        """Detailed format for the planner prompt."""
        k = self._knowledge
        parts: list[str] = []

        if k.project_summary:
            parts.append(f"Project: {k.project_summary}")

        ts = k.tech_stack
        stack_parts = []
        if ts.language:
            stack_parts.append(ts.language)
        if ts.framework:
            stack_parts.append(ts.framework)
        if ts.test_framework:
            stack_parts.append(f"tests: {ts.test_framework}")
        if ts.package_manager:
            stack_parts.append(f"pkg: {ts.package_manager}")
        if stack_parts:
            parts.append(f"Stack: {', '.join(stack_parts)}")

        if k.installed_packages:
            pkgs = k.installed_packages[-30:]
            parts.append(f"Installed: {', '.join(pkgs)}")

        if k.file_purposes:
            fp_lines = [f"  {p}: {desc}" for p, desc in list(k.file_purposes.items())[-10:]]
            parts.append("Key files:\n" + "\n".join(fp_lines))

        if k.patterns:
            for p in k.patterns[-5:]:
                parts.append(f"  [pattern] {p}")

        if k.fixes:
            for f in k.fixes[-5:]:
                parts.append(f"  [fix] {f}")

        if not parts:
            return ""

        return "=== PROJECT KNOWLEDGE ===\n" + "\n".join(parts) + "\n=== END KNOWLEDGE ==="

    def format_for_agents(self) -> str:
        """Compact format for all agent prompts (~200 tokens max)."""
        k = self._knowledge
        parts: list[str] = []

        if k.project_summary:
            parts.append(f"Project: {k.project_summary}")

        ts = k.tech_stack
        stack_parts = []
        if ts.language:
            stack_parts.append(ts.language)
        if ts.framework:
            stack_parts.append(ts.framework)
        if ts.test_framework:
            stack_parts.append(f"tests: {ts.test_framework}")
        if stack_parts:
            parts.append(f"Stack: {', '.join(stack_parts)}")

        if k.installed_packages:
            pkgs = k.installed_packages[-15:]
            parts.append(f"Packages: {', '.join(pkgs)}")

        if not parts:
            return ""

        return "[Project Info] " + " | ".join(parts)

    @property
    def size(self) -> int:
        k = self._knowledge
        return (
            (1 if k.project_summary else 0)
            + len(k.installed_packages)
            + len(k.file_purposes)
            + len(k.patterns)
            + len(k.fixes)
        )

    @property
    def knowledge(self) -> ProjectKnowledge:
        return self._knowledge


# ── Helpers ──────────────────────────────────────────────────────

# Common English words that are NOT package names — prevents LLM
# free-form text from polluting installed_packages.
_NOT_PACKAGES = {
    "a", "an", "the", "to", "in", "on", "at", "by", "of", "for",
    "and", "or", "but", "not", "is", "it", "as", "be", "do", "if",
    "so", "no", "up", "we", "he", "my", "am", "are", "was", "has",
    "had", "use", "used", "using", "uses", "with", "from", "into",
    "that", "this", "then", "them", "than", "will", "can", "may",
    "all", "any", "each", "both", "more", "most", "much", "many",
    "some", "such", "very", "also", "just", "only", "well", "even",
    "still", "back", "out", "now", "new", "old", "own", "set",
    "run", "way", "get", "got", "let", "put", "end", "too", "yet",
    "how", "why", "who", "what", "when", "where", "which",
    # common verbs LLMs use in dependency descriptions
    "always", "never", "ensure", "make", "keep", "need", "must",
    "should", "would", "could", "might", "shall", "want", "like",
    "add", "create", "update", "delete", "remove", "change", "modify",
    "install", "uninstall", "upgrade", "downgrade", "integrate",
    "integrating", "implement", "implementing", "configure", "setup",
    "before", "after", "during", "between", "about", "above", "below",
    "project", "code", "file", "module", "package", "library",
    "function", "class", "method", "variable", "import", "export",
    "test", "testing", "tests", "debug", "build", "deploy", "start",
    "stop", "version", "latest", "required", "dependency", "dependencies",
}


def _extract_package_name(text: str) -> str:
    """Extract a real package name from a dependency description.

    Validates that the extracted token looks like an actual package
    (not a common English word that LLMs love to produce).
    """
    text = text.strip().lower()

    # Strategy 1: Look for explicit install command patterns
    install_match = re.search(
        r'(?:pip|npm|yarn|pnpm|gem|cargo|go)\s+(?:install|add|get)\s+'
        r'([a-zA-Z][a-zA-Z0-9_.-]*)',
        text,
    )
    if install_match:
        pkg = re.split(r'[=<>~!@]', install_match.group(1))[0]
        if pkg.lower() not in _NOT_PACKAGES:
            return pkg.lower()

    # Strategy 2: If the entire text looks like a bare package name
    # e.g. "flask", "react-dom", "pytest_mock"
    bare = re.match(r'^([a-z][a-z0-9_-]*)(?:[=<>~!@].*)?$', text)
    if bare:
        candidate = bare.group(1)
        # Must not be a common English word, must be 2+ chars
        if len(candidate) >= 2 and candidate not in _NOT_PACKAGES:
            return candidate

    return ""


_INSTALL_PATTERNS = [
    re.compile(r'`(?:pip3?\s+install|pip3?\s+install\s+-U)\s+(.+?)`', re.IGNORECASE),
    re.compile(r'`(?:npm\s+install|npm\s+i|yarn\s+add|pnpm\s+add)\s+(.+?)`', re.IGNORECASE),
    re.compile(r'`(?:gem\s+install)\s+(.+?)`', re.IGNORECASE),
    re.compile(r'`(?:cargo\s+add)\s+(.+?)`', re.IGNORECASE),
    re.compile(r'`(?:go\s+get)\s+(.+?)`', re.IGNORECASE),
]


def _extract_packages_from_step(step_text: str) -> list[str]:
    """Extract package names from an install command step.

    Only matches backtick-wrapped install commands to avoid false
    positives from free-form step descriptions.
    """
    packages: list[str] = []
    for pattern in _INSTALL_PATTERNS:
        m = pattern.search(step_text)
        if m:
            raw = m.group(1)
            for token in raw.split():
                token = token.strip()
                # Skip flags
                if token.startswith("-"):
                    continue
                # Skip scoped npm packages like @types/node
                if token.startswith("@") and "/" in token:
                    # But still record them (they're real packages)
                    packages.append(token.lower())
                    continue
                # Strip version specifiers
                pkg = re.split(r'[=<>~!@]', token)[0]
                if pkg and re.match(r'^[a-zA-Z]', pkg):
                    pkg_lower = pkg.lower()
                    # Validate: skip common English words
                    if pkg_lower not in _NOT_PACKAGES and len(pkg_lower) >= 2:
                        packages.append(pkg_lower)
    return packages


def _infer_file_purpose(filepath: str) -> str:
    """Infer a meaningful purpose string from a file path.

    Extracts specifics from the filename rather than just returning
    generic labels like 'test file'.
    """
    basename = os.path.basename(filepath)
    name = basename.lower()
    stem = os.path.splitext(basename)[0]
    path_lower = filepath.lower().replace("\\", "/")

    # Exact filename matches
    purpose_map = {
        "main.py": "application entry point",
        "app.py": "application entry point",
        "index.js": "application entry point",
        "index.ts": "application entry point",
        "server.js": "server setup",
        "server.ts": "server setup",
        "setup.py": "package setup config",
        "conftest.py": "pytest fixtures and test config",
        "manage.py": "Django management commands",
        "settings.py": "Django settings",
        "wsgi.py": "WSGI application entry",
        "asgi.py": "ASGI application entry",
        "urls.py": "URL routing",
        "views.py": "view handlers",
        "models.py": "database models",
        "forms.py": "form definitions",
        "serializers.py": "API serializers",
        "tasks.py": "background tasks",
        "admin.py": "admin configuration",
    }
    if name in purpose_map:
        return purpose_map[name]

    # Test files — extract what they test from the name
    if "test" in path_lower or "spec" in path_lower:
        # "test_game_flow.py" → "tests for game_flow"
        # "test_snake_and_main.py" → "tests for snake_and_main"
        subject = stem
        for prefix in ("test_", "test-", "tests_", "spec_", "spec-"):
            if subject.startswith(prefix):
                subject = subject[len(prefix):]
                break
        for suffix in ("_test", "-test", "_spec", "-spec", ".test", ".spec"):
            if subject.endswith(suffix):
                subject = subject[:-len(suffix)]
                break
        subject = subject.replace("_", " ").replace("-", " ").strip()
        if subject:
            return f"tests for {subject}"
        return "test suite"

    # Directory-based purpose
    if "util" in path_lower or "helper" in path_lower:
        return f"utility: {stem.replace('_', ' ')}"
    if "model" in path_lower:
        return f"data model: {stem.replace('_', ' ')}"
    if "route" in path_lower or "controller" in path_lower:
        return f"route handler: {stem.replace('_', ' ')}"
    if "config" in path_lower:
        return f"configuration: {stem.replace('_', ' ')}"
    if "middleware" in path_lower:
        return f"middleware: {stem.replace('_', ' ')}"
    if "component" in path_lower:
        return f"UI component: {stem.replace('_', ' ')}"
    if "service" in path_lower:
        return f"service: {stem.replace('_', ' ')}"
    if "schema" in path_lower:
        return f"schema: {stem.replace('_', ' ')}"
    if "migration" in path_lower:
        return f"database migration"

    # Generic: use the stem as description
    readable = stem.replace("_", " ").replace("-", " ").strip()
    if readable and readable != name:
        return f"{readable} module"

    return ""
