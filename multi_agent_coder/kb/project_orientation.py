"""
Project orientation — detect project DNA and produce grounding context.

Reads the KB graph and config files to build a ``ProjectProfile`` that
is injected into EVERY LLM prompt as a mandatory grounding block.  This
prevents the LLM from falling back to training defaults (e.g. Python
instead of the actual project language, ``/src/`` instead of the real
source root).

The profile is built once at session start and cached for the entire
session.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProjectProfile
# ---------------------------------------------------------------------------

@dataclass
class ProjectProfile:
    """Structured representation of a project's identity and layout."""

    # Identity
    project_name: str = "unknown"
    project_version: str = "unknown"
    language: str = "unknown"
    framework: Optional[str] = None
    package_manager: str = "npm"

    # Layout
    project_root: str = ""
    source_root: str = "src"
    source_root_absolute: str = ""
    test_root: Optional[str] = None
    entry_points: list[str] = field(default_factory=list)

    # Commands
    test_command: Optional[str] = None
    build_command: Optional[str] = None
    dev_command: Optional[str] = None

    # Test frameworks
    test_frameworks: list[str] = field(default_factory=list)

    @property
    def source_extensions(self) -> list[str]:
        """Return file extensions appropriate for the detected language."""
        return {
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "python": [".py"],
            "java": [".java"],
            "go": [".go"],
            "rust": [".rs"],
        }.get(self.language, [".js"])

    def format_for_prompt(self) -> str:
        """Format the profile as a grounding block for LLM prompt injection.

        This is the most important output of the orientation layer — it
        anchors the LLM to the actual project structure.
        """
        lines = [
            "=== PROJECT CONTEXT (read carefully before acting) ===",
            "",
            f"Project: {self.project_name} v{self.project_version}",
            f"Language: {self.language}",
        ]

        if self.framework:
            lines.append(f"Framework: {self.framework}")

        lines += [
            "",
            "DIRECTORY STRUCTURE (CRITICAL — always use these paths):",
            f"  Project root:  {self.project_root}",
            f"  Source files:   {self.source_root_absolute}",
        ]

        if self.test_root:
            lines.append(
                f"  Test files:    "
                f"{os.path.join(self.project_root, self.test_root)}"
            )

        if self.entry_points:
            lines.append(f"  Entry points:  {', '.join(self.entry_points)}")

        lines += ["", "COMMANDS:"]
        if self.test_command:
            lines.append(f"  Run tests:  {self.package_manager} run test")
            lines.append(f"              (or: {self.test_command})")
        if self.dev_command:
            lines.append(f"  Run dev:    {self.package_manager} run dev")
        if self.build_command:
            lines.append(f"  Build:      {self.package_manager} run build")

        if self.test_frameworks:
            lines += [
                "",
                f"TEST FRAMEWORK: {', '.join(self.test_frameworks)}",
            ]

        ext_str = " or ".join(self.source_extensions)
        tf_str = (
            ", ".join(self.test_frameworks)
            if self.test_frameworks
            else self.language
        )

        lines += [
            "",
            "STRICT RULES — you MUST follow these:",
            f"  1. NEVER create files outside {self.source_root_absolute}"
            f" unless explicitly told to.",
            f"  2. ALL new source files must use {ext_str} extension.",
            f"  3. ALL test files must use the project's test framework: {tf_str}.",
            f"  4. NEVER use a different language. This is a {self.language} project.",
            f"  5. Run tests with: "
            f"{self.test_command or 'check package.json scripts'}",
            "",
            "=== END PROJECT CONTEXT ===",
        ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ProjectOrientation
# ---------------------------------------------------------------------------

class ProjectOrientation:
    """Detect project structure from the KB graph and config files.

    Parameters
    ----------
    graph:
        The ``CodeGraph`` instance (may be ``None`` if KB is unavailable).
    project_root:
        Absolute path to the project root directory.
    """

    def __init__(self, graph, project_root: str) -> None:
        self._graph = graph
        self._root = os.path.abspath(project_root)
        self._profile: Optional[ProjectProfile] = None

    def get_profile(self) -> ProjectProfile:
        """Return the cached project profile, building it on first call.

        The profile is built once and cached for the entire session.
        Target: < 200 ms.
        """
        if self._profile is None:
            t0 = time.perf_counter()
            self._profile = self._build_profile()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                "[ProjectOrientation] Profile built in %.1fms: "
                "lang=%s framework=%s source=%s tests=%s",
                elapsed_ms,
                self._profile.language,
                self._profile.framework,
                self._profile.source_root,
                self._profile.test_frameworks,
            )
        return self._profile

    # ------------------------------------------------------------------
    # Profile construction
    # ------------------------------------------------------------------

    def _build_profile(self) -> ProjectProfile:
        profile = ProjectProfile()
        profile.project_root = self._root

        # 1. Detect project type and language from config files
        self._detect_from_config_files(profile)

        # 2. Detect source layout from graph
        self._detect_source_layout(profile)

        # 3. Detect test framework (supplement if not found via config)
        self._detect_test_framework(profile)

        # 4. Detect entry points
        self._detect_entry_points(profile)

        return profile

    # ------------------------------------------------------------------
    # Config file detection
    # ------------------------------------------------------------------

    def _detect_from_config_files(self, profile: ProjectProfile) -> None:
        """Read package.json, tsconfig, pyproject.toml, etc."""

        pkg = self._read_json("package.json")
        if pkg:
            profile.language = "javascript"
            profile.package_manager = self._detect_package_manager()

            deps: dict = {
                **pkg.get("dependencies", {}),
                **pkg.get("devDependencies", {}),
            }

            # Framework detection
            if "react" in deps:
                profile.framework = "react"
                if "typescript" in deps or self._file_exists("tsconfig.json"):
                    profile.language = "typescript"

            if "next" in deps:
                profile.framework = "nextjs"
                if "typescript" in deps or self._file_exists("tsconfig.json"):
                    profile.language = "typescript"

            if "vue" in deps:
                profile.framework = "vue"

            if any(d in deps for d in ("express", "fastify", "koa")):
                if profile.framework is None:
                    profile.framework = "node-backend"

            # TypeScript detection (even without React/Next)
            if profile.language == "javascript":
                if "typescript" in deps or self._file_exists("tsconfig.json"):
                    profile.language = "typescript"

            # Test framework detection from dependencies
            for test_fw in [
                "jest", "vitest", "mocha", "jasmine", "cypress",
                "@testing-library/react",
            ]:
                if test_fw in deps:
                    profile.test_frameworks.append(test_fw)

            # Scripts
            scripts = pkg.get("scripts", {})
            profile.test_command = scripts.get("test")
            profile.build_command = scripts.get("build")
            profile.dev_command = scripts.get("dev") or scripts.get("start")

            profile.project_name = pkg.get("name", "unknown")
            profile.project_version = pkg.get("version", "unknown")
            return

        # Python project
        if (
            self._file_exists("pyproject.toml")
            or self._file_exists("setup.py")
            or self._file_exists("requirements.txt")
        ):
            profile.language = "python"
            profile.package_manager = "pip"
            self._detect_python_framework(profile)
            return

        # Java project
        if self._file_exists("pom.xml") or self._file_exists("build.gradle"):
            profile.language = "java"
            if self._file_contains("pom.xml", "spring-boot"):
                profile.framework = "spring"
            elif self._file_contains("build.gradle", "spring"):
                profile.framework = "spring"
            return

        # Go project
        if self._file_exists("go.mod"):
            profile.language = "go"
            return

        # Rust project
        if self._file_exists("Cargo.toml"):
            profile.language = "rust"
            return

    def _detect_python_framework(self, profile: ProjectProfile) -> None:
        """Detect Python web frameworks and test tools."""
        # Check requirements.txt
        reqs_text = self._read_text("requirements.txt") or ""
        pyproject_text = self._read_text("pyproject.toml") or ""
        combined = reqs_text + pyproject_text

        if "django" in combined.lower():
            profile.framework = "django"
        elif "flask" in combined.lower():
            profile.framework = "flask"
        elif "fastapi" in combined.lower():
            profile.framework = "fastapi"

        # Detect Python test frameworks
        if "pytest" in combined.lower():
            profile.test_frameworks.append("pytest")
            profile.test_command = "pytest"
        elif "unittest" in combined.lower():
            profile.test_frameworks.append("unittest")
            profile.test_command = "python -m unittest discover"

        if self._file_exists("setup.py"):
            setup_text = self._read_text("setup.py") or ""
            if "pytest" in setup_text:
                if "pytest" not in profile.test_frameworks:
                    profile.test_frameworks.append("pytest")
                    profile.test_command = "pytest"

    # ------------------------------------------------------------------
    # Source layout detection
    # ------------------------------------------------------------------

    def _detect_source_layout(self, profile: ProjectProfile) -> None:
        """Determine source and test root directories."""

        # First try graph-based detection
        if self._graph is not None:
            try:
                file_nodes = self._graph.get_all_file_nodes()
                dir_counts: dict[str, int] = {}
                for node in file_nodes:
                    node_path = node.get("path", "")
                    if not node_path:
                        continue
                    try:
                        rel_path = os.path.relpath(node_path, self._root)
                    except ValueError:
                        continue
                    parts = Path(rel_path).parts
                    if len(parts) > 1:
                        top_dir = parts[0]
                        dir_counts[top_dir] = dir_counts.get(top_dir, 0) + 1

                if dir_counts:
                    profile.source_root = max(dir_counts, key=dir_counts.get)
                    profile.source_root_absolute = os.path.join(
                        self._root, profile.source_root,
                    )
            except Exception as exc:
                logger.debug(
                    "[ProjectOrientation] Graph-based layout detection "
                    "failed: %s", exc,
                )

        # Override with common well-known patterns (more specific)
        for candidate in [
            "src", "my-app/src", "app/src", "client/src",
            "frontend/src", "web/src",
        ]:
            full = os.path.join(self._root, candidate)
            if os.path.isdir(full):
                profile.source_root = candidate
                profile.source_root_absolute = full
                break
        else:
            # If no known candidate matched, ensure absolute path is set
            if not profile.source_root_absolute:
                profile.source_root_absolute = os.path.join(
                    self._root, profile.source_root,
                )

        # Detect test directories
        test_candidates = [
            "__tests__", "tests", "test", "spec",
            f"{profile.source_root}/__tests__",
            f"{profile.source_root}/tests",
        ]
        for candidate in test_candidates:
            if os.path.isdir(os.path.join(self._root, candidate)):
                profile.test_root = candidate
                break

    # ------------------------------------------------------------------
    # Test framework detection (filesystem fallback)
    # ------------------------------------------------------------------

    def _detect_test_framework(self, profile: ProjectProfile) -> None:
        """Supplement test framework detection from filesystem clues."""
        if profile.test_frameworks:
            return  # Already detected from config files

        # Check for jest config files
        jest_configs = [
            "jest.config.js", "jest.config.ts", "jest.config.mjs",
            "jest.config.cjs", "jest.config.json",
        ]
        for cfg in jest_configs:
            if self._file_exists(cfg):
                profile.test_frameworks.append("jest")
                break

        # Check for vitest config
        if self._file_exists("vitest.config.ts") or self._file_exists("vitest.config.js"):
            profile.test_frameworks.append("vitest")

        # Check for pytest.ini / conftest.py
        if self._file_exists("pytest.ini") or self._file_exists("conftest.py"):
            if "pytest" not in profile.test_frameworks:
                profile.test_frameworks.append("pytest")
                if not profile.test_command:
                    profile.test_command = "pytest"

    # ------------------------------------------------------------------
    # Entry point detection
    # ------------------------------------------------------------------

    def _detect_entry_points(self, profile: ProjectProfile) -> None:
        """Find main entry files."""
        candidates = [
            f"{profile.source_root}/index.tsx",
            f"{profile.source_root}/index.ts",
            f"{profile.source_root}/index.jsx",
            f"{profile.source_root}/index.js",
            f"{profile.source_root}/main.tsx",
            f"{profile.source_root}/main.ts",
            f"{profile.source_root}/App.tsx",
            f"{profile.source_root}/App.jsx",
            "index.js", "index.ts", "main.py", "app.py",
        ]
        for candidate in candidates:
            if self._file_exists(candidate):
                profile.entry_points.append(candidate)

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    def _file_exists(self, rel_path: str) -> bool:
        """Check if a file exists relative to project root."""
        return os.path.isfile(os.path.join(self._root, rel_path))

    def _file_contains(self, rel_path: str, substring: str) -> bool:
        """Check if a file contains a substring (case-insensitive)."""
        full = os.path.join(self._root, rel_path)
        if not os.path.isfile(full):
            return False
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                return substring.lower() in f.read().lower()
        except OSError:
            return False

    def _read_json(self, rel_path: str) -> Optional[dict]:
        """Read and parse a JSON file relative to project root."""
        full = os.path.join(self._root, rel_path)
        if not os.path.isfile(full):
            return None
        try:
            with open(full, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _read_text(self, rel_path: str) -> Optional[str]:
        """Read a text file relative to project root."""
        full = os.path.join(self._root, rel_path)
        if not os.path.isfile(full):
            return None
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None

    def _detect_package_manager(self) -> str:
        """Detect the JavaScript/Node package manager in use."""
        if self._file_exists("yarn.lock"):
            return "yarn"
        if self._file_exists("pnpm-lock.yaml"):
            return "pnpm"
        if self._file_exists("bun.lockb"):
            return "bun"
        return "npm"
