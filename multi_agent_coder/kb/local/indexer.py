"""
Indexer — orchestrates full and incremental Knowledge Base indexing.

Full index:
  1. Walk the project directory (respecting .gitignore patterns)
  2. Parse each supported source file via tree-sitter
  3. Build a NetworkX code graph
  4. Persist graph.pkl, index.db, and graph_meta.json

Incremental index:
  Triggered by the file watcher; re-parses only changed/deleted files.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directory / file exclusion rules
# ---------------------------------------------------------------------------

_SKIP_DIRS: frozenset[str] = frozenset({
    "node_modules", "dist", "build", "__pycache__",
    ".git", "vendor", ".agentchanti",
    ".venv", "venv", "env", ".env",
    ".tox", ".mypy_cache", ".pytest_cache",
    "target",           # Rust/Java build output
    "bin", "obj",       # C# build output
    "coverage",
    ".next", ".nuxt",   # JS frameworks
    "out", ".output",
    "eggs", ".eggs",
    ".cache",
})

_GITIGNORE_PATTERNS: list[str] = []  # populated per-project at runtime


def _load_gitignore_patterns(project_root: str) -> list[str]:
    """Read .gitignore from *project_root* and return glob patterns."""
    gi_path = os.path.join(project_root, ".gitignore")
    patterns: list[str] = []
    if not os.path.exists(gi_path):
        return patterns
    with open(gi_path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def _is_ignored(path: str, gitignore_patterns: list[str]) -> bool:
    """Return True if *path* matches any gitignore pattern."""
    name = os.path.basename(path)
    for pattern in gitignore_patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
        if fnmatch.fnmatch(path, pattern):
            return True
    return False


# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

def _kb_dir(project_root: str) -> str:
    return os.path.join(project_root, ".agentchanti", "kb", "local")


def _graph_path(project_root: str) -> str:
    return os.path.join(_kb_dir(project_root), "graph.pkl")


def _manifest_path(project_root: str) -> str:
    return os.path.join(_kb_dir(project_root), "index.db")


def _meta_path(project_root: str) -> str:
    return os.path.join(_kb_dir(project_root), "graph_meta.json")


# ---------------------------------------------------------------------------
# Meta helpers
# ---------------------------------------------------------------------------

KB_VERSION = "1.0.0"


def _write_meta(project_root: str, graph_stats: dict) -> None:
    """Write graph_meta.json with indexing statistics."""
    meta = {
        "last_indexed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_count": graph_stats.get("by_node_type", {}).get("FILE", 0),
        "symbol_count": sum(
            v for k, v in graph_stats.get("by_node_type", {}).items()
            if k in ("FUNCTION", "CLASS", "VARIABLE")
        ),
        "edge_count": graph_stats.get("edge_count", 0),
        "kb_version": KB_VERSION,
    }
    os.makedirs(_kb_dir(project_root), exist_ok=True)
    with open(_meta_path(project_root), "w") as fh:
        json.dump(meta, fh, indent=2)


def read_meta(project_root: str) -> Optional[dict]:
    """
    Read and return graph_meta.json, or None if it does not exist.

    Parameters
    ----------
    project_root:
        Project root directory.
    """
    p = _meta_path(project_root)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# File walker
# ---------------------------------------------------------------------------

def _walk_source_files(project_root: str) -> list[str]:
    """
    Walk *project_root* and return paths of all indexable source files.

    Skips excluded directories and gitignore-matched paths.
    Paths are returned relative to *project_root*.
    """
    from .parser import EXTENSION_TO_LANGUAGE

    gi_patterns = _load_gitignore_patterns(project_root)
    results: list[str] = []

    for dirpath, dirnames, filenames in os.walk(project_root, topdown=True):
        # Prune excluded directories in-place (modifies the walk)
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS
            and not d.startswith(".")
            and not _is_ignored(os.path.join(dirpath, d), gi_patterns)
        ]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in EXTENSION_TO_LANGUAGE:
                continue
            abs_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(abs_path, project_root)
            if _is_ignored(rel_path, gi_patterns):
                continue
            results.append(rel_path)

    return sorted(results)


# ---------------------------------------------------------------------------
# Module-name → file-path mapping (for import edge resolution)
# ---------------------------------------------------------------------------

def _build_module_map(project_root: str, source_files: list[str]) -> dict[str, str]:
    """
    Build a mapping from Python module dotted name → relative file path.

    Only handles Python files for now.
    """
    module_map: dict[str, str] = {}
    for rel_path in source_files:
        if not rel_path.endswith(".py"):
            continue
        # Convert path to dotted module name
        without_ext = rel_path[:-3]  # strip .py
        module_name = without_ext.replace(os.sep, ".").replace("/", ".")
        if module_name.endswith(".__init__"):
            module_name = module_name[:-9]  # strip .__init__
        module_map[module_name] = rel_path
    return module_map


# ---------------------------------------------------------------------------
# Public indexing API
# ---------------------------------------------------------------------------

class Indexer:
    """
    Orchestrates full and incremental indexing of a project.

    Parameters
    ----------
    project_root:
        Absolute path to the project root directory.
    """

    def __init__(self, project_root: str) -> None:
        self.project_root = os.path.abspath(project_root)
        os.makedirs(_kb_dir(self.project_root), exist_ok=True)

    # ------------------------------------------------------------------
    # Full index
    # ------------------------------------------------------------------

    def full_index(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """
        Perform a full re-index of the project.

        1. Walk project directory, collect source files
        2. Parse each file via tree-sitter
        3. Build code graph
        4. Persist graph.pkl, index.db, graph_meta.json

        Parameters
        ----------
        progress_callback:
            Optional callable called with (current, total, filename) for each
            processed file.

        Returns
        -------
        dict
            Summary: file_count, symbol_count, edge_count, elapsed_seconds,
            error_count.
        """
        from .parser import parse_file, compute_file_hash
        from .graph import CodeGraph
        from .manifest import Manifest, SymbolRecord

        start_time = time.time()
        graph = CodeGraph()
        manifest = Manifest(_manifest_path(self.project_root))
        manifest.clear()

        source_files = _walk_source_files(self.project_root)
        total = len(source_files)
        error_count = 0

        for idx, rel_path in enumerate(source_files):
            abs_path = os.path.join(self.project_root, rel_path)
            if progress_callback:
                progress_callback(idx + 1, total, rel_path)

            try:
                parsed = parse_file(abs_path)
                if parsed.parse_error and not (parsed.functions or parsed.classes):
                    logger.warning("Parse error in %s: %s", rel_path, parsed.parse_error)
                    error_count += 1
                    continue

                # Add to graph using relative path as node identifier
                # Re-create ParsedFile with relative path for consistent IDs
                parsed.path = rel_path
                graph.add_parsed_file(parsed)

                # Update manifest
                try:
                    last_mod = os.path.getmtime(abs_path)
                except OSError:
                    last_mod = 0.0

                symbols: list[SymbolRecord] = []
                for fn in parsed.functions:
                    symbols.append(SymbolRecord(fn.name, "function", fn.line_start, fn.line_end))
                for cls in parsed.classes:
                    symbols.append(SymbolRecord(cls.name, "class", cls.line_start, cls.line_end))
                for var in parsed.variables:
                    symbols.append(SymbolRecord(var.name, "variable", 0, 0))

                manifest.upsert_file(
                    path=rel_path,
                    hash_=parsed.hash,
                    language=parsed.language,
                    last_modified=last_mod,
                    symbols=symbols,
                )

            except Exception as exc:
                logger.warning("Unexpected error indexing %s: %s", rel_path, exc)
                error_count += 1

        # Resolve import edges once all files are indexed
        module_map = _build_module_map(self.project_root, source_files)
        try:
            graph.resolve_import_edges(module_map)
        except Exception as exc:
            logger.warning("Import edge resolution failed: %s", exc)

        # Persist
        graph.save(_graph_path(self.project_root))
        stats = graph.stats()
        _write_meta(self.project_root, stats)

        elapsed = time.time() - start_time
        summary = {
            "file_count": total,
            "symbol_count": stats.get("by_node_type", {}).get("FUNCTION", 0)
                           + stats.get("by_node_type", {}).get("CLASS", 0)
                           + stats.get("by_node_type", {}).get("VARIABLE", 0),
            "edge_count": stats.get("edge_count", 0),
            "elapsed_seconds": round(elapsed, 2),
            "error_count": error_count,
        }
        logger.info(
            "Full index complete: %d files, %d symbols, %d edges in %.1fs",
            summary["file_count"],
            summary["symbol_count"],
            summary["edge_count"],
            elapsed,
        )
        return summary

    # ------------------------------------------------------------------
    # Incremental update (called by file watcher)
    # ------------------------------------------------------------------

    def update_file(self, rel_path: str) -> None:
        """
        Re-parse a single changed or newly created file and update the graph.

        Parameters
        ----------
        rel_path:
            File path relative to *project_root*.
        """
        from .parser import parse_file, compute_file_hash
        from .manifest import Manifest, SymbolRecord

        abs_path = os.path.join(self.project_root, rel_path)
        if not os.path.exists(abs_path):
            return

        graph = self._load_graph()
        manifest = Manifest(_manifest_path(self.project_root))

        new_hash = compute_file_hash(abs_path)
        if not manifest.is_file_changed(rel_path, new_hash):
            logger.debug("File unchanged, skipping: %s", rel_path)
            return

        # Remove old nodes
        graph.remove_file(rel_path)

        try:
            parsed = parse_file(abs_path)
            parsed.path = rel_path
            if parsed.parse_error and not (parsed.functions or parsed.classes):
                logger.warning("Parse error updating %s: %s", rel_path, parsed.parse_error)
                manifest.remove_file(rel_path)
                self._save_graph(graph)
                return

            graph.add_parsed_file(parsed)

            try:
                last_mod = os.path.getmtime(abs_path)
            except OSError:
                last_mod = 0.0

            symbols: list[SymbolRecord] = []
            for fn in parsed.functions:
                symbols.append(SymbolRecord(fn.name, "function", fn.line_start, fn.line_end))
            for cls in parsed.classes:
                symbols.append(SymbolRecord(cls.name, "class", cls.line_start, cls.line_end))
            for var in parsed.variables:
                symbols.append(SymbolRecord(var.name, "variable", 0, 0))

            manifest.upsert_file(
                path=rel_path,
                hash_=parsed.hash,
                language=parsed.language,
                last_modified=last_mod,
                symbols=symbols,
            )
        except Exception as exc:
            logger.warning("Error updating %s: %s", rel_path, exc)

        self._save_graph(graph)
        _write_meta(self.project_root, graph.stats())
        logger.info("Incremental update complete for %s", rel_path)

    def remove_file(self, rel_path: str) -> None:
        """
        Remove all graph data for a deleted file.

        Parameters
        ----------
        rel_path:
            File path relative to *project_root*.
        """
        from .manifest import Manifest

        graph = self._load_graph()
        graph.remove_file(rel_path)
        self._save_graph(graph)

        manifest = Manifest(_manifest_path(self.project_root))
        manifest.remove_file(rel_path)
        _write_meta(self.project_root, graph.stats())
        logger.info("Removed file from index: %s", rel_path)

    # ------------------------------------------------------------------
    # Graph access helpers
    # ------------------------------------------------------------------

    def load_graph(self):
        """
        Load and return the persisted CodeGraph.

        Returns
        -------
        CodeGraph

        Raises
        ------
        FileNotFoundError
            If the graph has not been built yet (run `kb index` first).
        """
        return self._load_graph()

    def _load_graph(self):
        from .graph import CodeGraph
        path = _graph_path(self.project_root)
        if os.path.exists(path):
            try:
                return CodeGraph.load(path)
            except Exception as exc:
                logger.warning("Failed to load graph, starting fresh: %s", exc)
        from .graph import CodeGraph
        return CodeGraph()

    def _save_graph(self, graph) -> None:
        graph.save(_graph_path(self.project_root))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_indexed(self) -> bool:
        """Return True if a graph has been built for this project."""
        return os.path.exists(_graph_path(self.project_root))
