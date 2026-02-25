"""
Scope resolver — uses the Phase 1 code graph to determine the minimal
set of symbols and line ranges the LLM needs to see for a given edit task.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)

# Keywords that suggest a multi-file change
_MULTI_FILE_KEYWORDS = re.compile(
    r"\b(all|everywhere|wherever|across|also\s+update|rename|every)\b", re.IGNORECASE
)

# Patterns for line-number references in task descriptions
_LINE_PATTERNS = [
    re.compile(r"lines?\s+(\d+)\s*[-–]\s*(\d+)", re.IGNORECASE),  # line 42-67
    re.compile(r"around\s+line\s+(\d+)", re.IGNORECASE),           # around line 50
    re.compile(r"lines?\s+(\d+)", re.IGNORECASE),                  # line 42
]

# Stack trace patterns (Python-style and generic file:line)
_TRACEBACK_PATTERNS = [
    re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+)'),              # Python traceback
    re.compile(r"(\S+\.(?:py|js|ts|java|go|rs|rb|php|cs)):(\d+)"),  # file.ext:42
]


@dataclass
class SymbolRange:
    """A symbol's location and editability within a file."""
    symbol_name: str
    symbol_type: str  # "function"|"class"|"method"|"variable"
    file_path: str
    line_start: int
    line_end: int
    editable: bool  # True for primary, False for context-only
    parent_class: Optional[str] = None


@dataclass
class EditScope:
    """The resolved scope of an edit operation."""
    primary_symbols: list[SymbolRange] = field(default_factory=list)
    context_symbols: list[SymbolRange] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)
    resolution_method: str = "fallback"  # "graph_lookup"|"line_mention"|"error_location"|"semantic"|"fallback"
    confidence: float = 0.0


class ScopeResolver:
    """Resolve the minimal edit scope for a task using the code graph."""

    def __init__(self, graph) -> None:
        """
        Parameters
        ----------
        graph:
            A ``CodeGraph`` instance (from ``multi_agent_coder.kb.local.graph``).
        """
        self._graph = graph

    def resolve(
        self,
        task_description: str,
        target_file: str,
        graph=None,
    ) -> EditScope:
        """Resolve the edit scope for *task_description* targeting *target_file*.

        Parameters
        ----------
        task_description:
            The natural-language task/step description.
        target_file:
            The file being edited (relative path).
        graph:
            Optional override graph; defaults to the one passed at construction.

        Returns
        -------
        EditScope
            The resolved scope with primary/context symbols, affected files,
            resolution method, and confidence score.
        """
        g = graph or self._graph
        if g is None:
            logger.warning("[DiffEdit] No code graph available, falling back to full file")
            return EditScope(resolution_method="fallback", confidence=0.0)

        scope = EditScope()
        scope.affected_files = [target_file]

        # Get all symbols in the target file
        file_symbols = g.get_file_symbols(target_file)

        # --- 1. Explicit symbol mention (highest confidence) ---
        resolved = self._resolve_explicit_symbols(task_description, target_file, file_symbols, g)
        if resolved:
            scope.primary_symbols = resolved
            scope.resolution_method = "graph_lookup"
            scope.confidence = 0.95

        # --- 2. Line number mention ---
        if not scope.primary_symbols:
            resolved = self._resolve_line_numbers(task_description, target_file, file_symbols)
            if resolved:
                scope.primary_symbols = resolved
                scope.resolution_method = "line_mention"
                scope.confidence = 0.90

        # --- 3. Error / stack trace location ---
        if not scope.primary_symbols:
            resolved = self._resolve_error_location(task_description, target_file, file_symbols, g)
            if resolved:
                scope.primary_symbols = resolved
                scope.resolution_method = "error_location"
                scope.confidence = 0.88

        # --- 4. Semantic fuzzy match ---
        if not scope.primary_symbols:
            resolved = self._resolve_semantic(task_description, target_file, file_symbols)
            if resolved:
                scope.primary_symbols = resolved
                scope.resolution_method = "semantic"
                scope.confidence = max(0.70, min(0.85, 0.70 + 0.05 * len(resolved)))

        # --- 5. Multi-file impact analysis ---
        if scope.primary_symbols and _MULTI_FILE_KEYWORDS.search(task_description):
            self._expand_multi_file(scope, target_file, g)

        # --- 6. Gather context symbols ---
        if scope.primary_symbols:
            self._add_context_symbols(scope, g)

        # --- 7. Fallback ---
        if not scope.primary_symbols or scope.confidence < 0.60:
            logger.warning(
                "[DiffEdit] Low confidence scope resolution (%.2f), "
                "falling back to full file for %s",
                scope.confidence, target_file,
            )
            scope.resolution_method = "fallback"
            scope.confidence = 0.0

        return scope

    # ------------------------------------------------------------------
    # Resolution strategies
    # ------------------------------------------------------------------

    def _resolve_explicit_symbols(
        self,
        task: str,
        target_file: str,
        file_symbols: list[dict],
        graph,
    ) -> list[SymbolRange]:
        """Scan task for known symbol names from the file's graph."""
        found: list[SymbolRange] = []
        seen: set[str] = set()
        task_lower = task.lower()

        for sym in file_symbols:
            name = sym.get("name", "")
            if not name or len(name) < 2:
                continue
            # Check if the symbol name appears in the task (word-boundary aware)
            pattern = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)
            if pattern.search(task):
                key = f"{name}:{sym.get('file_path', '')}:{sym.get('line_start', 0)}"
                if key not in seen:
                    seen.add(key)
                    found.append(self._sym_dict_to_range(sym, editable=True))

        # Also try graph-wide find_symbol for names mentioned in task
        # that might not be in this file (for cross-references)
        if not found:
            words = set(re.findall(r'\b[A-Za-z_]\w+\b', task))
            for word in words:
                if len(word) < 3:
                    continue
                matches = graph.find_symbol(word)
                for m in matches:
                    if m.get("file_path") == target_file:
                        key = f"{m['name']}:{m['file_path']}:{m.get('line_start', 0)}"
                        if key not in seen:
                            seen.add(key)
                            found.append(self._sym_dict_to_range(m, editable=True))

        return found

    def _resolve_line_numbers(
        self,
        task: str,
        target_file: str,
        file_symbols: list[dict],
    ) -> list[SymbolRange]:
        """Detect line number references and map to containing symbols."""
        lines: list[int] = []

        for pat in _LINE_PATTERNS:
            for m in pat.finditer(task):
                groups = m.groups()
                if len(groups) == 2 and groups[1]:
                    # Range: line 42-67
                    lines.extend(range(int(groups[0]), int(groups[1]) + 1))
                else:
                    lines.append(int(groups[0]))

        if not lines:
            return []

        found: list[SymbolRange] = []
        seen: set[str] = set()

        for line_num in lines:
            for sym in file_symbols:
                ls = sym.get("line_start", 0)
                le = sym.get("line_end", 0)
                if ls <= line_num <= le:
                    key = f"{sym['name']}:{sym.get('file_path', '')}:{ls}"
                    if key not in seen:
                        seen.add(key)
                        found.append(self._sym_dict_to_range(sym, editable=True))

        return found

    def _resolve_error_location(
        self,
        task: str,
        target_file: str,
        file_symbols: list[dict],
        graph,
    ) -> list[SymbolRange]:
        """Extract file:line from stack traces and map to symbols."""
        found: list[SymbolRange] = []
        seen: set[str] = set()

        for pat in _TRACEBACK_PATTERNS:
            for m in pat.finditer(task):
                file_ref, line_str = m.group(1), m.group(2)
                line_num = int(line_str)

                # Check if the file reference matches the target file
                if not (target_file.endswith(file_ref) or file_ref.endswith(target_file)):
                    # Try finding symbols in the referenced file instead
                    ref_symbols = graph.get_file_symbols(file_ref)
                    if ref_symbols:
                        for sym in ref_symbols:
                            ls = sym.get("line_start", 0)
                            le = sym.get("line_end", 0)
                            if ls <= line_num <= le:
                                key = f"{sym['name']}:{sym.get('file_path', '')}:{ls}"
                                if key not in seen:
                                    seen.add(key)
                                    found.append(self._sym_dict_to_range(sym, editable=True))
                    continue

                for sym in file_symbols:
                    ls = sym.get("line_start", 0)
                    le = sym.get("line_end", 0)
                    if ls <= line_num <= le:
                        key = f"{sym['name']}:{sym.get('file_path', '')}:{ls}"
                        if key not in seen:
                            seen.add(key)
                            found.append(self._sym_dict_to_range(sym, editable=True))

        return found

    def _resolve_semantic(
        self,
        task: str,
        target_file: str,
        file_symbols: list[dict],
    ) -> list[SymbolRange]:
        """Fuzzy-match task keywords against symbol names."""
        # Extract candidate words (3+ chars, not common English)
        words = set(re.findall(r'\b[A-Za-z_]\w{2,}\b', task.lower()))
        _common = {
            "the", "and", "for", "that", "this", "with", "from", "not", "but",
            "are", "was", "were", "been", "have", "has", "had", "will", "can",
            "should", "would", "could", "may", "might", "must", "shall",
            "fix", "add", "remove", "update", "change", "make", "handle",
            "function", "method", "class", "file", "code", "error", "bug",
            "implement", "create", "delete", "modify", "return", "value",
            "line", "lines", "new", "old", "use", "using", "also",
        }
        words -= _common

        found: list[SymbolRange] = []
        seen: set[str] = set()

        for sym in file_symbols:
            name = sym.get("name", "")
            if not name:
                continue
            # Split camelCase/snake_case into parts
            name_parts = set(re.findall(r'[a-z]+', name.lower()))

            best_score = 0.0
            for word in words:
                # Direct fuzzy match against full name
                score = SequenceMatcher(None, word, name.lower()).ratio()
                best_score = max(best_score, score)
                # Also match against individual name parts
                for part in name_parts:
                    if len(part) >= 3:
                        part_score = SequenceMatcher(None, word, part).ratio()
                        best_score = max(best_score, part_score)

            if best_score >= 0.75:
                key = f"{name}:{sym.get('file_path', '')}:{sym.get('line_start', 0)}"
                if key not in seen:
                    seen.add(key)
                    found.append(self._sym_dict_to_range(sym, editable=True))

        return found

    def _expand_multi_file(self, scope: EditScope, target_file: str, graph) -> None:
        """Use impact_analysis to find other affected files."""
        affected = graph.impact_analysis(target_file)
        primary_names = {s.symbol_name for s in scope.primary_symbols}

        for af in affected:
            if af == target_file:
                continue
            # Check if any symbols in this file reference our primary symbols
            af_symbols = graph.get_file_symbols(af)
            for sym in af_symbols:
                if sym.get("name") in primary_names:
                    if af not in scope.affected_files:
                        scope.affected_files.append(af)
                    scope.primary_symbols.append(
                        self._sym_dict_to_range(sym, editable=True)
                    )
                    break

    def _add_context_symbols(self, scope: EditScope, graph) -> None:
        """Fetch 1-hop neighbors of primary symbols as read-only context."""
        seen_primary = {
            (s.symbol_name, s.file_path, s.line_start)
            for s in scope.primary_symbols
        }
        seen_context: set[tuple[str, str, int]] = set()

        for primary in list(scope.primary_symbols):
            related = graph.get_related_symbols(primary.symbol_name, depth=1)
            for rel in related:
                key = (rel.get("name", ""), rel.get("file_path", ""), rel.get("line_start", 0))
                if key not in seen_primary and key not in seen_context:
                    seen_context.add(key)
                    # Only include symbols from the same files we're editing
                    if rel.get("file_path") in scope.affected_files:
                        scope.context_symbols.append(
                            self._sym_dict_to_range(rel, editable=False)
                        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sym_dict_to_range(sym: dict, editable: bool) -> SymbolRange:
        """Convert a graph node summary dict to a SymbolRange."""
        node_type = sym.get("node_type", "").upper()
        if node_type == "FUNCTION":
            stype = "method" if sym.get("parent_class") else "function"
        elif node_type == "CLASS":
            stype = "class"
        else:
            stype = "variable"

        return SymbolRange(
            symbol_name=sym.get("name", ""),
            symbol_type=stype,
            file_path=sym.get("file_path", ""),
            line_start=sym.get("line_start", 0),
            line_end=sym.get("line_end", 0),
            editable=editable,
            parent_class=sym.get("parent_class"),
        )
