"""
NetworkX-based code graph for the Local Knowledge Base.

Builds a directed multi-graph where nodes represent code entities (files,
modules, classes, functions, variables) and edges represent structural
relationships (imports, calls, inheritance, containment, etc.).
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Optional

try:
    import networkx as nx  # type: ignore
except ImportError:
    nx = None  # type: ignore

from .parser import ParsedFile, ParsedFunction, ParsedClass, ParsedVariable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node / Edge type constants
# ---------------------------------------------------------------------------

class NodeType:
    FILE = "FILE"
    MODULE = "MODULE"
    CLASS = "CLASS"
    FUNCTION = "FUNCTION"
    VARIABLE = "VARIABLE"


class EdgeType:
    CONTAINS = "CONTAINS"
    CALLS = "CALLS"
    INHERITS = "INHERITS"
    IMPORTS = "IMPORTS"
    REFERENCES = "REFERENCES"
    OVERRIDES = "OVERRIDES"


# ---------------------------------------------------------------------------
# Node-ID helpers
# ---------------------------------------------------------------------------

def _file_id(path: str) -> str:
    return f"FILE:{path}"


def _func_id(name: str, file_path: str, parent_class: Optional[str] = None) -> str:
    if parent_class:
        return f"FUNC:{file_path}::{parent_class}.{name}"
    return f"FUNC:{file_path}::{name}"


def _class_id(name: str, file_path: str) -> str:
    return f"CLASS:{file_path}::{name}"


def _var_id(name: str, file_path: str, scope: str) -> str:
    return f"VAR:{file_path}::{scope}::{name}"


def _module_id(name: str) -> str:
    return f"MODULE:{name}"


# ---------------------------------------------------------------------------
# CodeGraph
# ---------------------------------------------------------------------------

class CodeGraph:
    """
    Directed graph representing the structural relationships in a codebase.

    Nodes represent code entities; edges represent relationships.  The graph
    can be serialised to / deserialised from a pickle file.

    Node types: FILE, MODULE, CLASS, FUNCTION, VARIABLE
    Edge types: CONTAINS, CALLS, INHERITS, IMPORTS, REFERENCES, OVERRIDES
    """

    def __init__(self) -> None:
        if nx is None:
            raise RuntimeError(
                "networkx is not installed. Install with: pip install networkx"
            )
        self._g: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------

    def _add_node(self, node_id: str, **attrs: Any) -> None:
        """Add or update a node, merging attributes."""
        if self._g.has_node(node_id):
            self._g.nodes[node_id].update(attrs)
        else:
            self._g.add_node(node_id, **attrs)

    def _add_edge(self, src: str, dst: str, edge_type: str, **attrs: Any) -> None:
        """Add a directed edge if both endpoints exist."""
        if not self._g.has_node(src) or not self._g.has_node(dst):
            return
        # Avoid duplicate edges of same type
        if self._g.has_edge(src, dst) and self._g[src][dst].get("type") == edge_type:
            return
        self._g.add_edge(src, dst, type=edge_type, **attrs)

    # ------------------------------------------------------------------
    # Building from parsed files
    # ------------------------------------------------------------------

    def add_parsed_file(self, parsed: ParsedFile) -> None:
        """
        Incorporate all structural information from *parsed* into the graph.

        This method is idempotent for the same file: re-calling it after
        removing the file's nodes (via :meth:`remove_file`) re-adds them.

        Parameters
        ----------
        parsed:
            The result of :func:`~agentchanti.kb.local.parser.parse_file`.
        """
        if parsed.parse_error and not (parsed.functions or parsed.classes):
            logger.debug("Skipping %s due to parse error: %s", parsed.path, parsed.parse_error)
            return

        fid = _file_id(parsed.path)
        self._add_node(
            fid,
            node_type=NodeType.FILE,
            path=parsed.path,
            language=parsed.language,
            hash=parsed.hash,
        )

        # --- Classes ---
        for cls in parsed.classes:
            cid = _class_id(cls.name, parsed.path)
            self._add_node(
                cid,
                node_type=NodeType.CLASS,
                name=cls.name,
                file_path=parsed.path,
                line_start=cls.line_start,
                line_end=cls.line_end,
                docstring=cls.docstring,
                bases=cls.bases,
            )
            self._add_edge(fid, cid, EdgeType.CONTAINS)

        # --- Functions / methods ---
        for fn in parsed.functions:
            fnid = _func_id(fn.name, parsed.path, fn.parent_class)
            self._add_node(
                fnid,
                node_type=NodeType.FUNCTION,
                name=fn.name,
                file_path=parsed.path,
                line_start=fn.line_start,
                line_end=fn.line_end,
                docstring=fn.docstring,
                params=fn.params,
                return_type=fn.return_type,
                parent_class=fn.parent_class,
            )
            if fn.parent_class:
                parent_cid = _class_id(fn.parent_class, parsed.path)
                if self._g.has_node(parent_cid):
                    self._add_edge(parent_cid, fnid, EdgeType.CONTAINS)
                else:
                    self._add_edge(fid, fnid, EdgeType.CONTAINS)
            else:
                self._add_edge(fid, fnid, EdgeType.CONTAINS)

        # --- Variables ---
        for var in parsed.variables:
            vid = _var_id(var.name, parsed.path, var.scope)
            self._add_node(
                vid,
                node_type=NodeType.VARIABLE,
                name=var.name,
                file_path=parsed.path,
                scope=var.scope,
                type_hint=var.type_hint,
            )
            self._add_edge(fid, vid, EdgeType.CONTAINS)

        # --- Imports ---
        for imp in parsed.imports:
            mid = _module_id(imp.imported_name)
            # Only add a MODULE node if we don't already have a FILE for it
            if not self._g.has_node(mid):
                self._add_node(mid, node_type=NodeType.MODULE, name=imp.imported_name)
            self._add_edge(fid, mid, EdgeType.IMPORTS)

        # --- Inheritance edges ---
        for cls in parsed.classes:
            cid = _class_id(cls.name, parsed.path)
            for base in cls.bases:
                # Try to find the base class node in the graph
                base_node = self._find_class_node(base)
                if base_node:
                    self._add_edge(cid, base_node, EdgeType.INHERITS)

        # --- Call edges ---
        # Map function name → node_id for functions defined in this file
        local_func_map: dict[str, str] = {}
        for fn in parsed.functions:
            fnid = _func_id(fn.name, parsed.path, fn.parent_class)
            local_func_map[fn.name] = fnid

        for call in parsed.calls:
            # Resolve caller
            caller_fn = parsed.functions[0] if parsed.functions else None
            for fn in parsed.functions:
                if fn.name == call.caller_function:
                    caller_fn = fn
                    break
            if caller_fn is None:
                continue
            caller_id = _func_id(
                caller_fn.name, parsed.path, caller_fn.parent_class
            )
            # Resolve callee — look in local file first, then anywhere in graph
            callee_id = self._resolve_callee(call.callee_name, parsed.path)
            if callee_id and self._g.has_node(caller_id):
                self._add_edge(caller_id, callee_id, EdgeType.CALLS)

    def _find_class_node(self, class_name: str) -> Optional[str]:
        """Find the first CLASS node with the given name."""
        for nid, attrs in self._g.nodes(data=True):
            if attrs.get("node_type") == NodeType.CLASS and attrs.get("name") == class_name:
                return nid
        return None

    def _resolve_callee(self, callee_name: str, caller_file: str) -> Optional[str]:
        """
        Try to find the FUNCTION node for *callee_name*.
        First searches within *caller_file*, then across the whole graph.
        """
        # Local search
        for nid, attrs in self._g.nodes(data=True):
            if (
                attrs.get("node_type") == NodeType.FUNCTION
                and attrs.get("name") == callee_name
                and attrs.get("file_path") == caller_file
            ):
                return nid
        # Global search
        for nid, attrs in self._g.nodes(data=True):
            if (
                attrs.get("node_type") == NodeType.FUNCTION
                and attrs.get("name") == callee_name
            ):
                return nid
        return None

    # ------------------------------------------------------------------
    # File removal (for incremental updates)
    # ------------------------------------------------------------------

    def remove_file(self, file_path: str) -> None:
        """
        Remove all nodes and edges associated with *file_path* from the graph.

        Safe to call even if the file was never indexed.

        Parameters
        ----------
        file_path:
            Path of the file whose nodes should be removed.
        """
        nodes_to_remove = [
            nid for nid, attrs in self._g.nodes(data=True)
            if attrs.get("file_path") == file_path or attrs.get("path") == file_path
        ]
        self._g.remove_nodes_from(nodes_to_remove)
        logger.debug("Removed %d nodes for file %s", len(nodes_to_remove), file_path)

    # ------------------------------------------------------------------
    # Wiring import edges after all files are indexed
    # ------------------------------------------------------------------

    def resolve_import_edges(self, path_by_module: dict[str, str]) -> None:
        """
        Replace MODULE→FILE import edges once all files are known.

        After a full index, call this method with a mapping from module name
        to its file path to convert abstract module nodes to file-to-file
        IMPORTS edges.

        Parameters
        ----------
        path_by_module:
            dict mapping dotted module name → file path.
        """
        edges_to_add: list[tuple[str, str]] = []
        for src, dst, data in list(self._g.edges(data=True)):
            if data.get("type") != EdgeType.IMPORTS:
                continue
            dst_attrs = self._g.nodes.get(dst, {})
            if dst_attrs.get("node_type") == NodeType.MODULE:
                mod_name = dst_attrs.get("name", "")
                if mod_name in path_by_module:
                    target_fid = _file_id(path_by_module[mod_name])
                    if self._g.has_node(target_fid):
                        edges_to_add.append((src, target_fid))
        for src, dst in edges_to_add:
            self._add_edge(src, dst, EdgeType.IMPORTS)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def find_callers(self, function_name: str) -> list[dict]:
        """
        Return all functions that call *function_name*.

        Parameters
        ----------
        function_name:
            The name of the function to look up.

        Returns
        -------
        list[dict]
            Each item contains: name, file_path, line_start, line_end.
        """
        target_ids = self._find_function_nodes(function_name)
        results: list[dict] = []
        seen: set[str] = set()
        for tid in target_ids:
            for pred in self._g.predecessors(tid):
                if self._g[pred][tid].get("type") != EdgeType.CALLS:
                    continue
                if pred in seen:
                    continue
                seen.add(pred)
                attrs = self._g.nodes[pred]
                results.append(self._node_summary(pred, attrs))
        return results

    def find_callees(self, function_name: str) -> list[dict]:
        """
        Return all functions called by *function_name*.

        Parameters
        ----------
        function_name:
            The name of the calling function.

        Returns
        -------
        list[dict]
            Each item contains: name, file_path, line_start, line_end.
        """
        source_ids = self._find_function_nodes(function_name)
        results: list[dict] = []
        seen: set[str] = set()
        for sid in source_ids:
            for succ in self._g.successors(sid):
                if self._g[sid][succ].get("type") != EdgeType.CALLS:
                    continue
                if succ in seen:
                    continue
                seen.add(succ)
                attrs = self._g.nodes[succ]
                results.append(self._node_summary(succ, attrs))
        return results

    def find_references(self, symbol_name: str) -> list[dict]:
        """
        Return all locations where *symbol_name* is referenced (called or
        used as a variable).

        Parameters
        ----------
        symbol_name:
            The symbol name to look up.

        Returns
        -------
        list[dict]
            Each item contains node summary data.
        """
        target_ids = set(
            self._find_function_nodes(symbol_name)
            + self._find_class_nodes(symbol_name)
            + self._find_variable_nodes(symbol_name)
        )
        results: list[dict] = []
        seen: set[str] = set()
        for tid in target_ids:
            for pred in self._g.predecessors(tid):
                edge_type = self._g[pred][tid].get("type")
                if edge_type not in (EdgeType.CALLS, EdgeType.REFERENCES, EdgeType.IMPORTS):
                    continue
                if pred in seen:
                    continue
                seen.add(pred)
                attrs = self._g.nodes[pred]
                results.append(self._node_summary(pred, attrs))
        return results

    def get_inheritance_chain(self, class_name: str) -> list[dict]:
        """
        Return the full inheritance hierarchy for *class_name*.

        Traverses INHERITS edges upward (toward base classes).

        Parameters
        ----------
        class_name:
            The name of the class to start from.

        Returns
        -------
        list[dict]
            Ordered list from immediate parent to root, each containing
            node summary data.
        """
        start_ids = self._find_class_nodes(class_name)
        chain: list[dict] = []
        visited: set[str] = set()
        queue = list(start_ids)
        while queue:
            cid = queue.pop(0)
            if cid in visited:
                continue
            visited.add(cid)
            for succ in self._g.successors(cid):
                if self._g[cid][succ].get("type") == EdgeType.INHERITS:
                    attrs = self._g.nodes[succ]
                    chain.append(self._node_summary(succ, attrs))
                    queue.append(succ)
        return chain

    def get_file_symbols(self, file_path: str) -> list[dict]:
        """
        Return all symbols (functions, classes, variables) defined in
        *file_path*.

        Parameters
        ----------
        file_path:
            Relative or absolute path to the file.

        Returns
        -------
        list[dict]
            Each item is a node summary dict.
        """
        results: list[dict] = []
        for nid, attrs in self._g.nodes(data=True):
            if attrs.get("file_path") == file_path and attrs.get("node_type") in (
                NodeType.FUNCTION, NodeType.CLASS, NodeType.VARIABLE
            ):
                results.append(self._node_summary(nid, attrs))
        return results

    def impact_analysis(self, file_path: str) -> list[str]:
        """
        Return all files that may be affected if *file_path* changes.

        Performs a reverse traversal of IMPORTS edges starting from the
        file's node.

        Parameters
        ----------
        file_path:
            Path of the file that changed.

        Returns
        -------
        list[str]
            File paths of files that directly or indirectly import *file_path*.
        """
        fid = _file_id(file_path)
        if not self._g.has_node(fid):
            return []
        affected: set[str] = set()
        # BFS over reversed import edges
        queue = [fid]
        while queue:
            current = queue.pop(0)
            for pred in self._g.predecessors(current):
                edge_type = self._g[pred][current].get("type")
                if edge_type != EdgeType.IMPORTS:
                    continue
                pred_attrs = self._g.nodes[pred]
                if pred_attrs.get("node_type") == NodeType.FILE:
                    fp = pred_attrs.get("path", "")
                    if fp and fp != file_path and fp not in affected:
                        affected.add(fp)
                        queue.append(pred)
        return sorted(affected)

    def get_all_file_nodes(self) -> list[dict]:
        """Return all FILE nodes in the graph.

        Returns
        -------
        list[dict]
            Each item contains: id, path, language, hash, and node_type.
        """
        return [
            {"id": nid, "path": attrs.get("path", ""), **attrs}
            for nid, attrs in self._g.nodes(data=True)
            if attrs.get("node_type") == NodeType.FILE
        ]

    def find_symbol(
        self,
        name: str,
        symbol_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Locate any symbol by name across the whole graph.

        Parameters
        ----------
        name:
            Symbol name to search for.
        symbol_type:
            Optional filter: "FUNCTION", "CLASS", or "VARIABLE".

        Returns
        -------
        list[dict]
            Matching node summaries.
        """
        results: list[dict] = []
        for nid, attrs in self._g.nodes(data=True):
            if attrs.get("name") != name:
                continue
            if symbol_type and attrs.get("node_type") != symbol_type:
                continue
            results.append(self._node_summary(nid, attrs))
        return results

    def get_related_symbols(self, symbol_name: str, depth: int = 2) -> list[dict]:
        """
        Return all graph neighbours of *symbol_name* up to *depth* hops away.

        Parameters
        ----------
        symbol_name:
            Starting symbol name.
        depth:
            Number of hops to traverse.

        Returns
        -------
        list[dict]
            Deduplicated node summaries for all reachable neighbours.
        """
        start_ids = (
            self._find_function_nodes(symbol_name)
            + self._find_class_nodes(symbol_name)
            + self._find_variable_nodes(symbol_name)
        )
        if not start_ids:
            return []
        visited: set[str] = set(start_ids)
        frontier = set(start_ids)
        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                for nbr in list(self._g.predecessors(nid)) + list(self._g.successors(nid)):
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.add(nbr)
            frontier = next_frontier
        # Exclude the starting nodes themselves
        related_ids = visited - set(start_ids)
        results = []
        for nid in related_ids:
            attrs = self._g.nodes[nid]
            if attrs.get("node_type") in (NodeType.FUNCTION, NodeType.CLASS, NodeType.VARIABLE):
                results.append(self._node_summary(nid, attrs))
        return results

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise the graph to *path* using pickle.

        Parameters
        ----------
        path:
            Destination file path (e.g. .agentchanti/kb/local/graph.pkl).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._g, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("Saved graph (%d nodes, %d edges) to %s",
                     self._g.number_of_nodes(), self._g.number_of_edges(), path)

    @classmethod
    def load(cls, path: str) -> "CodeGraph":
        """
        Deserialise a graph previously saved with :meth:`save`.

        Parameters
        ----------
        path:
            Path to the pickle file.

        Returns
        -------
        CodeGraph
            Loaded graph instance.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph file not found: {path}")
        instance = cls.__new__(cls)
        if nx is None:
            raise RuntimeError("networkx is not installed")
        with open(path, "rb") as fh:
            instance._g = pickle.load(fh)
        return instance

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """
        Return aggregate statistics about the graph.

        Returns
        -------
        dict
            Keys: node_count, edge_count, by_node_type (dict), by_edge_type (dict).
        """
        by_node: dict[str, int] = {}
        for _, attrs in self._g.nodes(data=True):
            nt = attrs.get("node_type", "unknown")
            by_node[nt] = by_node.get(nt, 0) + 1

        by_edge: dict[str, int] = {}
        for _, _, attrs in self._g.edges(data=True):
            et = attrs.get("type", "unknown")
            by_edge[et] = by_edge.get(et, 0) + 1

        return {
            "node_count": self._g.number_of_nodes(),
            "edge_count": self._g.number_of_edges(),
            "by_node_type": by_node,
            "by_edge_type": by_edge,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_function_nodes(self, name: str) -> list[str]:
        return [
            nid for nid, attrs in self._g.nodes(data=True)
            if attrs.get("node_type") == NodeType.FUNCTION and attrs.get("name") == name
        ]

    def _find_class_nodes(self, name: str) -> list[str]:
        return [
            nid for nid, attrs in self._g.nodes(data=True)
            if attrs.get("node_type") == NodeType.CLASS and attrs.get("name") == name
        ]

    def _find_variable_nodes(self, name: str) -> list[str]:
        return [
            nid for nid, attrs in self._g.nodes(data=True)
            if attrs.get("node_type") == NodeType.VARIABLE and attrs.get("name") == name
        ]

    @staticmethod
    def _node_summary(node_id: str, attrs: dict) -> dict:
        """Return a compact, serialisable summary of a node."""
        return {
            "id": node_id,
            "node_type": attrs.get("node_type", ""),
            "name": attrs.get("name", ""),
            "file_path": attrs.get("file_path", attrs.get("path", "")),
            "line_start": attrs.get("line_start", 0),
            "line_end": attrs.get("line_end", 0),
            "docstring": attrs.get("docstring", ""),
            "parent_class": attrs.get("parent_class"),
        }
