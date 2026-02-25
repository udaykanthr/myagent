"""
Unit tests for multi_agent_coder.kb.local.graph

Tests all public query methods with a deterministic in-memory graph.
"""

from __future__ import annotations

import os
import sys
import tempfile
import pickle
import pytest

# ---------------------------------------------------------------------------
# Helpers — build a small synthetic graph without tree-sitter
# ---------------------------------------------------------------------------

def _make_graph():
    """
    Return a CodeGraph populated with synthetic nodes / edges.

    Graph layout (Python project):

        files/a.py  contains:
            class Foo (inherits Bar from b.py)
                method __init__
                method process   ← calls helper (b.py)

        files/b.py  contains:
            class Bar
                method helper

        files/c.py  imports a.py
    """
    from multi_agent_coder.kb.local.graph import (
        CodeGraph, NodeType, EdgeType,
        _file_id, _func_id, _class_id,
    )
    from multi_agent_coder.kb.local.parser import (
        ParsedFile, ParsedFunction, ParsedClass,
        ParsedImport, ParsedCall,
    )

    g = CodeGraph()

    # --- b.py ---
    b = ParsedFile(
        path="files/b.py",
        language="python",
        hash="bbbb",
        classes=[
            ParsedClass(name="Bar", file_path="files/b.py", line_start=1, line_end=20),
        ],
        functions=[
            ParsedFunction(
                name="helper",
                file_path="files/b.py",
                line_start=5, line_end=15,
                parent_class="Bar",
            ),
        ],
    )
    g.add_parsed_file(b)

    # --- a.py ---
    a = ParsedFile(
        path="files/a.py",
        language="python",
        hash="aaaa",
        classes=[
            ParsedClass(
                name="Foo",
                file_path="files/a.py",
                line_start=1, line_end=30,
                bases=["Bar"],
            ),
        ],
        functions=[
            ParsedFunction(
                name="__init__",
                file_path="files/a.py",
                line_start=5, line_end=10,
                parent_class="Foo",
            ),
            ParsedFunction(
                name="process",
                file_path="files/a.py",
                line_start=12, line_end=25,
                parent_class="Foo",
            ),
        ],
        imports=[
            ParsedImport(source_file="files/a.py", imported_name="files.b"),
        ],
        calls=[
            ParsedCall(
                caller_function="process",
                callee_name="helper",
                file_path="files/a.py",
                line=15,
            ),
        ],
    )
    g.add_parsed_file(a)

    # --- c.py ---
    c = ParsedFile(
        path="files/c.py",
        language="python",
        hash="cccc",
        imports=[
            ParsedImport(source_file="files/c.py", imported_name="files.a"),
        ],
    )
    g.add_parsed_file(c)

    # Resolve import edges
    g.resolve_import_edges({
        "files.a": "files/a.py",
        "files.b": "files/b.py",
    })

    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCodeGraphBasic:
    """Verify nodes and edges are built correctly."""

    def setup_method(self):
        pytest.importorskip("networkx")
        self.g = _make_graph()

    def test_node_count(self):
        stats = self.g.stats()
        # FILES: a.py, b.py, c.py
        assert stats["by_node_type"].get("FILE", 0) == 3

    def test_function_nodes(self):
        stats = self.g.stats()
        # __init__, process, helper
        assert stats["by_node_type"].get("FUNCTION", 0) == 3

    def test_class_nodes(self):
        stats = self.g.stats()
        assert stats["by_node_type"].get("CLASS", 0) == 2  # Foo, Bar

    def test_edge_count_positive(self):
        stats = self.g.stats()
        assert stats["edge_count"] > 0


class TestFindCallers:
    """find_callers returns functions that call the target."""

    def setup_method(self):
        pytest.importorskip("networkx")
        self.g = _make_graph()

    def test_helper_called_by_process(self):
        callers = self.g.find_callers("helper")
        names = [c["name"] for c in callers]
        assert "process" in names

    def test_unknown_function(self):
        callers = self.g.find_callers("does_not_exist")
        assert callers == []


class TestFindCallees:
    """find_callees returns functions called by the source."""

    def setup_method(self):
        pytest.importorskip("networkx")
        self.g = _make_graph()

    def test_process_calls_helper(self):
        callees = self.g.find_callees("process")
        names = [c["name"] for c in callees]
        assert "helper" in names

    def test_init_no_callees(self):
        # __init__ has no calls registered
        callees = self.g.find_callees("__init__")
        assert callees == []


class TestGetInheritanceChain:
    """get_inheritance_chain traverses INHERITS edges upward."""

    def setup_method(self):
        pytest.importorskip("networkx")
        self.g = _make_graph()

    def test_foo_inherits_bar(self):
        chain = self.g.get_inheritance_chain("Foo")
        names = [c["name"] for c in chain]
        assert "Bar" in names

    def test_bar_has_empty_chain(self):
        chain = self.g.get_inheritance_chain("Bar")
        assert chain == []


class TestGetFileSymbols:
    """get_file_symbols returns all symbols from a file."""

    def setup_method(self):
        pytest.importorskip("networkx")
        self.g = _make_graph()

    def test_a_symbols(self):
        syms = self.g.get_file_symbols("files/a.py")
        names = {s["name"] for s in syms}
        assert "Foo" in names
        assert "__init__" in names
        assert "process" in names

    def test_b_symbols(self):
        syms = self.g.get_file_symbols("files/b.py")
        names = {s["name"] for s in syms}
        assert "Bar" in names
        assert "helper" in names

    def test_missing_file(self):
        syms = self.g.get_file_symbols("nonexistent.py")
        assert syms == []


class TestImpactAnalysis:
    """impact_analysis returns files affected when a file changes."""

    def setup_method(self):
        pytest.importorskip("networkx")
        self.g = _make_graph()

    def test_b_affects_a(self):
        affected = self.g.impact_analysis("files/b.py")
        assert "files/a.py" in affected

    def test_a_affects_c(self):
        affected = self.g.impact_analysis("files/a.py")
        assert "files/c.py" in affected

    def test_c_affects_nothing(self):
        affected = self.g.impact_analysis("files/c.py")
        assert affected == []


class TestFindSymbol:
    """find_symbol locates any symbol by name."""

    def setup_method(self):
        pytest.importorskip("networkx")
        self.g = _make_graph()

    def test_find_function(self):
        results = self.g.find_symbol("process")
        assert any(r["node_type"] == "FUNCTION" for r in results)

    def test_find_class(self):
        results = self.g.find_symbol("Foo")
        assert any(r["node_type"] == "CLASS" for r in results)

    def test_type_filter(self):
        results = self.g.find_symbol("process", symbol_type="FUNCTION")
        assert all(r["node_type"] == "FUNCTION" for r in results)

    def test_not_found(self):
        results = self.g.find_symbol("xyz_not_here")
        assert results == []


class TestGetRelatedSymbols:
    """get_related_symbols traverses up to N hops."""

    def setup_method(self):
        pytest.importorskip("networkx")
        self.g = _make_graph()

    def test_related_to_process_depth1(self):
        related = self.g.get_related_symbols("process", depth=1)
        names = {r["name"] for r in related}
        # helper is a direct callee of process
        assert "helper" in names

    def test_related_depth2(self):
        # From helper (depth 2), should reach process and potentially Bar
        related = self.g.get_related_symbols("helper", depth=2)
        names = {r["name"] for r in related}
        assert len(names) > 0


class TestSerialisation:
    """Graph can be saved and reloaded."""

    def setup_method(self):
        pytest.importorskip("networkx")

    def test_save_and_load(self, tmp_path):
        g = _make_graph()
        pkl_path = str(tmp_path / "graph.pkl")
        g.save(pkl_path)
        assert os.path.exists(pkl_path)

        from multi_agent_coder.kb.local.graph import CodeGraph
        g2 = CodeGraph.load(pkl_path)
        assert g2.stats()["node_count"] == g.stats()["node_count"]
        assert g2.stats()["edge_count"] == g.stats()["edge_count"]

    def test_load_missing_file(self, tmp_path):
        from multi_agent_coder.kb.local.graph import CodeGraph
        with pytest.raises(FileNotFoundError):
            CodeGraph.load(str(tmp_path / "missing.pkl"))


class TestRemoveFile:
    """remove_file cleans up all associated nodes."""

    def setup_method(self):
        pytest.importorskip("networkx")

    def test_remove_reduces_node_count(self):
        g = _make_graph()
        before = g.stats()["node_count"]
        g.remove_file("files/a.py")
        after = g.stats()["node_count"]
        assert after < before

    def test_symbols_gone_after_remove(self):
        g = _make_graph()
        g.remove_file("files/a.py")
        syms = g.get_file_symbols("files/a.py")
        assert syms == []
