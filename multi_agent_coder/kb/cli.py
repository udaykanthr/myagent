"""
`agentchanti kb` subcommand CLI.

Provides commands for building, querying and watching the Local
Knowledge Base code graph.

Commands
--------
agentchanti kb index                           -- full re-index of current project
agentchanti kb index --watch                   -- full index then start file watcher
agentchanti kb status                          -- show graph_meta.json summary
agentchanti kb query find-callers <function_name>
agentchanti kb query find-callees <function_name>
agentchanti kb query find-refs    <symbol_name>
agentchanti kb query impact       <file_path>
agentchanti kb query symbol       <name>

Phase 2 — Semantic Layer
agentchanti kb embed               -- embed all symbols (full)
agentchanti kb embed --incremental -- embed only changed symbols
agentchanti kb search "<query>"    -- semantic search
agentchanti kb search "<query>" --top-k 5
agentchanti kb search "<query>" --filter file=src/auth
agentchanti kb search "<query>" --filter language=python
agentchanti kb qdrant start        -- start Qdrant Docker container
agentchanti kb qdrant stop         -- stop Qdrant Docker container
agentchanti kb qdrant status       -- show Qdrant status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> str:
    """Return the current working directory as project root."""
    return os.getcwd()


def _require_index(project_root: str) -> None:
    """Exit with an informative message if the KB has not been indexed."""
    from .local.indexer import Indexer
    indexer = Indexer(project_root)
    if not indexer.is_indexed():
        print(
            "No KB index found. Run `agentchanti kb index` first.",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_results(results: list[dict], title: str) -> None:
    """Pretty-print a list of node summary dicts."""
    if not results:
        print(f"  (no results for: {title})")
        return
    print(f"\n{title}  [{len(results)} result(s)]")
    print("-" * 60)
    for r in results:
        ntype = r.get("node_type", "")
        name = r.get("name", "")
        fpath = r.get("file_path", "")
        ls = r.get("line_start", 0)
        le = r.get("line_end", 0)
        parent = r.get("parent_class")
        location = f"{fpath}:{ls}-{le}" if ls else fpath
        label = f"{ntype:<10}  {name}"
        if parent:
            label += f"  (in {parent})"
        print(f"  {label:<50}  {location}")


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def _cmd_index(args: argparse.Namespace) -> None:
    """Run a full index of the current project."""
    project_root = _project_root()

    try:
        from tqdm import tqdm  # type: ignore
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    print(f"Indexing project: {project_root}")

    from .local.indexer import Indexer
    indexer = Indexer(project_root)

    if _tqdm_available:
        pbar = tqdm(total=None, unit="file", desc="Parsing")

        def _progress(current: int, total: int, filename: str) -> None:
            if pbar.total != total:
                pbar.total = total
                pbar.refresh()
            pbar.set_postfix_str(os.path.basename(filename), refresh=False)
            pbar.update(1)

        summary = indexer.full_index(progress_callback=_progress)
        pbar.close()
    else:
        # Fallback without tqdm
        _last_pct = [0]

        def _progress(current: int, total: int, filename: str) -> None:
            pct = int(100 * current / total) if total else 0
            if pct >= _last_pct[0] + 10:
                _last_pct[0] = pct
                print(f"  [{pct:3d}%] {filename}")

        summary = indexer.full_index(progress_callback=_progress)

    print(
        f"\nIndex complete:\n"
        f"  Files:   {summary['file_count']}\n"
        f"  Symbols: {summary['symbol_count']}\n"
        f"  Edges:   {summary['edge_count']}\n"
        f"  Errors:  {summary['error_count']}\n"
        f"  Time:    {summary['elapsed_seconds']:.1f}s"
    )

    if args.watch:
        print("\nStarting file watcher... (Ctrl+C to stop)")
        from .local.watcher import KBWatcher
        watcher = KBWatcher(indexer, project_root)
        try:
            watcher.start()  # blocking
        except KeyboardInterrupt:
            print("\nFile watcher stopped.")


def _cmd_status(args: argparse.Namespace) -> None:
    """Print the graph_meta.json summary."""
    project_root = _project_root()
    from .local.indexer import read_meta
    meta = read_meta(project_root)
    if meta is None:
        print("No KB index found. Run `agentchanti kb index` first.")
        return
    print("\nKnowledge Base Status")
    print("=" * 40)
    for k, v in meta.items():
        print(f"  {k:<20} {v}")
    print()


def _cmd_query(args: argparse.Namespace) -> None:
    """Dispatch graph query subcommands."""
    project_root = _project_root()
    _require_index(project_root)

    from .local.indexer import Indexer
    indexer = Indexer(project_root)
    try:
        graph = indexer.load_graph()
    except Exception as exc:
        print(f"Failed to load graph: {exc}", file=sys.stderr)
        sys.exit(1)

    query_cmd = args.query_cmd

    if query_cmd == "find-callers":
        if not args.name:
            print("Usage: agentchanti kb query find-callers <function_name>", file=sys.stderr)
            sys.exit(1)
        t0 = time.perf_counter()
        results = graph.find_callers(args.name)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _print_results(results, f"Callers of '{args.name}'")
        print(f"\n  Query time: {elapsed_ms:.1f}ms")

    elif query_cmd == "find-callees":
        if not args.name:
            print("Usage: agentchanti kb query find-callees <function_name>", file=sys.stderr)
            sys.exit(1)
        t0 = time.perf_counter()
        results = graph.find_callees(args.name)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _print_results(results, f"Callees of '{args.name}'")
        print(f"\n  Query time: {elapsed_ms:.1f}ms")

    elif query_cmd == "find-refs":
        if not args.name:
            print("Usage: agentchanti kb query find-refs <symbol_name>", file=sys.stderr)
            sys.exit(1)
        t0 = time.perf_counter()
        results = graph.find_references(args.name)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _print_results(results, f"References to '{args.name}'")
        print(f"\n  Query time: {elapsed_ms:.1f}ms")

    elif query_cmd == "impact":
        if not args.name:
            print("Usage: agentchanti kb query impact <file_path>", file=sys.stderr)
            sys.exit(1)
        t0 = time.perf_counter()
        affected = graph.impact_analysis(args.name)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if affected:
            print(f"\nFiles affected by changes to '{args.name}'  [{len(affected)} file(s)]")
            print("-" * 60)
            for fp in affected:
                print(f"  {fp}")
        else:
            print(f"  No files depend on '{args.name}'")
        print(f"\n  Query time: {elapsed_ms:.1f}ms")

    elif query_cmd == "symbol":
        if not args.name:
            print("Usage: agentchanti kb query symbol <name>", file=sys.stderr)
            sys.exit(1)
        t0 = time.perf_counter()
        results = graph.find_symbol(args.name)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _print_results(results, f"Symbol '{args.name}'")
        print(f"\n  Query time: {elapsed_ms:.1f}ms")

    else:
        print(f"Unknown query command: {query_cmd}", file=sys.stderr)
        print("Available: find-callers, find-callees, find-refs, impact, symbol", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Phase 2 sub-command handlers
# ---------------------------------------------------------------------------

def _cmd_embed(args: argparse.Namespace) -> None:
    """Embed project symbols into Qdrant."""
    project_root = _project_root()
    _require_index(project_root)

    from .local.indexer import Indexer, _manifest_path
    from .local.manifest import Manifest
    from .local.vector_store import QdrantStore, is_qdrant_running
    from .local.embedder import embed_project

    if not is_qdrant_running():
        print(
            "Qdrant is not running. Start it with:\n"
            "  agentchanti kb qdrant start"
        )
        sys.exit(1)

    indexer = Indexer(project_root)
    try:
        graph = indexer.load_graph()
    except FileNotFoundError:
        print("Graph not found. Run `agentchanti kb index` first.", file=sys.stderr)
        sys.exit(1)

    manifest = Manifest(_manifest_path(project_root))
    vector_store = QdrantStore(project_root)

    incremental = getattr(args, "incremental", False)
    mode = "incremental" if incremental else "full"
    print(f"Embedding project symbols ({mode} mode): {project_root}")

    import time as _time
    t0 = _time.perf_counter()
    summary = embed_project(
        graph=graph,
        manifest=manifest,
        vector_store=vector_store,
        project_root=project_root,
        incremental=incremental,
    )
    elapsed = _time.perf_counter() - t0

    print(
        f"\nEmbed complete:\n"
        f"  Total symbols : {summary['total_symbols']}\n"
        f"  Embedded      : {summary['embedded']}\n"
        f"  Skipped       : {summary['skipped']}\n"
        f"  Errors        : {summary['errors']}\n"
        f"  Time          : {elapsed:.1f}s"
    )


def _cmd_search(args: argparse.Namespace) -> None:
    """Semantic search over the knowledge base."""
    project_root = _project_root()
    _require_index(project_root)

    query: str = args.query
    top_k: int = getattr(args, "top_k", 10)
    filter_str: Optional[str] = getattr(args, "filter", None)

    filters: Optional[dict] = None
    if filter_str:
        try:
            key, val = filter_str.split("=", 1)
            filters = {key.strip(): val.strip()}
        except ValueError:
            print(
                f"Invalid --filter format '{filter_str}'. "
                "Use: --filter key=value  (e.g. --filter language=python)",
                file=sys.stderr,
            )
            sys.exit(1)

    from .local.indexer import Indexer, _manifest_path
    from .local.manifest import Manifest
    from .local.vector_store import QdrantStore
    from .local.searcher import Searcher

    indexer = Indexer(project_root)
    try:
        graph = indexer.load_graph()
    except FileNotFoundError:
        print("Graph not found. Run `agentchanti kb index` first.", file=sys.stderr)
        sys.exit(1)

    manifest = Manifest(_manifest_path(project_root))
    vector_store = QdrantStore(project_root)
    searcher = Searcher(
        graph=graph,
        manifest=manifest,
        vector_store=vector_store,
        project_root=project_root,
    )

    import time as _time
    t0 = _time.perf_counter()
    results = searcher.search(query=query, filters=filters, top_k=top_k)
    elapsed_ms = (_time.perf_counter() - t0) * 1000

    if not results:
        print(f"No results found for: {query!r}")
        return

    print(f"\nSearch results for: {query!r}  [{len(results)} result(s)]")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r.symbol_type}: {r.symbol_name}")
        print(f"       File   : {r.file}:{r.line_start}-{r.line_end}")
        print(f"       Score  : {r.score:.4f}")
        if r.code_snippet:
            snippet_lines = r.code_snippet.splitlines()
            preview = "\n         ".join(snippet_lines[:5])
            if len(snippet_lines) > 5:
                preview += f"\n         ... ({len(snippet_lines) - 5} more lines)"
            print(f"       Code   :\n         {preview}")
        if r.related_symbols:
            print(f"       Related ({len(r.related_symbols)}):")
            for rel in r.related_symbols[:5]:
                node_type = rel.get("node_type", "")
                name = rel.get("name", "")
                fp = rel.get("file_path", "")
                ls = rel.get("line_start", 0)
                print(f"         {node_type}: {name} ({fp}:{ls})")

    print(f"\n  Search time: {elapsed_ms:.1f}ms")


def _cmd_qdrant(args: argparse.Namespace) -> None:
    """Dispatch qdrant subcommands."""
    from .local.vector_store import qdrant_start, qdrant_stop, qdrant_status

    qdrant_cmd = args.qdrant_cmd

    if qdrant_cmd == "start":
        project_root = _project_root()
        qdrant_start(project_root)
    elif qdrant_cmd == "stop":
        qdrant_stop()
    elif qdrant_cmd == "status":
        qdrant_status()
    else:
        print(f"Unknown qdrant command: {qdrant_cmd}", file=sys.stderr)
        print("Available: start, stop, status", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the `kb` subcommand argument parser."""
    parser = argparse.ArgumentParser(
        prog="agentchanti kb",
        description="AgentChanti Knowledge Base — code graph and structural analysis",
    )
    subparsers = parser.add_subparsers(dest="kb_cmd", metavar="COMMAND")
    subparsers.required = True

    # --- index ---
    index_p = subparsers.add_parser("index", help="Full re-index of current project")
    index_p.add_argument(
        "--watch", action="store_true",
        help="After indexing, start a file watcher for incremental updates",
    )
    index_p.set_defaults(func=_cmd_index)

    # --- status ---
    status_p = subparsers.add_parser("status", help="Show KB index summary")
    status_p.set_defaults(func=_cmd_status)

    # --- query ---
    query_p = subparsers.add_parser("query", help="Query the code graph")
    query_sub = query_p.add_subparsers(dest="query_cmd", metavar="QUERY")
    query_sub.required = True
    query_p.set_defaults(func=_cmd_query)

    for qname, qhelp in [
        ("find-callers", "List all functions that call FUNCTION_NAME"),
        ("find-callees", "List all functions called by FUNCTION_NAME"),
        ("find-refs",    "List all references to SYMBOL_NAME"),
        ("impact",       "List files affected if FILE_PATH changes"),
        ("symbol",       "Find any symbol by name"),
    ]:
        qp = query_sub.add_parser(qname, help=qhelp)
        qp.add_argument("name", nargs="?", help="Target name to look up")

    # --- embed ---
    embed_p = subparsers.add_parser(
        "embed", help="Embed project symbols into Qdrant (Phase 2)"
    )
    embed_p.add_argument(
        "--incremental", action="store_true",
        help="Only embed symbols from files that changed since last embed",
    )
    embed_p.set_defaults(func=_cmd_embed)

    # --- search ---
    search_p = subparsers.add_parser(
        "search", help="Semantic search over the knowledge base (Phase 2)"
    )
    search_p.add_argument("query", help="Natural-language search query")
    search_p.add_argument(
        "--top-k", dest="top_k", type=int, default=10,
        help="Number of results to return (default: 10)",
    )
    search_p.add_argument(
        "--filter", dest="filter", default=None, metavar="KEY=VALUE",
        help="Payload filter, e.g. --filter language=python or --filter file=src/auth",
    )
    search_p.set_defaults(func=_cmd_search)

    # --- qdrant ---
    qdrant_p = subparsers.add_parser(
        "qdrant", help="Manage the local Qdrant Docker container (Phase 2)"
    )
    qdrant_sub = qdrant_p.add_subparsers(dest="qdrant_cmd", metavar="ACTION")
    qdrant_sub.required = True
    qdrant_p.set_defaults(func=_cmd_qdrant)

    for qname, qhelp in [
        ("start",  "Start the Qdrant Docker container"),
        ("stop",   "Stop the Qdrant Docker container"),
        ("status", "Show Qdrant container status"),
    ]:
        qdrant_sub.add_parser(qname, help=qhelp)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def kb_main(argv: Optional[list[str]] = None) -> None:
    """
    Main entry point for the `agentchanti kb` subcommand.

    Parameters
    ----------
    argv:
        Argument list (without the leading ``agentchanti kb`` tokens).
        Defaults to sys.argv if None.
    """
    # Configure logging if not already configured
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s  %(name)s  %(message)s",
        )

    parser = _build_parser()
    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
