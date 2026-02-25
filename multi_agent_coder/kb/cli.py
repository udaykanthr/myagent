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

Phase 3 — Global Knowledge Base
agentchanti kb update                       -- pull latest from GitHub registry
agentchanti kb update --category errors     -- update specific category only
agentchanti kb update --check               -- check for updates, don't download
agentchanti kb version                      -- show current global KB version
agentchanti kb error-lookup "<message>"     -- lookup error fix
agentchanti kb error-lookup "<message>" --language python
agentchanti kb global-search "<query>"      -- search global KB
agentchanti kb global-search "<query>" --category patterns
agentchanti kb seed                         -- (dev) re-seed sample data
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
# Phase 3 sub-command handlers
# ---------------------------------------------------------------------------

def _cmd_seed(args: argparse.Namespace) -> None:
    """Seed the global KB with sample data."""
    from .global_kb.seeder import seed

    embed = not getattr(args, "no_embed", False)
    project_root = _project_root()

    print("Seeding global knowledge base...")
    t0 = time.perf_counter()
    summary = seed(embed=embed, project_root=project_root)
    elapsed = time.perf_counter() - t0

    print(
        f"\nSeed complete:\n"
        f"  Errors seeded   : {summary['errors_seeded']}\n"
        f"  Docs seeded     : {summary['docs_seeded']}\n"
        f"  Chunks embedded : {summary['chunks_embedded']}\n"
        f"  Time            : {elapsed:.1f}s"
    )

    # Print per-language breakdown
    from .global_kb.error_dict import ErrorDict
    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "global_kb", "core", "errors.db",
    )
    edict = ErrorDict(db_path)
    counts = edict.count_by_language()
    if counts:
        print("\n  Errors by language:")
        for lang, cnt in sorted(counts.items()):
            print(f"    {lang:<14} {cnt}")


def _cmd_version(args: argparse.Namespace) -> None:
    """Show the current global KB version."""
    from .global_kb.updater import get_manifest_info

    info = get_manifest_info()
    print(f"\nGlobal Knowledge Base")
    print("=" * 40)
    print(f"  Version    : {info.get('version', 'unknown')}")
    print(f"  Created    : {info.get('created_at', 'unknown')}")
    cats = info.get("categories", [])
    if cats:
        print(f"  Categories : {', '.join(cats)}")

    # Show error count
    try:
        from .global_kb.error_dict import ErrorDict
        db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "global_kb", "core", "errors.db",
        )
        if os.path.isfile(db_path):
            edict = ErrorDict(db_path)
            total = edict.count()
            print(f"  Errors     : {total}")
    except Exception:
        pass

    # Show registry doc count
    try:
        registry_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "global_kb", "registry",
        )
        doc_count = 0
        for dirpath, _, filenames in os.walk(registry_dir):
            doc_count += sum(1 for f in filenames if f.endswith(".md"))
        if doc_count:
            print(f"  Documents  : {doc_count}")
    except Exception:
        pass
    print()


def _cmd_error_lookup(args: argparse.Namespace) -> None:
    """Look up error fixes in the global KB."""
    from .global_kb.store import GlobalKBStore

    message: str = args.message
    language: Optional[str] = getattr(args, "language", None)

    store = GlobalKBStore()

    t0 = time.perf_counter()
    results = store.search_errors(message, language=language)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not results:
        print(f"No fixes found for: {message!r}")
        print(f"\n  Lookup time: {elapsed_ms:.1f}ms")
        return

    lang_label = f" (language={language})" if language else ""
    print(f"\nError fixes for: {message!r}{lang_label}  [{len(results)} result(s)]")
    print("-" * 70)

    for i, ef in enumerate(results, 1):
        print(f"\n  [{i}] {ef.error_type} ({ef.language})")
        print(f"       Severity : {ef.severity}")
        if ef.cause:
            print(f"       Cause    : {ef.cause}")
        print(f"       Fix      : {ef.fix_template}")
        if ef.tags:
            print(f"       Tags     : {ef.tags}")

    print(f"\n  Lookup time: {elapsed_ms:.1f}ms")


def _cmd_global_search(args: argparse.Namespace) -> None:
    """Search the global knowledge base."""
    from .global_kb.store import GlobalKBStore

    query: str = args.query
    category: Optional[str] = getattr(args, "category", None)
    language: Optional[str] = getattr(args, "language", None)
    top_k: int = getattr(args, "top_k", 5)

    categories = [category] if category else None

    store = GlobalKBStore()

    t0 = time.perf_counter()
    results = store.search(query, categories=categories, language=language, top_k=top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not results:
        print(f"No results found for: {query!r}")
        print(f"\n  Search time: {elapsed_ms:.1f}ms")
        return

    cat_label = f" (category={category})" if category else ""
    print(f"\nGlobal KB results for: {query!r}{cat_label}  [{len(results)} result(s)]")
    print("-" * 70)

    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r.title}")
        print(f"       Category : {r.category}")
        print(f"       File     : {r.file}")
        print(f"       Score    : {r.score:.4f}")
        if r.tags:
            print(f"       Tags     : {', '.join(r.tags)}")
        if r.language != "all":
            print(f"       Language : {r.language}")

    print(f"\n  Search time: {elapsed_ms:.1f}ms")


def _cmd_edit_stats(args: argparse.Namespace) -> None:
    """Show rolling DiffEdit statistics."""
    from ..editing.metrics import read_edit_stats

    project_root = _project_root()
    last_n = getattr(args, "last_n", 50)
    stats = read_edit_stats(last_n=last_n, project_root=project_root)

    if stats["total_edits"] == 0:
        print("No DiffEdit metrics found yet.")
        print("Metrics are recorded when diff-aware editing is used during task execution.")
        return

    print(f"\n┌─────────────────────────────────────┐")
    print(f"│ DiffEdit Stats (last {last_n} edits){' ' * max(0, 8 - len(str(last_n)))}│")
    print(f"├─────────────────────────────────────┤")
    print(f"│ Total edits:          {stats['total_edits']:<13}│")
    print(f"│ Avg token reduction:  {stats['avg_token_reduction']:<6.0f}%{' ' * 6}│")
    print(f"│ Success rate:         {stats['success_rate']:<6.0f}%{' ' * 6}│")
    print(f"│ Fallback rate:        {stats['fallback_rate']:<6.0f}%{' ' * 6}│")
    print(f"│ Avg confidence:       {stats['avg_confidence']:<13.2f}│")

    methods = stats.get("resolution_methods", {})
    if methods:
        print(f"│ Resolution methods:{' ' * 18}│")
        for method, pct in methods.items():
            label = f"  {method}:"
            print(f"│ {label:<22}{pct:<6.0f}%{' ' * 6}│")

    print(f"└─────────────────────────────────────┘")
    print()


def _cmd_health(args: argparse.Namespace) -> None:
    """Show overall KB health report."""
    from .health import check, format_health, to_json

    project_root = _project_root()
    health = check(project_root)

    use_json = getattr(args, "json", False)
    if use_json:
        print(to_json(health))
    else:
        print(format_health(health))


def _cmd_update(args: argparse.Namespace) -> None:
    """Check for or download global KB updates from GitHub registry."""
    from ..config import Config
    from .global_kb.updater import check_for_updates, download_update

    cfg = Config.load()
    owner = cfg.KB_REGISTRY_OWNER
    repo = cfg.KB_REGISTRY_REPO

    if not owner:
        print(
            "KB registry owner is not configured.\n"
            "Set kb_registry_owner in .agentchanti.yaml or "
            "KB_REGISTRY_OWNER environment variable."
        )
        return

    check_only = getattr(args, "check", False)
    category = getattr(args, "category", None)

    if check_only:
        print(f"Checking for updates from {owner}/{repo}...")
        status = check_for_updates(owner, repo)
        print(f"\n  Current version : {status.current_version}")
        print(f"  Latest version  : {status.latest_version}")
        if status.update_available:
            print(f"  Update available!")
            if status.changelog:
                print(f"\n  Changelog:\n  {status.changelog[:500]}")
        else:
            print(f"  You are up to date.")
        return

    print(f"Downloading update from {owner}/{repo}...")
    categories = [category] if category else None
    try:
        summary = download_update(owner, repo, categories=categories)
        print(
            f"\nUpdate complete:\n"
            f"  Version        : {summary['version']}\n"
            f"  Files updated  : {summary['files_updated']}\n"
            f"  Errors updated : {summary['errors_updated']}"
        )
    except ConnectionError as exc:
        print(f"Update failed: {exc}", file=sys.stderr)
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

    # =======================================================================
    # Phase 3 — Global Knowledge Base commands
    # =======================================================================

    # --- seed ---
    seed_p = subparsers.add_parser(
        "seed", help="(Dev) Seed global KB with sample data"
    )
    seed_p.add_argument(
        "--no-embed", action="store_true",
        help="Skip embedding into Qdrant (seed errors.db and .md files only)",
    )
    seed_p.set_defaults(func=_cmd_seed)

    # --- version ---
    version_p = subparsers.add_parser(
        "version", help="Show current global KB version"
    )
    version_p.set_defaults(func=_cmd_version)

    # --- error-lookup ---
    elookup_p = subparsers.add_parser(
        "error-lookup", help="Look up error fixes in the global KB"
    )
    elookup_p.add_argument("message", help="Error message to look up")
    elookup_p.add_argument(
        "--language", default=None,
        help="Filter by language (e.g. python, java, go)",
    )
    elookup_p.set_defaults(func=_cmd_error_lookup)

    # --- global-search ---
    gsearch_p = subparsers.add_parser(
        "global-search", help="Search the global knowledge base"
    )
    gsearch_p.add_argument("query", help="Natural-language search query")
    gsearch_p.add_argument(
        "--category", default=None,
        help="Filter by category (pattern, adr, doc, behavioral)",
    )
    gsearch_p.add_argument(
        "--language", default=None,
        help="Filter by language",
    )
    gsearch_p.add_argument(
        "--top-k", dest="top_k", type=int, default=5,
        help="Number of results (default: 5)",
    )
    gsearch_p.set_defaults(func=_cmd_global_search)

    # --- update ---
    update_p = subparsers.add_parser(
        "update", help="Pull latest updates from GitHub registry"
    )
    update_p.add_argument(
        "--check", action="store_true",
        help="Check for updates without downloading",
    )
    update_p.add_argument(
        "--category", default=None,
        help="Update only a specific category (errors, patterns, adrs, docs, behavioral)",
    )
    update_p.set_defaults(func=_cmd_update)

    # =======================================================================
    # Phase 4 — Health check
    # =======================================================================

    # --- health ---
    health_p = subparsers.add_parser(
        "health", help="Show overall KB health report"
    )
    health_p.add_argument(
        "--json", action="store_true",
        help="Machine-readable JSON output",
    )
    health_p.set_defaults(func=_cmd_health)

    # =======================================================================
    # Phase 5 — DiffEdit stats
    # =======================================================================

    # --- edit-stats ---
    editstats_p = subparsers.add_parser(
        "edit-stats", help="Show rolling DiffEdit performance statistics"
    )
    editstats_p.add_argument(
        "--last-n", dest="last_n", type=int, default=50,
        help="Number of recent edits to include in stats (default: 50)",
    )
    editstats_p.set_defaults(func=_cmd_edit_stats)

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
