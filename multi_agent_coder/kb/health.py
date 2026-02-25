"""
KB health check — utility for checking overall Knowledge Base health.

Used by the CLI (``agentchanti kb health``) and at agent startup.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KBHealth:
    """Overall KB health status."""

    local_kb_indexed: bool = False
    local_symbol_count: int = 0
    local_last_indexed: Optional[str] = None
    local_index_stale: bool = False
    qdrant_running: bool = False
    global_kb_version: str = "unknown"
    global_kb_last_updated: Optional[str] = None
    registry_update_available: bool = False


def check(project_root: Optional[str] = None) -> KBHealth:
    """
    Check the health of all KB components.

    Parameters
    ----------
    project_root:
        Absolute path to the project root.  Defaults to ``os.getcwd()``.

    Returns
    -------
    KBHealth
        Aggregated health status.
    """
    project_root = os.path.abspath(project_root or os.getcwd())
    health = KBHealth()

    # --- Local KB ---
    try:
        from .local.indexer import read_meta

        meta = read_meta(project_root)
        if meta is not None:
            health.local_kb_indexed = True
            health.local_symbol_count = meta.get("symbol_count", 0)
            health.local_last_indexed = meta.get("last_indexed")

            # Check staleness (> 1 hour)
            if health.local_last_indexed:
                try:
                    from datetime import datetime, timezone
                    indexed_time = datetime.fromisoformat(
                        health.local_last_indexed.replace("Z", "+00:00")
                    )
                    now = datetime.now(timezone.utc)
                    age_seconds = (now - indexed_time).total_seconds()
                    health.local_index_stale = age_seconds > 3600
                except Exception:
                    health.local_index_stale = True
    except Exception as exc:
        logger.debug("[KB health] Local KB check failed: %s", exc)

    # --- Qdrant ---
    try:
        from .local.vector_store import is_qdrant_running

        health.qdrant_running = is_qdrant_running()
    except Exception as exc:
        logger.debug("[KB health] Qdrant check failed: %s", exc)

    # --- Global KB ---
    try:
        from .global_kb.updater import get_version, get_manifest_info

        health.global_kb_version = get_version()
        info = get_manifest_info()
        health.global_kb_last_updated = info.get("created_at")
    except Exception as exc:
        logger.debug("[KB health] Global KB version check failed: %s", exc)

    # --- Registry update check ---
    try:
        from ..config import Config

        cfg = Config.load()
        if cfg.KB_REGISTRY_OWNER:
            from .global_kb.updater import check_for_updates

            status = check_for_updates(cfg.KB_REGISTRY_OWNER, cfg.KB_REGISTRY_REPO)
            health.registry_update_available = status.update_available
    except Exception as exc:
        logger.debug("[KB health] Registry update check failed: %s", exc)

    return health


def format_health(health: KBHealth) -> str:
    """
    Format a :class:`KBHealth` into a human-readable report.

    Parameters
    ----------
    health:
        The health status to format.

    Returns
    -------
    str
        Multi-line human-readable report.
    """
    def _status(ok: bool) -> str:
        return "OK" if ok else "NOT OK"

    def _stale(stale: bool) -> str:
        return " (stale)" if stale else ""

    lines = [
        "",
        "Knowledge Base Health Report",
        "=" * 40,
        "",
        "Local KB:",
        f"  Indexed       : {_status(health.local_kb_indexed)}",
        f"  Symbols       : {health.local_symbol_count}",
        f"  Last indexed  : {health.local_last_indexed or 'never'}"
        f"{_stale(health.local_index_stale)}",
        "",
        "Qdrant:",
        f"  Running       : {_status(health.qdrant_running)}",
        "",
        "Global KB:",
        f"  Version       : {health.global_kb_version}",
        f"  Last updated  : {health.global_kb_last_updated or 'unknown'}",
        f"  Update avail. : {'Yes' if health.registry_update_available else 'No'}",
        "",
    ]
    return "\n".join(lines)


def to_json(health: KBHealth) -> str:
    """
    Serialise a :class:`KBHealth` to JSON.

    Parameters
    ----------
    health:
        The health status to serialise.

    Returns
    -------
    str
        JSON string.
    """
    return json.dumps({
        "local_kb_indexed": health.local_kb_indexed,
        "local_symbol_count": health.local_symbol_count,
        "local_last_indexed": health.local_last_indexed,
        "local_index_stale": health.local_index_stale,
        "qdrant_running": health.qdrant_running,
        "global_kb_version": health.global_kb_version,
        "global_kb_last_updated": health.global_kb_last_updated,
        "registry_update_available": health.registry_update_available,
    }, indent=2)
