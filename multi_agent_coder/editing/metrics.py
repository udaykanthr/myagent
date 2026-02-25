"""
Edit metrics — tracks diff-edit performance in a JSONL log file.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_METRICS_DIR = ".agentchanti/kb"
_METRICS_FILE = "edit_metrics.jsonl"


def _metrics_path(project_root: str | None = None) -> str:
    """Return the absolute path to the metrics file."""
    base = project_root or os.getcwd()
    return os.path.join(base, _METRICS_DIR, _METRICS_FILE)


def log_edit_metric(data: dict, project_root: str | None = None) -> None:
    """Append a single edit metric entry to the JSONL log.

    Parameters
    ----------
    data:
        Metric fields to log (file, confidence, token_reduction_pct, etc.).
    project_root:
        Optional project root directory. Defaults to CWD.
    """
    path = _metrics_path(project_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    entry = {"timestamp": datetime.now(timezone.utc).isoformat()}
    entry.update(data)

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.warning("[DiffEdit] Failed to write metrics: %s", exc)


def read_edit_stats(
    last_n: int = 50,
    project_root: str | None = None,
) -> dict:
    """Compute rolling statistics from the metrics log.

    Parameters
    ----------
    last_n:
        Number of most-recent entries to include.
    project_root:
        Optional project root directory.

    Returns
    -------
    dict
        Statistics including avg_token_reduction, success_rate,
        fallback_rate, avg_confidence, resolution_methods, total_edits.
    """
    path = _metrics_path(project_root)

    entries: list[dict] = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            pass

    # Take last N entries
    entries = entries[-last_n:]

    if not entries:
        return {
            "total_edits": 0,
            "avg_token_reduction": 0.0,
            "success_rate": 0.0,
            "fallback_rate": 0.0,
            "avg_confidence": 0.0,
            "resolution_methods": {},
        }

    total = len(entries)
    token_reductions = [
        e.get("token_reduction_pct", 0) for e in entries
        if "token_reduction_pct" in e
    ]
    confidences = [
        e.get("confidence", 0) for e in entries
        if "confidence" in e
    ]
    fallbacks = sum(1 for e in entries if e.get("fallback_used", False))
    successes = sum(
        1 for e in entries
        if not e.get("fallback_used", False) and e.get("hunks_failed", 0) == 0
    )

    methods = Counter(e.get("resolution_method", "unknown") for e in entries)

    return {
        "total_edits": total,
        "avg_token_reduction": (
            sum(token_reductions) / len(token_reductions)
            if token_reductions else 0.0
        ),
        "success_rate": successes / total * 100 if total else 0.0,
        "fallback_rate": fallbacks / total * 100 if total else 0.0,
        "avg_confidence": (
            sum(confidences) / len(confidences)
            if confidences else 0.0
        ),
        "resolution_methods": {
            method: count / total * 100
            for method, count in methods.most_common()
        },
    }
