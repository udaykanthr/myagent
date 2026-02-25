"""Tests for edit metrics logging and stats."""

import json
import os
import tempfile

import pytest

from multi_agent_coder.editing.metrics import log_edit_metric, read_edit_stats


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temp project root with the metrics directory."""
    metrics_dir = tmp_path / ".agentchanti" / "kb"
    metrics_dir.mkdir(parents=True)
    return str(tmp_path)


class TestLogEditMetric:
    def test_creates_file_and_writes_entry(self, tmp_project):
        log_edit_metric(
            {"file": "src/auth.py", "confidence": 0.95, "fallback_used": False},
            project_root=tmp_project,
        )

        path = os.path.join(tmp_project, ".agentchanti", "kb", "edit_metrics.jsonl")
        assert os.path.isfile(path)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["file"] == "src/auth.py"
        assert entry["confidence"] == 0.95
        assert "timestamp" in entry

    def test_appends_multiple_entries(self, tmp_project):
        log_edit_metric({"file": "a.py"}, project_root=tmp_project)
        log_edit_metric({"file": "b.py"}, project_root=tmp_project)
        log_edit_metric({"file": "c.py"}, project_root=tmp_project)

        path = os.path.join(tmp_project, ".agentchanti", "kb", "edit_metrics.jsonl")
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3


class TestReadEditStats:
    def test_empty_stats(self, tmp_project):
        stats = read_edit_stats(project_root=tmp_project)

        assert stats["total_edits"] == 0
        assert stats["avg_token_reduction"] == 0.0
        assert stats["success_rate"] == 0.0
        assert stats["fallback_rate"] == 0.0

    def test_stats_from_entries(self, tmp_project):
        entries = [
            {
                "file": "a.py",
                "resolution_method": "graph_lookup",
                "confidence": 0.95,
                "token_reduction_pct": 85,
                "fallback_used": False,
                "hunks_failed": 0,
            },
            {
                "file": "b.py",
                "resolution_method": "graph_lookup",
                "confidence": 0.90,
                "token_reduction_pct": 90,
                "fallback_used": False,
                "hunks_failed": 0,
            },
            {
                "file": "c.py",
                "resolution_method": "fallback",
                "confidence": 0.50,
                "token_reduction_pct": 0,
                "fallback_used": True,
                "hunks_failed": 0,
            },
        ]
        for e in entries:
            log_edit_metric(e, project_root=tmp_project)

        stats = read_edit_stats(last_n=50, project_root=tmp_project)

        assert stats["total_edits"] == 3
        # (85 + 90 + 0) / 3 ≈ 58.3
        assert 58 <= stats["avg_token_reduction"] <= 59
        # 2 successes / 3 total ≈ 66.7%
        assert 66 <= stats["success_rate"] <= 67
        # 1 fallback / 3 total ≈ 33.3%
        assert 33 <= stats["fallback_rate"] <= 34
        # (0.95 + 0.90 + 0.50) / 3 ≈ 0.783
        assert 0.78 <= stats["avg_confidence"] <= 0.79

        methods = stats["resolution_methods"]
        assert "graph_lookup" in methods
        assert "fallback" in methods

    def test_last_n_limits(self, tmp_project):
        for i in range(10):
            log_edit_metric(
                {"file": f"f{i}.py", "confidence": 0.9, "fallback_used": False,
                 "hunks_failed": 0, "token_reduction_pct": 80,
                 "resolution_method": "graph_lookup"},
                project_root=tmp_project,
            )

        stats = read_edit_stats(last_n=5, project_root=tmp_project)
        assert stats["total_edits"] == 5
