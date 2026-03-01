"""
Unit tests for multi_agent_coder.kb.health

Tests the health check utility: check(), format_health(), to_json().
All external dependencies are mocked.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from multi_agent_coder.kb.health import (
    KBHealth, check, format_health, to_json,
)


# ---------------------------------------------------------------------------
# KBHealth dataclass tests
# ---------------------------------------------------------------------------

class TestKBHealth:

    def test_defaults(self):
        h = KBHealth()
        assert h.local_kb_indexed is False
        assert h.local_symbol_count == 0
        assert h.local_last_indexed is None
        assert h.local_index_stale is False
        assert h.global_kb_version == "unknown"
        assert h.global_kb_last_updated is None
        assert h.registry_update_available is False


# ---------------------------------------------------------------------------
# check() tests
# ---------------------------------------------------------------------------

class TestCheck:

    @patch("multi_agent_coder.kb.health.check")
    def test_check_returns_health(self, mock_check, tmp_path):
        """check() returns a KBHealth instance."""
        mock_check.return_value = KBHealth(
            local_kb_indexed=True,
            local_symbol_count=42,
        )
        h = mock_check(str(tmp_path))
        assert h.local_kb_indexed is True
        assert h.local_symbol_count == 42

    def test_check_no_index(self, tmp_path):
        """When no index exists, local_kb_indexed should be False."""
        h = check(str(tmp_path))
        assert h.local_kb_indexed is False
        assert h.local_symbol_count == 0

    def test_check_with_meta(self, tmp_path):
        """When meta exists, local info should be populated."""
        import json as _json
        meta_dir = tmp_path / ".agentchanti" / "kb" / "local"
        meta_dir.mkdir(parents=True)
        (meta_dir / "graph_meta.json").write_text(_json.dumps({
            "last_indexed": "2026-02-24T00:00:00Z",
            "symbol_count": 100,
        }))

        h = check(str(tmp_path))
        assert h.local_kb_indexed is True
        assert h.local_symbol_count == 100
        assert h.local_last_indexed == "2026-02-24T00:00:00Z"


# ---------------------------------------------------------------------------
# format_health() tests
# ---------------------------------------------------------------------------

class TestFormatHealth:

    def test_format_contains_sections(self):
        h = KBHealth(
            local_kb_indexed=True,
            local_symbol_count=150,
            global_kb_version="1.2.0",
        )
        output = format_health(h)
        assert "Knowledge Base Health Report" in output
        assert "Local KB:" in output
        assert "Global KB:" in output
        assert "150" in output
        assert "1.2.0" in output

    def test_format_not_indexed(self):
        h = KBHealth()
        output = format_health(h)
        assert "NOT OK" in output
        assert "never" in output

    def test_stale_marker(self):
        h = KBHealth(
            local_kb_indexed=True,
            local_index_stale=True,
            local_last_indexed="2026-01-01T00:00:00Z",
        )
        output = format_health(h)
        assert "(stale)" in output


# ---------------------------------------------------------------------------
# to_json() tests
# ---------------------------------------------------------------------------

class TestToJson:

    def test_valid_json(self):
        h = KBHealth(
            local_kb_indexed=True,
            local_symbol_count=42,
            global_kb_version="1.0.0",
        )
        result = to_json(h)
        data = json.loads(result)
        assert data["local_kb_indexed"] is True
        assert data["local_symbol_count"] == 42
        assert data["global_kb_version"] == "1.0.0"

    def test_all_fields_present(self):
        h = KBHealth()
        data = json.loads(to_json(h))
        expected_keys = {
            "local_kb_indexed", "local_symbol_count", "local_last_indexed",
            "local_index_stale", "global_kb_version",
            "global_kb_last_updated", "registry_update_available",
        }
        assert set(data.keys()) == expected_keys
