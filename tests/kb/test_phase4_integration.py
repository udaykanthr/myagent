"""
Integration tests for Phase 4 — Context Injection.

Tests config loading with KB settings, pipeline integration with
kb_context_builder parameter, and the --no-kb flag.
"""

from __future__ import annotations

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Config KB settings tests
# ---------------------------------------------------------------------------

class TestConfigKBSettings:

    def test_default_kb_settings(self):
        """Default config should have KB settings enabled."""
        from multi_agent_coder.config import Config

        cfg = Config.load(None)
        assert cfg.KB_ENABLED is True
        assert cfg.KB_MAX_CONTEXT_TOKENS == 4000
        assert cfg.KB_AUTO_INDEX_ON_START is True
        assert cfg.KB_WATCHER_DEBOUNCE_SECONDS == 1.0
        assert cfg.KB_VERBOSE_LOGGING is False

    def test_kb_settings_in_yaml(self, tmp_path):
        """KB settings from YAML should be loaded correctly."""
        from multi_agent_coder.config import Config

        yaml_content = """\
kb:
  enabled: false
  max_context_tokens: 2000
  auto_index_on_start: false
  watcher_debounce_seconds: 2.5
  verbose_logging: true
"""
        yaml_path = tmp_path / ".agentchanti.yaml"
        yaml_path.write_text(yaml_content)

        cfg = Config.load(str(yaml_path))
        assert cfg.KB_ENABLED is False
        assert cfg.KB_MAX_CONTEXT_TOKENS == 2000
        assert cfg.KB_AUTO_INDEX_ON_START is False
        assert cfg.KB_WATCHER_DEBOUNCE_SECONDS == 2.5
        assert cfg.KB_VERBOSE_LOGGING is True

    def test_kb_settings_in_to_dict(self):
        """KB settings should be present in to_dict() output."""
        from multi_agent_coder.config import Config

        cfg = Config.load(None)
        d = cfg.to_dict()
        assert "kb" in d
        assert d["kb"]["enabled"] is True
        assert d["kb"]["max_context_tokens"] == 4000


# ---------------------------------------------------------------------------
# Pipeline kb_context_builder parameter tests
# ---------------------------------------------------------------------------

class TestPipelineKBIntegration:

    def test_execute_step_accepts_kb_context_builder(self):
        """_execute_step should accept kb_context_builder kwarg without error."""
        from multi_agent_coder.orchestrator.pipeline import _execute_step

        # We need to mock everything — just verify the function accepts the param
        import inspect
        sig = inspect.signature(_execute_step)
        assert "kb_context_builder" in sig.parameters

    def test_run_diagnosis_loop_accepts_kb_context_builder(self):
        """_run_diagnosis_loop should accept kb_context_builder kwarg."""
        from multi_agent_coder.orchestrator.pipeline import _run_diagnosis_loop

        import inspect
        sig = inspect.signature(_run_diagnosis_loop)
        assert "kb_context_builder" in sig.parameters


# ---------------------------------------------------------------------------
# api.py KB integration tests
# ---------------------------------------------------------------------------

class TestAPIKBIntegration:

    def test_run_task_accepts_no_kb(self):
        """run_task should accept no_kb parameter."""
        from multi_agent_coder.api import run_task

        import inspect
        sig = inspect.signature(run_task)
        assert "no_kb" in sig.parameters

    def test_run_task_impl_accepts_no_kb(self):
        """_run_task_impl should accept no_kb parameter."""
        from multi_agent_coder.api import _run_task_impl

        import inspect
        sig = inspect.signature(_run_task_impl)
        assert "no_kb" in sig.parameters


# ---------------------------------------------------------------------------
# KB CLI health command test
# ---------------------------------------------------------------------------

class TestKBCLIHealth:

    def test_health_command_registered(self):
        """The 'health' subcommand should be registered in the KB CLI parser."""
        from multi_agent_coder.kb.cli import _build_parser

        parser = _build_parser()
        # Parse a minimal health command
        args = parser.parse_args(["health"])
        assert args.kb_cmd == "health"
        assert hasattr(args, "func")

    def test_health_json_flag(self):
        """The --json flag should be available on health command."""
        from multi_agent_coder.kb.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["health", "--json"])
        assert args.json is True
