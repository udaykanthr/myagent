"""
Plugin System — extensible step type handlers.

Users can create custom step handlers by subclassing :class:`StepPlugin`
and registering them via .agentchanti.yaml or setuptools entry points.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..executor import Executor
from ..cli_display import CLIDisplay
from ..orchestrator.memory import FileMemory


@dataclass
class PluginContext:
    """Context passed to plugin step handlers."""
    executor: Executor
    memory: FileMemory
    display: CLIDisplay
    llm_client: Any
    step_idx: int
    task: str
    language: str | None = None


class StepPlugin(ABC):
    """Base class for custom step type handlers.

    Example::

        class LintPlugin(StepPlugin):
            name = "LINT"

            def can_handle(self, step_text: str) -> bool:
                return "lint" in step_text.lower()

            def handle(self, step_text: str, ctx: PluginContext) -> tuple[bool, str]:
                success, output = ctx.executor.run_command("ruff check .")
                return success, output
    """

    name: str = ""  # Step type name (e.g., "LINT", "DEPLOY")

    @abstractmethod
    def can_handle(self, step_text: str) -> bool:
        """Return True if this plugin should handle the given step."""

    @abstractmethod
    def handle(self, step_text: str, ctx: PluginContext) -> tuple[bool, str]:
        """Execute the step. Returns (success, error_info)."""


__all__ = ["StepPlugin", "PluginContext"]
