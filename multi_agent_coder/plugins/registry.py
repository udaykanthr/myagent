"""
Plugin Registry — discovers and manages step-type plugins.
"""

from __future__ import annotations

import importlib

from ..cli_display import log
from . import StepPlugin


class PluginRegistry:
    """Discovers and manages step-type plugins.

    Plugins can be registered via:
    1. Config file (``plugins`` list of Python import paths)
    2. Setuptools entry points (``agentchanti.plugins`` group)
    """

    def __init__(self):
        self._plugins: list[StepPlugin] = []

    def discover(self, config_plugins: list[str] | None = None):
        """Discover and load plugins from config and entry points."""
        # 1. Config-specified plugins (Python import paths)
        if config_plugins:
            for path in config_plugins:
                plugin = self._load_from_path(path)
                if plugin:
                    self._plugins.append(plugin)

        # 2. Entry points (setuptools-based discovery)
        try:
            if hasattr(importlib.metadata, "entry_points"):
                eps = importlib.metadata.entry_points()
                # Python 3.12+ returns a SelectableGroups or dict
                if hasattr(eps, "select"):
                    plugin_eps = eps.select(group="agentchanti.plugins")
                elif isinstance(eps, dict):
                    plugin_eps = eps.get("agentchanti.plugins", [])
                else:
                    plugin_eps = []

                for ep in plugin_eps:
                    try:
                        cls = ep.load()
                        if isinstance(cls, type) and issubclass(cls, StepPlugin):
                            instance = cls()
                            self._plugins.append(instance)
                            log.info(f"[PluginRegistry] Loaded entry point plugin: "
                                     f"{ep.name} ({cls.__name__})")
                    except Exception as e:
                        log.warning(f"[PluginRegistry] Failed to load entry point "
                                    f"'{ep.name}': {e}")
        except Exception as e:
            log.debug(f"[PluginRegistry] Entry point discovery skipped: {e}")

        log.info(f"[PluginRegistry] {len(self._plugins)} plugin(s) loaded")

    def _load_from_path(self, dotted_path: str) -> StepPlugin | None:
        """Load a plugin class from a dotted Python import path.

        Example: ``my_package.plugins.LintPlugin``
        """
        try:
            module_path, cls_name = dotted_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, cls_name)
            if isinstance(cls, type) and issubclass(cls, StepPlugin):
                instance = cls()
                log.info(f"[PluginRegistry] Loaded plugin: {dotted_path} "
                         f"(type={instance.name})")
                return instance
            else:
                log.warning(f"[PluginRegistry] {dotted_path} is not a StepPlugin subclass")
        except (ImportError, AttributeError, ValueError) as e:
            log.warning(f"[PluginRegistry] Failed to load '{dotted_path}': {e}")
        return None

    def find_handler(self, step_text: str) -> StepPlugin | None:
        """Find the first plugin that can handle the given step text."""
        for plugin in self._plugins:
            try:
                if plugin.can_handle(step_text):
                    return plugin
            except Exception as e:
                log.warning(f"[PluginRegistry] Error in {plugin.name}.can_handle(): {e}")
        return None

    @property
    def plugins(self) -> list[StepPlugin]:
        return list(self._plugins)

    @property
    def size(self) -> int:
        return len(self._plugins)
