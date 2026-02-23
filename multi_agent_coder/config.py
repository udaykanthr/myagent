"""
Configuration — loads settings from .agentchanti.yaml, environment variables,
and built-in defaults (in that priority order: CLI args > env > YAML > defaults).
"""

import os

try:
    import yaml
except ImportError:
    yaml = None


_DEFAULTS = {
    "provider": "lm_studio",
    "model": "deepseek-coder-v2-lite-instruct",
    "context_window": 8192,
    "embedding_model": "nomic-embed-text",
    "embedding_top_k": 5,
    "stream": True,
    "llm_max_retries": 3,
    "llm_retry_delay": 2.0,
    "checkpoint_file": ".agentchanti_checkpoint.json",
    "ollama_base_url": "http://localhost:11434/api/generate",
    "lm_studio_base_url": "http://localhost:1234/v1",
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "gemini_api_key": "",
    "gemini_base_url": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic_api_key": "",
    "anthropic_base_url": "https://api.anthropic.com/v1",
    "models": {},
    "embedding_cache_dir": ".agentchanti",
    "report_dir": ".agentchanti/reports",
    "step_cache_ttl_hours": 24,
    "planner_context_chars": 6000,
    "plugins": [],
    "planner_suffix": "Do not create meta-steps (e.g., 'Review code', 'Identify issues'). Focus on implementation. Combine analysis and action.",
    "budget_limit": 0.0,
    "search_enabled": True,
    "search_provider": "duckduckgo",
    "search_api_key": "",
    "search_api_url": "",
    "search_max_results": 3,
    "search_max_page_chars": 3000,
    "pricing": {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "deepseek-coder": {"input": 0.14, "output": 0.28},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
        "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
        "claude-sonnet-4": {"input": 3.00, "output": 15.00},
        "claude-haiku-4": {"input": 0.80, "output": 4.00},
    }
}

# Config file search locations
_CONFIG_FILENAMES = [".agentchanti.yaml", ".agentchanti.yml"]


def _find_config_file(explicit_path: str | None = None) -> str | None:
    """Find the config file. Checks explicit path, CWD, then user home."""
    if explicit_path:
        if os.path.isfile(explicit_path):
            return explicit_path
        return None

    # Search CWD first, then home directory
    search_dirs = [os.getcwd(), os.path.expanduser("~")]
    for d in search_dirs:
        for name in _CONFIG_FILENAMES:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path
    return None


def _load_yaml(path: str) -> dict:
    """Load YAML file, returns empty dict on failure."""
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


class Config:
    """Application configuration.

    Settings are resolved in priority order:
    1. CLI arguments (handled by caller)
    2. Environment variables
    3. .agentchanti.yaml config file
    4. Built-in defaults
    """

    def __init__(self, yaml_data: dict | None = None):
        yd = yaml_data or {}

        # Helper: env var > yaml > default
        def _get(env_key: str, yaml_key: str, default, cast=str):
            env_val = os.getenv(env_key)
            if env_val is not None:
                return cast(env_val)
            yaml_val = yd.get(yaml_key)
            if yaml_val is not None:
                return cast(yaml_val)
            return default

        def _get_bool(env_key: str, yaml_key: str, default: bool) -> bool:
            env_val = os.getenv(env_key)
            if env_val is not None:
                return env_val.lower() == "true"
            yaml_val = yd.get(yaml_key)
            if yaml_val is not None:
                return bool(yaml_val)
            return default

        self.PROVIDER = _get("PROVIDER", "provider", _DEFAULTS["provider"])
        self.DEFAULT_MODEL = _get("DEFAULT_MODEL", "model", _DEFAULTS["model"])
        self.CONTEXT_WINDOW = _get("CONTEXT_WINDOW", "context_window",
                                   _DEFAULTS["context_window"], cast=int)
        self.EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "embedding_model",
                                    _DEFAULTS["embedding_model"])
        self.EMBEDDING_TOP_K = _get("EMBEDDING_TOP_K", "embedding_top_k",
                                    _DEFAULTS["embedding_top_k"], cast=int)

        self.LLM_MAX_RETRIES = _get("LLM_MAX_RETRIES", "llm_max_retries",
                                    _DEFAULTS["llm_max_retries"], cast=int)
        self.LLM_RETRY_DELAY = _get("LLM_RETRY_DELAY", "llm_retry_delay",
                                    _DEFAULTS["llm_retry_delay"], cast=float)
        self.STREAM_RESPONSES = _get_bool("STREAM_RESPONSES", "stream",
                                          _DEFAULTS["stream"])

        self.CHECKPOINT_FILE = _get("CHECKPOINT_FILE", "checkpoint_file",
                                    _DEFAULTS["checkpoint_file"])

        self.OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "ollama_base_url",
                                    _DEFAULTS["ollama_base_url"])
        self.LM_STUDIO_BASE_URL = _get("LM_STUDIO_BASE_URL", "lm_studio_base_url",
                                       _DEFAULTS["lm_studio_base_url"])

        # OpenAI / cloud provider
        openai_section = yd.get("openai", {}) if isinstance(yd.get("openai"), dict) else {}
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or openai_section.get(
            "api_key", _DEFAULTS["openai_api_key"])
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or openai_section.get(
            "base_url", _DEFAULTS["openai_base_url"])

        # Gemini
        gemini_section = yd.get("gemini", {}) if isinstance(yd.get("gemini"), dict) else {}
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or gemini_section.get(
            "api_key", _DEFAULTS["gemini_api_key"])
        self.GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") or gemini_section.get(
            "base_url", _DEFAULTS["gemini_base_url"])

        # Anthropic
        anthropic_section = yd.get("anthropic", {}) if isinstance(yd.get("anthropic"), dict) else {}
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or anthropic_section.get(
            "api_key", _DEFAULTS["anthropic_api_key"])
        self.ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL") or anthropic_section.get(
            "base_url", _DEFAULTS["anthropic_base_url"])

        # Per-agent model overrides
        self._agent_models: dict[str, str] = {}
        models_section = yd.get("models", {})
        if isinstance(models_section, dict):
            for agent_name in ("planner", "coder", "reviewer", "tester"):
                if agent_name in models_section:
                    self._agent_models[agent_name] = str(models_section[agent_name])

        # Custom agent prompt suffixes
        self.PROMPT_SUFFIXES: dict[str, str] = {}
        prompts_section = yd.get("prompts", {})
        if isinstance(prompts_section, dict):
            for key in ("planner_suffix", "coder_suffix",
                        "reviewer_suffix", "tester_suffix"):
                val = prompts_section.get(key)
                if val is not None:
                    self.PROMPT_SUFFIXES[key] = str(val)
                elif key in _DEFAULTS:
                    # Load from defaults if not in YAML
                    self.PROMPT_SUFFIXES[key] = _DEFAULTS[key]

        # Persistent embedding cache
        self.EMBEDDING_CACHE_DIR = _get("EMBEDDING_CACHE_DIR",
                                        "embedding_cache_dir",
                                        _DEFAULTS["embedding_cache_dir"])

        # HTML report output directory
        self.REPORT_DIR = _get("REPORT_DIR", "report_dir",
                               _DEFAULTS["report_dir"])

        # Step cache TTL
        self.STEP_CACHE_TTL_HOURS = _get("STEP_CACHE_TTL_HOURS",
                                         "step_cache_ttl_hours",
                                         _DEFAULTS["step_cache_ttl_hours"],
                                         cast=int)

        # Planner context size
        self.PLANNER_CONTEXT_CHARS = _get(
            "PLANNER_CONTEXT_CHARS", "planner_context_chars",
            _DEFAULTS["planner_context_chars"], cast=int)

        # Budget and Pricing
        self.BUDGET_LIMIT = _get("BUDGET_LIMIT", "budget_limit",
                                 _DEFAULTS["budget_limit"], cast=float)
        self.PRICING = yd.get("pricing", _DEFAULTS["pricing"])
        if not isinstance(self.PRICING, dict):
            self.PRICING = _DEFAULTS["pricing"]

        # Search agent
        self.SEARCH_ENABLED = _get_bool("SEARCH_ENABLED", "search_enabled",
                                         _DEFAULTS["search_enabled"])
        self.SEARCH_PROVIDER = _get("SEARCH_PROVIDER", "search_provider",
                                     _DEFAULTS["search_provider"])
        self.SEARCH_API_KEY = _get("SEARCH_API_KEY", "search_api_key",
                                    _DEFAULTS["search_api_key"])
        self.SEARCH_API_URL = _get("SEARCH_API_URL", "search_api_url",
                                    _DEFAULTS["search_api_url"])
        self.SEARCH_MAX_RESULTS = _get("SEARCH_MAX_RESULTS",
                                        "search_max_results",
                                        _DEFAULTS["search_max_results"],
                                        cast=int)
        self.SEARCH_MAX_PAGE_CHARS = _get("SEARCH_MAX_PAGE_CHARS",
                                           "search_max_page_chars",
                                           _DEFAULTS["search_max_page_chars"],
                                           cast=int)

        # Plugins
        self.PLUGINS: list[str] = yd.get("plugins", _DEFAULTS["plugins"])
        if not isinstance(self.PLUGINS, list):
            self.PLUGINS = []

    def to_dict(self) -> dict:
        """Return the current configuration as a dictionary."""
        return {
            "provider": self.PROVIDER,
            "model": self.DEFAULT_MODEL,
            "context_window": self.CONTEXT_WINDOW,
            "embedding_model": self.EMBEDDING_MODEL,
            "embedding_top_k": self.EMBEDDING_TOP_K,
            "stream": self.STREAM_RESPONSES,
            "llm_max_retries": self.LLM_MAX_RETRIES,
            "llm_retry_delay": self.LLM_RETRY_DELAY,
            "checkpoint_file": self.CHECKPOINT_FILE,
            "ollama_base_url": self.OLLAMA_BASE_URL,
            "lm_studio_base_url": self.LM_STUDIO_BASE_URL,
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "base_url": self.OPENAI_BASE_URL,
            },
            "gemini": {
                "api_key": self.GEMINI_API_KEY,
                "base_url": self.GEMINI_BASE_URL,
            },
            "anthropic": {
                "api_key": self.ANTHROPIC_API_KEY,
                "base_url": self.ANTHROPIC_BASE_URL,
            },
            "models": self._agent_models,
            "prompts": self.PROMPT_SUFFIXES,
            "embedding_cache_dir": self.EMBEDDING_CACHE_DIR,
            "report_dir": self.REPORT_DIR,
            "step_cache_ttl_hours": self.STEP_CACHE_TTL_HOURS,
            "planner_context_chars": self.PLANNER_CONTEXT_CHARS,
            "plugins": self.PLUGINS,
            "budget_limit": self.BUDGET_LIMIT,
            "pricing": self.PRICING,
            "search_enabled": self.SEARCH_ENABLED,
            "search_provider": self.SEARCH_PROVIDER,
            "search_api_key": self.SEARCH_API_KEY,
            "search_api_url": self.SEARCH_API_URL,
            "search_max_results": self.SEARCH_MAX_RESULTS,
            "search_max_page_chars": self.SEARCH_MAX_PAGE_CHARS,
        }

    def to_yaml(self) -> str:
        """Return the current configuration as a YAML string."""
        if yaml is None:
            # Fallback if yaml is not installed
            import json
            return json.dumps(self.to_dict(), indent=2)
        return yaml.dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def get_agent_model(self, agent_name: str) -> str | None:
        """Return the per-agent model override, or None to use the default."""
        return self._agent_models.get(agent_name.lower())

    @classmethod
    def load(cls, config_path: str | None = None) -> "Config":
        """Load config from YAML file (if found) + env vars + defaults."""
        path = _find_config_file(config_path)
        yaml_data = _load_yaml(path) if path else {}
        return cls(yaml_data)

    # Safety: Critical files that require extra care when editing
    CRITICAL_FILES = {
        "package.json",
        "pyproject.toml",
        "go.mod",
        "pom.xml",
        "requirements.txt",
        "Gemfile",
        "composer.json",
        "mix.exs",
        "Cargo.toml",
    }