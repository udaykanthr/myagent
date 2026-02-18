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
    "models": {},
    "embedding_cache_dir": ".agentchanti",
    "report_dir": ".agentchanti/reports",
    "step_cache_ttl_hours": 24,
    "plugins": [],
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
                if key in prompts_section:
                    self.PROMPT_SUFFIXES[key] = str(prompts_section[key])

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

        # Plugins
        self.PLUGINS: list[str] = yd.get("plugins", _DEFAULTS["plugins"])
        if not isinstance(self.PLUGINS, list):
            self.PLUGINS = []

    def get_agent_model(self, agent_name: str) -> str | None:
        """Return the per-agent model override, or None to use the default."""
        return self._agent_models.get(agent_name.lower())

    @classmethod
    def load(cls, config_path: str | None = None) -> "Config":
        """Load config from YAML file (if found) + env vars + defaults."""
        path = _find_config_file(config_path)
        yaml_data = _load_yaml(path) if path else {}
        return cls(yaml_data)