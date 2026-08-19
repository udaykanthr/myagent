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
    "max_output_tokens": 16384,
    "embedding_model": "nomic-embed-text",
    "embedding_provider": None,  # if set, overrides 'provider' for KB embeddings only
    "embedding_top_k": 5,
    "stream": True,
    "llm_max_retries": 3,
    "llm_retry_delay": 2.0,
    # Seconds to wait for the next byte of a generation. Streaming resets
    # it per chunk, so this really guards the wait before the FIRST token —
    # which for a reasoning model is however long it thinks. 300 cost a
    # measured run 12 timed-out calls and two abandoned steps.
    "llm_read_timeout": 900,
    "checkpoint_file": ".agentchanti_checkpoint.json",
    "ollama_base_url": "http://localhost:11434/api/generate",
    "lm_studio_base_url": "http://localhost:1234/v1",
    "lm_studio_reasoning_effort": None,  # None | "low" | "medium" | "high"
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "openai_reasoning_effort": None,  # None | "low" | "medium" | "high"
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
    "language_backends": [],
    "planner_suffix": "Do not create meta-steps (e.g., 'Review code', 'Identify issues'). Focus on implementation. Combine analysis and action.",
    "budget_limit": 0.0,
    "search_enabled": True,
    "search_provider": "duckduckgo",
    "search_api_key": "",
    "search_api_url": "",
    "search_max_results": 3,
    "search_max_page_chars": 3000,
    "kb_registry_owner": "udaykanthr",
    "kb_registry_repo": "agentchanti-kb-registry",
    "kb_registry_auto_update": True,
    "kb_enabled": True,
    "kb_vector_backend": "local",
    "kb_max_context_tokens": 4000,
    "kb_auto_index_on_start": True,
    "kb_watcher_debounce_seconds": 1.0,
    "kb_verbose_logging": False,
    "editing_diff_mode": True,
    "editing_min_confidence": 0.60,
    "editing_context_lines": 5,
    "editing_fuzzy_match_window": 3,
    "editing_validate_syntax": True,
    "editing_track_metrics": True,
    "editing_fallback_on_syntax_error": True,
    "editing_chunk_mode": True,
    "editing_slim_context": True,
    "editing_reviewer_diff_mode": True,
    "editing_max_chunk_files": 3,
    "review_mode": "static",
    "agent_loop": True,
    "agent_loop_max_turns": 8,
    "wave_snapshots": True,
    "ghost_shadow": True,
    "require_independent_evidence": False,
    # Rounds of repair allowed when a user acceptance command fails.
    # 0 disables retrying entirely (the pre-2026-08-20 behaviour).
    "acceptance_repair_rounds": 3,
    "seed_acceptance_tests": True,
    "ghost_heal": True,
    "ghost_heal_source_edits": True,
    "plan_mode": "content",
    "dependency_check_enabled": True,
    "analyser_enabled": False,
    "wiring_verification_enabled": True,
    "smoke_test_enabled": True,
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
        self.MAX_OUTPUT_TOKENS = _get("MAX_OUTPUT_TOKENS", "max_output_tokens",
                                      _DEFAULTS["max_output_tokens"], cast=int)
        self.EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "embedding_model",
                                    _DEFAULTS["embedding_model"])
        self.EMBEDDING_PROVIDER = _get("EMBEDDING_PROVIDER", "embedding_provider",
                                       _DEFAULTS["embedding_provider"]) or None
        self.EMBEDDING_TOP_K = _get("EMBEDDING_TOP_K", "embedding_top_k",
                                    _DEFAULTS["embedding_top_k"], cast=int)

        self.LLM_MAX_RETRIES = _get("LLM_MAX_RETRIES", "llm_max_retries",
                                    _DEFAULTS["llm_max_retries"], cast=int)
        self.LLM_RETRY_DELAY = _get("LLM_RETRY_DELAY", "llm_retry_delay",
                                    _DEFAULTS["llm_retry_delay"], cast=float)
        self.LLM_READ_TIMEOUT = _get("LLM_READ_TIMEOUT", "llm_read_timeout",
                                     _DEFAULTS["llm_read_timeout"], cast=int)
        self.STREAM_RESPONSES = _get_bool("STREAM_RESPONSES", "stream",
                                          _DEFAULTS["stream"])

        self.CHECKPOINT_FILE = _get("CHECKPOINT_FILE", "checkpoint_file",
                                    _DEFAULTS["checkpoint_file"])

        self.OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "ollama_base_url",
                                    _DEFAULTS["ollama_base_url"])
        self.LM_STUDIO_BASE_URL = _get("LM_STUDIO_BASE_URL", "lm_studio_base_url",
                                       _DEFAULTS["lm_studio_base_url"])
        self.LM_STUDIO_REASONING_EFFORT = (
            os.getenv("LM_STUDIO_REASONING_EFFORT")
            or (yd.get("lm_studio", {}) or {}).get("reasoning_effort")
            or yd.get("reasoning_effort")  # top-level fallback
            or _DEFAULTS["lm_studio_reasoning_effort"]
        )

        # OpenAI / cloud provider
        openai_section = yd.get("openai", {}) if isinstance(yd.get("openai"), dict) else {}
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or openai_section.get(
            "api_key", _DEFAULTS["openai_api_key"])
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or openai_section.get(
            "base_url", _DEFAULTS["openai_base_url"])
        # Reasoning effort for OpenAI reasoning models. The top-level
        # `reasoning_effort:` key previously only reached LM Studio, so a
        # config that set it saw no effect on an OpenAI run at all.
        self.OPENAI_REASONING_EFFORT = (
            os.getenv("OPENAI_REASONING_EFFORT")
            or openai_section.get("reasoning_effort")
            or yd.get("reasoning_effort")   # top-level fallback
            or _DEFAULTS["openai_reasoning_effort"]
        )

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

        # Per-agent model overrides. Accept every key: a hardcoded
        # allowlist here silently dropped `escalation` (and `intent` /
        # `analyser`), so a correctly configured escalation model never
        # fired — the loop failed at its turn budget with the stronger
        # model sitting unused. Lowercased to match get_agent_model().
        #
        # A `<agent>_provider` key overrides the provider used to build
        # that agent's client, e.g.:
        #     models:
        #       escalation: gpt-5.4
        #       escalation_provider: openai
        # Without it a per-agent model inherits the run provider, so a
        # cross-provider escalation model (gpt-5.4 on an ollama run) is
        # POSTed to the wrong endpoint and 404s on every attempt.
        self._agent_models: dict[str, str] = {}
        self._agent_providers: dict[str, str] = {}
        models_section = yd.get("models", {})
        if isinstance(models_section, dict):
            for k, v in models_section.items():
                key = str(k).lower()
                if key.endswith("_provider"):
                    self._agent_providers[key[:-len("_provider")]] = str(v).lower()
                else:
                    self._agent_models[key] = str(v)

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

        # Global KB registry
        self.KB_REGISTRY_OWNER = _get(
            "KB_REGISTRY_OWNER", "kb_registry_owner",
            _DEFAULTS["kb_registry_owner"])
        self.KB_REGISTRY_REPO = _get(
            "KB_REGISTRY_REPO", "kb_registry_repo",
            _DEFAULTS["kb_registry_repo"])
        self.KB_REGISTRY_AUTO_UPDATE = _get_bool(
            "KB_REGISTRY_AUTO_UPDATE", "kb_registry_auto_update",
            _DEFAULTS["kb_registry_auto_update"])

        # KB context injection (Phase 4)
        kb_section = yd.get("kb", {}) if isinstance(yd.get("kb"), dict) else {}
        self.KB_ENABLED = _get_bool(
            "KB_ENABLED",
            "kb_enabled",
            kb_section.get("enabled", _DEFAULTS["kb_enabled"]),
        )
        self.KB_MAX_CONTEXT_TOKENS = int(
            os.getenv("KB_MAX_CONTEXT_TOKENS")
            or kb_section.get("max_context_tokens", _DEFAULTS["kb_max_context_tokens"])
        )
        self.KB_AUTO_INDEX_ON_START = _get_bool(
            "KB_AUTO_INDEX_ON_START",
            "kb_auto_index_on_start",
            kb_section.get("auto_index_on_start", _DEFAULTS["kb_auto_index_on_start"]),
        )
        self.KB_WATCHER_DEBOUNCE_SECONDS = float(
            os.getenv("KB_WATCHER_DEBOUNCE_SECONDS")
            or kb_section.get("watcher_debounce_seconds",
                              _DEFAULTS["kb_watcher_debounce_seconds"])
        )
        self.KB_VERBOSE_LOGGING = _get_bool(
            "KB_VERBOSE_LOGGING",
            "kb_verbose_logging",
            kb_section.get("verbose_logging", _DEFAULTS["kb_verbose_logging"]),
        )
        self.KB_VECTOR_BACKEND = _get(
            "KB_VECTOR_BACKEND", "kb_vector_backend",
            kb_section.get("vector_backend", _DEFAULTS["kb_vector_backend"]))

        # Diff-aware editing (Phase 5)
        editing_section = yd.get("editing", {}) if isinstance(yd.get("editing"), dict) else {}
        self.EDITING_DIFF_MODE = _get_bool(
            "EDITING_DIFF_MODE", "editing_diff_mode",
            editing_section.get("diff_mode", _DEFAULTS["editing_diff_mode"]),
        )
        self.EDITING_MIN_CONFIDENCE = float(
            os.getenv("EDITING_MIN_CONFIDENCE")
            or editing_section.get("min_confidence_threshold",
                                   _DEFAULTS["editing_min_confidence"])
        )
        self.EDITING_CONTEXT_LINES = int(
            os.getenv("EDITING_CONTEXT_LINES")
            or editing_section.get("context_lines",
                                   _DEFAULTS["editing_context_lines"])
        )
        self.EDITING_FUZZY_MATCH_WINDOW = int(
            os.getenv("EDITING_FUZZY_MATCH_WINDOW")
            or editing_section.get("fuzzy_match_window",
                                   _DEFAULTS["editing_fuzzy_match_window"])
        )
        self.EDITING_VALIDATE_SYNTAX = _get_bool(
            "EDITING_VALIDATE_SYNTAX", "editing_validate_syntax",
            editing_section.get("validate_syntax_after_patch",
                                _DEFAULTS["editing_validate_syntax"]),
        )
        self.EDITING_TRACK_METRICS = _get_bool(
            "EDITING_TRACK_METRICS", "editing_track_metrics",
            editing_section.get("track_metrics",
                                _DEFAULTS["editing_track_metrics"]),
        )
        self.EDITING_FALLBACK_ON_SYNTAX_ERROR = _get_bool(
            "EDITING_FALLBACK_ON_SYNTAX_ERROR", "editing_fallback_on_syntax_error",
            editing_section.get("fallback_on_syntax_error",
                                _DEFAULTS["editing_fallback_on_syntax_error"]),
        )
        self.EDITING_CHUNK_MODE = _get_bool(
            "EDITING_CHUNK_MODE", "editing_chunk_mode",
            editing_section.get("chunk_mode",
                                _DEFAULTS["editing_chunk_mode"]),
        )
        self.EDITING_SLIM_CONTEXT = _get_bool(
            "EDITING_SLIM_CONTEXT", "editing_slim_context",
            editing_section.get("slim_context",
                                _DEFAULTS["editing_slim_context"]),
        )
        self.EDITING_REVIEWER_DIFF_MODE = _get_bool(
            "EDITING_REVIEWER_DIFF_MODE", "editing_reviewer_diff_mode",
            editing_section.get("reviewer_diff_mode",
                                _DEFAULTS["editing_reviewer_diff_mode"]),
        )
        self.EDITING_MAX_CHUNK_FILES = int(
            os.getenv("EDITING_MAX_CHUNK_FILES")
            or editing_section.get("max_chunk_files",
                                   _DEFAULTS["editing_max_chunk_files"])
        )

        # Review mode: "static" (default) skips LLM reviewer when offline
        # lint + import checks pass; "full" always runs LLM reviewer.
        self.REVIEW_MODE = _get(
            "REVIEW_MODE", "review_mode",
            editing_section.get("review_mode", _DEFAULTS["review_mode"]),
        )

        # Agent loop: run CODE/TEST steps as a bounded tool-calling loop
        # instead of the generate → review → retry pipeline. Default ON —
        # A/B benchmarked at parity on success rate and ~14% cheaper on
        # tokens. Only takes effect when the provider supports native tool
        # calling (Ollama/OpenAI/Anthropic); other providers automatically
        # use the classic pipeline. Set `agent_loop: false` to opt out.
        self.AGENT_LOOP = _get_bool(
            "AGENT_LOOP", "agent_loop", _DEFAULTS["agent_loop"])
        self.AGENT_LOOP_MAX_TURNS = _get(
            "AGENT_LOOP_MAX_TURNS", "agent_loop_max_turns",
            _DEFAULTS["agent_loop_max_turns"], cast=int)

        # Per-wave git snapshots of the target project + monotonic
        # rollback when fix rounds break previously-passing gates.
        # Only activates in a workdir that is NOT already a git repo.
        self.WAVE_SNAPSHOTS = _get_bool(
            "WAVE_SNAPSHOTS", "wave_snapshots", _DEFAULTS["wave_snapshots"])

        # Read-only reconciliation of the plan's declared effects against
        # the real tree (orchestrator/ghost.py). Costs no LLM call and
        # changes no verdict — it only reports where the evidence and the
        # pipeline's own claim disagree.
        self.GHOST_SHADOW = _get_bool(
            "GHOST_SHADOW", "ghost_shadow", _DEFAULTS["ghost_shadow"])

        # Acceptance commands the USER wrote (orchestrator/evidence.py).
        # The only instrument in a run the model neither authored nor can
        # edit, which is why these are also the only checks allowed to
        # fail the run on their own. Absent by default: a run without them
        # is reported as completed-but-unverified, never as failed.
        _acc = os.getenv("ACCEPTANCE_CMDS")
        if _acc is not None:
            self.ACCEPTANCE_CMDS = [c for c in _acc.split(";") if c.strip()]
        else:
            _acc_yaml = yd.get("acceptance_cmds") or []
            if isinstance(_acc_yaml, str):
                _acc_yaml = [_acc_yaml]
            self.ACCEPTANCE_CMDS = [str(c) for c in _acc_yaml if str(c).strip()]

        # Turn "completed but nothing independent verified it" into a
        # non-zero exit. Off by default so greenfield builds, which
        # legitimately have no pre-existing suite, do not all start
        # failing — the honest default is to say so, not to fail.
        self.REQUIRE_INDEPENDENT_EVIDENCE = _get_bool(
            "REQUIRE_INDEPENDENT_EVIDENCE", "require_independent_evidence",
            _DEFAULTS["require_independent_evidence"])
        self.ACCEPTANCE_REPAIR_ROUNDS = _get(
            "ACCEPTANCE_REPAIR_ROUNDS", "acceptance_repair_rounds",
            _DEFAULTS["acceptance_repair_rounds"], cast=int)
        self.SEED_ACCEPTANCE_TESTS = _get_bool(
            "SEED_ACCEPTANCE_TESTS", "seed_acceptance_tests",
            _DEFAULTS["seed_acceptance_tests"])

        # Deterministic repair of the gaps the shadow finds
        # (orchestrator/ghost_heal.py): installing a declared dependency
        # into the interpreter that runs the app, creating an absent
        # package marker, adding a declared-but-missing import. Never
        # writes content the plan did not supply — a healer that invents
        # a CSS rule or a function body to satisfy its own check turns a
        # detectable defect into an undetectable one.
        self.GHOST_HEAL = _get_bool(
            "GHOST_HEAL", "ghost_heal", _DEFAULTS["ghost_heal"])
        # Narrows healing to environment actions only (installs), leaving
        # every project file untouched.
        self.GHOST_HEAL_SOURCE_EDITS = _get_bool(
            "GHOST_HEAL_SOURCE_EDITS", "ghost_heal_source_edits",
            _DEFAULTS["ghost_heal_source_edits"])

        # Planning mode: "content" (planner emits full file bodies,
        # classic) or "intent" (planner emits goals + verify: gates; the
        # tool-calling agent loop authors the files against real project
        # state). Intent mode requires a tool-capable provider — steps
        # without inline code route through the agent loop.
        _plan_mode = str(_get(
            "PLAN_MODE", "plan_mode", _DEFAULTS["plan_mode"])).lower()
        self.PLAN_MODE = _plan_mode if _plan_mode in ("content", "intent") \
            else _DEFAULTS["plan_mode"]

        # Dependency check (post-step integration validation)
        dep_section = yd.get("dependency_check", {}) if isinstance(yd.get("dependency_check"), dict) else {}
        self.DEPENDENCY_CHECK_ENABLED = _get_bool(
            "DEPENDENCY_CHECK_ENABLED", "dependency_check_enabled",
            dep_section.get("enabled", _DEFAULTS["dependency_check_enabled"]),
        )

        # Analyser LLM enrichment (off by default — costs 1 LLM call per run)
        analyser_section = yd.get("analyser", {}) if isinstance(yd.get("analyser"), dict) else {}
        self.ANALYSER_ENABLED = _get_bool(
            "ANALYSER_ENABLED", "analyser_enabled",
            analyser_section.get("enabled", _DEFAULTS["analyser_enabled"]),
        )

        # Wiring verification — cross-file integration check after all steps
        # (on by default; disable via `wiring_verification: {enabled: false}`
        # or WIRING_VERIFICATION_ENABLED=false env var)
        _wv_section = yd.get("wiring_verification", {})
        if not isinstance(_wv_section, dict):
            _wv_section = {}
        self.WIRING_VERIFICATION_ENABLED = _get_bool(
            "WIRING_VERIFICATION_ENABLED", "wiring_verification_enabled",
            _wv_section.get("enabled", _DEFAULTS["wiring_verification_enabled"]),
        )

        # Runtime smoke test — launch the app entry point after the pipeline
        # succeeds and feed launch crashes into the fix loop (on by default;
        # disable via `smoke_test: {enabled: false}` or SMOKE_TEST_ENABLED=false)
        _st_section = yd.get("smoke_test", {})
        if not isinstance(_st_section, dict):
            _st_section = {}
        self.SMOKE_TEST_ENABLED = _get_bool(
            "SMOKE_TEST_ENABLED", "smoke_test_enabled",
            _st_section.get("enabled", _DEFAULTS["smoke_test_enabled"]),
        )

        # Plugins
        self.PLUGINS: list[str] = yd.get("plugins", _DEFAULTS["plugins"])
        if not isinstance(self.PLUGINS, list):
            self.PLUGINS = []

        # Language backends (custom LanguageBackend subclass paths)
        self.LANGUAGE_BACKENDS: list[str] = yd.get(
            "language_backends", _DEFAULTS["language_backends"])
        if not isinstance(self.LANGUAGE_BACKENDS, list):
            self.LANGUAGE_BACKENDS = []

    def to_dict(self) -> dict:
        """Return the current configuration as a dictionary."""
        return {
            "provider": self.PROVIDER,
            "model": self.DEFAULT_MODEL,
            "context_window": self.CONTEXT_WINDOW,
            "embedding_model": self.EMBEDDING_MODEL,
            "embedding_provider": self.EMBEDDING_PROVIDER,
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
            "models": {
                **self._agent_models,
                **{f"{k}_provider": v for k, v in self._agent_providers.items()},
            },
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
            "kb_registry_owner": self.KB_REGISTRY_OWNER,
            "kb_registry_repo": self.KB_REGISTRY_REPO,
            "kb_registry_auto_update": self.KB_REGISTRY_AUTO_UPDATE,
            "kb": {
                "enabled": self.KB_ENABLED,
                "vector_backend": self.KB_VECTOR_BACKEND,
                "max_context_tokens": self.KB_MAX_CONTEXT_TOKENS,
                "auto_index_on_start": self.KB_AUTO_INDEX_ON_START,
                "watcher_debounce_seconds": self.KB_WATCHER_DEBOUNCE_SECONDS,
                "verbose_logging": self.KB_VERBOSE_LOGGING,
            },
            "editing": {
                "diff_mode": self.EDITING_DIFF_MODE,
                "min_confidence_threshold": self.EDITING_MIN_CONFIDENCE,
                "context_lines": self.EDITING_CONTEXT_LINES,
                "fuzzy_match_window": self.EDITING_FUZZY_MATCH_WINDOW,
                "validate_syntax_after_patch": self.EDITING_VALIDATE_SYNTAX,
                "track_metrics": self.EDITING_TRACK_METRICS,
                "fallback_on_syntax_error": self.EDITING_FALLBACK_ON_SYNTAX_ERROR,
                "chunk_mode": self.EDITING_CHUNK_MODE,
                "slim_context": self.EDITING_SLIM_CONTEXT,
                "reviewer_diff_mode": self.EDITING_REVIEWER_DIFF_MODE,
                "max_chunk_files": self.EDITING_MAX_CHUNK_FILES,
                "review_mode": self.REVIEW_MODE,
            },
            "dependency_check": {
                "enabled": self.DEPENDENCY_CHECK_ENABLED,
            },
            "analyser": {
                "enabled": self.ANALYSER_ENABLED,
            },
            "wiring_verification": {
                "enabled": self.WIRING_VERIFICATION_ENABLED,
            },
            "smoke_test": {
                "enabled": self.SMOKE_TEST_ENABLED,
            },
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

    def get_agent_provider(self, agent_name: str) -> str | None:
        """Return the per-agent provider override, or None to use the run provider."""
        return self._agent_providers.get(agent_name.lower())

    @classmethod
    def load(cls, config_path: str | None = None) -> "Config":
        """Load config from YAML file (if found) + env vars + defaults."""
        path = _find_config_file(config_path)
        yaml_data = _load_yaml(path) if path else {}
        return cls(yaml_data)

    # Safety: Critical files that require extra care when editing
    CRITICAL_FILES = {
        # "package.json",
        "pyproject.toml",
        "go.mod",
        "pom.xml",
        "requirements.txt",
        "Gemfile",
        "composer.json",
        "mix.exs",
        "Cargo.toml",
    }