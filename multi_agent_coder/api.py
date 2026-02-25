"""
Programmatic API for AgentChanti — use as a library from Python code.

Example usage::

    from multi_agent_coder import run_task

    result = run_task(
        task="Add input validation to all endpoints",
        provider="ollama",
        model="deepseek-coder-v2:16b",
        auto=True,
    )
    print(result.success)
    print(result.files_written)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

from .config import Config
from .llm.ollama import OllamaClient
from .llm.lm_studio import LMStudioClient
from .agents.planner import PlannerAgent
from .agents.coder import CoderAgent
from .agents.reviewer import ReviewerAgent
from .agents.tester import TesterAgent
from .executor import Executor
from .embedding_store import EmbeddingStore
from .cli_display import CLIDisplay, token_tracker, log
from .language import (
    detect_language, detect_language_from_task, get_language_name, get_code_block_lang,
)
from .project_scanner import scan_project, format_scan_for_planner, collect_source_files
from .checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint

from .orchestrator.memory import FileMemory
from .orchestrator.pipeline import build_step_waves, _execute_step, _run_diagnosis_loop


@dataclass
class TaskResult:
    """Structured result returned by :func:`run_task`."""
    success: bool
    files_written: list[str] = field(default_factory=list)
    plan_steps: list[str] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)
    log_file: str = ""
    error: str = ""


def run_task(
    task: str,
    *,
    provider: str = "lm_studio",
    model: str | None = None,
    embed_model: str | None = None,
    language: str | None = None,
    no_embeddings: bool = False,
    no_git: bool = True,
    no_kb: bool = False,
    auto: bool = True,
    config_path: str | None = None,
    working_dir: str | None = None,
) -> TaskResult:
    """Run an AgentChanti task programmatically.

    Args:
        task: The coding task description.
        provider: LLM provider — ``"ollama"``, ``"lm_studio"``, or ``"openai"``.
        model: Model name override (default: from config).
        embed_model: Embedding model name (default: from config).
        language: Programming language override (auto-detected if None).
        no_embeddings: Disable semantic embeddings.
        no_git: Disable git integration (default True for library use).
        no_kb: Disable KB context injection.
        auto: Non-interactive mode (default True for library use).
        config_path: Explicit path to ``.agentchanti.yaml``.
        working_dir: Working directory for the task (default: CWD).

    Returns:
        A :class:`TaskResult` with success status, files written, plan steps,
        and token usage.
    """
    # Change to working directory if specified
    original_dir = os.getcwd()
    if working_dir:
        os.chdir(working_dir)

    try:
        return _run_task_impl(
            task, provider=provider, model=model, embed_model=embed_model,
            language=language, no_embeddings=no_embeddings,
            no_git=no_git, no_kb=no_kb, auto=auto, config_path=config_path,
        )
    finally:
        os.chdir(original_dir)


def _run_task_impl(
    task: str, *, provider: str, model: str | None,
    embed_model: str | None, language: str | None,
    no_embeddings: bool, no_git: bool, no_kb: bool,
    auto: bool, config_path: str | None,
) -> TaskResult:
    """Internal implementation of run_task."""

    cfg = Config.load(config_path)
    model = model or cfg.DEFAULT_MODEL
    embed_model = embed_model or cfg.EMBEDDING_MODEL

    # Detect language
    if not language:
        language = detect_language_from_task(task) or detect_language()

    # Init LLM
    llm_kwargs = dict(
        max_retries=cfg.LLM_MAX_RETRIES,
        retry_delay=cfg.LLM_RETRY_DELAY,
        stream=cfg.STREAM_RESPONSES,
    )

    if provider == "ollama":
        llm_client = OllamaClient(
            base_url=cfg.OLLAMA_BASE_URL, model=model, **llm_kwargs)
    elif provider == "openai":
        from .llm.openai_client import OpenAIClient
        api_key = cfg.OPENAI_API_KEY
        if not api_key:
            return TaskResult(
                success=False,
                error="OpenAI provider requires an API key. "
                      "Set OPENAI_API_KEY or add it to .agentchanti.yaml.",
            )
        llm_client = OpenAIClient(
            base_url=cfg.OPENAI_BASE_URL, model=model,
            api_key=api_key, **llm_kwargs)
    else:
        llm_client = LMStudioClient(
            base_url=cfg.LM_STUDIO_BASE_URL, model=model, **llm_kwargs)

    # Project scan
    scan_result = scan_project(".")
    source_files = collect_source_files(".")
    project_context = format_scan_for_planner(
        scan_result, max_chars=cfg.PLANNER_CONTEXT_CHARS,
        source_files=source_files)

    # Embedding store
    embed_store = None
    if not no_embeddings:
        embed_store = EmbeddingStore(llm_client, embed_model=embed_model)

    # Per-agent model helper
    def _make_llm(agent_name: str):
        agent_model = cfg.get_agent_model(agent_name) or model
        if agent_model == model:
            return llm_client
        if provider == "ollama":
            return OllamaClient(
                base_url=cfg.OLLAMA_BASE_URL, model=agent_model, **llm_kwargs)
        elif provider == "openai":
            from .llm.openai_client import OpenAIClient
            return OpenAIClient(
                base_url=cfg.OPENAI_BASE_URL, model=agent_model,
                api_key=cfg.OPENAI_API_KEY, **llm_kwargs)
        else:
            return LMStudioClient(
                base_url=cfg.LM_STUDIO_BASE_URL, model=agent_model, **llm_kwargs)

    planner = PlannerAgent("Planner", "Senior Software Architect",
                           "Create a step-by-step plan for the coding task.",
                           _make_llm("planner"))
    coder = CoderAgent("Coder", "Senior Software Developer",
                       f"Write clean {get_language_name(language)} code for a single step.",
                       _make_llm("coder"))
    reviewer = ReviewerAgent("Reviewer", "Code Reviewer",
                             "Review code for errors and style issues.",
                             _make_llm("reviewer"))
    tester = TesterAgent("Tester", "Software Engineer in Test",
                         "Create unit tests for the provided code.",
                         _make_llm("tester"))
    executor = Executor()
    display = CLIDisplay(task)
    memory = FileMemory(embedding_store=embed_store, top_k=cfg.EMBEDDING_TOP_K)

    # Search agent
    search_agent = None
    if cfg.SEARCH_ENABLED:
        from .agents.search import SearchAgent
        search_agent = SearchAgent(
            provider=cfg.SEARCH_PROVIDER,
            api_key=cfg.SEARCH_API_KEY,
            api_url=cfg.SEARCH_API_URL,
            max_results=cfg.SEARCH_MAX_RESULTS,
            max_page_chars=cfg.SEARCH_MAX_PAGE_CHARS,
        )

    # KB context builder and runtime watcher (Phase 4)
    kb_context_builder = None
    kb_runtime_watcher = None
    if cfg.KB_ENABLED and not no_kb:
        try:
            from .kb.startup import KBStartupManager
            from .kb.context_builder import ContextBuilder
            from .kb.runtime_watcher import RuntimeWatcher

            # Smart startup check — handles Qdrant, global KB, local KB
            KBStartupManager().run(project_root=os.getcwd())

            kb_context_builder = ContextBuilder(project_root=os.getcwd())
            kb_runtime_watcher = RuntimeWatcher(
                debounce_seconds=cfg.KB_WATCHER_DEBOUNCE_SECONDS,
            )
            kb_runtime_watcher.start(project_root=os.getcwd())
            _logger.info("[KB] Context builder and runtime watcher initialised")
        except Exception as kb_exc:
            _logger.warning("[KB] Initialisation failed (non-fatal): %s", kb_exc)
            kb_context_builder = None
            kb_runtime_watcher = None

    # Project orientation — detect project DNA for LLM grounding
    project_profile = None
    try:
        from .kb.project_orientation import ProjectOrientation

        kb_graph = None
        if kb_context_builder is not None:
            kb_graph = getattr(kb_context_builder, "_graph", None)

        orientation = ProjectOrientation(
            graph=kb_graph,
            project_root=os.getcwd(),
        )
        project_profile = orientation.get_profile()
        _logger.info(
            "[KB] Project detected: %s / %s | source: %s | tests: %s",
            project_profile.language,
            project_profile.framework or "no framework",
            project_profile.source_root,
            ", ".join(project_profile.test_frameworks) or "unknown",
        )
    except Exception as orient_exc:
        _logger.warning(
            "[KB] Project orientation failed (non-fatal): %s", orient_exc,
        )

    # Pre-load existing source files into memory
    if source_files:
        memory.update(source_files)

    # Plan
    planner_context = f"Existing project:\n{project_context}" if project_context else ""
    plan = planner.process(task, context=planner_context)
    raw_steps = executor.parse_plan_steps(plan)
    if not raw_steps:
        return TaskResult(success=False, error="Could not parse any steps from the plan.")

    steps, dependencies = executor.parse_step_dependencies(raw_steps)
    display.set_steps(steps)

    # Execute
    waves = build_step_waves(steps, dependencies)
    checkpoint_file = cfg.CHECKPOINT_FILE
    step_results: dict[int, str] = {}
    pipeline_success = True

    for wave_idx, wave in enumerate(waves):
        for idx in wave:
            step_text = steps[idx]
            idx, success, error_info = _execute_step(
                idx, step_text,
                llm_client=llm_client, executor=executor,
                coder=coder, reviewer=reviewer, tester=tester,
                task=task, memory=memory, display=display,
                language=language, auto=auto,
                kb_context_builder=kb_context_builder,
                project_profile=project_profile,
            )

            if success:
                step_results[idx] = "done"
                save_checkpoint(checkpoint_file, task, steps, idx,
                                memory.as_dict(), step_results, language)
            else:
                fixed = _run_diagnosis_loop(
                    idx, step_text, error_info,
                    llm_client=llm_client, executor=executor,
                    coder=coder, reviewer=reviewer, tester=tester,
                    task=task, memory=memory, display=display,
                    language=language,
                    search_agent=search_agent,
                    kb_context_builder=kb_context_builder,
                    project_profile=project_profile,
                )
                if fixed:
                    step_results[idx] = "done"
                else:
                    pipeline_success = False
                    break

        if not pipeline_success:
            break

    # Stop KB runtime watcher
    if kb_runtime_watcher is not None:
        try:
            kb_runtime_watcher.stop()
        except Exception:
            pass

    if pipeline_success:
        clear_checkpoint(checkpoint_file)

    # Collect written files (exclude internal cmd/fix outputs)
    files_written = [f for f in memory.all_files().keys()
                     if not f.startswith("_")]

    return TaskResult(
        success=pipeline_success,
        files_written=files_written,
        plan_steps=steps,
        token_usage={
            "total": token_tracker.total_tokens,
            "prompt": token_tracker.total_prompt_tokens,
            "completion": token_tracker.total_completion_tokens,
        },
        log_file=log.handlers[0].baseFilename if log.handlers else "",
    )
