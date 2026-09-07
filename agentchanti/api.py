"""
Programmatic API for AgentChanti — use as a library from Python code.

Example usage::

    from agentchanti import run_task

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
from dataclasses import replace as _dc_replace
import os
import time
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

from .config import Config
from .llm.ollama import OllamaClient
from .llm.lm_studio import LMStudioClient
from .llm import build_embed_client
from .agents.planner import PlannerAgent
from .agents.coder import CoderAgent
from .agents.reviewer import ReviewerAgent
from .agents.tester import TesterAgent
from .agents.analyser import AnalyseAgent, ProjectContext, build_project_context, parse_briefing_packages
from .executor import Executor
from .embedding_store import EmbeddingStore
from .cli_display import CLIDisplay, token_tracker, log
from .language import (
    detect_language, detect_language_from_task, get_language_name, get_code_block_lang,
)
from .project_scanner import scan_project, format_scan_for_planner, collect_source_files
from .checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint

from .orchestrator.memory import FileMemory
from .orchestrator.pipeline import (
    build_step_waves, _execute_step, _run_diagnosis_loop,
    run_wiring_verification,
)
from .orchestrator.test_analyzer import perform_baseline_test_analysis
from .orchestrator.plan_step import build_waves as _build_plan_waves


@dataclass
class TaskResult:
    """Structured result returned by :func:`run_task`."""
    success: bool
    files_written: list[str] = field(default_factory=list)
    plan_steps: list[str] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)
    log_file: str = ""
    error: str = ""
    answer: str = ""  # populated for QUESTION tasks that produce no code changes
    # Did anything the agent did NOT write in this run agree that it
    # worked? `success` means the plan ran; this means someone else
    # checked. See orchestrator/evidence.py — a run whose only passing
    # tests it wrote itself is `success=True, verified=False`.
    verified: bool = False
    evidence: str = ""


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

    # Load custom language backends from config
    if cfg.LANGUAGE_BACKENDS:
        from .language_backend import load_custom_backends
        load_custom_backends(cfg.LANGUAGE_BACKENDS)

    # Init LLM
    llm_kwargs = dict(
        max_retries=cfg.LLM_MAX_RETRIES,
        retry_delay=cfg.LLM_RETRY_DELAY,
        stream=cfg.STREAM_RESPONSES,
        max_output_tokens=cfg.MAX_OUTPUT_TOKENS,
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
            api_key=api_key,
            reasoning_effort=cfg.OPENAI_REASONING_EFFORT, **llm_kwargs)
    else:
        llm_client = LMStudioClient(
            base_url=cfg.LM_STUDIO_BASE_URL, model=model,
            reasoning_effort=cfg.LM_STUDIO_REASONING_EFFORT, **llm_kwargs)

    # Project scan TODO: Dont scan entire project instead scan only the files that are relevant to the task or 1 layer of folders
    scan_result = scan_project(".")
    source_files = collect_source_files(".")
    project_context = format_scan_for_planner(
        scan_result, max_chars=cfg.PLANNER_CONTEXT_CHARS,
        source_files=source_files)

    # Embedding store — embed_client kept at top level for KB component reuse
    embed_client = None if no_embeddings else build_embed_client(cfg)
    embed_store = None
    if embed_client is not None:
        embed_store = EmbeddingStore(embed_client, embed_model=embed_model)

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
                api_key=cfg.OPENAI_API_KEY,
                reasoning_effort=cfg.OPENAI_REASONING_EFFORT, **llm_kwargs)
        else:
            return LMStudioClient(
                base_url=cfg.LM_STUDIO_BASE_URL, model=agent_model,
                reasoning_effort=cfg.LM_STUDIO_REASONING_EFFORT, **llm_kwargs)

    planner = PlannerAgent("Planner", "Senior Software Architect",
                           "Create a step-by-step plan for the coding task.",
                           _make_llm("planner"))
    from .agents.intent import IntentAgent, parse_intent_spec
    intent_agent = IntentAgent("IntentAnalyzer", "Requirements Analyst",
                               "Analyze the prompt and search the web if intent is ambiguous to produce a formal REQUIREMENTS_SPEC.",
                               _make_llm("intent"))
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

    # Search agent TODO: should be running only on a condition?
    search_agent = None
    if cfg.SEARCH_ENABLED:
        from .agents.search import SearchAgent
        search_agent = SearchAgent(
            provider=cfg.SEARCH_PROVIDER,
            api_key=cfg.SEARCH_API_KEY,
            api_url=cfg.SEARCH_API_URL,
            max_results=cfg.SEARCH_MAX_RESULTS,
            max_page_chars=cfg.SEARCH_MAX_PAGE_CHARS,
            llm_client=llm_client,
        )

    # KB context builder and runtime watcher (Phase 4)
    kb_context_builder = None
    kb_runtime_watcher = None
    if cfg.KB_ENABLED and not no_kb:
        try:
            from .kb.startup import KBStartupManager
            from .kb.context_builder import ContextBuilder
            from .kb.runtime_watcher import RuntimeWatcher

            # Use embed_client for KB vector ops; fall back to llm_client if unavailable
            kb_api_client = embed_client or llm_client

            # Smart startup check — handles global KB, local KB
            KBStartupManager().run(project_root=os.getcwd(), api_client=kb_api_client)

            kb_context_builder = ContextBuilder(project_root=os.getcwd(), api_client=kb_api_client)
            kb_runtime_watcher = RuntimeWatcher(
                debounce_seconds=cfg.KB_WATCHER_DEBOUNCE_SECONDS,
            )
            kb_runtime_watcher.start(project_root=os.getcwd(), api_client=kb_api_client)
            memory.watcher_created_files = kb_runtime_watcher.created_files
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

    # Knowledge base
    knowledge_base = None
    try:
        from .knowledge import KnowledgeBase
        kb_path = os.path.join(cfg.EMBEDDING_CACHE_DIR, "knowledge.json")
        knowledge_base = KnowledgeBase(path=kb_path)
        if project_profile is not None:
            knowledge_base.update_tech_stack(project_profile)
    except Exception:
        pass

    # Pre-analysis: map relevant files, classify intent, enrich context
    # Detect blank projects (no package manager / build config files)
    _has_project_config = bool(scan_result.get("key_files"))
    if _has_project_config:
        planner_context = f"Existing project:\n{project_context}"
    else:
        from .orchestrator.cli import _blank_project_scaffold_hint
        _scaffold_hint = _blank_project_scaffold_hint(language)
        planner_context = (
            f"PROJECT STATE: BLANK / EMPTY directory — no build config files found.\n"
            f"The plan MUST start with project scaffolding / initialization steps "
            f"({_scaffold_hint}) before writing any source code.\n"
        )
        if project_context:
            planner_context += f"\nCurrent directory contents:\n{project_context}"

    # KB injection deferred — done after pre_analyze so the IntentAgent's
    # "KB topics:" field can filter it down to relevant entries only.

    # Baseline test analysis before planning
    from .agents.planner import _classify_task_intent
    _task_intent = _classify_task_intent(task)
    test_analysis = ""
    if not no_kb:
        try:
            test_analysis = perform_baseline_test_analysis(
                memory, executor, language,
                project_profile=project_profile,
                task_intent=_task_intent,
            )
        except Exception as test_exc:
            _logger.warning("[Analysis] Baseline test analysis failed: %s", test_exc)

    _api_subproject: str | None = None
    try:
        from .orchestrator.step_handlers import _detect_subproject_root
        _api_subproject = _detect_subproject_root(memory)
    except Exception:
        pass
    analysis_context = planner.pre_analyze(
        task,
        source_files=source_files,
        kb_context_builder=kb_context_builder,
        knowledge_base=knowledge_base,
        test_analysis=test_analysis,
        language=language,
        intent_agent=intent_agent,
        search_agent=search_agent,
        subproject_cwd=_api_subproject,
        executor=executor,
    )
    if analysis_context:
        planner_context = analysis_context + "\n\n" + planner_context

    # ── QUESTION short-circuit ────────────────────────────────────────────────
    # The IntentAgent already produced the answer — no planner call needed.
    if getattr(planner, '_is_question_task', False):
        _answer = getattr(planner, '_question_answer', '')
        return TaskResult(success=True, answer=_answer)

    # Record whether the RAW task asks for tests before enrichment replaces
    # it (see cli.py — the enriched spec routinely adds testing language).
    from .language import task_requests_tests
    memory._task_requests_tests = task_requests_tests(task)

    # Update task if IntentAgent enriched it during pre_analyze
    task = getattr(planner, '_enriched_task', task)
    intent_spec = parse_intent_spec(task)

    # ── Filtered KB injection ─────────────────────────────────────────────────
    if knowledge_base and knowledge_base.size > 0:
        import re as _re_kb
        from .orchestrator.cli import _parse_kb_topics
        _kb_topics: list[str] = _parse_kb_topics(task, _re_kb)
        kb_ctx = (
            knowledge_base.format_for_task(_kb_topics)
            if _kb_topics
            else knowledge_base.format_stack_only()
        )
        if kb_ctx:
            planner_context += f"\n\n{kb_ctx}"

    # Apply LLM-corrected language (set by pre_analyze when heuristics were wrong)
    _llm_detected = getattr(planner, '_detected_language', None)
    if _llm_detected and _llm_detected != language:
        import logging as _lg
        _lg.getLogger(__name__).info(
            "Language corrected by LLM during pre-analysis: %s → %s",
            language, _llm_detected,
        )
        language = _llm_detected
        coder.role = f"Write clean {get_language_name(language)} code for a single step."

    # Propagate task briefing to memory so all downstream agents can use it
    _briefing_text = getattr(planner, '_task_briefing', '')
    if _briefing_text:
        memory._task_briefing = _briefing_text

    # Plan
    plan = planner.process(task, context=planner_context, language=language,
                           plan_mode=getattr(cfg, "PLAN_MODE", "content"))

    # ── Parse plan: try structured format first, fall back to legacy ──
    from .orchestrator.plan_step import (
        parse_structured_plan, is_structured_plan, validate_plan,
        fix_nested_workspace_collision,
        fix_import_dependencies,
        project_file_reader,
        steps_as_text_list, steps_dependencies_dict,
        from_legacy_steps, parse_heuristic_plan, PlanStep,
        reclassify_manifest_steps,
    )
    plan_steps: list[PlanStep] | None = None

    _is_structured = is_structured_plan(plan)
    _logger.info("[Plan] Structured plan detected: %s", _is_structured)
    if _is_structured:
        plan_steps = parse_structured_plan(plan)
        if plan_steps:
            _logger.info(
                "[Plan] Parsed %d structured steps: %s",
                len(plan_steps),
                [(s.id, s.step_type, s.index) for s in plan_steps],
            )
            errors = validate_plan(plan_steps)
            if errors:
                _logger.warning("[Plan] Validation warnings: %s", errors)
            ws_fixes = fix_nested_workspace_collision(plan_steps)
            if ws_fixes:
                _logger.info("[Plan] Auto-fixed workspace collision: %s", ws_fixes)
            dep_fixes = fix_import_dependencies(plan_steps, read_file=project_file_reader)
            if dep_fixes:
                _logger.info("[Plan] Auto-fixed import dependencies: %s", dep_fixes)
            steps = steps_as_text_list(plan_steps)
            dependencies = steps_dependencies_dict(plan_steps)
        else:
            _logger.warning("[Plan] Structured parse returned 0 steps, falling back to legacy")

    if plan_steps is None:
        # Heuristic fallback: handles weaker LLMs that output markdown headers
        # with **Key:** value metadata instead of the --STEP format.
        heuristic_steps = parse_heuristic_plan(plan)
        if heuristic_steps:
            _logger.info(
                "[Plan] Heuristic parser extracted %d steps from non-standard format",
                len(heuristic_steps),
            )
            dep_fixes = fix_import_dependencies(heuristic_steps, read_file=project_file_reader)
            if dep_fixes:
                _logger.info("[Plan] Auto-fixed import dependencies: %s", dep_fixes)
            plan_steps = heuristic_steps
            steps = steps_as_text_list(plan_steps)
            dependencies = steps_dependencies_dict(plan_steps)

    if plan_steps is None:
        _logger.info("[Plan] Using legacy step parser (no structured plan)")
        # Legacy numbered-list parsing
        raw_steps = executor.parse_plan_steps(plan)
        if not raw_steps:
            return TaskResult(success=False, error="Could not parse any steps from the plan.")
        steps, dependencies = executor.parse_step_dependencies(raw_steps)

    # ── Truncation guard ──
    # A plan can parse cleanly yet be a stub (planner ran out of output
    # tokens mid-plan). The library path is single-shot, so surface it
    # loudly rather than shipping the stub silently. (The CLI re-plans.)
    from .orchestrator.plan_step import plan_looks_truncated as _plan_truncated
    _prov_trunc = getattr(getattr(planner, "llm_client", None),
                          "_last_truncated", False)
    _struct_trunc, _trunc_why = _plan_truncated(plan, plan_steps)
    if _struct_trunc or (_prov_trunc and "==END==" not in (plan or "")):
        _logger.warning(
            "[Plan] Plan looks truncated (%s) — proceeding, but it may be "
            "incomplete; downstream recovery will have to backfill missing "
            "files", _trunc_why or "planner hit its output-token limit")

    # Post-plan optimization
    from .orchestrator.plan_optimizer import optimize_plan, optimize_structured_plan

    if plan_steps is not None:
        # Structured path: optimize directly on PlanStep objects,
        # preserving step_type, target_files, exports, imports_from, command
        plan_steps = optimize_structured_plan(
            plan_steps, knowledge_base=knowledge_base,
            kb_context_builder=kb_context_builder,
            language=language)
        # Rebuild legacy views for downstream display/compat
        steps = steps_as_text_list(plan_steps)
        dependencies = steps_dependencies_dict(plan_steps)
    else:
        # Legacy path: optimize on list[str]
        steps, dependencies = optimize_plan(
            steps, knowledge_base=knowledge_base,
            kb_context_builder=kb_context_builder,
            dependencies=dependencies,
            language=language)
        plan_steps = from_legacy_steps(steps, dependencies)

    # Reclassify CODE steps targeting only protected dependency manifests
    # (package.json, requirements.txt, etc.) as CMD install steps so the
    # package manager does the safe, atomic update instead of the LLM write
    # being silently blocked by the protected-file guard.
    plan_steps = reclassify_manifest_steps(plan_steps)
    steps = steps_as_text_list(plan_steps)
    dependencies = steps_dependencies_dict(plan_steps)

    # ── Project analysis phase ──────────────────────────────────
    # Build structured ProjectContext from static analysis (zero LLM cost).
    # Optionally enrich with LLM for deeper success criteria and test hints.
    project_context_obj = build_project_context(
        task, steps,
        source_files=source_files or {},
        language=language,
        project_profile=project_profile,
    )

    # Inject packages from the task briefing's "New packages:" line so that
    # _ensure_packages_installed runs the actual install before the first CODE
    # step — even when the plan has no explicit CMD install step.
    for _pkg in parse_briefing_packages(_briefing_text):
        if _pkg not in project_context_obj.required_packages:
            project_context_obj.required_packages.append(_pkg)
            _logger.info("[PreAnalysis] Briefing package injected: %s", _pkg)

    # LLM enrichment — adds deeper success criteria, assertion hints,
    # and testable unit identification.  Uses 1 LLM call.
    # Disabled by default; enable via `analyser: {enabled: true}` in .agentchanti.yaml
    # or ANALYSER_ENABLED=true env var.
    if cfg.ANALYSER_ENABLED:
        display.show_status("Analysing project...")
        try:
            analyser = AnalyseAgent(
                "Analyser", "Senior Technical Analyst",
                "Analyse the task and project to guide downstream agents.",
                _make_llm("analyser"))
            project_context_obj = analyser.enrich_context(
                project_context_obj, task, steps, source_files or {})
            _logger.info(
                "[Analysis] ProjectContext built: lang=%s, framework=%s, "
                "test_fw=%s, %d installed pkgs, %d testable units, %d criteria",
                project_context_obj.language,
                project_context_obj.framework,
                project_context_obj.test_framework,
                len(project_context_obj.installed_packages),
                len(project_context_obj.testable_units),
                len(project_context_obj.success_criteria),
            )
        except Exception as analyse_exc:
            _logger.warning("[Analysis] LLM enrichment failed (non-fatal): %s",
                            analyse_exc)
    else:
        _logger.info("[Analysis] LLM enrichment skipped (analyser_enabled: false)")

    display.set_steps(steps)

    # Execute — use phase-aware wave builder when structured plan steps exist
    if plan_steps:
        plan_waves = _build_plan_waves(plan_steps)
        waves = [[s.index for s in w] for w in plan_waves]
    else:
        waves = build_step_waves(steps, dependencies)
    checkpoint_file = cfg.CHECKPOINT_FILE
    step_results: dict[int, str] = {}
    pipeline_success = True

    # Test files that predate the run — the only ones whose passing is
    # not the agent grading its own homework (orchestrator/evidence.py).
    from .orchestrator.evidence import snapshot_test_files as _snap_tests
    _pre_existing_tests = _snap_tests(os.getcwd())

    # Clear any lingering planning/analysis status message before execution
    # starts. Without this, "Analysing project...", "Requesting steps from
    # planner...", etc. stay pinned to the STATUS panel for the entire run
    # because nothing inside _execute_step touches show_status. The wiring
    # verification phase sets/clears its own status independently.
    display.show_status("")

    for wave_idx, wave in enumerate(waves):
        for idx in wave:
            step_text = steps[idx]
            # Find matching PlanStep for this index
            _ps = next((s for s in plan_steps if s.index == idx), None) if plan_steps else None
            if _ps is None and plan_steps:
                log.warning(
                    "[PlanStep] No PlanStep found for idx=%d. "
                    "Available indices: %s",
                    idx, [s.index for s in plan_steps],
                )

            idx, success, error_info = _execute_step(
                idx, step_text,
                steps=steps,
                llm_client=llm_client, executor=executor,
                coder=coder, reviewer=reviewer, tester=tester,
                task=task, memory=memory, display=display,
                language=language, auto=auto,
                kb_context_builder=kb_context_builder,
                project_profile=project_profile,
                knowledge_base=knowledge_base,
                project_context=project_context_obj,
                plan_step=_ps,
                all_plan_steps=plan_steps,
                intent_spec=intent_spec,
            )

            if success:
                step_results[idx] = "done"
                ds = {"elapsed": time.monotonic() - display.start_time, "steps": display.steps}
                save_checkpoint(checkpoint_file, task, steps, idx,
                                memory.as_dict(), step_results, language,
                                display_state=ds,
                                plan_steps=plan_steps,
                                project_context=project_context_obj)
            else:
                fixed = _run_diagnosis_loop(
                    idx, step_text, error_info,
                    steps=steps,
                    llm_client=llm_client, executor=executor,
                    coder=coder, reviewer=reviewer, tester=tester,
                    task=task, memory=memory, display=display,
                    language=language,
                    search_agent=search_agent,
                    kb_context_builder=kb_context_builder,
                    project_profile=project_profile,
                    knowledge_base=knowledge_base,
                    project_context=project_context_obj,
                    plan_step=_ps,
                    all_plan_steps=plan_steps,
                    intent_spec=intent_spec,
                )
                if fixed:
                    display.complete_step(idx, "done")
                    step_results[idx] = "done"
                    ds = {"elapsed": time.monotonic() - display.start_time, "steps": display.steps}
                    save_checkpoint(checkpoint_file, task, steps, idx,
                                    memory.as_dict(), step_results, language,
                                    display_state=ds,
                                    plan_steps=plan_steps,
                                project_context=project_context_obj)
                else:
                    pipeline_success = False
                    break

        if not pipeline_success:
            break

    # ── Bulk test execution + per-file fix ──
    # All inline TEST steps deferred their runs — execute once here.
    verif_ok = False
    if pipeline_success:
        from .orchestrator.pipeline import run_bulk_test_execution_and_fix
        verif_ok, verif_err = run_bulk_test_execution_and_fix(
            memory=memory,
            executor=executor,
            coder=coder,
            display=display,
            language=language,
            task=task,
            cfg=cfg,
            project_context=project_context_obj,
            kb_context_builder=kb_context_builder,
            all_plan_steps=plan_steps,
        )
        if not verif_ok:
            pipeline_success = False
            log.warning(f"[BulkTest] Pipeline marked failed: {verif_err[:200]}")

    # ── Wiring verification ──
    # One LLM call that checks all fix-scope files together for cross-file
    # integration issues (entry-point mounts, import/export mismatches, etc.).
    # Skipped when the bulk test run just executed real tests and they all
    # passed — see should_run_wiring_verification() for the full rationale.
    from .orchestrator.pipeline import should_run_wiring_verification
    _run_wiring = should_run_wiring_verification(
        memory,
        pipeline_success=pipeline_success,
        bulk_test_verif_ok=verif_ok,
        wiring_enabled=cfg.WIRING_VERIFICATION_ENABLED,
    )
    if not _run_wiring and pipeline_success and cfg.WIRING_VERIFICATION_ENABLED:
        log.info(
            "[WiringVerification] Skipped — bulk tests just ran and "
            "passed; wiring is implicitly verified."
        )
    if _run_wiring:
        wv_ok, wv_err = run_wiring_verification(
            memory=memory,
            executor=executor,
            coder=coder,
            display=display,
            task=task,
            language=language,
            cfg=cfg,
            kb_context_builder=kb_context_builder,
            project_root=os.getcwd(),
        )
        if not wv_ok:
            log.warning(f"[WiringVerification] Fix failed: {wv_err[:200]}")

    # ── Runtime smoke verification ──
    # Tests can pass while the app crashes at launch (GUI apps especially).
    # Launch the entry point briefly and feed crashes into a bounded fix loop.
    if pipeline_success:
        from .orchestrator.smoke_test import run_smoke_verification
        smoke_ok, smoke_err = run_smoke_verification(
            memory=memory,
            executor=executor,
            coder=coder,
            display=display,
            task=task,
            language=language,
            cfg=cfg,
        )
        if not smoke_ok:
            pipeline_success = False
            log.warning(f"[SmokeTest] Pipeline marked failed: {smoke_err[:300]}")

    # Stop KB runtime watcher
    if kb_runtime_watcher is not None:
        try:
            kb_runtime_watcher.stop()
        except Exception:
            pass

    # Extract knowledge (runs on both success and failure) —
    # patterns/fixes from completed steps are valuable regardless,
    # especially fixes learned from failures.
    if knowledge_base is not None:
        try:
            knowledge_base.extract_from_run(
                task, steps, memory.as_dict(), llm_client)
        except Exception:
            pass

    if pipeline_success:
        clear_checkpoint(checkpoint_file)

    # Collect written files (exclude internal cmd/fix outputs)
    files_written = [f for f in memory.all_files().keys()
                     if not f.startswith("_")]

    from .orchestrator.pipeline import _is_test_file
    from .orchestrator.evidence import classify as _classify_evidence
    from .orchestrator.evidence import run_acceptance_commands as _run_acc
    _acceptance_cmds = list(getattr(cfg, "ACCEPTANCE_CMDS", []) or [])
    _acceptance_passed = None
    _acceptance_repaired = 0
    if pipeline_success and _acceptance_cmds:
        _acceptance_passed, _acc_failures = _run_acc(executor, _acceptance_cmds)
        if _acceptance_passed is False:
            # Same bounded repair the CLI path gets; the two must not
            # diverge on what a failing acceptance check means.
            from .orchestrator.evidence import (
                repair_failed_acceptance as _repair_acc)
            _acceptance_passed, _acceptance_repaired, _acc_failures = (
                _repair_acc(
                    executor=executor, cmds=_acceptance_cmds,
                    failures=_acc_failures,
                    llm_client=getattr(coder, "llm_client", None),
                    memory=memory, task=task, language=language,
                    max_rounds=getattr(cfg, "ACCEPTANCE_REPAIR_ROUNDS", 3),
                    max_turns=getattr(cfg, "AGENT_LOOP_MAX_TURNS", 8),
                    escalation_client=getattr(coder, "escalation_client",
                                              None),
                    project_root=os.getcwd()))
        if _acceptance_passed is False:
            pipeline_success = False
            log.error("Pipeline failed: user acceptance command(s) did not "
                      "pass — " + "; ".join(_acc_failures[:3]))
    _evidence = _classify_evidence(
        os.getcwd(), _pre_existing_tests,
        tests_ran=any(_is_test_file(f) for f in files_written),
        acceptance_passed=_acceptance_passed,
        acceptance_cmds=_acceptance_cmds)
    if _acceptance_repaired:
        _evidence = _dc_replace(_evidence, repaired=_acceptance_repaired)
    log.info(_evidence.log_line())
    if (pipeline_success and not _evidence.independent
            and getattr(cfg, "REQUIRE_INDEPENDENT_EVIDENCE", False)):
        log.error("Pipeline failed: require_independent_evidence is set and "
                  "nothing outside this run's own output verified it")
        pipeline_success = False

    return TaskResult(
        success=pipeline_success,
        verified=_evidence.independent,
        evidence=_evidence.log_line(),
        files_written=files_written,
        plan_steps=steps,
        token_usage={
            "total": token_tracker.total_tokens,
            "prompt": token_tracker.total_prompt_tokens,
            "completion": token_tracker.total_completion_tokens,
            "cached": token_tracker.total_cached_tokens,
        },
        log_file=log.handlers[0].baseFilename if log.handlers else "",
    )
