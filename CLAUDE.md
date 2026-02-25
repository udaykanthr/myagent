# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentChanti is a multi-agent AI coding CLI tool (`agentchanti` command) and Python library (`multi_agent_coder` package). It takes a plain English task description and autonomously plans, codes, reviews, and tests the solution using a pipeline of specialized LLM-powered agents. Supports local LLMs (Ollama, LM Studio) and cloud providers (OpenAI, Gemini, Anthropic).

## Common Commands

```bash
# Install in editable mode
pip install -e .

# Run tests
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_flow.py -v

# Run the CLI
agentchanti "your task" --provider ollama --model deepseek-coder-v2:16b

# Run via library API
python -c "from multi_agent_coder import run_task; run_task(task='...', auto=True)"
```

## Architecture

### Agent Pipeline

The system runs a sequential pipeline: **Planner -> Coder -> Reviewer -> Tester**. Each agent (`multi_agent_coder/agents/`) extends `Agent` base class and calls `self.llm_client.generate_response(prompt)`. The pipeline is orchestrated in two places:
- **CLI path**: `orchestrator/cli.py:main()` — parses args, builds agents, runs the pipeline
- **Library path**: `api.py:run_task()` — programmatic entry point returning `TaskResult`

Both paths share the same execution engine in `orchestrator/pipeline.py`.

### Step Execution Flow

1. **PlannerAgent** generates numbered steps from the task description
2. `pipeline.py:build_step_waves()` groups steps into dependency waves for parallel execution
3. Each step is classified by `classification.py:_classify_step()` via LLM into: **CMD**, **CODE**, **TEST**, or **IGNORE**
4. Step handlers in `orchestrator/step_handlers.py` execute each type:
   - **CMD**: Runs shell commands via `Executor.run_command()`
   - **CODE**: Coder generates code -> Reviewer checks -> retry loop (up to 3x) -> diagnosis on failure
   - **TEST**: TesterAgent generates tests -> runs them -> Coder fixes failures
5. `orchestrator/diagnosis.py` handles failure analysis and auto-fix

### Language Detection (multi_agent_coder/language.py)

Auto-detects project language by scanning file extensions (`detect_language()`) or parsing task keywords (`detect_language_from_task()`). Maps languages to test frameworks via `TEST_FRAMEWORKS` dict. **Known issue**: defaults to Python/pytest when language is `None`, which causes incorrect test generation for non-Python projects (e.g., TypeScript projects get Python tests). The TesterAgent at lines 10-12 and 41-44 hard-defaults to Python when `language` is None.

### LLM Client Layer (multi_agent_coder/llm/)

`LLMClient` base class with provider implementations: `OllamaClient`, `LMStudioClient`, `OpenAIClient`, `GeminiClient`, `AnthropicClient`. All expose `generate_response(prompt) -> str` with retry and streaming support.

### Key Subsystems

- **Config** (`config.py`): Priority resolution: CLI args > env vars > `.agentchanti.yaml` > defaults
- **Executor** (`executor.py`): File I/O, shell command execution, plan/code-block parsing
- **FileMemory** (`orchestrator/memory.py`): Thread-safe tracking of files written during a run
- **Knowledge Base** (`kb/`): Local code graph (tree-sitter based), project orientation, context injection for agents. Subcommand: `agentchanti kb ...`
- **Editing** (`editing/`): Diff-aware code editing with fuzzy matching and syntax validation
- **Plugins** (`plugins/`): Custom step handlers (LINT, DEPLOY, etc.) via `StepPlugin` base class, discovered from config or setuptools entry points
- **Step Cache** (`step_cache.py`): Hash-based LLM response caching with configurable TTL
- **Checkpoint** (`checkpoint.py`): Save/restore pipeline state for resume after interruption
- **Git Utils** (`git_utils.py`): Checkpoint branches, commit on success, rollback on failure

### Test Framework Mapping

Defined in `language.py:TEST_FRAMEWORKS`. The TesterAgent (`agents/tester.py`) builds language-specific prompts:
- `_python_test_rules()` for Python/pytest
- `_js_test_rules()` for JavaScript/TypeScript (Jest-oriented, no Vitest support yet)

The step handler `_handle_test_step()` in `step_handlers.py` detects JS project environment (ESM vs CJS) and auto-installs test runners.

## Configuration

Settings file: `.agentchanti.yaml` (project root or home directory). Key sections: `models` (per-agent model overrides), `prompts` (agent prompt suffixes), `openai`/`gemini`/`anthropic` (cloud API keys), `kb` (knowledge base), `editing` (diff-aware editing), `plugins`.

## Entry Points

- CLI: `agentchanti` -> `multi_agent_coder.orchestrator.cli:main` (defined in `setup.py`)
- Library: `from multi_agent_coder import run_task, TaskResult`
- KB subcommand: `agentchanti kb ...` -> `multi_agent_coder.kb.cli:kb_main`
