# AgentChanti Documentation

Complete reference for the AgentChanti multi-agent AI coding system.

## Table of Contents

1. [Architecture](#architecture)
   - [Agent Pipeline](#agent-pipeline)
   - [Execution Pipeline](#execution-pipeline)
   - [Step Classification & Handling](#step-classification--handling)
   - [Failure Diagnosis](#failure-diagnosis)
2. [CLI Reference](#cli-reference)
   - [All Options](#all-options)
   - [Examples](#examples)
3. [Configuration](#configuration)
   - [Priority Order](#priority-order)
   - [YAML Config File](#yaml-config-file)
   - [Environment Variables](#environment-variables)
4. [LLM Provider Setup](#llm-provider-setup)
   - [Ollama](#ollama)
   - [LM Studio](#lm-studio)
   - [OpenAI & Cloud Providers](#openai--cloud-providers)
   - [Google Gemini](#google-gemini)
   - [Anthropic Claude](#anthropic-claude)
   - [Embedding Models](#embedding-models)
5. [Key Features](#key-features)
   - [TUI Plan Editor](#tui-plan-editor)
   - [Diff-Aware Editing](#diff-aware-editing)
   - [Knowledge Base (KB)](#knowledge-base-kb)
   - [KB CLI Commands](#kb-cli-commands)
   - [Web Search Agent](#web-search-agent)
   - [Step Caching](#step-caching)
   - [Persistent Embeddings](#persistent-embeddings)
   - [Cost Tracking & Budgets](#cost-tracking--budgets)
   - [Checkpoint & Resume](#checkpoint--resume)
   - [Git Integration](#git-integration)
   - [HTML Reports](#html-reports)
   - [Protected Files](#protected-files)
6. [Library API](#library-api)
7. [Plugin System](#plugin-system)
8. [Supported Languages](#supported-languages)
9. [Troubleshooting](#troubleshooting)

---

## Architecture

### Agent Pipeline

AgentChanti runs a sequential pipeline of four specialized agents. Each extends the `Agent` base class and calls `self.llm_client.generate_response(prompt)`.

| Agent | Role | What It Does |
|-------|------|--------------|
| **Planner** | Software Architect | Breaks the task into numbered, actionable steps with dependency info |
| **Coder** | Senior Developer | Writes clean, idiomatic code for each step |
| **Reviewer** | Code Reviewer | Checks code for bugs, style issues, and correctness |
| **Tester** | QA Engineer | Generates and runs unit tests to verify the code works |
| **Search** | Research Assistant | Performs web searches for latest docs, error fixes, and best practices |

The pipeline is orchestrated from two entry points:
- **CLI path**: `orchestrator/cli.py:main()` — parses args, builds agents, runs the pipeline
- **Library path**: `api.py:run_task()` — programmatic entry point returning `TaskResult`

Both share the same execution engine in `orchestrator/pipeline.py`.

### Execution Pipeline

```
                    +-----------+
    Your Task ----->|  Planner  |-----> Step-by-step plan
                    +-----------+
                          |
              [Plan Approval / TUI Editor]
                          |
              +-----------+-----------+
              |           |           |
         +--------+  +--------+  +--------+
         | Step 1 |  | Step 2 |  | Step 3 |   (parallel if independent)
         +--------+  +--------+  +--------+
              |
     +--------+--------+
     |                  |
 +-------+        +----------+
 | Coder |------->| Reviewer |-----> Pass? Move on
 +-------+   ^    +----------+       Fail? Retry with feedback
              |
              +--- up to 3 retries
              |
     (still failing?)
              |
     +------------------+
     | Failure Diagnosis |-----> Diagnose root cause
     +------------------+        Apply fix, retry
              |
     +------------------+
     |     Tester       |-----> Generate tests
     +------------------+        Run tests
              |                  Fix code if tests fail
         [Checkpoint]            (saved after each step)
              |
     +------------------+
     |   HTML Report    |-----> Self-contained report
     +------------------+
```

**Detailed flow:**

1. **Scan** — Reads your project structure and key files
2. **Knowledge** — Loads learnings from previous runs and KB context
3. **Plan** — Planner agent creates numbered steps with dependency info
4. **Approve** — You review the plan (approve / TUI edit / text edit / replan)
5. **Wave Building** — Steps are grouped into dependency waves for parallel execution (via `build_step_waves()`)
6. **Execute** — Each step is classified and executed by type
7. **Diagnose** — If a step fails after retries, the LLM analyzes why and applies a fix
8. **Checkpoint** — Progress is saved after each successful step
9. **Learn** — Extracts patterns and conventions from the run
10. **Report** — Generates an HTML report with stats, diffs, and step breakdown
11. **Git** — On completion, offers to commit; on failure, offers to rollback

### Step Classification & Handling

Each step is classified via LLM into one of these types (`classification.py:_classify_step()`):

| Type | Handler | What Happens |
|------|---------|--------------|
| **CMD** | `_handle_cmd_step()` | Extracts and runs a shell command via `Executor.run_command()` |
| **CODE** | `_handle_code_step()` | Coder writes code -> Reviewer checks -> retry loop (up to 3x) -> diagnosis on failure |
| **TEST** | `_handle_test_step()` | Tester generates tests -> runs them -> Coder fixes failures |
| **IGNORE** | skipped | Non-actionable steps (meta-comments, summaries) |
| **PLUGIN** | custom handler | Matched to registered plugin via `can_handle()` |

### Failure Diagnosis

**External service failures** (diagnosis skipped — not fixable by code):
- MongoDB (27017), PostgreSQL (5432), MySQL (3306), Redis (6379), RabbitMQ (5672), Elasticsearch (9200)
- Generic connection refused / timeout patterns

**System-level issues** (diagnosis skipped — require manual setup):
- Missing runtimes: Python, Node.js, Java, Ruby, .NET, Docker
- Missing package managers: pip, npm, maven, gradle

**Code bugs** (diagnosis runs, up to 2 retries):
1. LLM analyzes error + code context
2. LLM proposes a fix
3. Fix is applied to file memory
4. Step is re-run
5. If still failing: next retry or halt

---

## CLI Reference

### All Options

```bash
agentchanti "<task description>" [options]
```

**Core:**

| Flag | Description | Default |
|------|-------------|---------|
| `"task"` | The coding task to perform (positional, required) | — |
| `--prompt-from-file FILE` | Read task description from a text file | — |

**Provider & Model:**

| Flag | Description | Default |
|------|-------------|---------|
| `--provider` | LLM provider: `ollama`, `lm_studio`, `openai`, `gemini`, `anthropic` | `lm_studio` |
| `--model` | Model name to use | `deepseek-coder-v2-lite-instruct` |
| `--embed-model` | Embedding model name | `nomic-embed-text` |
| `--language` | Override auto-detected language (e.g. `python`, `javascript`) | auto-detect |

**Feature Toggles:**

| Flag | Description | Default |
|------|-------------|---------|
| `--no-embeddings` | Disable semantic embeddings | off |
| `--no-stream` | Disable streaming responses | off |
| `--no-git` | Disable git checkpoint/rollback | off |
| `--no-diff` | Disable diff preview before writing files | off |
| `--no-cache` | Disable step-level caching | off |
| `--clear-cache` | Clear step cache before running | off |
| `--no-knowledge` | Disable project knowledge base | off |
| `--no-search` | Disable web search agent for planning and diagnosis | off |
| `--no-kb` | Disable KB context injection (debugging) | off |
| `--report` / `--no-report` | Enable/disable HTML report generation | on |

**Execution Mode:**

| Flag | Description | Default |
|------|-------------|---------|
| `--auto` | Non-interactive mode: auto-approve plan, skip all prompts | off |
| `--resume` | Force resume from last checkpoint | off |
| `--fresh` | Ignore any existing checkpoint and start fresh | off |

**Configuration:**

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to `.agentchanti.yaml` config file | auto-discover |
| `--generate-yaml` / `--generate-config` | Generate a `.agentchanti.yaml` file with current settings and exit | off |

### Examples

```bash
# Simple Python script
agentchanti "Create a Python script that reads a CSV and generates a bar chart"

# Web application with Ollama
agentchanti "Build a Flask REST API with CRUD" --provider ollama --model deepseek-coder-v2:16b

# Using OpenAI
OPENAI_API_KEY="sk-..." agentchanti "Build a CLI tool" --provider openai --model gpt-4o-mini

# Using Google Gemini
GEMINI_API_KEY="..." agentchanti "Build a REST API" --provider gemini --model gemini-2.5-flash

# Using Anthropic Claude
ANTHROPIC_API_KEY="sk-ant-..." agentchanti "Build a CLI tool" --provider anthropic --model claude-sonnet-4

# JavaScript / Node.js project
agentchanti "Create an Express.js REST API with JWT auth" --language javascript

# Working on existing project (auto-scans directory)
cd my-project/ && agentchanti "Add input validation to all API endpoints"

# Non-interactive mode (CI/scripts)
agentchanti "Generate unit tests for all modules" --auto --no-git --no-report

# Read task from file
agentchanti --prompt-from-file task.txt --provider ollama --model codellama:13b

# Resume a failed run
agentchanti "same task" --resume

# Start fresh, clearing cache
agentchanti "Refactor the database layer" --fresh --clear-cache

# Generate a configuration file
agentchanti --generate-yaml
agentchanti --provider openai --model gpt-4o-mini --generate-yaml
```

---

## Configuration

### Priority Order

Settings are resolved in this priority (highest first):
1. **CLI arguments**
2. **Environment variables**
3. **`.agentchanti.yaml`** (project root, then home directory)
4. **Built-in defaults**

### YAML Config File

Create `.agentchanti.yaml` in your project root or home directory. Use `agentchanti --generate-yaml` to generate one with defaults.

```yaml
# ── General ─────────────────────────────────────────────
provider: lm_studio              # ollama | lm_studio | openai | gemini | anthropic
model: deepseek-coder-v2-lite-instruct
context_window: 8192
stream: true
budget_limit: 5.0                # halt if cost exceeds $5.00 (0 = unlimited)
planner_context_chars: 6000

# ── Service URLs ────────────────────────────────────────
ollama_base_url: "http://localhost:11434/api/generate"
lm_studio_base_url: "http://localhost:1234/v1"

# ── Cloud Provider API Keys ────────────────────────────
openai:
  api_key: "sk-your-key-here"
  base_url: "https://api.openai.com/v1"

gemini:
  api_key: "your-gemini-api-key"
  base_url: "https://generativelanguage.googleapis.com/v1beta"

anthropic:
  api_key: "sk-ant-your-key-here"
  base_url: "https://api.anthropic.com/v1"

# ── Embeddings ──────────────────────────────────────────
embedding_model: nomic-embed-text
embedding_top_k: 5
embedding_cache_dir: .agentchanti    # SQLite cache location

# ── Per-Agent Model Overrides ───────────────────────────
models:
  planner: deepseek-coder-v2:16b     # smarter model for planning
  coder: qwen2.5-coder:7b            # fast model for coding
  reviewer: deepseek-coder-v2:16b    # smarter model for reviews
  tester: qwen2.5-coder:7b           # fast model for tests

# ── Custom Prompt Suffixes ──────────────────────────────
prompts:
  planner_suffix: "Focus on small, incremental steps."
  coder_suffix: "Always use type hints. Follow PEP 8."
  reviewer_suffix: "Check for SQL injection and XSS."
  tester_suffix: "Use pytest fixtures. Aim for 90% coverage."

# ── Web Search ──────────────────────────────────────────
search_enabled: true
search_provider: duckduckgo          # duckduckgo | google | bing
search_api_key: ""
search_api_url: ""
search_max_results: 3
search_max_page_chars: 3000

# ── Knowledge Base (KB) ────────────────────────────────
kb:
  enabled: true
  max_context_tokens: 4000
  auto_index_on_start: true
  watcher_debounce_seconds: 1.0
  verbose_logging: false

kb_registry_owner: udaykanthr
kb_registry_repo: agentchanti-kb-registry
kb_registry_auto_update: true

# ── Diff-Aware Editing ─────────────────────────────────
editing:
  diff_mode: true
  min_confidence_threshold: 0.60
  context_lines: 5
  fuzzy_match_window: 3
  validate_syntax_after_patch: true
  track_metrics: true
  fallback_on_syntax_error: true
  chunk_mode: true
  slim_context: true
  reviewer_diff_mode: true
  max_chunk_files: 3

# ── Step Caching ────────────────────────────────────────
step_cache_ttl_hours: 24

# ── HTML Reports ────────────────────────────────────────
report_dir: .agentchanti/reports

# ── Checkpoint ──────────────────────────────────────────
checkpoint_file: .agentchanti_checkpoint.json

# ── LLM Behavior ───────────────────────────────────────
llm_max_retries: 3
llm_retry_delay: 2.0

# ── Custom Pricing (per 1M tokens) ─────────────────────
pricing:
  gpt-4o: {input: 2.50, output: 10.00}
  gpt-4o-mini: {input: 0.15, output: 0.60}
  gemini-2.5-flash: {input: 0.15, output: 0.60}
  gemini-2.5-pro: {input: 1.25, output: 10.00}
  claude-sonnet-4: {input: 3.00, output: 15.00}
  claude-haiku-4: {input: 0.80, output: 4.00}

# ── Plugins ─────────────────────────────────────────────
plugins:
  - my_package.plugins.LintPlugin
  - my_package.plugins.DeployPlugin
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| **Core** | | |
| `PROVIDER` | `lm_studio` | LLM provider override |
| `DEFAULT_MODEL` | `deepseek-coder-v2-lite-instruct` | Default model name |
| `CONTEXT_WINDOW` | `8192` | Model context window size (tokens) |
| `STREAM_RESPONSES` | `true` | Enable/disable streaming |
| `BUDGET_LIMIT` | `0.0` | Cost limit in USD (0 = unlimited) |
| `PLANNER_CONTEXT_CHARS` | `6000` | Max chars for planner context |
| **Service URLs** | | |
| `OLLAMA_BASE_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `OPENAI_API_KEY` | — | API key for OpenAI-compatible providers |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API base URL |
| `GEMINI_API_KEY` | — | API key for Google Gemini |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta` | Gemini API base URL |
| `ANTHROPIC_API_KEY` | — | API key for Anthropic Claude |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com/v1` | Anthropic API base URL |
| **Embeddings** | | |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `EMBEDDING_TOP_K` | `5` | Number of files to retrieve for context |
| `EMBEDDING_CACHE_DIR` | `.agentchanti` | Directory for persistent embedding cache |
| **Search** | | |
| `SEARCH_ENABLED` | `true` | Enable web search agent |
| `SEARCH_PROVIDER` | `duckduckgo` | Search provider (duckduckgo, google, bing) |
| `SEARCH_API_KEY` | — | API key for search provider |
| `SEARCH_API_URL` | — | Custom search API URL |
| `SEARCH_MAX_RESULTS` | `3` | Max search results per query |
| `SEARCH_MAX_PAGE_CHARS` | `3000` | Max chars fetched per page |
| **Knowledge Base** | | |
| `KB_ENABLED` | `true` | Enable KB context injection |
| `KB_MAX_CONTEXT_TOKENS` | `4000` | Max tokens for KB context |
| `KB_AUTO_INDEX_ON_START` | `true` | Auto-index on startup |
| `KB_WATCHER_DEBOUNCE_SECONDS` | `1.0` | File watcher debounce delay |
| `KB_VERBOSE_LOGGING` | `false` | Enable verbose KB logging |
| `KB_REGISTRY_OWNER` | `udaykanthr` | GitHub registry owner |
| `KB_REGISTRY_REPO` | `agentchanti-kb-registry` | GitHub registry repo |
| `KB_REGISTRY_AUTO_UPDATE` | `true` | Auto-update global KB from registry |
| **Editing** | | |
| `EDITING_DIFF_MODE` | `true` | Enable diff-aware editing |
| `EDITING_MIN_CONFIDENCE` | `0.60` | Min confidence for fuzzy matching |
| `EDITING_CONTEXT_LINES` | `5` | Context lines around edits |
| `EDITING_FUZZY_MATCH_WINDOW` | `3` | Fuzzy match search window |
| `EDITING_VALIDATE_SYNTAX` | `true` | Validate syntax after patching |
| `EDITING_TRACK_METRICS` | `true` | Track edit success metrics |
| `EDITING_FALLBACK_ON_SYNTAX_ERROR` | `true` | Fallback to full rewrite on syntax error |
| `EDITING_CHUNK_MODE` | `true` | Enable chunk-based editing for large files |
| `EDITING_SLIM_CONTEXT` | `true` | Minimize context sent to LLM |
| `EDITING_REVIEWER_DIFF_MODE` | `true` | Send diffs to reviewer instead of full files |
| `EDITING_MAX_CHUNK_FILES` | `3` | Max files per chunk edit |
| **Other** | | |
| `REPORT_DIR` | `.agentchanti/reports` | HTML report output directory |
| `STEP_CACHE_TTL_HOURS` | `24` | Step cache TTL in hours |
| `LLM_MAX_RETRIES` | `3` | Max retries on LLM failure |
| `LLM_RETRY_DELAY` | `2.0` | Base delay (seconds) between retries |
| `CHECKPOINT_FILE` | `.agentchanti_checkpoint.json` | Checkpoint file path |

---

## LLM Provider Setup

### Prerequisites

- **Python 3.10+** ([python.org](https://www.python.org/downloads/))
- **Git** ([git-scm.com](https://git-scm.com/))

- A local LLM server **or** a cloud API key (see providers below)

AgentChanti supports five LLM providers. Choose one or configure multiple.

### Ollama

Easiest way to run LLMs locally.

1. Install from [ollama.com](https://ollama.com)
2. Pull a model:
   ```bash
   ollama pull deepseek-coder-v2:16b   # 16GB+ RAM/VRAM
   ollama pull qwen2.5-coder:7b        # 8GB RAM/VRAM
   ollama pull nomic-embed-text         # embedding model (optional)
   ```
3. Start server: `ollama serve`
4. Run:
   ```bash
   agentchanti "task" --provider ollama --model deepseek-coder-v2:16b
   ```

### LM Studio

GUI-based local LLM server with OpenAI-compatible API.

1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Load a model and click **Start Server** (Local Server tab)
3. Run (LM Studio is the default provider):
   ```bash
   agentchanti "task"
   ```

### OpenAI & Cloud Providers

Works with any OpenAI-compatible API (OpenAI, Groq, Together.ai, etc.).

```bash
export OPENAI_API_KEY="sk-your-key-here"
agentchanti "task" --provider openai --model gpt-4o-mini
```

Or configure in `.agentchanti.yaml`:
```yaml
openai:
  api_key: "sk-your-key-here"
  base_url: "https://api.openai.com/v1"  # or Groq, Together.ai, etc.
```

**Example with Groq:**
```bash
export OPENAI_API_KEY="gsk_your-groq-key"
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
agentchanti "task" --provider openai --model llama-3.1-70b-versatile
```

### Google Gemini

Native Gemini REST API support.

```bash
export GEMINI_API_KEY="your-gemini-api-key"
agentchanti "task" --provider gemini --model gemini-2.5-flash
```

Or configure in `.agentchanti.yaml`:
```yaml
gemini:
  api_key: "your-gemini-api-key"
  base_url: "https://generativelanguage.googleapis.com/v1beta"
```

Get an API key from [Google AI Studio](https://aistudio.google.com/). Gemini also supports embeddings via `text-embedding-004`.

### Anthropic Claude

Native Anthropic Messages API support.

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
agentchanti "task" --provider anthropic --model claude-sonnet-4
```

Or configure in `.agentchanti.yaml`:
```yaml
anthropic:
  api_key: "sk-ant-your-key-here"
  base_url: "https://api.anthropic.com/v1"
```

Get an API key from the [Anthropic Console](https://console.anthropic.com/). Anthropic does not provide an embedding API — use `--no-embeddings` or configure a different provider for embeddings.

### Embedding Models

Embeddings enable smarter file retrieval for context. Cached in SQLite (`.agentchanti/embeddings.db`).

```bash
# Ollama:
ollama pull nomic-embed-text

# Disable:
agentchanti "task" --no-embeddings
```

---

## Key Features

### TUI Plan Editor

When the planner generates a plan, you can interactively edit it before execution:

| Key | Action |
|-----|--------|
| **Arrow keys** | Navigate steps |
| **e** | Edit a step's text |
| **d** | Delete a step |
| **a** | Add a new step |
| **Shift+K / Shift+J** | Reorder steps up/down |
| **Enter** | Approve and start execution |

Falls back to a text editor (notepad/vi) on unsupported terminals.

### Diff-Aware Editing

When enabled (`editing.diff_mode: true`), the Coder generates diffs instead of full file content. This reduces LLM context usage and improves accuracy on large files.

**Components:**
- **ScopeResolver** — Extracts code scopes (functions, classes) using tree-sitter
- **ContextSlicer** — Creates file slices with context lines around edits
- **DiffParser** — Parses LLM-generated diffs into structured patches
- **PatchApplier** — Applies patches with fuzzy matching and fallback
- **ChunkEditor** — Splits large files into chunks for per-chunk editing
- **Metrics** — Tracks edit success rates and confidence scores

Configure in `.agentchanti.yaml` under the `editing:` section.

### Knowledge Base (KB)

A four-phase system that provides context to agents:

**Phase 1: Code Graph** — Tree-sitter based AST parsing. Extracts classes, functions, imports, and definitions. Builds a symbol index for the project.

**Phase 2: Local Semantic KB** — Indexes code symbols using embeddings. Semantic search with file/directory filters for fast lookup during step execution.

**Phase 3: Global KB** — Error dictionary (maps error patterns to fixes), ADRs, language-specific patterns. Bundled with codebase + auto-updated from GitHub registry.

**Phase 4: Context Builder** — Single entry point for all KB injection. Called per-step to gather local symbols, related symbols, error fixes, global patterns, and behavioral instructions.

**Additional components:**
- **Project Orientation** — Detects project DNA (language, framework, package manager) and injects it as a grounding block in every LLM prompt
- **Runtime Watcher** — Monitors file changes during execution and triggers incremental indexing
- **Startup Manager** — Smart initialization with < 10ms target for common case

**Embedding:** The semantic layer (Phase 2) uses a local SQLite vector store for embeddings. No external services required.

### KB CLI Commands

The Knowledge Base is managed via the `agentchanti kb` subcommand.

```bash
agentchanti kb <command> [options]
```

#### Phase 1 — Local Code Graph

| Command | Description | Options |
|---------|-------------|---------|
| `kb index` | Full re-index of the current project | `--watch` — start file watcher after indexing |
| `kb status` | Show graph metadata summary | — |
| `kb query find-callers <name>` | Find all callers of a function/method | — |
| `kb query find-callees <name>` | Find all callees of a function/method | — |
| `kb query find-refs <name>` | Find all references to a symbol | — |
| `kb query impact <file_path>` | Show impact analysis for a file | — |
| `kb query symbol <name>` | Look up a symbol definition | — |

#### Phase 2 — Semantic Layer (requires Docker)

| Command | Description | Options |
|---------|-------------|---------|
| `kb embed` | Embed project symbols into the vector store | `--incremental` — only embed changed symbols |
| `kb search <query>` | Semantic search over the KB | `--top-k N` (default: 10), `--filter KEY=VALUE` |


#### Phase 3 — Global Knowledge Base

| Command | Description | Options |
|---------|-------------|---------|
| `kb seed` | Seed global KB with sample data (dev) | `--no-embed` — skip embedding step |
| `kb version` | Show current global KB version | — |
| `kb error-lookup <message>` | Look up error fixes in global KB | `--language` — filter by language |
| `kb global-search <query>` | Search the global knowledge base | `--category`, `--language`, `--top-k N` (default: 5) |
| `kb update` | Pull updates from GitHub registry | `--check` — check only, `--category` — specific category |

#### Health & Metrics

| Command | Description | Options |
|---------|-------------|---------|
| `kb health` | Show overall KB health report | `--json` — machine-readable output |
| `kb edit-stats` | Show DiffEdit performance statistics | `--last-n N` (default: 50) — recent edits to include |

#### Example Workflow

```bash
# First time: index the project and embed
agentchanti kb index
agentchanti kb embed

# Query the code graph
agentchanti kb query find-callers "run_task"
agentchanti kb query impact "multi_agent_coder/api.py"

# Semantic search
agentchanti kb search "how does step classification work" --top-k 5

# Look up an error
agentchanti kb error-lookup "ModuleNotFoundError: No module named 'flask'" --language python

# Update global KB from registry
agentchanti kb update

# Check overall health
agentchanti kb health
```

### Web Search Agent

A non-LLM agent that performs web searches during planning and diagnosis phases.

- Fetches latest documentation, error fixes, and best practices
- Uses DuckDuckGo by default (also supports Google, Bing)
- Returns formatted search results and page excerpts
- Disable with `--no-search`

Configure in `.agentchanti.yaml`:
```yaml
search_enabled: true
search_provider: duckduckgo
search_max_results: 3
search_max_page_chars: 3000
```

### Step Caching

Hashes step inputs (step text + language + file memory) and caches LLM responses. Re-running the same task skips completed steps instantly.

- Cache location: `.agentchanti/cache/`
- Default TTL: 24 hours (configurable via `step_cache_ttl_hours`)
- Clear cache: `agentchanti "task" --clear-cache`
- Disable: `agentchanti "task" --no-cache`

### Persistent Embeddings

Project file embeddings are cached in a SQLite database (`.agentchanti/embeddings.db`). Unchanged files are never re-embedded, saving ~30s per run on large projects.

### Cost Tracking & Budgets

Monitors token usage and calculates costs for cloud providers.

- **Budget enforcement**: Set `budget_limit` in config. The system halts if exceeded.
- **Pricing**: Configured via the `pricing` dictionary in YAML. Default prices for common models are built-in.
- **Reporting**: Total cost is displayed in terminal output and HTML reports.

### Checkpoint & Resume

Saves progress after each successful step. Resume exactly where you left off after interruption.

**Checkpoint file** (`.agentchanti_checkpoint.json`):
- Stores: task, steps, completed step index, file memory, step results, language

**Usage:**
```bash
# Auto-detects checkpoint and prompts to resume
agentchanti "same task"

# Force resume
agentchanti "same task" --resume

# Start fresh, ignoring checkpoint
agentchanti "same task" --fresh
```

### Git Integration

Creates a safety net before execution.

- **Checkpoint branch**: `agentchanti/pre-{slug}-{timestamp}` created before execution
- **On success**: Offers to commit changes with message `AgentChanti: {task}`
- **On failure**: Offers to rollback to the checkpoint branch
- Disable with `--no-git`

### HTML Reports

Auto-generated after every run in `.agentchanti/reports/`. Contains:
- Task summary, token usage, and cost
- Step-by-step execution status with icons
- Colored unified diffs for all modified files
- Dark theme, self-contained HTML

Disable with `--no-report`.

### Protected Files

AgentChanti protects certain files from being overwritten:

**Lock files (never touched):**
- `package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`
- `go.sum`, `Cargo.lock`, `Gemfile.lock`
- `composer.lock`, `Pipfile.lock`, `poetry.lock`

**Manifests (smart merge only — additive deps, no removals):**
- `package.json`, `requirements.txt`, `go.mod`, `Cargo.toml`, `Gemfile`

---

## Library API

Use AgentChanti programmatically for backend integration, CI pipelines, or custom tooling.

```python
from multi_agent_coder import run_task, TaskResult

result: TaskResult = run_task(
    task="Create a Python CLI calculator",
    provider="ollama",
    model="deepseek-coder-v2:16b",
    language="python",
    auto=True,           # non-interactive (default for library)
    no_git=True,         # default for library use
)

print(f"Success: {result.success}")
print(f"Files written: {result.files_written}")
print(f"Plan steps: {result.plan_steps}")
print(f"Token usage: {result.token_usage}")
print(f"Log file: {result.log_file}")

if not result.success:
    print(f"Error: {result.error}")
```

### `run_task()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | required | The coding task description |
| `provider` | `str` | `"lm_studio"` | LLM provider |
| `model` | `str \| None` | `None` | Model name (uses config default) |
| `embed_model` | `str \| None` | `None` | Embedding model name |
| `language` | `str \| None` | `None` | Override auto-detected language |
| `no_embeddings` | `bool` | `False` | Disable semantic embeddings |
| `no_git` | `bool` | `True` | Disable git integration |
| `no_kb` | `bool` | `False` | Disable KB context |
| `auto` | `bool` | `True` | Non-interactive mode |
| `config_path` | `str \| None` | `None` | Path to config file |
| `working_dir` | `str \| None` | `None` | Working directory for the task |

### `TaskResult` Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the pipeline completed successfully |
| `files_written` | `list[str]` | Paths of all files created or modified |
| `plan_steps` | `list[str]` | The executed plan steps |
| `token_usage` | `dict` | Token counts (`total`, `prompt`, `completion`, `cost`) |
| `log_file` | `str` | Path to the detailed log file |
| `error` | `str \| None` | Error message if pipeline failed |

---

## Plugin System

Extend AgentChanti with custom step type handlers.

### Creating a Plugin

```python
from multi_agent_coder.plugins import StepPlugin, PluginContext

class LintPlugin(StepPlugin):
    name = "LINT"

    def can_handle(self, step_text: str) -> bool:
        return "lint" in step_text.lower() or "format" in step_text.lower()

    def handle(self, step_text: str, ctx: PluginContext) -> tuple[bool, str]:
        success, output = ctx.executor.run_command("ruff check . --fix")
        return success, output if not success else ""
```

### `PluginContext` Fields

| Field | Type | Description |
|-------|------|-------------|
| `executor` | `Executor` | File I/O and command execution |
| `memory` | `FileMemory` | Thread-safe file tracking |
| `display` | `CLIDisplay` | Terminal UI access |
| `llm_client` | `Any` | LLM client for generating responses |
| `step_idx` | `int` | Current step index |
| `task` | `str` | Original task description |
| `language` | `str \| None` | Detected language |

### Registering Plugins

**Via config file:**
```yaml
plugins:
  - my_package.plugins.LintPlugin
  - my_package.plugins.DeployPlugin
```

**Via setuptools entry points:**
```toml
# pyproject.toml
[project.entry-points."agentchanti.plugins"]
lint = "my_package.plugins:LintPlugin"
```

---

## Supported Languages

AgentChanti auto-detects your project's language from file extensions and task keywords.

| Language | Test Framework | File Extensions | Task Keywords |
|----------|---------------|-----------------|---------------|
| Python | pytest | `.py` | flask, django, fastapi |
| JavaScript | Jest / Vitest | `.js`, `.jsx` | node, express, react |
| TypeScript | Jest / Vitest | `.ts`, `.tsx` | angular, nest |
| Go | go test | `.go` | golang, gin |
| Rust | cargo test | `.rs` | cargo, tokio |
| Java | mvn test | `.java` | spring, maven |
| Ruby | RSpec | `.rb` | rails, sinatra |
| C++ | — | `.cpp`, `.hpp` | — |
| C | — | `.c`, `.h` | — |
| C# | — | `.cs` | dotnet |
| Swift | — | `.swift` | — |
| Kotlin | — | `.kt` | — |
| PHP | — | `.php` | — |
| Scala | — | `.scala` | — |
| R | — | `.r`, `.R` | — |
| Lua | — | `.lua` | — |
| Bash | — | `.sh` | — |
| PowerShell | — | `.ps1` | — |

**Test runner auto-detection**: For JS/TS projects, AgentChanti checks for `vitest.config.*`, `jest.config.*`, and `package.json` devDependencies to select the correct runner.

Override with `--language`:
```bash
agentchanti "Build a REST API" --language go
```

---

## Troubleshooting

### "Connection refused" or "Connection error"
Your LLM server isn't running. Start Ollama (`ollama serve`) or LM Studio (Local Server tab → Start Server).

### Streaming errors / socket timeouts
```bash
agentchanti "task" --no-stream
```

### "Empty response" after retries
The model may be too small or the context overflowing. Try a larger model, increasing `context_window`, or `--no-embeddings`.

### Authentication errors (OpenAI / Gemini / Anthropic)
Ensure your API key is set via environment variable or `.agentchanti.yaml`:
```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Anthropic: no embeddings
Anthropic doesn't offer an embedding API. Use `--no-embeddings` or configure a different provider's embedding model.

### Plan has too many / wrong steps
Use the plan approval prompt: **A** (approve), **E** (TUI editor), **T** (text editor), **R** (replan).

### Resuming a failed run
```bash
agentchanti "same task" --resume       # force resume
agentchanti "same task" --fresh        # start over
```

### Cache issues (stale results)
```bash
agentchanti "task" --clear-cache       # clear before running
agentchanti "task" --no-cache          # disable caching entirely
```

---

## Project Structure

```
agentchanti/
  multi_agent_coder/
    agents/
      base.py                  # Agent base class with custom prompt suffix support
      planner.py               # PlannerAgent — creates step-by-step plans
      coder.py                 # CoderAgent — writes source code
      reviewer.py              # ReviewerAgent — reviews code quality
      tester.py                # TesterAgent — generates unit tests
      search.py                # SearchAgent — web search for docs and error fixes
    llm/
      base.py                  # LLMClient base with retry + streaming
      ollama.py                # Ollama HTTP client
      lm_studio.py             # LM Studio (OpenAI-compatible) client
      openai_client.py         # Cloud LLM client (any OpenAI-compatible API)
      gemini_client.py         # Google Gemini native API client
      anthropic_client.py      # Anthropic Claude native API client
    orchestrator/
      __init__.py              # Package init with backward-compatible exports
      cli.py                   # CLI entry point with argparse
      memory.py                # FileMemory — thread-safe file tracking
      classification.py        # Step classification (CMD/CODE/TEST/IGNORE)
      step_handlers.py         # Step execution handlers
      diagnosis.py             # Failure diagnosis and fix application
      pipeline.py              # Wave-based parallel/sequential execution
    kb/
      cli.py                   # KB subcommand entry point
      context_builder.py       # Single entry point for KB context injection
      project_orientation.py   # Project DNA detection and grounding
      runtime_watcher.py       # File change monitoring during execution
      startup.py               # Smart KB initialization manager
      local/
        graph.py               # Tree-sitter based code graph
        ...                    # Semantic indexing, search
      global_kb/
        ...                    # Error dictionary, ADRs, patterns
    editing/
      scope_resolver.py        # Code scope extraction (tree-sitter)
      context_slicer.py        # File slicing with context
      diff_parser.py           # LLM diff parsing
      patch_applier.py         # Fuzzy patch application
      chunk_editor.py          # Large file chunk editing
      metrics.py               # Edit success tracking
    plugins/
      __init__.py              # StepPlugin base class + PluginContext
      registry.py              # Plugin discovery (config + entry points)
    executor.py                # File I/O, command execution, plan parsing
    cli_display.py             # Terminal UI, progress bars, token tracking
    config.py                  # YAML config + env vars + defaults
    api.py                     # Library API: run_task() + TaskResult
    diff_display.py            # Colored diff preview before file writes
    embedding_store.py         # In-memory vector store
    embedding_store_sqlite.py  # Persistent SQLite-backed vector store
    step_cache.py              # Hash-based LLM response cache with TTL
    knowledge.py               # Cross-run project knowledge base
    report.py                  # HTML report generator
    tui_editor.py              # Curses-based interactive plan editor
    language.py                # Language detection + test framework mapping
    project_scanner.py         # Project scanner for planner context
    checkpoint.py              # Save/restore pipeline state for resume
    git_utils.py               # Git checkpoint, commit, rollback
    __init__.py                # Public API exports (run_task, TaskResult)
  tests/
    test_flow.py               # Core agent flow tests
    test_embeddings.py         # Embedding store + FileMemory tests
  setup.py                     # Package configuration
  install.sh                   # Linux/macOS installer
  install.bat                  # Windows installer
```
