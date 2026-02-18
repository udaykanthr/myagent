<p align="center">
<pre>
     _                    _      ____ _                 _   _
    / \   __ _  ___ _ __ | |_   / ___| |__   __ _ _ __ | |_(_)
   / _ \ / _` |/ _ \ '_ \| __| | |   | '_ \ / _` | '_ \| __| |
  / ___ \ (_| |  __/ | | | |_  | |___| | | | (_| | | | | |_| |
 /_/   \_\__, |\___|_| |_|\__|  \____|_| |_|\__,_|_| |_|\__|_|
         |___/
                    ━━  L o c a l   C o d e r  ━━
</pre>
</p>

<p align="center">
  <b>A fully offline, multi-agent AI coding system powered by local LLMs.</b><br>
  Plans. Codes. Reviews. Tests. All on your machine.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/LLM-Local%20%2B%20Cloud-orange" alt="Local + Cloud">
  <img src="https://img.shields.io/badge/providers-Ollama%20%7C%20LM%20Studio%20%7C%20OpenAI-blueviolet" alt="Multiple Providers">
</p>



## What is AgentChanti?

AgentChanti is a **command-line tool and Python library** that takes a plain English description of a coding task and autonomously builds the software for you using a team of specialized AI agents:

| Agent | Role | What it does |
|-------|------|--------------|
| **Planner** | Software Architect | Breaks your task into numbered, actionable steps |
| **Coder** | Senior Developer | Writes clean, idiomatic code for each step |
| **Reviewer** | Code Reviewer | Checks code for bugs, style issues, and correctness |
| **Tester** | QA Engineer | Generates and runs unit tests to verify everything works |

The agents collaborate in a pipeline with built-in retry loops, automatic failure diagnosis, and self-healing capabilities.

---

## Features

### Core
- **Local + Cloud LLMs** — Works with [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), and any **OpenAI-compatible API** (OpenAI, Groq, Together.ai, etc.)
- **Multi-Language Support** — Auto-detects your project's language (Python, JavaScript, TypeScript, Go, Rust, Java, Ruby, C++, and more) and adapts prompts, code style, and test frameworks accordingly.
- **Existing Project Awareness** — Scans your current directory before planning so the agents understand your codebase, dependencies, and structure.
- **Parallel Execution** — Independent steps run in parallel when the planner marks them as having no dependencies.
- **Halt-on-Failure with Auto-Diagnosis** — If a step fails, the pipeline stops, diagnoses the root cause, applies a fix, and retries. No wasted work.

### Smart Features
- **Persistent Embedding Store** — SQLite-backed vector cache that persists embeddings across runs. Unchanged files are never re-embedded, saving ~30s per run on large projects.
- **Project Knowledge Base** — Learns from each run and stores patterns, fixes, conventions, and dependency info in `.agentchanti/knowledge.json`. Future runs benefit from accumulated knowledge.
- **Step-Level Caching** — Hashes step inputs and caches LLM responses. Re-running the same task skips completed steps instantly. Configurable TTL (default: 24h).
- **Custom Agent Prompts** — Override or extend agent behavior via `.agentchanti.yaml` config (coding conventions, review criteria, etc.)
- **Plugin System** — Extend with custom step types (DEPLOY, LINT, DOCS, etc.) via Python classes.
- **Cloud LLM (OpenAI-compatible)** — High-performance coding with **GPT-4o**, **Claude 3.5**, or **DeepSeek**.
- **TUI Plan Editor** — Interactively edit, add, or reorder steps before execution starts.
- **Cost Tracking & Budgets** — Monitor cloud API spending and set hard limits.

### Developer Experience
- **TUI Plan Editor** — Interactive curses-based plan editor with arrow-key navigation, reordering, inline editing, and step deletion. Falls back to text editor on unsupported terminals.
- **Multi-File Diff Preview** — See colored unified diffs of all file changes before they're written to disk.
- **HTML Reports** — Auto-generates self-contained HTML reports after every run with a dark theme, dashboard stats, step breakdown, and colored diffs.
- **Checkpoint & Resume** — Saves progress after each step. Resume exactly where you left off after interruption.
- **Git Safety Net** — Creates a checkpoint branch before execution. Offers to commit on success or rollback on failure.
- **Beautiful CLI** — Full-screen terminal UI with ASCII art banner, progress bar, per-step token tracking, and status icons.
- **Library API** — Use AgentChanti programmatically with `run_task()` for backend integration.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/udaykanth-rapeti/myagent.git
cd myagent

# 2. Install
pip install -e .

# 3. Start your LLM server (Ollama or LM Studio -- see below)

# 4. Run!
agentchanti "Create a Python CLI calculator that supports add, subtract, multiply, divide"
```

That's it. AgentChanti will plan the steps, write the code, review it, generate tests, and run them.

---

## Installation

### Prerequisites

- **Python 3.10+** ([python.org](https://www.python.org/downloads/))
- **Git** ([git-scm.com](https://git-scm.com/))
- A local LLM server **or** a cloud API key

### Option 1: Automatic Installer (Recommended)

**Linux / macOS:**
```bash
chmod +x install.sh
./install.sh
```

**Windows:**
```cmd
install.bat
```

### Option 2: Manual Install

```bash
# Clone
git clone https://github.com/udaykanth-rapeti/myagent.git
cd myagent

# (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install in editable mode
pip install -e .
```

### Verify Installation

```bash
agentchanti --help
```

---

## LLM Server Setup

AgentChanti supports three LLM providers. Choose **one** (or configure multiple):

### Option A: Ollama (Recommended for beginners)

Ollama is the easiest way to run LLMs locally.

**Step 1: Install Ollama**

| Platform | Command |
|----------|---------|
| **macOS** | `brew install ollama` or download from [ollama.com](https://ollama.com) |
| **Linux** | `curl -fsSL https://ollama.com/install.sh \| sh` |
| **Windows** | Download from [ollama.com/download](https://ollama.com/download) |

**Step 2: Download a Model**

```bash
# Recommended coding models (pick one):
ollama pull deepseek-coder-v2:16b       # Great for coding, 16B params
ollama pull qwen2.5-coder:7b            # Lighter, still good
ollama pull codellama:13b               # Meta's code-focused model

# For semantic search (optional but recommended):
ollama pull nomic-embed-text            # Small embedding model
```

> **Which model should I pick?** Bigger = smarter but slower. Start with a 7B model if you have 8GB RAM/VRAM, or a 13B-16B model if you have 16GB+.

**Step 3: Start & Run**

```bash
ollama serve
# In another terminal:
agentchanti "Create a REST API with Flask" --provider ollama --model deepseek-coder-v2:16b
```

---

### Option B: LM Studio

LM Studio provides a GUI for downloading and running models with an OpenAI-compatible API.

1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Search and download a coding model (e.g., `deepseek-coder-v2-lite-instruct`)
3. Go to **Local Server** tab → select model → click **Start Server**
4. Run (LM Studio is the default provider):

```bash
agentchanti "Build a todo app with SQLite backend"
```

---

### Option C: Cloud LLM (OpenAI-compatible)

Use any OpenAI-compatible API — OpenAI, Groq, Together.ai, or any other compatible service.

**Via environment variable:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
agentchanti "Create a REST API" --provider openai --model gpt-4o-mini
```

**Via config file (`.agentchanti.yaml`):**
```yaml
openai:
  api_key: "sk-your-key-here"
  base_url: "https://api.openai.com/v1"  # or Groq, Together.ai, etc.
```

```bash
agentchanti "Create a REST API" --provider openai --model gpt-4o-mini
```

**Example: Using Groq**
```bash
export OPENAI_API_KEY="gsk_your-groq-key"
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
agentchanti "your task" --provider openai --model llama-3.1-70b-versatile
```

---

### Optional: Embedding Model (for smarter context retrieval)

Embeddings help AgentChanti find the most relevant files for each step. Enabled by default with persistent SQLite caching.

```bash
# Ollama:
ollama pull nomic-embed-text

# Disable embeddings (saves resources):
agentchanti "your task" --no-embeddings
```

---

## Usage

### Basic Syntax

```bash
agentchanti "<task description>" [options]
```

### All Options

| Flag | Description | Default |
|------|-------------|---------|
| `"task"` | The coding task to perform (required) | — |
| `--provider` | LLM provider: `ollama`, `lm_studio`, or `openai` | `lm_studio` |
| `--model` | Model name to use | `deepseek-coder-v2-lite-instruct` |
| `--embed-model` | Embedding model name | `nomic-embed-text` |
| `--language` | Override auto-detected language (e.g. `python`, `javascript`) | auto-detect |
| `--config` | Path to `.agentchanti.yaml` config file | auto-discover |
| `--no-embeddings` | Disable semantic embeddings | off |
| `--no-stream` | Disable streaming responses | off |
| `--no-git` | Disable git checkpoint/rollback | off |
| `--no-diff` | Disable diff preview before writing files | off |
| `--no-cache` | Disable step-level caching | off |
| `--clear-cache` | Clear step cache before running | off |
| `--no-knowledge` | Disable project knowledge base | off |
| `--report` / `--no-report` | Enable/disable HTML report generation | on |
| `--resume` | Force resume from last checkpoint | off |
| `--fresh` | Ignore any existing checkpoint and start fresh | off |
| `--auto` | Non-interactive mode: auto-approve plan, skip all prompts | off |
| `--generate-yaml` | Generate a `.agentchanti.yaml` file with current settings and exit | off |

### Examples

**1. Simple Python script**
```bash
agentchanti "Create a Python script that reads a CSV and generates a bar chart using matplotlib"
```

**2. Web application**
```bash
agentchanti "Build a Flask REST API with CRUD operations on a books database using SQLite"
```

**3. Using a cloud provider**
```bash
OPENAI_API_KEY="sk-..." agentchanti "Build a CLI password generator" --provider openai --model gpt-4o-mini
```

**4. Using Ollama with a specific model**
```bash
agentchanti "Create a CLI password generator" --provider ollama --model codellama:13b
```

**5. JavaScript / Node.js project**
```bash
agentchanti "Create an Express.js REST API with JWT authentication" --language javascript
```

**6. Working on an existing project**
```bash
cd my-existing-project/
agentchanti "Add input validation to all API endpoints"
# AgentChanti scans the directory first and understands your codebase
```

**7. Non-interactive (CI/scripts)**
```bash
agentchanti "Generate unit tests for all modules" --auto --no-git --no-report
```

**8. Re-run with fresh cache**
```bash
agentchanti "Refactor the database layer" --clear-cache
```

**9. Generate a configuration file**
```bash
agentchanti --generate-yaml
# Creates .agentchanti.yaml with defaults

agentchanti --provider openai --model gpt-4o-mini --generate-yaml
# Creates .agentchanti.yaml with specific provider/model setup
```

---

## Configuration

AgentChanti is highly configurable. It looks for `.agentchanti.yaml` in your project root, or `~/.agentchanti.yaml` for global settings.

```yaml
# General
provider: "lm_studio"
model: "deepseek-coder-v2-lite-instruct"
budget_limit: 5.0  # Halt if cost exceeds $5.00

# Cloud Setup
openai:
  api_key: "sk-..."
  base_url: "https://api.openai.com/v1"

# Custom Pricing (per 1M tokens)
pricing:
  gpt-4o: {input: 2.50, output: 10.00}
```
### Config File (`.agentchanti.yaml`)

AgentChanti looks for a config file in the **current directory** first, then your **home directory**. Settings are resolved in priority order: **CLI args > env vars > YAML config > defaults**.

Create `.agentchanti.yaml` in your project or home directory:

```yaml
# LLM Settings
model: "deepseek-coder-v2:16b"
context_window: 16384
stream: true

# Provider URLs
ollama_base_url: "http://localhost:11434/api/generate"
lm_studio_base_url: "http://localhost:1234/v1"

# Cloud LLM (any OpenAI-compatible API)
openai:
  api_key: "sk-your-key-here"
  base_url: "https://api.openai.com/v1"

# Embeddings
embedding_model: "nomic-embed-text"
embedding_top_k: 5
embedding_cache_dir: ".agentchanti"   # SQLite cache location

# Per-agent model overrides (use different models for different agents)
models:
  planner: "deepseek-coder-v2:16b"     # Smarter model for planning
  coder: "qwen2.5-coder:7b"           # Fast model for coding
  reviewer: "deepseek-coder-v2:16b"    # Smarter model for reviews
  tester: "qwen2.5-coder:7b"          # Fast model for tests

# Custom prompt suffixes (append instructions to each agent's prompt)
prompts:
  planner_suffix: "Focus on small, incremental steps."
  coder_suffix: "Always use type hints. Follow PEP 8. Use dataclasses."
  reviewer_suffix: "Check for SQL injection and XSS vulnerabilities."
  tester_suffix: "Use pytest fixtures. Aim for 90% coverage."

# Step caching
step_cache_ttl_hours: 24

# HTML reports
report_dir: ".agentchanti/reports"

# Plugins (custom step handlers)
plugins:
  - my_package.plugins.LintPlugin
  - my_package.plugins.DeployPlugin

# Other
llm_max_retries: 3
llm_retry_delay: 2.0
checkpoint_file: ".agentchanti_checkpoint.json"
```

### Environment Variables

All settings can also be set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_MODEL` | `deepseek-coder-v2-lite-instruct` | Default model name |
| `CONTEXT_WINDOW` | `8192` | Model context window size (tokens) |
| `OLLAMA_BASE_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `OPENAI_API_KEY` | — | API key for OpenAI-compatible providers |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Cloud LLM API base URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `EMBEDDING_TOP_K` | `5` | Number of files to retrieve for context |
| `EMBEDDING_CACHE_DIR` | `.agentchanti` | Directory for persistent embedding cache |
| `REPORT_DIR` | `.agentchanti/reports` | HTML report output directory |
| `STEP_CACHE_TTL_HOURS` | `24` | Step cache TTL in hours |
| `LLM_MAX_RETRIES` | `3` | Max retries on LLM failure |
| `LLM_RETRY_DELAY` | `2.0` | Base delay (seconds) between retries |
| `STREAM_RESPONSES` | `true` | Enable/disable streaming |

---

## Library API (Programmatic Use)

Use AgentChanti as a Python library for backend integration, CI pipelines, or custom tooling:

```python
from multi_agent_coder import run_task, TaskResult

result: TaskResult = run_task(
    task="Create a Python CLI calculator",
    provider="lm_studio",
    model="deepseek-coder-v2-lite-instruct",
    language="python",
    auto=True,          # Non-interactive
)

print(f"Success: {result.success}")
print(f"Files written: {result.files_written}")
print(f"Plan steps: {result.plan_steps}")
print(f"Token usage: {result.token_usage}")
print(f"Log file: {result.log_file}")

if not result.success:
    print(f"Error: {result.error}")
```

### `TaskResult` Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the pipeline completed successfully |
| `files_written` | `list[str]` | Paths of all files created or modified |
| `plan_steps` | `list[str]` | The executed plan steps |
| `token_usage` | `dict` | Token counts (`sent`, `recv`, `total`) |
| `log_file` | `str` | Path to the detailed log file |
| `error` | `str \| None` | Error message if pipeline failed |

---

## Plugin System

Extend AgentChanti with custom step type handlers. Create a Python class that inherits from `StepPlugin`:

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

Register your plugin in `.agentchanti.yaml`:

```yaml
plugins:
  - my_package.plugins.LintPlugin
```

Or via setuptools entry points:

```toml
# pyproject.toml
[project.entry-points."agentchanti.plugins"]
lint = "my_package.plugins:LintPlugin"
```

---

## How It Works

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
     +------------------+        Apply fix
              |                  Retry the step
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

### Execution Flow

1. **Scan** — Reads your project structure and key files
2. **Knowledge** — Loads learnings from previous runs (if available)
3. **Plan** — Planner agent creates numbered steps with dependency info
4. **Approve** — You review the plan (approve / TUI edit / text edit / replan)
5. **Execute** — Each step is classified (CMD / CODE / TEST / IGNORE / PLUGIN) and executed:
   - **CMD**: Shell commands (installs, file operations)
   - **CODE**: Coder writes code → Reviewer checks → retry if needed
   - **TEST**: Tester generates tests → run them → Coder fixes failures
   - **PLUGIN**: Custom step handlers (LINT, DEPLOY, etc.)
   - **IGNORE**: Non-actionable steps are skipped
6. **Diagnose** — If a step fails after retries, the LLM analyzes why and applies a fix
7. **Checkpoint** — Progress is saved after each successful step
8. **Learn** — Extracts patterns and conventions from the run
9. **Report** — Generates an HTML report with stats, diffs, and step breakdown
10. **Git** — On completion, offers to commit; on failure, offers to rollback

---

## Project Structure

```
myagent/
  multi_agent_coder/
    agents/
      base.py                  # Agent base class with custom prompt suffix support
      planner.py               # PlannerAgent — creates step-by-step plans
      coder.py                 # CoderAgent — writes source code
      reviewer.py              # ReviewerAgent — reviews code quality
      tester.py                # TesterAgent — generates unit tests
    llm/
      base.py                  # LLMClient base with retry + streaming
      ollama.py                # Ollama HTTP client
      lm_studio.py             # LM Studio (OpenAI-compatible) client
      openai_client.py         # Cloud LLM client (any OpenAI-compatible API)
    orchestrator/
      __init__.py              # Package init with backward-compatible exports
      cli.py                   # CLI entry point with argparse
      memory.py                # FileMemory — thread-safe file tracking
      classification.py        # Step classification (CMD/CODE/TEST/IGNORE)
      step_handlers.py         # Step execution handlers
      diagnosis.py             # Failure diagnosis and fix application
      pipeline.py              # Wave-based parallel/sequential execution
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
  README.md                    # You are here
```

---

## Troubleshooting

### "Connection refused" or "Connection error"

Your LLM server isn't running. Start it:
```bash
# Ollama
ollama serve

# LM Studio
# Open the app -> Local Server tab -> Start Server
```

### Streaming errors / socket timeouts

Some models or server versions don't handle streaming well:
```bash
agentchanti "your task" --no-stream
```

### "Empty response" after retries

The model may be too small for the task, or the context window is overflowing. Try:
- A larger model
- Increasing `context_window` in `.agentchanti.yaml`
- Using `--no-embeddings` to reduce context size

### OpenAI provider: authentication errors

Make sure your API key is set:
```bash
export OPENAI_API_KEY="sk-your-key-here"
# Or add it to .agentchanti.yaml under openai.api_key
```

### Plan has too many / wrong steps

Use the plan approval prompt:
- Press `A` to approve the plan as-is
- Press `E` to open the **TUI editor** (reorder, edit, delete steps interactively)
- Press `T` to open a **text editor** (notepad on Windows, vi on Linux/macOS)
- Press `R` to request a completely new plan

### Resuming a failed run

```bash
# Same task — detects checkpoint and asks to resume
agentchanti "same task description"

# Force resume
agentchanti "same task" --resume

# Start completely fresh
agentchanti "same task" --fresh
```

### Cache issues (stale results)

If source files changed but cached results seem stale:
```bash
agentchanti "task" --clear-cache     # Clear cache before running
agentchanti "task" --no-cache        # Disable caching entirely
```

---

## Supported Languages

AgentChanti auto-detects your project's language and adapts accordingly:

| Language | Test Framework | Auto-Detected By |
|----------|---------------|------------------|
| Python | pytest | `.py` files, keywords: flask, django, fastapi |
| JavaScript | Jest | `.js` files, keywords: node, express, react |
| TypeScript | Jest | `.ts`/`.tsx` files, keywords: angular, nest |
| Go | go test | `.go` files, keywords: golang, gin |
| Rust | cargo test | `.rs` files, keywords: cargo, tokio |
| Java | Maven (mvn test) | `.java` files, keywords: spring, maven |
| Ruby | RSpec | `.rb` files, keywords: rails, sinatra |
| C++ | — | `.cpp`/`.hpp` files |
| C# | — | `.cs` files, keywords: dotnet |

Override with `--language`:
```bash
agentchanti "Build a REST API" --language go
```

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run the tests: `python -m pytest tests/ -v`
5. Commit and push
6. Open a pull request

### Creating Plugins

See the [Plugin System](#plugin-system) section above. You can distribute plugins as pip-installable packages with `agentchanti.plugins` entry points.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Disclaimer

> **This is a personal project by [Uday Kanth](https://github.com/udaykanthr).** It is not affiliated with, endorsed by, sponsored by, or in any way officially connected with my current or past employer(s), or any of their subsidiaries, clients, or affiliates. All opinions, code, and design decisions in this project are my own and do not represent the views or intellectual property of any organization I am or have been associated with. This project was built entirely on my own time using my own resources.

---
