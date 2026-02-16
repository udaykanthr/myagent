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
  <img src="https://img.shields.io/badge/LLM-100%25%20Local-orange" alt="100% Local">
  <img src="https://img.shields.io/badge/API%20keys-none%20required-brightgreen" alt="No API Keys">
</p>

---

## Disclaimer

> **This is a personal project by [Uday Kanth](https://github.com/udaykanth-rapeti).** It is not affiliated with, endorsed by, sponsored by, or in any way officially connected with my current or past employer(s), or any of their subsidiaries, clients, or affiliates. All opinions, code, and design decisions in this project are my own and do not represent the views or intellectual property of any organization I am or have been associated with. This project was built entirely on my own time using my own resources.

---

## What is AgentChanti?

AgentChanti is a **command-line tool** that takes a plain English description of a coding task and autonomously builds the software for you using a team of specialized AI agents, all running on your local machine with no cloud dependency:

| Agent | Role | What it does |
|-------|------|--------------|
| **Planner** | Software Architect | Breaks your task into numbered, actionable steps |
| **Coder** | Senior Developer | Writes clean, idiomatic code for each step |
| **Reviewer** | Code Reviewer | Checks code for bugs, style issues, and correctness |
| **Tester** | QA Engineer | Generates and runs unit tests to verify everything works |

The agents collaborate in a pipeline with built-in retry loops, automatic failure diagnosis, and self-healing capabilities.

---

## Features

- **100% Offline** -- Works with [Ollama](https://ollama.com) and [LM Studio](https://lmstudio.ai). No API keys, no cloud, no data leaves your machine.
- **Multi-Language Support** -- Auto-detects your project's language (Python, JavaScript, TypeScript, Go, Rust, Java, Ruby, C++, and more) and adapts prompts, code style, and test frameworks accordingly.
- **Existing Project Awareness** -- Scans your current directory before planning so the agents understand your codebase, dependencies, and structure.
- **Plan Approval** -- Shows you the proposed plan and lets you approve, edit (remove steps), or ask for a replan before any code is written.
- **Halt-on-Failure with Auto-Diagnosis** -- If a step fails, the pipeline stops, the LLM diagnoses the root cause, applies a fix, and retries from that step. No wasted work on downstream steps.
- **Checkpoint & Resume** -- Saves progress after each step. If interrupted, resume exactly where you left off.
- **Git Safety Net** -- Creates a checkpoint branch before execution. On success, offers to commit. On failure, offers to rollback to the clean state.
- **Streaming Responses** -- Live token-count progress during LLM generation (no more frozen screens). Falls back to non-streaming automatically if the server doesn't support it.
- **Context Window Protection** -- Automatically manages prompt size to stay within your model's context window.
- **LLM Resilience** -- Retries with exponential backoff on connection errors, timeouts, or empty responses.
- **Parallel Execution** -- Independent steps run in parallel when the planner marks them as having no dependencies.
- **Beautiful CLI** -- Full-screen terminal UI with ASCII art banner, progress bar, per-step token tracking, and status icons.

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
- A local LLM server: **Ollama** or **LM Studio** (setup guide below)

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

You should see the argument list with all available options.

---

## LLM Server Setup

AgentChanti needs a local LLM running on your machine. Choose **one** of the following:

### Option A: Ollama (Recommended for beginners)

Ollama is the easiest way to run LLMs locally. One command to install, one command to download a model.

**Step 1: Install Ollama**

| Platform | Command |
|----------|---------|
| **macOS** | `brew install ollama` or download from [ollama.com](https://ollama.com) |
| **Linux** | `curl -fsSL https://ollama.com/install.sh \| sh` |
| **Windows** | Download the installer from [ollama.com/download](https://ollama.com/download) |

**Step 2: Download a Model**

```bash
# Recommended coding models (pick one):
ollama pull deepseek-coder-v2:16b       # Great for coding, 16B params
ollama pull qwen2.5-coder:7b            # Lighter, still good
ollama pull llama3:8b                   # General purpose
ollama pull codellama:13b               # Meta's code-focused model

# For semantic search (optional but recommended):
ollama pull nomic-embed-text            # Small embedding model
```

> **Which model should I pick?** Bigger = smarter but slower. Start with a 7B model if you have 8GB RAM/VRAM, or a 13B-16B model if you have 16GB+.

**Step 3: Start the Server**

```bash
ollama serve
```

The server runs at `http://localhost:11434` by default. Leave this terminal open.

**Step 4: Run AgentChanti**

```bash
agentchanti "Create a REST API with Flask" --provider ollama --model deepseek-coder-v2:16b
```

---

### Option B: LM Studio

LM Studio provides a GUI for downloading and running models with an OpenAI-compatible API.

**Step 1: Install LM Studio**

Download from [lmstudio.ai](https://lmstudio.ai) and install. Available for Windows, macOS, and Linux.

**Step 2: Download a Model**

1. Open LM Studio
2. Go to the **Search** tab (magnifying glass icon)
3. Search for a coding model, for example:
   - `deepseek-coder-v2-lite-instruct` (default, good balance)
   - `Qwen2.5-Coder-7B-Instruct`
   - `CodeLlama-13B-Instruct`
4. Click **Download** on the quantization that fits your RAM (Q4_K_M is a good default)

**Step 3: Start the Local Server**

1. Go to the **Local Server** tab (the `<->` icon on the left)
2. Select your downloaded model from the dropdown
3. Click **Start Server**
4. Default URL: `http://localhost:1234/v1`

> **Important:** Make sure the server status shows **Running** (green indicator) before using AgentChanti.

**Step 4: Run AgentChanti**

```bash
# LM Studio is the default provider, so this just works:
agentchanti "Build a todo app with SQLite backend"
```

---

### Optional: Embedding Model (for smarter context retrieval)

AgentChanti can use embeddings to find the most relevant files when building context for each step. This is optional but improves quality on larger projects.

**Ollama:**
```bash
ollama pull nomic-embed-text
```

**LM Studio:**
Download `nomic-embed-text` from the model search.

If you don't want embeddings (saves resources):
```bash
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
| `"task"` | The coding task to perform (required) | -- |
| `--provider` | LLM provider: `ollama` or `lm_studio` | `lm_studio` |
| `--model` | Model name to use | `deepseek-coder-v2-lite-instruct` |
| `--embed-model` | Embedding model name | `nomic-embed-text` |
| `--language` | Override auto-detected language (e.g. `python`, `javascript`) | auto-detect |
| `--no-embeddings` | Disable semantic embeddings | off |
| `--no-stream` | Disable streaming (use if your server has issues) | off |
| `--no-git` | Disable git checkpoint/rollback integration | off |
| `--resume` | Force resume from last checkpoint | off |
| `--fresh` | Ignore any existing checkpoint and start fresh | off |
| `--auto` | Non-interactive mode: auto-approve plan, skip all prompts | off |

### Examples

**1. Simple Python script**
```bash
agentchanti "Create a Python script that reads a CSV file and generates a bar chart using matplotlib"
```

**2. Web application**
```bash
agentchanti "Build a Flask REST API with endpoints for CRUD operations on a books database using SQLite"
```

**3. Algorithm implementation**
```bash
agentchanti "Implement a binary search tree with insert, delete, search, and in-order traversal in Python"
```

**4. Using Ollama with a specific model**
```bash
agentchanti "Create a CLI password generator" --provider ollama --model codellama:13b
```

**5. JavaScript / Node.js project**
```bash
agentchanti "Create an Express.js REST API with user authentication using JWT" --language javascript
```

**6. Working on an existing project**
```bash
cd my-existing-project/
agentchanti "Add input validation to all API endpoints"
# AgentChanti scans the directory first and understands your codebase
```

**7. Resume after interruption**
```bash
# Task was interrupted at step 5 of 10...
agentchanti "Create a web scraper" --resume
# Picks up from step 6
```

**8. Non-interactive (CI/scripts)**
```bash
agentchanti "Generate unit tests for all modules" --auto --no-git
```

**9. Disable streaming (troubleshooting)**
```bash
agentchanti "Create a hello world app" --no-stream
```

---

## How It Works

```
                    +-----------+
    Your Task ----->|  Planner  |-----> Step-by-step plan
                    +-----------+
                          |
                    [User Approval]
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
```

### Execution Flow

1. **Scan** -- Reads your project structure and key files
2. **Plan** -- Planner agent creates numbered steps with dependency info
3. **Approve** -- You review the plan (approve / edit / replan)
4. **Execute** -- Each step is classified (CMD / CODE / TEST / IGNORE) and executed:
   - **CMD**: Shell commands (installs, file operations)
   - **CODE**: Coder writes code -> Reviewer checks -> retry if needed
   - **TEST**: Tester generates tests -> run them -> Coder fixes failures
   - **IGNORE**: Non-actionable steps are skipped
5. **Diagnose** -- If a step fails after retries, the LLM analyzes why and applies a fix
6. **Checkpoint** -- Progress is saved after each successful step
7. **Git** -- On completion, offers to commit; on failure, offers to rollback

---

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `DEFAULT_MODEL` | `deepseek-coder-v2-lite-instruct` | Default model name |
| `CONTEXT_WINDOW` | `8192` | Model context window size (tokens) |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `EMBEDDING_TOP_K` | `5` | Number of files to retrieve for context |
| `LLM_MAX_RETRIES` | `3` | Max retries on LLM failure |
| `LLM_RETRY_DELAY` | `2.0` | Base delay (seconds) between retries |
| `STREAM_RESPONSES` | `true` | Enable/disable streaming |
| `CHECKPOINT_FILE` | `.agentchanti_checkpoint.json` | Checkpoint file path |

**Example: Use a different Ollama port**
```bash
export OLLAMA_BASE_URL="http://localhost:12345/api/generate"
agentchanti "my task" --provider ollama
```

---

## Project Structure

```
myagent/
  multi_agent_coder/
    agents/
      base.py            # Abstract agent base class
      planner.py         # PlannerAgent -- creates step-by-step plans
      coder.py           # CoderAgent -- writes source code
      reviewer.py        # ReviewerAgent -- reviews code quality
      tester.py          # TesterAgent -- generates unit tests
    llm/
      base.py            # LLMClient base with retry + streaming
      ollama.py          # Ollama HTTP client
      lm_studio.py       # LM Studio (OpenAI-compatible) client
    orchestrator.py      # Main pipeline: plan -> execute -> diagnose
    executor.py          # File I/O, command execution, plan parsing
    cli_display.py       # Terminal UI, progress bars, token tracking
    config.py            # Environment-based configuration
    language.py          # Language detection + test framework mapping
    project_scanner.py   # Scans existing projects for planner context
    checkpoint.py        # Save/restore pipeline state for resume
    git_utils.py         # Git checkpoint branches, commit, rollback
    embedding_store.py   # In-memory vector store with cosine similarity
  setup.py               # Package configuration
  install.sh             # Linux/macOS installer
  install.bat            # Windows installer
  README.md              # You are here
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

Some models or server versions don't handle streaming well. Disable it:
```bash
agentchanti "your task" --no-stream
```

### "Empty response" after retries

The model may be too small for the task, or the context window is overflowing. Try:
- A larger model
- Increasing `CONTEXT_WINDOW` environment variable
- Using `--no-embeddings` to reduce context size

### LM Studio: model not responding

Make sure:
1. A model is **loaded** (selected in the dropdown, not just downloaded)
2. The server shows **Running** (green)
3. The port matches (default: `1234`)

### Plan has too many / wrong steps

Use the plan approval prompt to fix it:
- Press `R` to replan (generates a new plan)
- Press `E` then enter step numbers to remove (e.g., `E 3,5,7`)

### Resuming a failed run

If AgentChanti was interrupted or a step failed:
```bash
# Same task -- it will detect the checkpoint and ask to resume
agentchanti "same task description"

# Or force resume
agentchanti "same task" --resume

# Or start completely fresh (ignores checkpoint)
agentchanti "same task" --fresh
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
| C++ | -- | `.cpp`/`.hpp` files |
| C# | -- | `.cs` files, keywords: dotnet |

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
4. Run the syntax check: `python -c "import ast; ast.parse(open('multi_agent_coder/orchestrator.py').read())"`
5. Commit and push
6. Open a pull request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with coffee and local LLMs by <a href="https://github.com/udaykanth-rapeti">Uday Kanth</a>
</p>
