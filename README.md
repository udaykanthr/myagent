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
  <img src="https://img.shields.io/badge/providers-Ollama%20%7C%20LM%20Studio%20%7C%20OpenAI%20%7C%20Gemini%20%7C%20Claude-blueviolet" alt="Multiple Providers">
</p>

## What is AgentChanti?

AgentChanti is a **command-line tool and Python library** that takes a plain English description of a coding task and autonomously builds the software for you using a team of specialized AI agents:

| Agent | Role |
|-------|------|
| **Planner** | Breaks your task into numbered, actionable steps |
| **Coder** | Writes clean, idiomatic code for each step |
| **Reviewer** | Checks code for bugs, style issues, and correctness |
| **Tester** | Generates and runs unit tests to verify everything works |

Supports local LLMs ([Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai)) and cloud providers (OpenAI, Google Gemini, Anthropic Claude).

---

## Getting Started

### Prerequisites

- **Python 3.10+** ([python.org](https://www.python.org/downloads/))
- **Git** ([git-scm.com](https://git-scm.com/))

- A local LLM server **or** a cloud API key

### Installation

**Option 1: Automatic Installer**

```bash
# Linux / macOS
chmod +x install.sh && ./install.sh

# Windows
./install.bat
```

**Option 2: Manual Install**

```bash
git clone https://github.com/udaykanthr/agentchanti.git
cd agentchanti

# (Optional) Virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

pip install -e .
```

**Verify:**

```bash
agentchanti --help
```

---

## Usage

```bash
agentchanti "<task description>" [options]
```

### Quick Examples

```bash
# With Ollama
agentchanti "Create a Flask REST API with CRUD" --provider ollama --model deepseek-coder-v2:16b

# With OpenAI
OPENAI_API_KEY="sk-..." agentchanti "Build a CLI tool" --provider openai --model gpt-4o-mini

# With Gemini
GEMINI_API_KEY="..." agentchanti "Build a REST API" --provider gemini --model gemini-2.5-flash

# With Claude
ANTHROPIC_API_KEY="sk-ant-..." agentchanti "Build a CLI tool" --provider anthropic --model claude-sonnet-4

# Non-interactive (CI/scripts)
agentchanti "Generate unit tests" --auto --no-git --no-report
```

### All Options

| Flag | Description | Default |
|------|-------------|---------|
| `"task"` | The coding task to perform (required) | — |
| `--prompt-from-file` | Read task description from a file | — |
| `--provider` | `ollama`, `lm_studio`, `openai`, `gemini`, `anthropic` | `lm_studio` |
| `--model` | Model name | `deepseek-coder-v2-lite-instruct` |
| `--embed-model` | Embedding model name | `nomic-embed-text` |
| `--language` | Override auto-detected language | auto-detect |
| `--config` | Path to `.agentchanti.yaml` config file | auto-discover |
| `--auto` | Non-interactive mode (auto-approve plan) | off |
| `--no-embeddings` | Disable semantic embeddings | off |
| `--no-stream` | Disable streaming responses | off |
| `--no-git` | Disable git checkpoint/rollback | off |
| `--no-diff` | Disable diff preview before writing | off |
| `--no-cache` | Disable step-level caching | off |
| `--clear-cache` | Clear step cache before running | off |
| `--no-knowledge` | Disable project knowledge base | off |
| `--no-search` | Disable web search agent | off |
| `--no-kb` | Disable KB context injection | off |
| `--report` / `--no-report` | Enable/disable HTML report | on |
| `--resume` | Force resume from checkpoint | off |
| `--fresh` | Ignore checkpoint, start fresh | off |
| `--generate-yaml` | Generate `.agentchanti.yaml` and exit | off |

### Knowledge Base Commands

Manage the project knowledge base via `agentchanti kb`:

```bash
agentchanti kb embed                 # Embed symbols into the vector store
agentchanti kb search "query"        # Semantic search over KB
agentchanti kb query find-callers X  # Find all callers of a function
agentchanti kb error-lookup "msg"    # Look up error fixes
agentchanti kb health                # Show KB health report
agentchanti kb update                # Pull global KB updates
```

See [documentation.md](documentation.md) for the full list of KB commands.

---

## Documentation

For full documentation including architecture details, configuration reference, library API, plugin system, and troubleshooting, see **[documentation.md](documentation.md)**.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run the tests: `python -m pytest tests/ -v`
4. Commit and push
5. Open a pull request

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Disclaimer

> **This is a personal project by [Uday Kanth](https://github.com/udaykanthr).** It is not affiliated with, endorsed by, sponsored by, or in any way officially connected with my current or past employer(s), or any of their subsidiaries, clients, or affiliates. All opinions, code, and design decisions in this project are my own and do not represent the views or intellectual property of any organization I am or have been associated with. This project was built entirely on my own time using my own resources.
