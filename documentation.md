# AgentChanti Documentation

Welcome to the AgentChanti documentation! This guide will help you set up and use all the features of the multi-agent coding system.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
    - [YAML Config File](#yaml-config-file)
    - [Environment Variables](#environment-variables)
3. [LLM Provider Setup](#llm-provider-setup)
    - [Ollama](#ollama)
    - [LM Studio](#lm-studio)
    - [OpenAI & Cloud Providers](#openai--cloud-providers)
    - [Google Gemini](#google-gemini)
    - [Anthropic Claude](#anthropic-claude)
4. [Command Line Interface (CLI)](#command-line-interface-cli)
5. [Key Features](#key-features)
    - [TUI Plan Editor](#tui-plan-editor)
    - [HTML Reports](#html-reports)
    - [Step Caching](#step-caching)
    - [Persistent Embeddings](#persistent-embeddings)
    - [Knowledge Base](#knowledge-base)
6. [Plugin System](#plugin-system)

---

## Quick Start
Get up and running in 3 steps:

1. **Install AgentChanti**:
   ```bash
   pip install -e .
   ```
2. **Start your LLM**: Ensure Ollama or LM Studio is running.
3. **Run a task**:
   ```bash
   agentchanti "Create a simple Python hello world script"
   ```

---

## Configuration

AgentChanti uses a priority-based configuration system:
1. **CLI Arguments** (highest priority)
2. **Environment Variables**
3. **`.agentchanti.yaml`** (project or home directory)
4. **Built-in Defaults** (lowest priority)

### YAML Config File
Create a `.agentchanti.yaml` file manually, or use the generator command:
```bash
agentchanti --generate-yaml
```

This creates a file in your project root for project-specific settings, or in your home directory (`~/.agentchanti.yaml`) for global defaults.

```yaml
# General Settings
model: "deepseek-coder-v2-lite-instruct"
context_window: 8192
stream: true
budget_limit: 5.0  # Optional: Halt if cost exceeds $5.00

# Cloud / OpenAI-compatible Setup
openai:
  api_key: "sk-your-key-here"
  base_url: "https://api.openai.com/v1"

# Google Gemini
gemini:
  api_key: "your-gemini-api-key"
  base_url: "https://generativelanguage.googleapis.com/v1beta"

# Anthropic Claude
anthropic:
  api_key: "sk-ant-your-key-here"
  base_url: "https://api.anthropic.com/v1"

# Custom Pricing (Optional, per 1M tokens)
pricing:
  gpt-4o: {input: 2.50, output: 10.00}
  gpt-4o-mini: {input: 0.15, output: 0.60}
  gemini-2.5-flash: {input: 0.15, output: 0.60}
  gemini-2.5-pro: {input: 1.25, output: 10.00}
  claude-sonnet-4: {input: 3.00, output: 15.00}
  claude-haiku-4: {input: 0.80, output: 4.00}

# Custom Agent Behavior
prompts:
  planner_suffix: "Focus on small, incremental steps."
  coder_suffix: "Always use type hints and PEP 8 style."

# Feature Settings
embedding_cache_dir: ".agentchanti"
report_dir: ".agentchanti/reports"
step_cache_ttl_hours: 24
```

### Environment Variables
| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI-compatible providers |
| `OPENAI_BASE_URL` | Base URL for the OpenAI API (e.g., https://api.openai.com/v1) |
| `GEMINI_API_KEY` | API key for Google Gemini |
| `GEMINI_BASE_URL` | Base URL for the Gemini API (default: https://generativelanguage.googleapis.com/v1beta) |
| `ANTHROPIC_API_KEY` | API key for Anthropic Claude |
| `ANTHROPIC_BASE_URL` | Base URL for the Anthropic API (default: https://api.anthropic.com/v1) |
| `DEFAULT_MODEL` | Default model name |
| `CONTEXT_WINDOW` | Maximum context tokens (default: 8192) |

---

## LLM Provider Setup

### Ollama
1. Install from [ollama.com](https://ollama.com).
2. Pull a model: `ollama pull qwen2.5-coder:7b`.
3. Start the server: `ollama serve`.
4. Run: `agentchanti "task" --provider ollama --model qwen2.5-coder:7b`.

### LM Studio
1. Download from [lmstudio.ai](https://lmstudio.ai).
2. Load a model and click **Start Server**.
3. Default URL is `http://localhost:1234/v1`.
4. Run: `agentchanti "task"` (LM Studio is the default).

### OpenAI & Cloud Providers
Configure the `openai` section in your `.agentchanti.yaml`:
```yaml
openai:
  api_key: "your-api-key"
  base_url: "https://api.openai.com/v1"
```
Or use environment variables:
```bash
export OPENAI_API_KEY="your-key"
agentchanti "task" --provider openai --model gpt-4o
```

### Google Gemini
Google Gemini is supported as a first-class provider via its native REST API.
1. Get an API key from [Google AI Studio](https://aistudio.google.com/).
2. Configure in `.agentchanti.yaml`:
```yaml
gemini:
  api_key: "your-gemini-api-key"
  base_url: "https://generativelanguage.googleapis.com/v1beta"  # default
```
Or set the environment variable: `export GEMINI_API_KEY="your-key"`

3. Run:
```bash
agentchanti "task" --provider gemini --model gemini-2.5-flash
```

**Available models:** `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`

**Embeddings:** Gemini supports embeddings via the `text-embedding-004` model.

### Anthropic Claude
Anthropic Claude is supported as a first-class provider via the native Messages API.
1. Get an API key from the [Anthropic Console](https://console.anthropic.com/).
2. Configure in `.agentchanti.yaml`:
```yaml
anthropic:
  api_key: "sk-ant-your-key-here"
  base_url: "https://api.anthropic.com/v1"  # default
```
Or set the environment variable: `export ANTHROPIC_API_KEY="your-key"`

3. Run:
```bash
agentchanti "task" --provider anthropic --model claude-sonnet-4
```

**Available models:** `claude-sonnet-4`, `claude-haiku-4`

**Embeddings:** Anthropic does not provide an embedding API. Use `--no-embeddings` or configure a different provider for embeddings.

---

## Command Line Interface (CLI)

| Option | Description |
|--------|-------------|
| `--provider` | Choose `ollama`, `lm_studio`, `openai`, `gemini`, or `anthropic`. |
| `--model` | Model name (e.g., `gpt-4o`, `qwen2.5-coder:7b`). |
| `--config` | Path to a specific YAML config file. |
| `--no-diff` | Skip showing the file diff preview. |
| `--no-cache` | Disable step-level caching. |
| `--no-knowledge` | Disable the project knowledge base. |
| `--report` | Generate an HTML report (enabled by default). |
| `--auto` | Non-interactive mode (auto-approves plans). |
| `--generate-yaml` | Generate a `.agentchanti.yaml` file with current settings and exit. |

---

## Key Features

### TUI Plan Editor
When the planner generates a plan, you'll be offered an **[E]dit (TUI)** option.
- **Arrows**: Navigate steps.
- **'e'**: Edit a step's text.
- **'d'**: Delete a step.
- **'a'**: Add a new step.
- **Shift+K/J**: Reorder steps.
- **Enter**: Approve and start.

### HTML Reports
Detailed reports are generated in `.agentchanti/reports/`. They include:
- Task summary and token usage.
- Step-by-step execution status.
- Colored diffs for all modified files.

### Step Caching
AgentChanti caches successful LLM responses based on a hash of the step text and relevant context. This allows you to restart tasks or run similar tasks much faster.
- Cache location: `.agentchanti/cache/`
- Clear cache: `agentchanti --clear-cache "some task"`

### Persistent Embeddings
Project file embeddings are cached in a SQLite database (`.agentchanti/embeddings.db`). This avoids re-embedding unchanged files during the project scan phase.

### Knowledge Base
Learnings from successful runs (bug fixes, project patterns) are stored in `.agentchanti/knowledge.json`. These are injected into the Planner's context for future runs to improve accuracy.

### Cost Tracking & Budget Limits
AgentChanti monitors your token usage and calculates costs for cloud providers.
- **Budget Enforcment**: Set `budget_limit` in your config. The system will halt immediately if the limit is reached.
- **Pricing Configuration**: Costs are calculated based on the `pricing` dictionary in your YAML. Default prices for OpenAI, Gemini, and Claude models are built-in.
- **Report Integration**: Total cost is displayed in both the terminal summary and the generated HTML reports.

---

## Plugin System
You can add custom step handlers by creating a class:

```python
from multi_agent_coder.plugins import StepPlugin, PluginContext

class DeplopyPlugin(StepPlugin):
    name = "DEPLOY"
    def can_handle(self, step_text: str) -> bool:
        return "deploy" in step_text.lower()
    
    def handle(self, step_text: str, ctx: PluginContext) -> tuple[bool, str]:
        # Custom logic here
        return True, ""
```

Register it in `.agentchanti.yaml`:
```yaml
plugins:
  - my_app.plugins.DeployPlugin
```
