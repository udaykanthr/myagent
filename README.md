# Multi-Agent Coder Walkthrough

I have successfully created a modular multi-agent coder program in Python. It is designed to work with local LLMs like Ollama and LM Studio.

## Features
- **Modular Design**: Separate agents for Planning, Coding, and Reviewing.
- **Extensible LLM Interface**: Easy to add new LLM providers. Currently supports Ollama and LM Studio (OpenAI compatible).
- **Orchestration**: Automated workflow from user request to reviewed code.

## Setup

1.  **Install Dependencies**
    ```bash
    pip install requests
    ```

2.  **Configure LLM**
    -   **Ollama**: Ensure Ollama is running (`ollama serve`). The default URL is `http://localhost:11434/api/generate`.
    -   **LM Studio**: Start the local server in LM Studio. The default URL is `http://localhost:1234/v1`.

## Usage

Run the orchestrator script with your task:

```bash
# Using Ollama (Default)
python3 -m multi_agent_coder.orchestrator "Create a python script to scrape a website"

# Using LM Studio
python3 -m multi_agent_coder.orchestrator "Create a python script to scrape a website" --provider lm_studio
```

## Verification

I verified the system logic using a mock test that simulates LLM responses.

### Test Results
```
Ran 1 test in 0.001s

OK
```

The agents correctly pass data between each other:
1.  **Planner** receives the task and produces a plan.
2.  **Coder** receives the plan and produces code.
3.  **Reviewer** receives the code and produces a review.
