"""
Checkpoint & resume — saves and restores pipeline state so interrupted
runs can pick up where they left off.
"""

import json
import os

DEFAULT_CHECKPOINT_FILE = ".agentchanti_checkpoint.json"


def save_checkpoint(filepath: str, task: str, steps: list[str],
                    completed_step: int, file_memory_dict: dict[str, str],
                    step_results: dict[int, str], language: str) -> None:
    """Persist current pipeline state to *filepath* as JSON."""
    state = {
        "task": task,
        "steps": steps,
        "completed_step": completed_step,
        "file_memory": file_memory_dict,
        "step_results": {str(k): v for k, v in step_results.items()},
        "language": language,
    }
    tmp = filepath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    # Atomic-ish rename (Windows: replaces if exists on Python 3.3+)
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(tmp, filepath)


def load_checkpoint(filepath: str) -> dict | None:
    """Load checkpoint state from *filepath*.

    Returns the state dict, or ``None`` if the file is missing or invalid.
    """
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        # Validate required keys
        required = {"task", "steps", "completed_step", "file_memory",
                     "step_results", "language"}
        if not required.issubset(state.keys()):
            return None
        # Convert step_results keys back to int
        state["step_results"] = {int(k): v for k, v in state["step_results"].items()}
        return state
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def clear_checkpoint(filepath: str) -> None:
    """Remove the checkpoint file after a successful run."""
    try:
        if os.path.isfile(filepath):
            os.remove(filepath)
    except OSError:
        pass
