"""
multi_agent_coder — Multi-Agent Local Coder (AgentChanti).

Public API for library usage::

    from multi_agent_coder import run_task, TaskResult

    result = run_task(task="Add logging to all endpoints", auto=True)
"""

from .api import run_task, TaskResult

__all__ = ["run_task", "TaskResult"]
