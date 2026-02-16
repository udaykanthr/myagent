from abc import ABC, abstractmethod
from ..llm.base import LLMClient

class Agent(ABC):
    def __init__(self, name: str, role: str, goal: str, llm_client: LLMClient):
        self.name = name
        self.role = role
        self.goal = goal
        self.llm_client = llm_client

    @abstractmethod
    def process(self, task: str, context: str = "") -> str:
        """
        Process the given task and return the result.
        """
        pass

    def _build_prompt(self, task: str, context: str, language: str | None = None) -> str:
        prompt = f"Role: {self.role}\nGoal: {self.goal}\n\n"
        if language:
            from ..language import get_language_name
            prompt += f"Language: {get_language_name(language)}\n\n"
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += f"Task: {task}\n\nResponse:"
        return prompt
