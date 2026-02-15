import requests
import json
from .base import LLMClient

class OllamaClient(LLMClient):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def generate_response(self, prompt: str) -> str:
        # Estimate tokens sent (rough: ~1.3 tokens per word)
        est_tokens = int(len(prompt.split()) * 1.3)
        print(f"  [LLM] Sending ~{est_tokens} tokens...")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            data = response.json()
            result = data.get("response", "")

            # Ollama returns token counts in the response
            prompt_tokens = data.get("prompt_eval_count", "?")
            completion_tokens = data.get("eval_count", "?")
            print(f"  [LLM] Received: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")

            return result
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return ""
