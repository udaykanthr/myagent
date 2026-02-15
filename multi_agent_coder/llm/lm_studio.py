import requests
import json
from .base import LLMClient
from ..cli_display import token_tracker, log


class LMStudioClient(LLMClient):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def generate_response(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[LM Studio] Sending ~{est_tokens} est. tokens")
        log.debug(f"[LM Studio] Prompt:\n{prompt}")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
        headers = {"Content-Type": "application/json"}
        try:
            url = f"{self.base_url}/chat/completions"
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", est_tokens)
            completion_tokens = usage.get("completion_tokens", 0)
            token_tracker.record(
                prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
                completion_tokens if isinstance(completion_tokens, int) else 0
            )
            log.debug(f"[LM Studio] Usage: prompt={prompt_tokens} completion={completion_tokens}")

            response_text = data['choices'][0]['message']['content']
            log.debug(f"[LM Studio] Response:\n{response_text}")

            return response_text
        except requests.exceptions.RequestException as e:
            log.error(f"[LM Studio] Connection error: {e}")
            return ""
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            log.error(f"[LM Studio] Parse error: {e}")
            return ""
