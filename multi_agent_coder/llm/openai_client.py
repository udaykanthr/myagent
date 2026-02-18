"""
OpenAI-compatible LLM client — works with OpenAI, Groq, Together.ai,
and any other provider that implements the OpenAI chat/completions API.
"""

import json
import requests
from typing import List, Optional

from .base import LLMClient
from ..cli_display import token_tracker, log


class OpenAIClient(LLMClient):

    def __init__(self, base_url: str, model: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    # ── Non-streaming generation ──

    def _generate(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[OpenAI] Sending ~{est_tokens} est. tokens")
        log.debug(f"[OpenAI] Prompt:\n{prompt}")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "stream": False,
        }
        url = f"{self.base_url}/chat/completions"
        response = requests.post(url, headers=self._headers(), json=payload,
                                 timeout=(10, 300))
        response.raise_for_status()
        data = response.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", est_tokens)
        completion_tokens = usage.get("completion_tokens", 0)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else 0,
            model_name=self.model
        )
        log.debug(f"[OpenAI] Usage: prompt={prompt_tokens} completion={completion_tokens}")

        response_text = data["choices"][0]["message"]["content"]
        log.debug(f"[OpenAI] Response:\n{response_text}")
        return response_text

    # ── Streaming generation ──

    def _generate_stream(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[OpenAI] Streaming ~{est_tokens} est. tokens")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "stream": True,
        }
        url = f"{self.base_url}/chat/completions"

        content_parts: list[str] = []
        tokens_generated = 0

        response = requests.post(url, headers=self._headers(), json=payload,
                                 stream=True, timeout=(10, 120))
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        content_parts.append(token)
                        tokens_generated += 1
                        if self._stream_callback and tokens_generated % 10 == 0:
                            self._stream_callback(tokens_generated)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        result = "".join(content_parts)
        token_tracker.record(est_tokens, tokens_generated, model_name=self.model)
        log.debug(f"[OpenAI] Streamed {tokens_generated} tokens")
        log.debug(f"[OpenAI] Response:\n{result}")

        if self._stream_callback:
            self._stream_callback(tokens_generated)

        return result

    # ── Embeddings ──

    def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        embed_model = model or self.model
        url = f"{self.base_url}/embeddings"
        payload = {"model": embed_model, "input": text}
        try:
            response = requests.post(url, headers=self._headers(), json=payload)
            response.raise_for_status()
            data = response.json()
            items = data.get("data", [])
            if items:
                return items[0].get("embedding", [])
            return []
        except requests.exceptions.RequestException as e:
            log.error(f"[OpenAI] Embedding error: {e}")
            return []
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            log.error(f"[OpenAI] Embedding parse error: {e}")
            return []
