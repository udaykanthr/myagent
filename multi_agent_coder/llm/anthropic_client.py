"""
Anthropic Claude LLM client — calls the Anthropic Messages API directly.
"""

import json
import requests
from typing import List, Optional

from .base import LLMClient
from ..cli_display import token_tracker, log


class AnthropicClient(LLMClient):

    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, base_url: str, model: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
        }

    # ── Non-streaming generation ──

    def _generate(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[Anthropic] Sending ~{est_tokens} est. tokens")
        log.debug(f"[Anthropic] Prompt:\n{prompt}")

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        url = f"{self.base_url}/messages"
        response = requests.post(url, headers=self._headers(), json=payload,
                                 timeout=(10, 300))
        response.raise_for_status()
        data = response.json()

        # Extract token counts
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", est_tokens)
        completion_tokens = usage.get("output_tokens", 0)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else 0,
            model_name=self.model,
        )
        log.debug(f"[Anthropic] Usage: prompt={prompt_tokens} completion={completion_tokens}")

        # Extract text from content blocks
        content_blocks = data.get("content", [])
        response_text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )
        log.debug(f"[Anthropic] Response:\n{response_text}")
        return response_text

    # ── Streaming generation ──

    def _generate_stream(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[Anthropic] Streaming ~{est_tokens} est. tokens")

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }
        url = f"{self.base_url}/messages"

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
                try:
                    event = json.loads(data_str)
                    event_type = event.get("type", "")

                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            token = delta.get("text", "")
                            if token:
                                content_parts.append(token)
                                tokens_generated += 1
                                if self._stream_callback and tokens_generated % 10 == 0:
                                    self._stream_callback(tokens_generated)

                    elif event_type == "message_delta":
                        # Final usage info
                        usage = event.get("usage", {})
                        output_tokens = usage.get("output_tokens")
                        if output_tokens:
                            tokens_generated = output_tokens

                    elif event_type == "message_stop":
                        break

                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        result = "".join(content_parts)
        token_tracker.record(est_tokens, tokens_generated, model_name=self.model)
        log.debug(f"[Anthropic] Streamed {tokens_generated} tokens")
        log.debug(f"[Anthropic] Response:\n{result}")

        if self._stream_callback:
            self._stream_callback(tokens_generated)

        return result

    # ── Embeddings ──

    def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        log.warning(
            "[Anthropic] Anthropic does not provide an embedding API. "
            "Consider using a different provider (e.g. OpenAI or Gemini) for embeddings."
        )
        return []
