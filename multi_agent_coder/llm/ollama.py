import json
import requests
from typing import List, Optional

from .base import LLMClient
from ..cli_display import token_tracker, log


class OllamaClient(LLMClient):

    def __init__(self, base_url: str, model: str, **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url
        self.model = model
        # Derive the API root for endpoints like /api/embed
        if "/api/" in base_url:
            self._api_root = base_url.rsplit("/api/", 1)[0]
        else:
            self._api_root = base_url.rstrip("/")

    # ── Non-streaming generation ──

    def _generate(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[Ollama] Sending ~{est_tokens} est. tokens")
        log.debug(f"[Ollama] Prompt:\n{prompt}")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(self.base_url, json=payload, timeout=(10, 300))
        response.raise_for_status()
        data = response.json()
        result = data.get("response", "")

        prompt_tokens = data.get("prompt_eval_count", est_tokens)
        completion_tokens = data.get("eval_count", 0)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else 0,
        )
        log.debug(f"[Ollama] Usage: prompt={prompt_tokens} completion={completion_tokens}")
        log.debug(f"[Ollama] Response:\n{result}")
        return result

    # ── Streaming generation ──

    def _generate_stream(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[Ollama] Streaming ~{est_tokens} est. tokens")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }
        content_parts: list[str] = []
        tokens_generated = 0
        prompt_tokens = est_tokens

        response = requests.post(self.base_url, json=payload,
                                 stream=True, timeout=(10, 120))
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    content_parts.append(token)
                    tokens_generated += 1
                    if self._stream_callback and tokens_generated % 10 == 0:
                        self._stream_callback(tokens_generated)

                # Final chunk contains token counts
                if chunk.get("done", False):
                    prompt_tokens = chunk.get("prompt_eval_count", est_tokens)
                    eval_count = chunk.get("eval_count", tokens_generated)
                    tokens_generated = eval_count if isinstance(eval_count, int) else tokens_generated
            except (json.JSONDecodeError, KeyError):
                continue

        result = "".join(content_parts)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            tokens_generated,
        )
        log.debug(f"[Ollama] Streamed {tokens_generated} tokens")
        log.debug(f"[Ollama] Response:\n{result}")

        if self._stream_callback:
            self._stream_callback(tokens_generated)

        return result

    # ── Embeddings (unchanged) ──

    def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        embed_model = model or self.model
        url = f"{self._api_root}/api/embed"
        payload = {"model": embed_model, "input": text}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [[]])
            return embeddings[0] if embeddings else []
        except requests.exceptions.RequestException as e:
            log.error(f"[Ollama] Embedding error: {e}")
            return []
