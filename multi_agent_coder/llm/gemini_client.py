"""
Google Gemini LLM client — calls the Gemini REST API directly.
"""

import json
import requests
from typing import List, Optional

from .base import LLMClient
from ..cli_display import token_tracker, log


class GeminiClient(LLMClient):

    def __init__(self, base_url: str, model: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    # ── Non-streaming generation ──

    def _generate(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[Gemini] Sending ~{est_tokens} est. tokens")
        log.debug(f"[Gemini] Prompt:\n{prompt}")

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
            },
        }
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        response = requests.post(url, json=payload, timeout=(10, 300))
        response.raise_for_status()
        data = response.json()

        # Extract token counts from usageMetadata
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", est_tokens)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        token_tracker.record(
            prompt_tokens if isinstance(prompt_tokens, int) else est_tokens,
            completion_tokens if isinstance(completion_tokens, int) else 0,
            model_name=self.model,
        )
        log.debug(f"[Gemini] Usage: prompt={prompt_tokens} completion={completion_tokens}")

        # Extract text from candidates
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        response_text = "".join(p.get("text", "") for p in parts)
        log.debug(f"[Gemini] Response:\n{response_text}")
        return response_text

    # ── Streaming generation ──

    def _generate_stream(self, prompt: str) -> str:
        est_tokens = int(len(prompt.split()) * 1.3)
        log.debug(f"[Gemini] Streaming ~{est_tokens} est. tokens")

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
            },
        }
        url = (
            f"{self.base_url}/models/{self.model}"
            f":streamGenerateContent?alt=sse&key={self.api_key}"
        )

        content_parts: list[str] = []
        tokens_generated = 0

        response = requests.post(url, json=payload, stream=True, timeout=(10, 120))
        response.raise_for_status()
        # Logging the headers as the body stream can't be read before iter_lines
        log.debug(f"[Gemini] Response Status: {response.status_code}, Headers: {dict(response.headers)}")
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    log.debug(f"[Gemini] Chunk: {chunk}")
                    candidates = chunk.get("candidates", [])
                    if not candidates:
                        continue
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        token = part.get("text", "")
                        if token:
                            content_parts.append(token)
                            tokens_generated += 1
                            if self._stream_callback and tokens_generated % 10 == 0:
                                self._stream_callback(tokens_generated)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        result = "".join(content_parts)
        token_tracker.record(est_tokens, tokens_generated, model_name=self.model)
        log.debug(f"[Gemini] Streamed {tokens_generated} tokens")
        log.debug(f"[Gemini] Response:\n{result}")

        if self._stream_callback:
            self._stream_callback(tokens_generated)

        return result

    # ── Embeddings ──

    # Models that are only available locally (not on the Gemini REST API).
    # If one of these is passed as *model*, fall back to the Gemini default.
    _LOCAL_ONLY_EMBED_MODELS = {
        "nomic-embed-text", "all-minilm", "mxbai-embed-large",
        "snowflake-arctic-embed", "bge-large", "bge-small",
    }

    def generate_embedding(self, text: str, model: Optional[str] = None, **kwargs) -> List[float]:
        # Ignore local-only model names that aren't valid on the Gemini API
        if model and (model in self._LOCAL_ONLY_EMBED_MODELS
                      or not model.startswith(("text-embedding", "embedding-", "models/", "gemini-embedding"))):
            log.warning(f"[Gemini] Embedding model '{model}' is not a valid "
                        f"Gemini API model, using 'text-embedding-004' instead")
            model = None
        embed_model = model or "text-embedding-004"
        url = (
            f"{self.base_url}/models/{embed_model}"
            f":embedContent?key={self.api_key}"
        )
        payload = {
            "model": f"models/{embed_model}",
            "content": {
                "parts": [{"text": text}]
            },
        }
        
        # Support specifying the embedding output dimension directly
        dimensions = kwargs.get("dimensions")
        if dimensions:
            payload["outputDimensionality"] = dimensions

        try:
            response = requests.post(url, json=payload, timeout=(10, 60))
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", {}).get("values", [])
        except requests.exceptions.RequestException as e:
            log.error(f"[Gemini] Embedding error: {e}")
            return []
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            log.error(f"[Gemini] Embedding parse error: {e}")
            return []
