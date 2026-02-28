import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from ..cli_display import log


class LLMError(Exception):
    """Raised when all LLM retries are exhausted."""


class LLMClient(ABC):

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0,
                 stream: bool = True):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stream = stream
        self._stream_callback: Optional[Callable[[int], None]] = None

    def set_stream_callback(self, callback: Callable[[int], None]) -> None:
        """Set a callback that receives ``(tokens_generated)`` during streaming."""
        self._stream_callback = callback

    # ── Public entry point ──

    def generate_response(self, prompt: str) -> str:
        """Generate a response with automatic retry and exponential backoff.

        Calls ``_generate_stream`` when streaming is enabled, otherwise
        ``_generate``.  Raises :class:`LLMError` after all retries are
        exhausted.
        """
        last_error: Exception | None = None
        use_stream = self.stream  # mutable — falls back on failure

        for attempt in range(1, self.max_retries + 1):
            try:
                if use_stream:
                    result = self._generate_stream(prompt)
                else:
                    result = self._generate(prompt)

                if not result or not result.strip():
                    log.warning(
                        f"[LLM] Empty response on attempt {attempt}/{self.max_retries}")
                    if attempt < self.max_retries:
                        # Jittered exponential backoff
                        import random
                        wait = self.retry_delay * (2 ** (attempt - 1))
                        jitter = wait * 0.1 * random.random()
                        time.sleep(wait + jitter)
                        continue
                    raise LLMError("LLM returned empty response after all retries")

                return result

            except LLMError:
                raise
            except Exception as e:
                last_error = e
                log.warning(
                    f"[LLM] Error on attempt {attempt}/{self.max_retries}: {e}")
                
                # If streaming failed, fall back to non-streaming for next retry
                if use_stream:
                    log.warning("[LLM] Streaming failed — falling back to non-streaming")
                    use_stream = False
                
                if attempt < self.max_retries:
                    # Jittered exponential backoff
                    import random
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    jitter = wait * 0.1 * random.random()
                    
                    # Special handling for 429: wait longer
                    if "429" in str(e):
                        wait *= 2
                        log.info(f"[LLM] Rate limit detected (429). Backing off for {wait:.1f}s")
                    
                    time.sleep(wait + jitter)

        raise LLMError(
            f"LLM failed after {self.max_retries} retries: {last_error}")

    # ── Subclass hooks ──

    @abstractmethod
    def _generate(self, prompt: str) -> str:
        """Synchronous (non-streaming) generation."""

    @abstractmethod
    def _generate_stream(self, prompt: str) -> str:
        """Streaming generation. Should call ``self._stream_callback``
        periodically with the number of tokens generated so far."""

    @abstractmethod
    def generate_embedding(self, text: str, model: Optional[str] = None, **kwargs) -> List[float]:
        """Generate an embedding vector for the given text."""
