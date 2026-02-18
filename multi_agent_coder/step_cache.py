"""
Step-Level Cache — hash-based LLM response caching.

Caches LLM responses by hashing the step text + context + model name.
Cache entries are stored as JSON files and expire after a configurable TTL.
"""

import hashlib
import json
import os
import time

from .cli_display import log


class StepCache:
    """Disk-backed cache for LLM step responses."""

    def __init__(self, cache_dir: str = ".agentchanti/cache",
                 ttl_hours: int = 24):
        self._cache_dir = cache_dir
        self._ttl_seconds = ttl_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)

    def _hash_key(self, step_text: str, context: str, model: str) -> str:
        """Generate a unique hash for the step inputs."""
        combined = f"{step_text}|{context}|{model}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _cache_path(self, hash_key: str) -> str:
        return os.path.join(self._cache_dir, f"{hash_key}.json")

    def get(self, step_text: str, context: str, model: str) -> str | None:
        """Return cached response or None if miss/expired."""
        hash_key = self._hash_key(step_text, context, model)
        path = self._cache_path(hash_key)

        if not os.path.isfile(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)

            # Check expiry
            timestamp = entry.get("timestamp", 0)
            if time.time() - timestamp > self._ttl_seconds:
                log.debug(f"[StepCache] Expired entry: {hash_key[:12]}...")
                os.remove(path)
                return None

            log.info(f"[StepCache] Cache hit: {hash_key[:12]}... "
                     f"(step: {step_text[:50]})")
            return entry.get("response")

        except (json.JSONDecodeError, OSError, KeyError) as e:
            log.warning(f"[StepCache] Read error: {e}")
            return None

    def put(self, step_text: str, context: str, model: str, response: str):
        """Store a response in the cache."""
        hash_key = self._hash_key(step_text, context, model)
        path = self._cache_path(hash_key)

        entry = {
            "step_text": step_text[:200],  # store truncated for debugging
            "model": model,
            "timestamp": time.time(),
            "response": response,
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2)
            log.debug(f"[StepCache] Stored: {hash_key[:12]}...")
        except OSError as e:
            log.warning(f"[StepCache] Write error: {e}")

    def clear(self):
        """Remove all cached entries."""
        count = 0
        try:
            for fname in os.listdir(self._cache_dir):
                if fname.endswith(".json"):
                    os.remove(os.path.join(self._cache_dir, fname))
                    count += 1
            log.info(f"[StepCache] Cleared {count} entries")
        except OSError as e:
            log.warning(f"[StepCache] Clear error: {e}")
        return count

    @property
    def size(self) -> int:
        """Number of cached entries."""
        try:
            return sum(1 for f in os.listdir(self._cache_dir)
                       if f.endswith(".json"))
        except OSError:
            return 0
