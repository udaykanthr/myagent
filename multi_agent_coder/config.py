import os


class Config:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
    LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-coder-v2-lite-instruct")
    CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "8192"))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    EMBEDDING_TOP_K = int(os.getenv("EMBEDDING_TOP_K", "5"))

    # LLM resilience
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "2.0"))
    STREAM_RESPONSES = os.getenv("STREAM_RESPONSES", "true").lower() == "true"

    # Checkpoint
    CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE", ".agentchanti_checkpoint.json")
