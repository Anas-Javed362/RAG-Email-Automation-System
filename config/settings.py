"""
Application settings — extended for production-grade features.

New in this version:
  - Prompt versioning
  - Retry + backoff configuration
  - Cache backend (in-memory / Redis)
  - LLM self-evaluation toggle
  - Thread history window size
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """Central configuration for the RAG Email Automation System."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider ───────────────────────────────────────────────────
    llm_provider: str = Field(default="openai", description="LLM backend: 'openai' or 'huggingface'")

    # ── OpenAI ─────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI chat model name")

    # ── HuggingFace ────────────────────────────────────────────────────
    huggingface_api_token: str = Field(default="", description="HuggingFace API token")
    huggingface_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description="HuggingFace model for inference"
    )

    # ── Embeddings ─────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model for text embeddings"
    )

    # ── Vector Store ───────────────────────────────────────────────────
    vector_store: str = Field(default="faiss", description="Vector DB backend: 'faiss' or 'chromadb'")
    faiss_index_path: str = Field(default="data/faiss_index", description="Path to persist FAISS index")
    chroma_persist_dir: str = Field(default="data/chroma_db", description="ChromaDB persistence directory")

    # ── Retrieval ──────────────────────────────────────────────────────
    top_k_retrieval: int = Field(default=5, description="Number of similar emails to retrieve")

    # ── Thread Awareness ───────────────────────────────────────────────
    thread_history_limit: int = Field(
        default=10,
        description="Max number of past messages to include from the same thread"
    )

    # ── Confidence Scoring ─────────────────────────────────────────────
    confidence_threshold: float = Field(
        default=0.65,
        description="Emails below this score are flagged for human review"
    )
    enable_llm_self_eval: bool = Field(
        default=True,
        description="Whether to ask the LLM to rate its own response confidence"
    )

    # ── Prompt Versioning ──────────────────────────────────────────────
    prompt_version: str = Field(
        default="v2",
        description="Which prompt template to use: 'v1' or 'v2'"
    )
    prompts_dir: str = Field(
        default="rag/prompts",
        description="Directory containing versioned prompt templates"
    )

    # ── Retry & Rate Limiting ──────────────────────────────────────────
    llm_max_retries: int = Field(default=3, description="Max retry attempts for LLM API calls")
    llm_retry_base_delay: float = Field(
        default=1.0,
        description="Base delay in seconds for exponential backoff"
    )
    llm_timeout_seconds: float = Field(default=30.0, description="LLM API call timeout in seconds")

    # ── Cache ──────────────────────────────────────────────────────────
    cache_backend: str = Field(
        default="memory",
        description="Cache backend: 'memory' (in-process dict) or 'redis'"
    )
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")

    # ── Database ───────────────────────────────────────────────────────
    database_url: str = Field(default="sqlite:///./data/emails.db", description="SQLAlchemy DB URL")

    # ── Application ────────────────────────────────────────────────────
    app_env: str = Field(default="development", description="Application environment")
    log_level: str = Field(default="INFO", description="Logging level")
    app_version: str = Field(default="2.0.0", description="Application version string")

    @property
    def data_dir(self) -> Path:
        """Ensure the data directory exists and return its path."""
        path = Path("data")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_openai(self) -> bool:
        return self.llm_provider.lower() == "openai"

    @property
    def is_huggingface(self) -> bool:
        return self.llm_provider.lower() == "huggingface"

    @property
    def is_redis_cache(self) -> bool:
        return self.cache_backend.lower() == "redis"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
