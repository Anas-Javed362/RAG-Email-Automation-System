"""
In-memory + Redis caching layer.

Two cache types:
  1. EmbeddingCache  - Stores text → numpy vector mappings (avoids re-encoding)
  2. ResponseCache   - Stores hash(email_body) → full pipeline output (avoids LLM re-calls)

Backend auto-selects based on settings.cache_backend:
  - "memory" → thread-safe Python dict with TTL simulation (default)
  - "redis"  → Redis via the `redis` package (requires: pip install redis)
"""

import hashlib
import json
import time
import threading
from typing import Any, Optional

import numpy as np

from config.settings import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── In-Memory Cache ────────────────────────────────────────────────────────────

class _MemoryCache:
    """
    Thread-safe in-memory key/value store with TTL expiration.

    Items are lazily expired on access (no background thread needed).
    """

    def __init__(self, ttl_seconds: int) -> None:
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (value, time.monotonic() + self._ttl)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)


# ── Redis Cache ────────────────────────────────────────────────────────────────

class _RedisCache:
    """
    Redis-backed cache with automatic JSON serialization.

    Falls back to no-op if the Redis package isn't installed or connection fails.
    """

    def __init__(self, url: str, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        try:
            import redis
            self._client = redis.from_url(url, decode_responses=True)
            self._client.ping()
            logger.info(f"Redis cache connected: {url}")
        except Exception as exc:
            logger.warning(f"Redis cache unavailable ({exc}). Cache will be disabled.")
            self._client = None

    def get(self, key: str) -> Optional[Any]:
        if not self._client:
            return None
        try:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        if not self._client:
            return
        try:
            self._client.setex(key, self._ttl, json.dumps(value))
        except Exception:
            pass

    def delete(self, key: str) -> None:
        if not self._client:
            return
        try:
            self._client.delete(key)
        except Exception:
            pass


# ── Cache Service ──────────────────────────────────────────────────────────────

class CacheService:
    """
    Unified caching service for embeddings and LLM responses.

    Usage:
        cache = get_cache_service()

        # Embedding cache
        vec = cache.get_embedding("some text")
        cache.set_embedding("some text", vector)

        # Response cache
        result = cache.get_response(email_body)
        cache.set_response(email_body, result_dict)
    """

    _EMBEDDING_PREFIX = "emb:"
    _RESPONSE_PREFIX = "rsp:"

    def __init__(self) -> None:
        settings = get_settings()
        ttl = settings.cache_ttl_seconds

        if settings.is_redis_cache:
            self._backend = _RedisCache(settings.redis_url, ttl)
            logger.info("CacheService initialized with Redis backend.")
        else:
            self._backend = _MemoryCache(ttl)
            logger.info("CacheService initialized with in-memory backend.")

    # ── Embedding Cache ────────────────────────────────────────────────

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Retrieve a cached embedding vector for the given text.

        Returns None on cache miss.
        """
        key = self._EMBEDDING_PREFIX + self._hash(text)
        cached = self._backend.get(key)
        if cached is None:
            return None
        # Deserialize from list back to numpy array
        arr = np.array(cached, dtype=np.float32)
        logger.debug(f"Embedding cache HIT for text of length {len(text)}")
        return arr

    def set_embedding(self, text: str, vector: np.ndarray) -> None:
        """Cache a text embedding vector."""
        key = self._EMBEDDING_PREFIX + self._hash(text)
        # Serialize numpy array as a regular list (JSON-serializable)
        self._backend.set(key, vector.tolist())

    # ── Response Cache ─────────────────────────────────────────────────

    def get_response(self, email_body: str) -> Optional[dict]:
        """
        Retrieve a cached pipeline response for an email body.

        Cache key is the SHA-256 hash of the cleaned body, so identical
        emails don't trigger a full LLM call.

        Returns None on cache miss.
        """
        key = self._RESPONSE_PREFIX + self._hash(email_body)
        cached = self._backend.get(key)
        if cached:
            logger.debug(f"Response cache HIT for body hash={self._hash(email_body)[:8]}")
        return cached

    def set_response(self, email_body: str, response: dict) -> None:
        """Cache the full pipeline response for an email body."""
        key = self._RESPONSE_PREFIX + self._hash(email_body)
        self._backend.set(key, response)

    @staticmethod
    def _hash(text: str) -> str:
        """Compute a stable SHA-256 hex digest of a string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @property
    def backend_name(self) -> str:
        """Human-readable name of the active cache backend."""
        return type(self._backend).__name__.replace("_", "").replace("Cache", "").lower()


# ── Singleton ──────────────────────────────────────────────────────────────────

_cache_instance: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Return a lazily-initialized singleton CacheService."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheService()
    return _cache_instance
