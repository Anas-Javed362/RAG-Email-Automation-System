"""
Text embedding module using sentence-transformers.

Converts cleaned email text into dense vector representations
suitable for FAISS similarity search.
"""

import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from config.settings import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    """
    Load and cache the sentence-transformer model.

    Loading is expensive (~1–3s), so we cache it for the lifetime of the process.

    Args:
        model_name: HuggingFace model identifier (e.g. 'all-MiniLM-L6-v2').

    Returns:
        Loaded SentenceTransformer instance.
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded successfully.")
    return model


def get_embedding_model() -> SentenceTransformer:
    """Return the cached embedding model using settings."""
    settings = get_settings()
    return _load_model(settings.embedding_model)


def embed_text(text: str) -> np.ndarray:
    """
    Convert a single text string into an embedding vector.

    Args:
        text: Cleaned email body or any text to embed.

    Returns:
        1D numpy array of shape (embedding_dim,), dtype float32.
    """
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    logger.debug(f"Embedded text of length {len(text)} → vector shape {vector.shape}")
    return vector.astype(np.float32)


def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts in a single batch pass (more efficient than looping).

    Args:
        texts: List of cleaned text strings.

    Returns:
        2D numpy array of shape (n, embedding_dim), dtype float32.

    Raises:
        ValueError: If texts list is empty.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    model = get_embedding_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 10,
        batch_size=32,
    )
    logger.info(f"Batch embedded {len(texts)} texts → shape {vectors.shape}")
    return vectors.astype(np.float32)


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding vectors."""
    return get_embedding_model().get_sentence_embedding_dimension()
