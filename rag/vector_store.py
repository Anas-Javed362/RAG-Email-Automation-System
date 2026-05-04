"""
Vector store abstraction supporting FAISS (default) and ChromaDB.

Manages:
  - Index creation and loading from disk
  - Adding new email vectors with metadata
  - Top-k similarity search
  - Persistence to disk
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

from config.settings import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class FaissVectorStore:
    """
    FAISS-backed vector store with a parallel metadata registry.

    Attributes:
        index           FAISS IndexFlatIP (inner-product / cosine for normalized vecs)
        metadata_store  List of dicts, indexed by FAISS row ID
    """

    def __init__(self, dimension: int, index_path: str) -> None:
        """
        Initialize or load an existing FAISS index from disk.

        Args:
            dimension:   Dimensionality of embedding vectors.
            index_path:  Base path (without extension) for index + metadata files.
        """
        self.dimension = dimension
        self._index_path = Path(index_path)
        self._meta_path = self._index_path.with_suffix(".meta.pkl")

        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        if self._index_path.with_suffix(".idx").exists():
            self._load()
        else:
            logger.info(f"Creating new FAISS index (dim={dimension})")
            # IndexFlatIP with normalized vectors ≈ cosine similarity
            self.index = faiss.IndexFlatIP(dimension)
            self.metadata_store: list[dict] = []

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load FAISS index and metadata from disk."""
        idx_file = self._index_path.with_suffix(".idx")
        logger.info(f"Loading FAISS index from {idx_file}")
        self.index = faiss.read_index(str(idx_file))

        if self._meta_path.exists():
            with open(self._meta_path, "rb") as f:
                self.metadata_store = pickle.load(f)
        else:
            self.metadata_store = []

        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        idx_file = self._index_path.with_suffix(".idx")
        faiss.write_index(self.index, str(idx_file))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.metadata_store, f)
        logger.info(f"FAISS index saved: {self.index.ntotal} vectors at {idx_file}")

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(self, vector: np.ndarray, metadata: dict) -> int:
        """
        Add a single embedding with associated metadata.

        Args:
            vector:    1D float32 numpy array.
            metadata:  Dict to store alongside the vector (e.g., body_cleaned, category).

        Returns:
            FAISS row ID (== position in metadata_store).
        """
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        vector = vector.astype(np.float32)
        self.index.add(vector)

        vector_id = len(self.metadata_store)
        self.metadata_store.append(metadata)

        logger.debug(f"Added vector ID {vector_id} | meta keys: {list(metadata.keys())}")
        self.save()
        return vector_id

    def add_batch(self, vectors: np.ndarray, metadatas: list[dict]) -> list[int]:
        """
        Add multiple embeddings in one shot (much faster than looping add()).

        Args:
            vectors:    2D float32 array of shape (n, dim).
            metadatas:  Parallel list of metadata dicts.

        Returns:
            List of assigned FAISS row IDs.
        """
        if len(vectors) != len(metadatas):
            raise ValueError("vectors and metadatas must have the same length.")

        vectors = vectors.astype(np.float32)
        start_id = len(self.metadata_store)
        self.index.add(vectors)
        self.metadata_store.extend(metadatas)
        ids = list(range(start_id, start_id + len(metadatas)))

        logger.info(f"Batch added {len(ids)} vectors (IDs {ids[0]}–{ids[-1]})")
        self.save()
        return ids

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Find the top-k most similar vectors to the query.

        Args:
            query_vector:  1D float32 query embedding.
            top_k:         Number of results to return.

        Returns:
            List of dicts, each containing:
                - 'vector_id'  (int)
                - 'score'      (float, higher = more similar, max 1.0 for cosine)
                - all keys from the stored metadata
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty — no results to retrieve.")
            return []

        k = min(top_k, self.index.ntotal)
        query = query_vector.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for padded empty slots
            entry = {"vector_id": int(idx), "score": float(score)}
            entry.update(self.metadata_store[idx])
            results.append(entry)

        logger.debug(f"FAISS search returned {len(results)} results (top score: {results[0]['score']:.4f})")
        return results

    @property
    def total_vectors(self) -> int:
        """Total number of vectors stored in the index."""
        return self.index.ntotal


# ── Singleton ──────────────────────────────────────────────────────────────────

_vector_store_instance: Optional[FaissVectorStore] = None


def get_vector_store() -> FaissVectorStore:
    """
    Return a lazily-initialized singleton vector store.

    Uses settings to determine index path and embedding dimension.
    """
    global _vector_store_instance

    if _vector_store_instance is None:
        from rag.embedder import get_embedding_dimension
        settings = get_settings()
        dim = get_embedding_dimension()
        _vector_store_instance = FaissVectorStore(
            dimension=dim,
            index_path=settings.faiss_index_path,
        )
        logger.info(f"Vector store initialized: {_vector_store_instance.total_vectors} existing vectors.")

    return _vector_store_instance
