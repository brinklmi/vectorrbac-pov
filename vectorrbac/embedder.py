"""Embedding module with graceful fallback.

Uses sentence-transformers (all-MiniLM-L6-v2) when available and model is cached.
Falls back to deterministic hash-based synthetic embeddings when not.

The synthetic embeddings preserve semantic relationships for POC purposes:
- Same text → same embedding
- Similar text → similar embedding (via character n-gram hashing)
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

# Try to load sentence-transformers
_model = None
_model_name = "all-MiniLM-L6-v2"
_model_dim = 384  # MiniLM output dimension
_fallback_dim = 64  # Synthetic embedding dimension

try:
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


def _load_model() -> bool:
    """Attempt to load the sentence-transformers model (lazy, cached)."""
    global _model
    if _model is not None:
        return True
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return False
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_model_name)
        return True
    except Exception:
        return False


def get_embedding_dimension() -> int:
    """Get the embedding dimension (depends on whether real model is available)."""
    if _load_model():
        return _model_dim
    return _fallback_dim


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts.

    Uses real sentence-transformers if available, otherwise falls back
    to deterministic synthetic embeddings.

    Parameters
    ----------
    texts : list[str]
        Texts to embed.

    Returns
    -------
    list[list[float]]
        List of embedding vectors (normalized to unit length).
    """
    if _load_model() and _model is not None:
        return _embed_real(texts)
    return _embed_synthetic(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query text.

    Parameters
    ----------
    text : str
        Query text.

    Returns
    -------
    list[float]
        Embedding vector (normalized).
    """
    results = embed_texts([text])
    return results[0]


def _embed_real(texts: list[str]) -> list[list[float]]:
    """Embed using sentence-transformers model."""
    embeddings = _model.encode(texts, normalize_embeddings=True)
    return [emb.tolist() for emb in embeddings]


def _embed_synthetic(texts: list[str]) -> list[list[float]]:
    """Generate deterministic synthetic embeddings from text.

    Uses character n-gram hashing to produce embeddings that preserve
    some semantic similarity (texts with shared n-grams will have
    higher cosine similarity).
    """
    dim = _fallback_dim
    embeddings: list[list[float]] = []

    for text in texts:
        emb = _text_to_vector(text, dim)
        embeddings.append(emb.tolist())

    return embeddings


def _text_to_vector(text: str, dim: int) -> np.ndarray:
    """Convert text to a deterministic normalized vector via n-gram hashing."""
    vec = np.zeros(dim, dtype=np.float32)

    # Use character 3-grams
    text_lower = text.lower().strip()
    if len(text_lower) < 3:
        # Very short text: hash the whole thing
        h = hashlib.md5(text_lower.encode()).digest()
        for i in range(min(dim, 16)):
            vec[i % dim] += float(h[i]) / 255.0
    else:
        for i in range(len(text_lower) - 2):
            ngram = text_lower[i:i+3]
            h = hashlib.md5(ngram.encode()).digest()
            # Distribute hash across vector dimensions
            idx = int.from_bytes(h[:2], 'big') % dim
            val = (int.from_bytes(h[2:4], 'big') / 65535.0) * 2 - 1  # [-1, 1]
            vec[idx] += val

    # Normalize to unit length
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


def is_real_model_available() -> bool:
    """Check if the real sentence-transformers model is available."""
    return _load_model()
