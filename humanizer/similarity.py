"""
Semantic similarity evaluation with embedding-based scoring and lexical fallback.

Provides `SimilarityEvaluator` — a lazy-loading, caching evaluator that computes
cosine similarity between text pairs using sentence-transformer embeddings. When the
embedding model cannot be loaded (missing dependency, corrupt model, etc.), the
evaluator transparently falls back to a lexical-overlap proxy (token Jaccard) so that
similarity floors are still enforceable throughout the pipeline.

Requirements: 6.2, 14.1
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional, Tuple


class SimilarityEvaluator:
    """Compute semantic similarity between text pairs.

    Uses sentence-transformers embeddings with cosine similarity when available,
    falling back to token Jaccard similarity when the model cannot be loaded.

    Parameters
    ----------
    model_name : str
        The sentence-transformers model identifier to load (default: all-MiniLM-L6-v2).
    cache_size : int
        Maximum number of text embeddings to cache via LRU (default: 512).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 512,
    ) -> None:
        self._model_name = model_name
        self._cache_size = cache_size
        self._model: Optional[object] = None
        self._model_load_attempted: bool = False
        self._model_loaded: bool = False
        self.last_source: str = "lexical"  # "embedding" or "lexical"

        # Build a cache closure with the specified size.
        # We use a nested function so @lru_cache respects the configured size.
        @lru_cache(maxsize=cache_size)
        def _cached_encode(text: str) -> Optional[object]:
            """Encode text to an embedding vector, or None on failure."""
            if not self._model_loaded or self._model is None:
                return None
            try:
                return self._model.encode(text, convert_to_numpy=True)
            except Exception:
                return None

        self._cached_encode = _cached_encode

    def _load_model(self) -> None:
        """Attempt to load the sentence-transformers model (once)."""
        if self._model_load_attempted:
            return
        self._model_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self._model_loaded = True
        except Exception:
            # Any failure (ImportError, OSError, download failure, etc.)
            self._model = None
            self._model_loaded = False

    def is_available(self) -> bool:
        """Return True only if the embedding model loaded successfully.

        Triggers a lazy load on first call.
        """
        self._load_model()
        return self._model_loaded

    def score(self, a: str, b: str) -> float:
        """Compute similarity between two texts.

        Tries embedding-based cosine similarity first. Falls back to
        lexical-overlap proxy (token Jaccard) on model failure.

        The result is clamped to [0.0, 1.0]. After calling, inspect
        `self.last_source` to determine which method was used
        ("embedding" or "lexical").

        Parameters
        ----------
        a : str
            First text.
        b : str
            Second text.

        Returns
        -------
        float
            Similarity score in [0.0, 1.0].
        """
        # Attempt embedding-based scoring
        self._load_model()
        if self._model_loaded:
            try:
                emb_a = self._cached_encode(a)
                emb_b = self._cached_encode(b)
                if emb_a is not None and emb_b is not None:
                    cos_sim = self._cosine_similarity(emb_a, emb_b)
                    self.last_source = "embedding"
                    return self._clamp(cos_sim)
            except Exception:
                pass

        # Fallback to lexical proxy
        self.last_source = "lexical"
        return self._clamp(self._lexical_similarity(a, b))

    @staticmethod
    def _cosine_similarity(vec_a, vec_b) -> float:
        """Compute cosine similarity between two numpy vectors."""
        import numpy as np

        dot = float(np.dot(vec_a, vec_b))
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp a value to [0.0, 1.0]."""
        return max(0.0, min(1.0, value))

    @staticmethod
    def _tokenize(text: str) -> set:
        """Tokenize text into a set of lowercased word tokens."""
        return set(re.findall(r"\b\w+\b", text.lower()))

    @classmethod
    def _lexical_similarity(cls, a: str, b: str) -> float:
        """Compute token Jaccard similarity (intersection / union).

        Returns 1.0 for identical token sets, 0.0 for completely disjoint.
        Handles empty texts gracefully (returns 1.0 if both empty, 0.0 otherwise).
        """
        tokens_a = cls._tokenize(a)
        tokens_b = cls._tokenize(b)

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)
