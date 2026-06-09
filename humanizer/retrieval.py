"""
Retrieval service for reference-augmented humanization.

Provides `ReferenceEntry` — a single passage from the reference corpus — and
`RetrievalService` — an in-memory retrieval engine that ranks corpus entries
by cosine similarity to a query using precomputed embeddings and NumPy
brute-force top-k.

Requirements: 7.1, 7.2, 7.4
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np


# Default corpus path: humanizer/data/reference_corpus.json
_DEFAULT_CORPUS_PATH = str(
    Path(__file__).parent / "data" / "reference_corpus.json"
)


class Embedder(Protocol):
    """Protocol for an embedder compatible with SimilarityEvaluator's model.

    Any object exposing ``encode(text) -> np.ndarray`` satisfies this.
    """

    def encode(self, text: str) -> np.ndarray: ...


@dataclass
class ReferenceEntry:
    """A single passage from the reference corpus.

    Attributes:
        id: Unique identifier for the entry.
        text: The human-written passage text.
        source: Provenance or license note for the passage.
        embedding: Precomputed embedding vector; lazily built if missing.
    """

    id: str
    text: str
    source: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


class RetrievalService:
    """In-memory retrieval engine over a reference corpus.

    Loads a JSON corpus file, optionally precomputes embeddings via an
    injected embedder, and serves top-k retrieval queries ranked by cosine
    similarity using NumPy brute-force search.

    Parameters
    ----------
    corpus_path : str or None
        Path to the JSON corpus file. Defaults to the bundled
        ``humanizer/data/reference_corpus.json``.
    embedder : Embedder or None
        An object with an ``encode(text) -> np.ndarray`` method used to
        compute embeddings. If None, a simple bag-of-words fallback is used.
    max_results : int
        Maximum number of results to return per query (default: 10).
    """

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        max_results: int = 10,
    ) -> None:
        self._corpus_path = corpus_path or _DEFAULT_CORPUS_PATH
        self._embedder = embedder
        self._max_results = max_results
        self._entries: List[ReferenceEntry] = []
        self._embedding_matrix: Optional[np.ndarray] = None

        self._load_corpus()
        self._build_embeddings()

    @property
    def corpus(self) -> List[ReferenceEntry]:
        """Return the loaded corpus entries."""
        return list(self._entries)

    def retrieve(self, query_text: str) -> List[ReferenceEntry]:
        """Retrieve up to ``max_results`` entries ranked by cosine relevance.

        Parameters
        ----------
        query_text : str
            The text to use as a retrieval query.

        Returns
        -------
        List[ReferenceEntry]
            Up to ``max_results`` entries sorted by non-increasing cosine
            similarity to the query embedding. Returns an empty list when the
            corpus is empty, the query is empty, or no results can be computed.
        """
        if not self._entries or not query_text or not query_text.strip():
            return []

        if self._embedding_matrix is None:
            return []

        # Compute query embedding
        query_embedding = self._compute_embedding(query_text)
        if query_embedding is None:
            return []

        # Compute cosine similarities via NumPy brute-force
        scores = self._cosine_similarities(query_embedding, self._embedding_matrix)

        # Get top-k indices sorted by non-increasing score
        k = min(self._max_results, len(self._entries))
        # Use argpartition for efficiency, then sort the top-k
        if k >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [self._entries[i] for i in top_indices]

    def _load_corpus(self) -> None:
        """Load corpus entries from the JSON file."""
        if not os.path.isfile(self._corpus_path):
            self._entries = []
            return

        try:
            with open(self._corpus_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._entries = []
            return

        if not isinstance(data, list):
            self._entries = []
            return

        entries = []
        for item in data:
            if not isinstance(item, dict):
                continue
            entry_id = item.get("id", "")
            text = item.get("text", "")
            source = item.get("source", "")
            if text:  # Skip entries with empty text
                entries.append(
                    ReferenceEntry(id=str(entry_id), text=text, source=source)
                )

        self._entries = entries

    def _build_embeddings(self) -> None:
        """Precompute embeddings for all corpus entries."""
        if not self._entries:
            self._embedding_matrix = None
            return

        embeddings = []
        for entry in self._entries:
            emb = self._compute_embedding(entry.text)
            if emb is not None:
                entry.embedding = emb
                embeddings.append(emb)
            else:
                # If any entry fails to embed, skip embedding-based retrieval
                self._embedding_matrix = None
                return

        if embeddings:
            self._embedding_matrix = np.vstack(embeddings)
        else:
            self._embedding_matrix = None

    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding for a single text using the embedder or fallback."""
        if self._embedder is not None:
            try:
                result = self._embedder.encode(text)
                if result is not None:
                    return np.asarray(result, dtype=np.float32).flatten()
            except Exception:
                pass
            return None

        # Fallback: simple bag-of-words vector (word frequency based)
        return self._bow_embedding(text)

    def _bow_embedding(self, text: str) -> np.ndarray:
        """Simple bag-of-words embedding fallback.

        Creates a fixed-size vector by hashing words into buckets and
        counting occurrences. This provides a basic similarity signal
        without requiring any ML model.
        """
        vocab_size = 1000  # Fixed vector dimension
        vec = np.zeros(vocab_size, dtype=np.float32)

        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            # Hash word to a bucket index
            bucket = hash(word) % vocab_size
            vec[bucket] += 1.0

        # L2 normalize to unit vector for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    @staticmethod
    def _cosine_similarities(
        query: np.ndarray, matrix: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between a query vector and a matrix of vectors.

        Parameters
        ----------
        query : np.ndarray
            1-D query embedding vector.
        matrix : np.ndarray
            2-D matrix where each row is a corpus embedding.

        Returns
        -------
        np.ndarray
            1-D array of cosine similarity scores, one per corpus entry.
        """
        # Ensure query is 1-D
        query = query.flatten()

        # Compute dot products
        dots = matrix @ query

        # Compute norms
        query_norm = np.linalg.norm(query)
        matrix_norms = np.linalg.norm(matrix, axis=1)

        # Avoid division by zero
        if query_norm == 0.0:
            return np.zeros(len(matrix), dtype=np.float32)

        # Cosine similarity = dot / (norm_a * norm_b)
        denominators = matrix_norms * query_norm
        # Replace zero denominators to avoid division by zero
        denominators = np.where(denominators == 0.0, 1.0, denominators)

        return dots / denominators
