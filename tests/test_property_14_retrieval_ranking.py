"""
Property 14: Retrieval top-k ranking.

For all non-empty queries and in-memory corpora with known embeddings, the
RetrievalService returns at most 10 passages ordered by non-increasing relevance
(cosine similarity).

Requirements: 7.2
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from humanizer.retrieval import RetrievalService, Embedder


# ---------------------------------------------------------------------------
# Fake embedder with pre-set embeddings
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """An embedder that returns pre-set embeddings for known texts and a fixed
    query embedding.

    For corpus entries, embeddings are looked up by text content.
    For query text, the designated query embedding is returned.
    """

    def __init__(
        self, corpus_embeddings: Dict[str, np.ndarray], query_embedding: np.ndarray
    ) -> None:
        self._corpus_embeddings = corpus_embeddings
        self._query_embedding = query_embedding
        self._query_texts: set = set()

    def set_query_text(self, query_text: str) -> None:
        """Register a text as a query so it gets the query embedding."""
        self._query_texts.add(query_text)

    def encode(self, text: str) -> np.ndarray:
        """Return the pre-set embedding for the given text."""
        if text in self._query_texts:
            return self._query_embedding
        if text in self._corpus_embeddings:
            return self._corpus_embeddings[text]
        # For unknown texts (queries not registered), return the query embedding
        return self._query_embedding


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy for corpus sizes: 1 to 20 entries (test both under and over 10)
corpus_size_strategy = st.integers(min_value=1, max_value=20)

# Strategy for embedding dimensions
embedding_dim_strategy = st.integers(min_value=3, max_value=16)

# Strategy for query text (non-empty)
query_text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
    min_size=3,
    max_size=50,
).filter(lambda t: t.strip() != "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _build_corpus_file(entries: list, tmp_dir: str) -> str:
    """Write a corpus JSON file and return the path."""
    corpus_path = os.path.join(tmp_dir, "test_corpus.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(entries, f)
    return corpus_path


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

# Feature: ultimate-humanizer, Property 14: Retrieval top-k ranking


@given(
    corpus_size=corpus_size_strategy,
    dim=embedding_dim_strategy,
    query_text=query_text_strategy,
    data=st.data(),
)
@settings(max_examples=100)
def test_retrieval_top_k_ranking(
    corpus_size: int, dim: int, query_text: str, data: st.DataObject
) -> None:
    """Property 14: Retrieval top-k ranking.

    For all non-empty queries and in-memory corpora with known embeddings,
    the service returns at most 10 passages ordered by non-increasing relevance.

    Validates: Requirements 7.2
    """
    # Generate random embeddings for corpus entries and a query embedding
    # Use data strategy for reproducibility with hypothesis
    rng = np.random.default_rng(
        data.draw(st.integers(min_value=0, max_value=2**31 - 1), label="seed")
    )

    # Build corpus entries with unique texts and known embeddings
    corpus_entries = []
    corpus_embeddings: Dict[str, np.ndarray] = {}

    for i in range(corpus_size):
        text = f"Passage {i}: This is a unique corpus entry number {i}."
        # Generate a random non-zero embedding
        emb = rng.standard_normal(dim).astype(np.float32)
        # Ensure non-zero norm
        if np.linalg.norm(emb) < 1e-8:
            emb[0] = 1.0
        corpus_entries.append(
            {"id": f"entry_{i}", "text": text, "source": f"source_{i}"}
        )
        corpus_embeddings[text] = emb

    # Generate query embedding (non-zero)
    query_embedding = rng.standard_normal(dim).astype(np.float32)
    if np.linalg.norm(query_embedding) < 1e-8:
        query_embedding[0] = 1.0

    # Create fake embedder
    fake_embedder = FakeEmbedder(corpus_embeddings, query_embedding)

    # Write corpus to a temp file
    with tempfile.TemporaryDirectory() as tmp_dir:
        corpus_path = _build_corpus_file(corpus_entries, tmp_dir)

        # Instantiate the RetrievalService with the fake embedder
        service = RetrievalService(
            corpus_path=corpus_path, embedder=fake_embedder, max_results=10
        )

        # Perform retrieval
        results = service.retrieve(query_text)

    # --- Property assertions ---

    # 1. At most 10 results returned
    assert len(results) <= 10, (
        f"Expected at most 10 results, got {len(results)}"
    )

    # 2. Results are ordered by non-increasing cosine similarity
    if len(results) >= 2:
        similarities = []
        for entry in results:
            assert entry.embedding is not None, (
                f"Entry {entry.id} has no embedding"
            )
            sim = _cosine_similarity(query_embedding, entry.embedding)
            similarities.append(sim)

        for i in range(len(similarities) - 1):
            assert similarities[i] >= similarities[i + 1] - 1e-6, (
                f"Results not in non-increasing similarity order: "
                f"index {i} has similarity {similarities[i]:.6f} but "
                f"index {i+1} has similarity {similarities[i+1]:.6f}"
            )

    # 3. Number of results does not exceed corpus size
    assert len(results) <= corpus_size, (
        f"Got {len(results)} results from a corpus of size {corpus_size}"
    )
