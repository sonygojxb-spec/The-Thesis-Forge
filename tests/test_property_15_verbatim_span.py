"""
Property 15: Verbatim-span bound.

For all stage outputs and retrieved passages, no span of more than 8
consecutive words from any retrieved passage appears verbatim in the output.

Requirements: 7.7

# Feature: ultimate-humanizer, Property 15: Verbatim-span bound
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from humanizer.retrieval import ReferenceEntry
from humanizer.stage_retrieval_augmented import RetrievalAugmentedRewriter


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRetrievalService:
    """A retrieval service that returns pre-configured passages.

    Provides a fixed set of passages for testing the verbatim-span guard
    without network or embedding dependencies.
    """

    def __init__(self, passages: List[ReferenceEntry]) -> None:
        self._passages = passages

    @property
    def corpus(self) -> List[ReferenceEntry]:
        """Return the stored passages as the corpus."""
        return list(self._passages)

    def retrieve(self, query_text: str) -> List[ReferenceEntry]:
        """Return all stored passages regardless of query."""
        return list(self._passages)


class FakeSimilarityEvaluator:
    """A similarity evaluator that always returns a high score (>= 0.85).

    Ensures the similarity floor does not interfere with the verbatim-span
    property test.
    """

    def __init__(self, score: float = 0.95) -> None:
        self._score = score

    def score(self, a: str, b: str) -> float:
        return self._score

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Known passages for testing
# ---------------------------------------------------------------------------

_KNOWN_PASSAGES = [
    ReferenceEntry(
        id="p1",
        text=(
            "The experimental results demonstrate a significant improvement "
            "in accuracy when compared to the baseline model across all "
            "evaluation metrics and test conditions."
        ),
        source="test",
        embedding=None,
    ),
    ReferenceEntry(
        id="p2",
        text=(
            "Our methodology leverages advanced machine learning techniques "
            "to effectively capture the underlying patterns in complex "
            "datasets while maintaining computational efficiency."
        ),
        source="test",
        embedding=None,
    ),
    ReferenceEntry(
        id="p3",
        text=(
            "Previous research has established that natural language processing "
            "models can achieve remarkable performance on downstream tasks "
            "when properly fine-tuned on domain-specific corpora."
        ),
        source="test",
        embedding=None,
    ),
]


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy to generate words that might overlap with passage vocabulary
_COMMON_WORDS = [
    "the", "a", "in", "of", "to", "and", "is", "that", "for", "on",
    "with", "as", "it", "are", "was", "from", "be", "has", "have",
    "results", "demonstrate", "significant", "improvement", "accuracy",
    "compared", "baseline", "model", "evaluation", "metrics", "test",
    "methodology", "advanced", "machine", "learning", "techniques",
    "capture", "patterns", "complex", "datasets", "computational",
    "research", "established", "natural", "language", "processing",
    "performance", "downstream", "tasks", "domain", "specific",
    "novel", "approach", "study", "analysis", "data", "method",
    "system", "framework", "algorithm", "experiment", "outcome",
]


@st.composite
def generated_output_text(draw: st.DrawFn) -> str:
    """Generate random text that may or may not share word spans with passages.

    Sometimes deliberately copies spans from passages to test that the guard
    rejects them, sometimes generates independent text.
    """
    # Decide whether to include a verbatim span from a passage
    include_verbatim = draw(st.booleans())

    if include_verbatim:
        # Pick a passage and extract a span of > 8 words
        passage_idx = draw(st.integers(min_value=0, max_value=len(_KNOWN_PASSAGES) - 1))
        passage_text = _KNOWN_PASSAGES[passage_idx].text
        passage_words = passage_text.split()

        # We need at least 9 words to form a span > 8
        assume(len(passage_words) >= 9)

        span_length = draw(st.integers(min_value=9, max_value=min(15, len(passage_words))))
        max_start = len(passage_words) - span_length
        start_idx = draw(st.integers(min_value=0, max_value=max_start))
        verbatim_span = " ".join(passage_words[start_idx:start_idx + span_length])

        # Build surrounding text
        prefix_len = draw(st.integers(min_value=0, max_value=5))
        suffix_len = draw(st.integers(min_value=0, max_value=5))

        prefix_words = draw(
            st.lists(st.sampled_from(_COMMON_WORDS), min_size=prefix_len, max_size=prefix_len)
        )
        suffix_words = draw(
            st.lists(st.sampled_from(_COMMON_WORDS), min_size=suffix_len, max_size=suffix_len)
        )

        parts = []
        if prefix_words:
            parts.append(" ".join(prefix_words))
        parts.append(verbatim_span)
        if suffix_words:
            parts.append(" ".join(suffix_words))

        return " ".join(parts)
    else:
        # Generate independent text from common words
        word_count = draw(st.integers(min_value=5, max_value=30))
        words = draw(
            st.lists(st.sampled_from(_COMMON_WORDS), min_size=word_count, max_size=word_count)
        )
        return " ".join(words)


# ---------------------------------------------------------------------------
# Property test: verbatim-span bound on process_measured
# ---------------------------------------------------------------------------


@given(output_text=generated_output_text())
@settings(max_examples=100)
def test_verbatim_span_bound_on_accepted_output(output_text: str) -> None:
    """Property 15: Verbatim-span bound.

    For all stage outputs, if the stage returns changed=True (accepted output),
    then no span of more than 8 consecutive words from any retrieved passage
    appears verbatim in the output.

    Validates: Requirements 7.7
    """
    fake_retrieval = FakeRetrievalService(_KNOWN_PASSAGES)
    fake_similarity = FakeSimilarityEvaluator(score=0.95)

    rewriter = RetrievalAugmentedRewriter(
        aggression=0.5,
        seed=42,
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:9999",
        retrieval_service=fake_retrieval,
        similarity=fake_similarity,
        floor=0.85,
        timeout_s=5,
    )

    # Mock _llm_rewrite to return the generated output text directly
    with patch.object(rewriter, "_llm_rewrite", return_value=output_text):
        result = rewriter.process_measured("Some input text for testing purposes.")

    # If the stage accepted the output (changed=True), the verbatim-span guard
    # must have ensured no span > 8 consecutive words from any passage appears
    if result.changed:
        passage_texts = [p.text for p in _KNOWN_PASSAGES]
        _assert_no_verbatim_span(result.text, passage_texts, max_span=8)


# ---------------------------------------------------------------------------
# Property test: _contains_verbatim_span method directly
# ---------------------------------------------------------------------------


@given(output_text=generated_output_text())
@settings(max_examples=100)
def test_contains_verbatim_span_correctness(output_text: str) -> None:
    """Property 15: _contains_verbatim_span correctness.

    When _contains_verbatim_span returns False for an output text, then no
    span of more than 8 consecutive words from any passage appears in the
    output. When it returns True, there exists such a span.

    Validates: Requirements 7.7
    """
    rewriter = RetrievalAugmentedRewriter(
        aggression=0.5,
        seed=42,
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:9999",
    )

    passage_texts = [p.text for p in _KNOWN_PASSAGES]
    detected = rewriter._contains_verbatim_span(output_text, passage_texts)

    if not detected:
        # If the guard says no violation, verify independently
        _assert_no_verbatim_span(output_text, passage_texts, max_span=8)
    else:
        # If the guard says violation, verify a span > 8 words actually exists
        assert _has_verbatim_span(output_text, passage_texts, max_span=8), (
            f"_contains_verbatim_span returned True but no span of >8 words "
            f"from any passage was found in: {output_text!r}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words matching the rewriter's approach."""
    return re.findall(r"\b\w+\b", text.lower())


def _assert_no_verbatim_span(
    output: str, passage_texts: List[str], max_span: int
) -> None:
    """Assert that no span of > max_span consecutive words from any passage
    appears in the output text."""
    output_words = _tokenize(output)
    span_length = max_span + 1  # We check for spans *exceeding* the limit

    for passage_text in passage_texts:
        passage_words = _tokenize(passage_text)
        if len(passage_words) < span_length:
            continue

        # Build set of all passage spans of length (max_span + 1)
        passage_spans = set()
        for i in range(len(passage_words) - span_length + 1):
            span = " ".join(passage_words[i:i + span_length])
            passage_spans.add(span)

        # Check no output span matches
        for i in range(len(output_words) - span_length + 1):
            output_span = " ".join(output_words[i:i + span_length])
            assert output_span not in passage_spans, (
                f"Found verbatim span of {span_length} words from passage in "
                f"output: '{output_span}'"
            )


def _has_verbatim_span(
    output: str, passage_texts: List[str], max_span: int
) -> bool:
    """Check whether a span of > max_span consecutive words from any passage
    exists in the output text."""
    output_words = _tokenize(output)
    span_length = max_span + 1

    for passage_text in passage_texts:
        passage_words = _tokenize(passage_text)
        if len(passage_words) < span_length:
            continue

        passage_spans = set()
        for i in range(len(passage_words) - span_length + 1):
            span = " ".join(passage_words[i:i + span_length])
            passage_spans.add(span)

        for i in range(len(output_words) - span_length + 1):
            output_span = " ".join(output_words[i:i + span_length])
            if output_span in passage_spans:
                return True

    return False
