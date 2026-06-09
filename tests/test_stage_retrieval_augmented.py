"""
Unit tests for RetrievalAugmentedRewriter stage.

Validates: Requirements 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from humanizer.retrieval import ReferenceEntry
from humanizer.stage_retrieval_augmented import RetrievalAugmentedRewriter
from tests.conftest import FakeSimilarityEvaluator


# ---------------------------------------------------------------------------
# Fake RetrievalService for testing
# ---------------------------------------------------------------------------


class FakeRetrievalService:
    """A fake retrieval service with a configurable corpus and retrieve behavior."""

    def __init__(
        self,
        entries: Optional[List[ReferenceEntry]] = None,
        retrieve_results: Optional[List[ReferenceEntry]] = None,
        retrieve_error: Optional[Exception] = None,
    ) -> None:
        self._entries = entries or []
        self._retrieve_results = retrieve_results
        self._retrieve_error = retrieve_error

    @property
    def corpus(self) -> List[ReferenceEntry]:
        return list(self._entries)

    def retrieve(self, query_text: str) -> List[ReferenceEntry]:
        if self._retrieve_error:
            raise self._retrieve_error
        if self._retrieve_results is not None:
            return self._retrieve_results
        return self._entries[:10]


def make_entry(text: str, entry_id: str = "e1") -> ReferenceEntry:
    """Helper to create a ReferenceEntry."""
    return ReferenceEntry(id=entry_id, text=text, source="test")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetrievalAugmentedRewriterEmptyInput:
    """Req 7.8: empty/whitespace input → return unchanged."""

    def test_empty_string_unchanged(self):
        rewriter = RetrievalAugmentedRewriter()
        result = rewriter.process_measured("")
        assert result.text == ""
        assert result.changed is False

    def test_whitespace_only_unchanged(self):
        rewriter = RetrievalAugmentedRewriter()
        result = rewriter.process_measured("   \n  ")
        assert result.text == "   \n  "
        assert result.changed is False


class TestRetrievalAugmentedRewriterNoRetrievalService:
    """No retrieval service → return unchanged."""

    def test_no_service_returns_unchanged(self):
        rewriter = RetrievalAugmentedRewriter(retrieval_service=None)
        result = rewriter.process_measured("Some academic text about methodology.")
        assert result.text == "Some academic text about methodology."
        assert result.changed is False
        assert result.fell_back is True


class TestRetrievalAugmentedRewriterEmptyCorpus:
    """Req 7.4: corpus empty → input unchanged."""

    def test_empty_corpus_returns_unchanged(self):
        service = FakeRetrievalService(entries=[])
        rewriter = RetrievalAugmentedRewriter(retrieval_service=service)
        result = rewriter.process_measured("Text to rewrite.")
        assert result.text == "Text to rewrite."
        assert result.changed is False
        assert result.fell_back is True

    def test_no_results_returns_unchanged(self):
        """Corpus has entries but retrieve returns empty."""
        service = FakeRetrievalService(
            entries=[make_entry("Some human text")],
            retrieve_results=[],
        )
        rewriter = RetrievalAugmentedRewriter(retrieval_service=service)
        result = rewriter.process_measured("Text to rewrite.")
        assert result.text == "Text to rewrite."
        assert result.changed is False
        assert result.fell_back is True


class TestRetrievalAugmentedRewriterLLMFailure:
    """Req 7.8: LLM error/empty/timeout → input unchanged."""

    def test_llm_error_returns_unchanged(self):
        """LLM raises RuntimeError → input unchanged."""
        service = FakeRetrievalService(
            entries=[make_entry("Human writing example passage for style.")],
        )
        sim = FakeSimilarityEvaluator(default=0.92)
        rewriter = RetrievalAugmentedRewriter(
            retrieval_service=service,
            similarity=sim,
            model="test",
            api_key="test-key",
            base_url="http://fake",
        )

        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Connection refused")
            result = rewriter.process_measured("Academic text to rewrite.")

        assert result.text == "Academic text to rewrite."
        assert result.changed is False
        assert result.fell_back is True
        assert result.error is not None

    def test_llm_empty_response_returns_unchanged(self):
        """LLM returns empty → input unchanged."""
        service = FakeRetrievalService(
            entries=[make_entry("Human writing example passage for style.")],
        )
        sim = FakeSimilarityEvaluator(default=0.92)
        rewriter = RetrievalAugmentedRewriter(
            retrieval_service=service,
            similarity=sim,
            model="test",
            api_key="test-key",
            base_url="http://fake",
        )

        # Simulate SSE response with empty content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":""}}]}',
            b'data: [DONE]',
        ]

        with patch("requests.post", return_value=mock_response):
            result = rewriter.process_measured("Academic text to rewrite.")

        assert result.text == "Academic text to rewrite."
        assert result.changed is False
        assert result.fell_back is True

    def test_llm_rewrite_raises_runtime_error_returns_unchanged(self):
        """Mock _llm_rewrite to raise RuntimeError → output == input, changed=False, fell_back=True.

        Validates: Requirements 7.8
        """
        service = FakeRetrievalService(
            entries=[make_entry("Human writing example passage for style.")],
        )
        sim = FakeSimilarityEvaluator(default=0.92)
        rewriter = RetrievalAugmentedRewriter(
            retrieval_service=service,
            similarity=sim,
            model="test",
            api_key="test-key",
            base_url="http://fake",
        )

        input_text = "The experimental results clearly demonstrate the hypothesis."

        with patch.object(
            rewriter, "_llm_rewrite", side_effect=RuntimeError("LLM service unavailable")
        ):
            result = rewriter.process_measured(input_text)

        assert result.text == input_text
        assert result.changed is False
        assert result.fell_back is True
        assert result.error is not None

    def test_llm_rewrite_returns_empty_string_returns_unchanged(self):
        """Mock _llm_rewrite to return empty string → output == input, changed=False, fell_back=True.

        Validates: Requirements 7.8
        """
        service = FakeRetrievalService(
            entries=[make_entry("Human writing example passage for style.")],
        )
        sim = FakeSimilarityEvaluator(default=0.92)
        rewriter = RetrievalAugmentedRewriter(
            retrieval_service=service,
            similarity=sim,
            model="test",
            api_key="test-key",
            base_url="http://fake",
        )

        input_text = "The experimental results clearly demonstrate the hypothesis."

        with patch.object(rewriter, "_llm_rewrite", return_value=""):
            result = rewriter.process_measured(input_text)

        assert result.text == input_text
        assert result.changed is False
        assert result.fell_back is True
        assert result.error is not None


class TestRetrievalAugmentedRewriterVerbatimSpanGuard:
    """Req 7.7: output containing > 8 consecutive-word span from passage → rejected."""

    def test_verbatim_span_detected(self):
        """If the LLM copies >8 words from a passage, output is rejected."""
        passage_text = (
            "The methodology employed in this research demonstrates a clear "
            "understanding of the theoretical framework underlying the study"
        )
        service = FakeRetrievalService(
            entries=[make_entry(passage_text)],
        )
        sim = FakeSimilarityEvaluator(default=0.92)
        rewriter = RetrievalAugmentedRewriter(
            retrieval_service=service,
            similarity=sim,
            model="test",
            api_key="test-key",
            base_url="http://fake",
        )

        # LLM output that contains a 9-word span from the passage
        llm_output = (
            "The methodology employed in this research demonstrates a clear "
            "understanding of the experimental design."
        )

        # Simulate successful SSE streaming response
        lines = []
        for word in llm_output.split():
            lines.append(
                f'data: {{"choices":[{{"delta":{{"content":"{word} "}}}}]}}'.encode()
            )
        lines.append(b'data: [DONE]')

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = lines

        with patch("requests.post", return_value=mock_response):
            result = rewriter.process_measured("Some academic text about theory.")

        assert result.text == "Some academic text about theory."
        assert result.changed is False
        assert result.fell_back is True
        assert "verbatim" in result.error.lower()

    def test_short_span_allowed(self):
        """A span of exactly 8 words (not more than 8) should be allowed."""
        rewriter = RetrievalAugmentedRewriter()
        # "the methodology employed in this study is clear" = 8 words
        passage_text = "the methodology employed in this study is clear and well developed"
        output_text = "the methodology employed in this study is clear in its approach"

        # 8-word span: "the methodology employed in this study is clear"
        # This is exactly 8, NOT more than 8, so it should be allowed
        assert not rewriter._contains_verbatim_span(
            output_text, [passage_text]
        )

    def test_nine_word_span_rejected(self):
        """A span of 9 consecutive words from a passage should be detected."""
        rewriter = RetrievalAugmentedRewriter()
        passage_text = "the methodology employed in this study is clear and well developed"
        # 9-word span: "the methodology employed in this study is clear and"
        output_text = "the methodology employed in this study is clear and something else"

        assert rewriter._contains_verbatim_span(output_text, [passage_text])


class TestRetrievalAugmentedRewriterSimilarityFloor:
    """Req 7.6, 7.9: rewrite similarity < 0.85 → input unchanged."""

    def test_low_similarity_returns_unchanged(self):
        """If similarity score is below floor, return input unchanged."""
        passage_text = "Human written academic prose example."
        service = FakeRetrievalService(
            entries=[make_entry(passage_text)],
        )
        # Similarity below floor
        sim = FakeSimilarityEvaluator(default=0.70)
        rewriter = RetrievalAugmentedRewriter(
            retrieval_service=service,
            similarity=sim,
            model="test",
            api_key="test-key",
            base_url="http://fake",
        )

        # LLM returns something that doesn't trigger verbatim guard
        llm_output = "A completely different rewrite of the academic text."
        lines = [
            f'data: {{"choices":[{{"delta":{{"content":"{llm_output}"}}}}]}}'.encode(),
            b'data: [DONE]',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = lines

        with patch("requests.post", return_value=mock_response):
            result = rewriter.process_measured("Original academic text here.")

        assert result.text == "Original academic text here."
        assert result.changed is False
        assert result.fell_back is True
        assert result.similarity == 0.70

    def test_above_floor_accepts_rewrite(self):
        """If similarity >= floor, accept the rewrite."""
        passage_text = "Human written academic prose example with enough words."
        service = FakeRetrievalService(
            entries=[make_entry(passage_text)],
        )
        sim = FakeSimilarityEvaluator(default=0.92)
        rewriter = RetrievalAugmentedRewriter(
            retrieval_service=service,
            similarity=sim,
            model="test",
            api_key="test-key",
            base_url="http://fake",
        )

        llm_output = "A well crafted academic rewrite."
        lines = [
            f'data: {{"choices":[{{"delta":{{"content":"{llm_output}"}}}}]}}'.encode(),
            b'data: [DONE]',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = lines

        with patch("requests.post", return_value=mock_response):
            result = rewriter.process_measured("Original academic text.")

        assert result.text == llm_output
        assert result.changed is True
        assert result.fell_back is False
        assert result.similarity == 0.92


class TestRetrievalAugmentedRewriterRetrievalError:
    """Retrieval service error → input unchanged."""

    def test_retrieve_raises_error(self):
        """If retrieval service.retrieve() raises, return input unchanged."""
        service = FakeRetrievalService(
            entries=[make_entry("Some text")],
            retrieve_error=RuntimeError("embedding service down"),
        )
        rewriter = RetrievalAugmentedRewriter(retrieval_service=service)
        result = rewriter.process_measured("Test text.")
        assert result.text == "Test text."
        assert result.changed is False
        assert result.fell_back is True
        assert "Retrieval error" in result.error


class TestRetrievalAugmentedRewriterProcessMethod:
    """process() delegates to process_measured() and returns text only."""

    def test_process_returns_text(self):
        rewriter = RetrievalAugmentedRewriter(retrieval_service=None)
        text = "Some text."
        assert rewriter.process(text) == text
