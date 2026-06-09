"""
Property-based tests for SimilarityEvaluator score-range validity
and integration test for the real embedding model.

Validates that both the embedding path (via FakeSimilarityEvaluator) and the
lexical-proxy path always return scores in [0.0, 1.0] for all text pairs.

Requirements: 6.2, 14.1
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from humanizer.similarity import SimilarityEvaluator
from tests.conftest import FakeSimilarityEvaluator
from tests.strategies import academic_text


# Feature: ultimate-humanizer, Property 5: Score-range validity (similarity portion)
# — for all text pairs, both the embedding path (via injected fake) and the
# lexical-proxy path return a score in [0.0, 1.0]


class TestSimilarityScoreRangeEmbeddingPath:
    """Verify score range [0.0, 1.0] via the FakeSimilarityEvaluator (embedding path)."""

    # Validates: Requirements 6.2, 14.1

    @given(
        text_a=st.text(min_size=0, max_size=500),
        text_b=st.text(min_size=0, max_size=500),
        fake_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_fake_evaluator_returns_score_in_unit_interval(
        self, text_a: str, text_b: str, fake_score: float
    ) -> None:
        """FakeSimilarityEvaluator always returns scores in [0.0, 1.0]."""
        evaluator = FakeSimilarityEvaluator(scores=[fake_score])
        result = evaluator.score(text_a, text_b)
        assert 0.0 <= result <= 1.0, (
            f"FakeSimilarityEvaluator returned {result}, expected in [0.0, 1.0]"
        )

    @given(
        text_a=academic_text(),
        text_b=academic_text(),
        fake_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_fake_evaluator_academic_text_score_in_unit_interval(
        self, text_a: str, text_b: str, fake_score: float
    ) -> None:
        """FakeSimilarityEvaluator with academic text always returns scores in [0.0, 1.0]."""
        evaluator = FakeSimilarityEvaluator(scores=[fake_score])
        result = evaluator.score(text_a, text_b)
        assert 0.0 <= result <= 1.0, (
            f"FakeSimilarityEvaluator returned {result}, expected in [0.0, 1.0]"
        )


class TestSimilarityScoreRangeLexicalPath:
    """Verify score range [0.0, 1.0] via the lexical-proxy fallback path."""

    # Validates: Requirements 6.2, 14.1

    @given(
        text_a=st.text(min_size=0, max_size=500),
        text_b=st.text(min_size=0, max_size=500),
    )
    @settings(max_examples=100)
    def test_lexical_proxy_returns_score_in_unit_interval(
        self, text_a: str, text_b: str
    ) -> None:
        """Lexical-proxy path (model load fails) returns scores in [0.0, 1.0] for all text pairs."""
        # Use a non-existent model name to force the lexical fallback
        evaluator = SimilarityEvaluator(model_name="nonexistent-model-xyz-999")
        result = evaluator.score(text_a, text_b)
        assert 0.0 <= result <= 1.0, (
            f"Lexical-proxy returned {result}, expected in [0.0, 1.0]"
        )
        assert evaluator.last_source == "lexical", (
            f"Expected 'lexical' source, got '{evaluator.last_source}'"
        )

    @given(
        text_a=academic_text(),
        text_b=academic_text(),
    )
    @settings(max_examples=100)
    def test_lexical_proxy_academic_text_score_in_unit_interval(
        self, text_a: str, text_b: str
    ) -> None:
        """Lexical-proxy with academic text returns scores in [0.0, 1.0]."""
        evaluator = SimilarityEvaluator(model_name="nonexistent-model-xyz-999")
        result = evaluator.score(text_a, text_b)
        assert 0.0 <= result <= 1.0, (
            f"Lexical-proxy returned {result}, expected in [0.0, 1.0]"
        )
        assert evaluator.last_source == "lexical"

    @given(
        text_a=st.text(min_size=0, max_size=500),
        text_b=st.text(min_size=0, max_size=500),
    )
    @settings(max_examples=100)
    def test_lexical_proxy_via_import_failure_returns_score_in_unit_interval(
        self, text_a: str, text_b: str
    ) -> None:
        """Lexical fallback triggered by import failure still returns [0.0, 1.0]."""
        # Patch sentence_transformers import to raise ImportError
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            evaluator = SimilarityEvaluator()
            # Reset load state so it re-attempts import
            evaluator._model_load_attempted = False
            evaluator._model_loaded = False
            evaluator._model = None
            result = evaluator.score(text_a, text_b)
            assert 0.0 <= result <= 1.0, (
                f"Lexical-proxy (import failure) returned {result}, expected in [0.0, 1.0]"
            )
            assert evaluator.last_source == "lexical"


# ---------------------------------------------------------------------------
# Integration test for the real embedding model (task 3.3)
# ---------------------------------------------------------------------------

# Skip the entire class if sentence-transformers is not installed
_sentence_transformers_available = False
try:
    import sentence_transformers  # noqa: F401

    _sentence_transformers_available = True
except ImportError:
    pass


@pytest.mark.integration
@pytest.mark.skipif(
    not _sentence_transformers_available,
    reason="sentence-transformers not installed",
)
class TestSimilarityEvaluatorRealModel:
    """Integration test: real embedding model sanity checks.

    Validates: Requirements 14.1
    """

    def test_near_duplicate_scores_higher_than_unrelated(self) -> None:
        """Near-duplicate texts should score higher than completely unrelated texts."""
        evaluator = SimilarityEvaluator()

        # Near-duplicate pair: same sentence with one word changed
        text_a = "The experiment demonstrated significant improvements in model accuracy."
        text_b = "The experiment demonstrated notable improvements in model accuracy."

        # Completely unrelated pair
        text_c = "The experiment demonstrated significant improvements in model accuracy."
        text_d = "Banana pancakes are best served with maple syrup on Sunday mornings."

        score_near_duplicate = evaluator.score(text_a, text_b)
        score_unrelated = evaluator.score(text_c, text_d)

        # Near-duplicate score must be higher than unrelated score
        assert score_near_duplicate > score_unrelated, (
            f"Near-duplicate score ({score_near_duplicate:.4f}) should be greater "
            f"than unrelated score ({score_unrelated:.4f})"
        )

        # Both scores must be in [0.0, 1.0]
        assert 0.0 <= score_near_duplicate <= 1.0, (
            f"Near-duplicate score {score_near_duplicate} not in [0.0, 1.0]"
        )
        assert 0.0 <= score_unrelated <= 1.0, (
            f"Unrelated score {score_unrelated} not in [0.0, 1.0]"
        )

        # Since the model loaded successfully, source should be "embedding"
        assert evaluator.last_source == "embedding", (
            f"Expected last_source='embedding', got '{evaluator.last_source}'"
        )
