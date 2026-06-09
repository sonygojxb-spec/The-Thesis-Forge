"""
Example-based tests for AdversarialRewriter fallback behavior.

Tests the following adversarial-stage invariants:
- LLM error/empty/timeout → input unchanged (Req 4.7)
- Candidate risk > input risk → input unchanged (Req 4.6)

Uses FakeSimilarityEvaluator and FakeClassifier from conftest.py to provide
deterministic, offline testing without real LLM or classifier dependencies.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from humanizer.stage_adversarial import AdversarialRewriter
from tests.conftest import FakeClassifier, FakeSimilarityEvaluator


# ---------------------------------------------------------------------------
# Requirement 4.7: LLM error / empty / timeout → input unchanged
# ---------------------------------------------------------------------------


class TestLLMErrorFallback:
    """Validates Req 4.7: LLM error → return input unchanged."""

    def test_llm_runtime_error_returns_input_unchanged(self):
        """WHEN _llm_rewrite raises RuntimeError, output == input, changed=False,
        fell_back=True, error is not None.

        **Validates: Requirements 4.7**
        """
        input_text = "The neural network architecture demonstrates superior performance."

        classifier = FakeClassifier(scores=[40.0])  # input risk score
        similarity = FakeSimilarityEvaluator(scores=[0.95])

        rewriter = AdversarialRewriter(
            aggression=0.5,
            seed=42,
            model="test-model",
            api_key="test-key",
            base_url="http://localhost",
            similarity=similarity,
            classifier=classifier,
        )

        with patch.object(
            rewriter, "_llm_rewrite", side_effect=RuntimeError("LLM API timeout")
        ):
            result = rewriter.process_measured(input_text)

        assert result.text == input_text
        assert result.changed is False
        assert result.fell_back is True
        assert result.error is not None
        assert "LLM API timeout" in result.error

    def test_llm_empty_result_returns_input_unchanged(self):
        """WHEN _llm_rewrite returns empty string, output == input, changed=False,
        fell_back=True.

        **Validates: Requirements 4.7**
        """
        input_text = "Empirical results indicate a statistically significant improvement."

        classifier = FakeClassifier(scores=[50.0])  # input risk score
        similarity = FakeSimilarityEvaluator(scores=[0.95])

        rewriter = AdversarialRewriter(
            aggression=0.5,
            seed=42,
            model="test-model",
            api_key="test-key",
            base_url="http://localhost",
            similarity=similarity,
            classifier=classifier,
        )

        with patch.object(rewriter, "_llm_rewrite", return_value=""):
            result = rewriter.process_measured(input_text)

        assert result.text == input_text
        assert result.changed is False
        assert result.fell_back is True


# ---------------------------------------------------------------------------
# Requirement 4.6: Candidate risk > input risk → input unchanged
# ---------------------------------------------------------------------------


class TestRiskIncreaseFallback:
    """Validates Req 4.6: candidate risk > input risk → return input unchanged."""

    def test_candidate_higher_risk_returns_input_unchanged(self):
        """WHEN candidate Detection_Risk_Score (80) > input Detection_Risk_Score (40),
        output == input, changed=False.

        **Validates: Requirements 4.6**
        """
        input_text = "The methodology employs a controlled experimental design."
        candidate_text = "The approach uses a controlled experimental setup."

        # FakeClassifier scores: first call for input risk (40.0),
        # second call for candidate risk (80.0).
        classifier = FakeClassifier(scores=[40.0, 80.0])
        # Similarity above floor so we reach the risk comparison step.
        similarity = FakeSimilarityEvaluator(scores=[0.92])

        rewriter = AdversarialRewriter(
            aggression=0.5,
            seed=42,
            model="test-model",
            api_key="test-key",
            base_url="http://localhost",
            similarity=similarity,
            classifier=classifier,
        )

        with patch.object(rewriter, "_llm_rewrite", return_value=candidate_text):
            result = rewriter.process_measured(input_text)

        assert result.text == input_text
        assert result.changed is False
        assert result.risk_before == 40.0
        assert result.risk_after == 80.0
