"""
Example-based tests for PerplexityOptimizer.

Validates:
- Req 3.5: Within-tolerance input returned unchanged
- Req 3.8: Empty/whitespace input returned unchanged
- Req 3.9: Unmeasurable perplexity returned unchanged
"""

from unittest.mock import patch

import pytest

from humanizer.results import TargetPerplexityProfile
from humanizer.stage_perplexity_optimize import PerplexityOptimizer
from humanizer.text_analysis import estimate_perplexity_score

from tests.conftest import FakeSimilarityEvaluator


class TestWithinToleranceUnchanged:
    """Req 3.5: When the measured mean perplexity of the input text is within
    the configured mean-perplexity tolerance of the target mean AND the measured
    variance is within the configured variance tolerance, the optimizer returns
    the input text unchanged."""

    def test_within_tolerance_returns_input_unchanged(self):
        """**Validates: Requirements 3.5**

        Create a target profile whose target_mean matches the actual mean
        perplexity of the input text (within 5% tolerance), and verify the
        text is returned unchanged with changed == False.
        """
        input_text = "The research methodology was sound. Data analysis revealed patterns."

        # Measure the actual mean perplexity of this text so we can set a
        # target that's within tolerance.
        from humanizer.text_analysis import split_sentences, estimate_perplexity_score

        sentences = split_sentences(input_text)
        scores = [estimate_perplexity_score(s) for s in sentences]
        actual_mean = sum(scores) / len(scores)
        actual_variance = (
            sum((s - actual_mean) ** 2 for s in scores) / len(scores)
            if len(scores) >= 2
            else 0.0
        )

        # Set target to match actual values (well within default 5%/10% tolerances)
        profile = TargetPerplexityProfile(
            target_mean=actual_mean,
            target_variance=actual_variance,
        )

        similarity = FakeSimilarityEvaluator(default=0.99)
        optimizer = PerplexityOptimizer(
            aggression=0.5,
            seed=42,
            similarity=similarity,
            target_profile=profile,
        )

        result = optimizer.process_measured(input_text)

        assert result.text == input_text
        assert result.changed is False


class TestEmptyWhitespaceUnchanged:
    """Req 3.8: If the input text is empty or contains only whitespace
    characters, the optimizer returns the input text unchanged."""

    def test_empty_string_returned_unchanged(self):
        """**Validates: Requirements 3.8**

        Empty string input is returned unchanged with changed == False.
        """
        similarity = FakeSimilarityEvaluator(default=0.99)
        optimizer = PerplexityOptimizer(
            aggression=0.5,
            seed=42,
            similarity=similarity,
        )

        result = optimizer.process_measured("")

        assert result.text == ""
        assert result.changed is False

    def test_whitespace_only_returned_unchanged(self):
        """**Validates: Requirements 3.8**

        Whitespace-only input is returned unchanged with changed == False.
        """
        similarity = FakeSimilarityEvaluator(default=0.99)
        optimizer = PerplexityOptimizer(
            aggression=0.5,
            seed=42,
            similarity=similarity,
        )

        result = optimizer.process_measured("   ")

        assert result.text == "   "
        assert result.changed is False


class TestUnmeasurablePerplexityUnchanged:
    """Req 3.9: If the perplexity profile of the input text cannot be measured,
    the optimizer returns the input text unchanged."""

    def test_unmeasurable_perplexity_returns_input_unchanged(self):
        """**Validates: Requirements 3.9**

        When estimate_perplexity_score always returns 50.0 (the neutral fallback
        indicating wordfreq isn't available), perplexity is unmeasurable and
        the text is returned unchanged.
        """
        input_text = "The experiment yielded interesting results. Further investigation is needed."

        similarity = FakeSimilarityEvaluator(default=0.99)
        optimizer = PerplexityOptimizer(
            aggression=0.5,
            seed=42,
            similarity=similarity,
        )

        # Mock estimate_perplexity_score to always return 50.0 (neutral default)
        with patch(
            "humanizer.stage_perplexity_optimize.estimate_perplexity_score",
            return_value=50.0,
        ):
            result = optimizer.process_measured(input_text)

        assert result.text == input_text
        assert result.changed is False
