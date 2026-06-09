"""
Property 9: Stylometric adjustment monotonicity.

For all inputs and seeds, adjustment magnitude at higher aggression >=
magnitude at any lower aggression.

Requirements: 2.5

# Feature: ultimate-humanizer, Property 9: Stylometric adjustment monotonicity
"""

from __future__ import annotations

import re

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from humanizer.stage_stylometric import StylometricObfuscator
from humanizer.text_analysis import (
    compute_sentence_length_variance,
    compute_type_token_ratio,
)
from tests.conftest import FakeSimilarityEvaluator


# ---------------------------------------------------------------------------
# Helpers: compute stylometric attributes
# ---------------------------------------------------------------------------

_FUNCTION_WORDS = {
    "the", "a", "an", "of", "in", "to", "for", "on", "with", "at",
    "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "this", "that", "these", "those", "it", "its", "they", "their",
    "which", "who", "whom", "what", "where", "when", "how",
    "not", "no", "nor", "but", "or", "and", "if", "then", "than",
    "so", "yet", "both", "either", "neither", "each", "every",
    "all", "any", "few", "more", "most", "some", "such",
    "very", "quite", "rather", "also", "just", "only", "still",
}


def compute_function_word_frequency(text: str) -> float:
    """Compute the proportion of function words in text."""
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return 0.0
    count = sum(1 for w in words if w in _FUNCTION_WORDS)
    return count / len(words)


def compute_punctuation_density(text: str) -> float:
    """Compute punctuation characters per word as a density metric."""
    words = text.split()
    if not words:
        return 0.0
    punct_count = sum(1 for ch in text if ch in ".,;:!?\u2014-\"'()")
    return punct_count / len(words)


def compute_adjustment_magnitude(input_text: str, output_text: str) -> float:
    """Compute the sum of absolute relative differences across all four
    stylometric attributes between input and output.

    Attributes: variance, TTR, function-word freq, punctuation density.
    """
    var_before = compute_sentence_length_variance(input_text)
    var_after = compute_sentence_length_variance(output_text)

    ttr_before = compute_type_token_ratio(input_text)
    ttr_after = compute_type_token_ratio(output_text)

    fw_before = compute_function_word_frequency(input_text)
    fw_after = compute_function_word_frequency(output_text)

    punct_before = compute_punctuation_density(input_text)
    punct_after = compute_punctuation_density(output_text)

    total_magnitude = 0.0
    for before, after in [
        (var_before, var_after),
        (ttr_before, ttr_after),
        (fw_before, fw_after),
        (punct_before, punct_after),
    ]:
        if before == 0.0:
            total_magnitude += abs(after)
        else:
            total_magnitude += abs(after - before) / abs(before)

    return total_magnitude


# ---------------------------------------------------------------------------
# Fixed multi-sentence input text for deterministic comparison
# ---------------------------------------------------------------------------

_FIXED_INPUT_TEXT = (
    "The study demonstrates that results indicate a significant trend in the data. "
    "We observe a clear pattern emerging from the analysis of multiple variables. "
    "The framework addresses several key challenges identified in prior research. "
    "A comprehensive evaluation confirms these findings align with theoretical predictions. "
    "The proposed method uses advanced techniques to achieve robust and scalable outcomes."
)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

# Feature: ultimate-humanizer, Property 9: Stylometric adjustment monotonicity


@given(
    lo=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    hi_delta=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=100)
def test_stylometric_adjustment_monotonicity(lo: float, hi_delta: float, seed: int) -> None:
    """Property 9: Stylometric adjustment monotonicity.

    For all inputs and seeds, adjustment magnitude at higher aggression >=
    magnitude at any lower aggression.

    **Validates: Requirements 2.5**
    """
    # Ensure lo < hi, both > 0
    hi = min(lo + hi_delta, 1.0)
    assume(hi > lo)

    # Use FakeSimilarityEvaluator returning high scores (>=0.85) so candidates are accepted
    evaluator_lo = FakeSimilarityEvaluator(default=0.95)
    evaluator_hi = FakeSimilarityEvaluator(default=0.95)

    # Run with lower aggression
    obfuscator_lo = StylometricObfuscator(
        aggression=lo,
        seed=seed,
        similarity=evaluator_lo,
        floor=0.85,
    )
    result_lo = obfuscator_lo.process_measured(_FIXED_INPUT_TEXT)

    # Run with higher aggression
    obfuscator_hi = StylometricObfuscator(
        aggression=hi,
        seed=seed,
        similarity=evaluator_hi,
        floor=0.85,
    )
    result_hi = obfuscator_hi.process_measured(_FIXED_INPUT_TEXT)

    # Only assert when BOTH runs produce changed=True
    assume(result_lo.changed)
    assume(result_hi.changed)

    # Measure the adjustment magnitude for both
    magnitude_lo = compute_adjustment_magnitude(_FIXED_INPUT_TEXT, result_lo.text)
    magnitude_hi = compute_adjustment_magnitude(_FIXED_INPUT_TEXT, result_hi.text)

    # Verify: magnitude at higher aggression >= magnitude at lower aggression
    assert magnitude_hi >= magnitude_lo, (
        f"Monotonicity violated: magnitude at higher aggression < magnitude at lower (Req 2.5)\n"
        f"  Aggression lo={lo:.4f}, hi={hi:.4f}\n"
        f"  Magnitude lo={magnitude_lo:.6f}, hi={magnitude_hi:.6f}\n"
        f"  Seed: {seed}\n"
        f"  Output lo: {result_lo.text!r}\n"
        f"  Output hi: {result_hi.text!r}"
    )
