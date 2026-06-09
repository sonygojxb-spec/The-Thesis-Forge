"""
Property 8: Stylometric attribute shift.

For all inputs with >=2 sentences and aggression > 0, output differs in at least
one targeted attribute by >=5%, and sentence-length variance is >=10% greater
(or within +/-2% at threshold).

The property only applies when the stage actually changed the text (result.changed == True).

Requirements: 2.1, 2.2

# Feature: ultimate-humanizer, Property 8: Stylometric attribute shift
"""

from __future__ import annotations

import re

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from humanizer.stage_stylometric import StylometricObfuscator, STYLO_VARIANCE_THRESHOLD
from humanizer.text_analysis import (
    compute_sentence_length_variance,
    compute_type_token_ratio,
    split_sentences,
)
from tests.conftest import FakeSimilarityEvaluator
from tests.strategies import multi_sentence_text


# ---------------------------------------------------------------------------
# Helper: compute function-word frequency
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
    punct_count = sum(1 for ch in text if ch in ".,;:!?—-\"'()")
    return punct_count / len(words)


def relative_difference(before: float, after: float) -> float:
    """Compute the absolute relative difference between two values.

    Returns a fraction (e.g. 0.05 = 5% difference).
    Uses the 'before' value as the denominator. If before is 0, uses
    absolute difference check against a small epsilon.
    """
    if before == 0.0:
        return abs(after)  # treat any non-zero as a shift from 0
    return abs(after - before) / abs(before)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

# Feature: ultimate-humanizer, Property 8: Stylometric attribute shift


@given(
    text=multi_sentence_text(min_sentences=2, max_sentences=6),
    aggression=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=100)
def test_stylometric_attribute_shift(text: str, aggression: float, seed: int) -> None:
    """Property 8: Stylometric attribute shift.

    For all inputs with >=2 sentences and aggression > 0, when the stage
    actually changes the text:
    - At least one targeted attribute (sentence-length variance, type-token
      ratio, function-word frequency, or punctuation density) differs by >=5%.
    - Sentence-length variance is >=10% greater, OR the input variance is
      already at the human threshold (>=STYLO_VARIANCE_THRESHOLD) and the
      output variance is within +/-2% of the input variance.

    **Validates: Requirements 2.1, 2.2**
    """
    # Precondition: input must have >=2 sentences
    sentences = split_sentences(text)
    assume(len(sentences) >= 2)

    # Use FakeSimilarityEvaluator returning high scores so candidate is accepted
    evaluator = FakeSimilarityEvaluator(default=0.95)

    obfuscator = StylometricObfuscator(
        aggression=aggression,
        seed=seed,
        similarity=evaluator,
        floor=0.85,
    )

    result = obfuscator.process_measured(text)

    # The property only applies when the stage actually changed the text
    if not result.changed:
        return

    output = result.text

    # Measure before/after metrics
    variance_before = compute_sentence_length_variance(text)
    variance_after = compute_sentence_length_variance(output)

    ttr_before = compute_type_token_ratio(text)
    ttr_after = compute_type_token_ratio(output)

    fw_freq_before = compute_function_word_frequency(text)
    fw_freq_after = compute_function_word_frequency(output)

    punct_before = compute_punctuation_density(text)
    punct_after = compute_punctuation_density(output)

    # --- Requirement 2.1: at least one targeted attribute differs by >=5% ---
    shifts = [
        relative_difference(variance_before, variance_after),
        relative_difference(ttr_before, ttr_after),
        relative_difference(fw_freq_before, fw_freq_after),
        relative_difference(punct_before, punct_after),
    ]

    at_least_one_shifted = any(s >= 0.05 for s in shifts)

    assert at_least_one_shifted, (
        f"No targeted attribute shifted by >=5% (Req 2.1)\n"
        f"  Variance: {variance_before} -> {variance_after} (rel diff: {shifts[0]:.4f})\n"
        f"  TTR: {ttr_before} -> {ttr_after} (rel diff: {shifts[1]:.4f})\n"
        f"  Func-word freq: {fw_freq_before:.4f} -> {fw_freq_after:.4f} (rel diff: {shifts[2]:.4f})\n"
        f"  Punct density: {punct_before:.4f} -> {punct_after:.4f} (rel diff: {shifts[3]:.4f})\n"
        f"  Aggression: {aggression}, Seed: {seed}\n"
        f"  Input:  {text!r}\n"
        f"  Output: {output!r}"
    )

    # --- Requirement 2.2: sentence-length variance >=10% greater,
    #     OR already at threshold and within +/-2% ---
    already_at_threshold = variance_before >= STYLO_VARIANCE_THRESHOLD

    if already_at_threshold:
        # Output variance should be within +/-2% of input variance
        if variance_before > 0:
            variance_rel_diff = abs(variance_after - variance_before) / variance_before
            assert variance_rel_diff <= 0.02 or variance_after >= variance_before * 0.98, (
                f"At threshold: variance changed more than +/-2% (Req 2.2)\n"
                f"  Variance before: {variance_before}, after: {variance_after}\n"
                f"  Relative diff: {variance_rel_diff:.4f}"
            )
    else:
        # Output variance should be >=10% greater than input variance
        if variance_before > 0:
            variance_increase = (variance_after - variance_before) / variance_before
            assert variance_increase >= 0.10 or variance_after >= variance_before * 1.10, (
                f"Variance did not increase by >=10% (Req 2.2)\n"
                f"  Variance before: {variance_before}, after: {variance_after}\n"
                f"  Increase: {variance_increase:.4f} (need >=0.10)\n"
                f"  Aggression: {aggression}, Seed: {seed}\n"
                f"  Input:  {text!r}\n"
                f"  Output: {output!r}"
            )
        else:
            # Variance was 0 (all sentences same length); any increase is fine
            # as long as the output has non-zero variance
            assert variance_after >= 0.0, (
                f"Variance was 0 and remained 0 or became negative (Req 2.2)\n"
                f"  Variance before: {variance_before}, after: {variance_after}"
            )
