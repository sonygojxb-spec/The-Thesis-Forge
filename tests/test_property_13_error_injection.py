"""
Property 13: Error-injection bound and monotonicity.

For all inputs:
1. Altered words <= floor(0.05 * word_count)
2. Zero alterations when floor(0.05 * word_count) < 1 (i.e., < 20 words)
3. Altered word count is non-decreasing in aggression (monotonicity), up to the bound

"Altered words" = count of words in output that differ from the corresponding
position in input (or extra/missing words).

Requirements: 5.1, 5.2, 5.8

# Feature: ultimate-humanizer, Property 13: Error-injection bound and monotonicity
"""

from __future__ import annotations

import math

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from humanizer.stage_error_injector import ErrorInjector
from tests.strategies import academic_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_altered_words(original: str, output: str) -> int:
    """Count the number of words that differ between original and output.

    Uses positional comparison. Extra or missing words also count as altered.
    """
    orig_words = original.split()
    out_words = output.split()

    changes = 0
    # Positional differences
    for i in range(min(len(orig_words), len(out_words))):
        if orig_words[i] != out_words[i]:
            changes += 1

    # Extra or missing words count as changes
    changes += abs(len(out_words) - len(orig_words))

    return changes


# ---------------------------------------------------------------------------
# Strategy: text with enough words (>= 20) for the bound test
# ---------------------------------------------------------------------------


@st.composite
def long_academic_text(draw: st.DrawFn) -> str:
    """Generate academic text guaranteed to have >= 20 words.

    This ensures floor(0.05 * word_count) >= 1 so the bound is active.
    """
    text = draw(academic_text(
        min_protected_terms=1,
        max_protected_terms=3,
        include_numbers=True,
        include_citations=True,
        include_quotes=True,
    ))
    # Ensure at least 20 words
    assume(len(text.split()) >= 20)
    return text


@st.composite
def short_text(draw: st.DrawFn) -> str:
    """Generate text with fewer than 20 words.

    This ensures floor(0.05 * word_count) < 1, so zero alterations should occur.
    """
    # Generate a short sentence of 3-19 words
    word_count = draw(st.integers(min_value=3, max_value=19))
    words = draw(st.lists(
        st.sampled_from([
            "the", "study", "results", "show", "data", "analysis",
            "method", "approach", "system", "model", "test", "value",
            "work", "case", "time", "based", "new", "first", "high",
        ]),
        min_size=word_count,
        max_size=word_count,
    ))
    return " ".join(words)


# ---------------------------------------------------------------------------
# Property 13.1: Bound — altered words <= floor(0.05 * word_count)
# ---------------------------------------------------------------------------


@given(
    text=long_academic_text(),
    aggression=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_error_injection_bound(text: str, aggression: float) -> None:
    """Property 13.1: Bound on altered words.

    For all inputs and aggression values, the count of altered words in
    the output <= floor(0.05 * word_count).

    **Validates: Requirements 5.1, 5.2, 5.8**
    """
    fixed_seed = 42

    injector = ErrorInjector(aggression=aggression, seed=fixed_seed)
    output = injector.process(text)

    word_count = len(text.split())
    max_altered = math.floor(0.05 * word_count)
    actual_altered = _count_altered_words(text, output)

    assert actual_altered <= max_altered, (
        f"Error injection bound violated: "
        f"altered {actual_altered} words but cap is floor(0.05 * {word_count}) = {max_altered}\n"
        f"aggression={aggression:.4f}, seed={fixed_seed}\n"
        f"Input:  {text!r}\n"
        f"Output: {output!r}"
    )


# ---------------------------------------------------------------------------
# Property 13.2: Zero when bound < 1 — text unchanged for < 20 words
# ---------------------------------------------------------------------------


@given(
    text=short_text(),
    aggression=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_error_injection_zero_when_bound_lt_one(text: str, aggression: float) -> None:
    """Property 13.2: Zero alterations when bound < 1.

    When input has < 20 words, floor(0.05 * word_count) < 1, so the
    output must equal the input (zero alterations).

    **Validates: Requirements 5.2, 5.8**
    """
    # Confirm our precondition
    word_count = len(text.split())
    assert word_count < 20, f"Expected < 20 words, got {word_count}"
    assert math.floor(0.05 * word_count) < 1

    fixed_seed = 42

    injector = ErrorInjector(aggression=aggression, seed=fixed_seed)
    output = injector.process(text)

    assert output == text, (
        f"Error injection should produce zero alterations when bound < 1: "
        f"word_count={word_count}, floor(0.05*{word_count})={math.floor(0.05*word_count)}\n"
        f"aggression={aggression:.4f}\n"
        f"Input:  {text!r}\n"
        f"Output: {output!r}"
    )


# ---------------------------------------------------------------------------
# Property 13.3: Monotonicity — altered words non-decreasing in aggression
# ---------------------------------------------------------------------------


@given(
    text=long_academic_text(),
    lo=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    hi=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_error_injection_monotonicity(text: str, lo: float, hi: float) -> None:
    """Property 13.3: Monotonicity of altered word count in aggression.

    For pairs (lo, hi) with lo < hi, altered word count at higher aggression
    >= altered word count at lower aggression, using a fixed seed.

    **Validates: Requirements 5.1**
    """
    assume(lo < hi)

    # Use a fixed seed so the only variable is aggression
    fixed_seed = 42

    injector_lo = ErrorInjector(aggression=lo, seed=fixed_seed)
    injector_hi = ErrorInjector(aggression=hi, seed=fixed_seed)

    output_lo = injector_lo.process(text)
    output_hi = injector_hi.process(text)

    altered_lo = _count_altered_words(text, output_lo)
    altered_hi = _count_altered_words(text, output_hi)

    assert altered_hi >= altered_lo, (
        f"Error injection monotonicity violated: "
        f"altered at aggression={hi:.4f} ({altered_hi}) < "
        f"altered at aggression={lo:.4f} ({altered_lo})\n"
        f"seed={fixed_seed}\n"
        f"Input:      {text!r}\n"
        f"Output lo:  {output_lo!r}\n"
        f"Output hi:  {output_hi!r}"
    )
