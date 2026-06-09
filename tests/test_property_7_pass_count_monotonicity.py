"""
Property 7: Paraphrase pass-count monotonicity.

For all aggression in [0, 1], pass count is >=1, <=5, equals 1 at 0.0,
reaches 5 at 1.0, and is non-decreasing in aggression.

Requirements: 1.2
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from humanizer.stage_iterative import IterativeParaphraser

# Feature: ultimate-humanizer, Property 7: Paraphrase pass-count monotonicity


@given(aggression=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_pass_count_range(aggression: float) -> None:
    """Property 7 (range): pass_count is always in [1, 5].

    Validates: Requirements 1.2
    """
    paraphraser = IterativeParaphraser(aggression=aggression)
    count = paraphraser.pass_count

    assert 1 <= count <= 5, (
        f"pass_count={count} is outside [1, 5] for aggression={aggression}"
    )


@given(aggression=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_pass_count_boundary_zero(aggression: float) -> None:
    """Property 7 (boundary): pass_count equals 1 at aggression=0.0.

    Validates: Requirements 1.2
    """
    paraphraser = IterativeParaphraser(aggression=0.0)
    count = paraphraser.pass_count

    assert count == 1, (
        f"pass_count={count} at aggression=0.0, expected 1"
    )


@given(aggression=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_pass_count_boundary_one(aggression: float) -> None:
    """Property 7 (boundary): pass_count reaches 5 at aggression=1.0.

    Validates: Requirements 1.2
    """
    paraphraser = IterativeParaphraser(aggression=1.0)
    count = paraphraser.pass_count

    assert count == 5, (
        f"pass_count={count} at aggression=1.0, expected 5"
    )


@given(
    a=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_pass_count_monotonicity(a: float, b: float) -> None:
    """Property 7 (monotonicity): pass_count is non-decreasing in aggression.

    For all pairs (a, b) where a <= b, pass_count(a) <= pass_count(b).

    Validates: Requirements 1.2
    """
    # Ensure a <= b
    lo, hi = min(a, b), max(a, b)

    paraphraser_lo = IterativeParaphraser(aggression=lo)
    paraphraser_hi = IterativeParaphraser(aggression=hi)

    count_lo = paraphraser_lo.pass_count
    count_hi = paraphraser_hi.pass_count

    assert count_lo <= count_hi, (
        f"Monotonicity violated: pass_count({lo})={count_lo} > pass_count({hi})={count_hi}"
    )
