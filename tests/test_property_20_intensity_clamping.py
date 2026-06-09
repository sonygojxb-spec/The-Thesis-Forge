"""
Property 20: Intensity application, clamping, and rounding.

For all requested intensity values, resolved enabled flags/aggression are applied;
<1 clamps to 1, >5 clamps to 5; non-integers round to nearest with halves up.

Requirements: 11.2, 11.5, 11.6

# Feature: ultimate-humanizer, Property 20: Intensity application, clamping, and rounding
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from humanizer.config import INTENSITY_PROFILES
from humanizer.pipeline import HumanizationPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_resolved_intensity(value: float) -> int:
    """Compute the expected resolved intensity level using the spec formula.

    1. Round non-integers with halves up: math.floor(value + 0.5)
    2. Clamp to [1, 5]: max(1, min(5, ...))
    """
    rounded = int(math.floor(value + 0.5))
    return max(1, min(5, rounded))


# ---------------------------------------------------------------------------
# Sub-property 1: Below 1 clamps to 1
# ---------------------------------------------------------------------------


@given(
    intensity=st.one_of(
        st.floats(min_value=-10.0, max_value=0.4, allow_nan=False, allow_infinity=False),
        st.integers(min_value=-10, max_value=0),
    ),
)
@settings(max_examples=100)
def test_below_1_clamps_to_1(intensity) -> None:
    """Property 20.1: Below 1 clamps to 1.

    intensity=0, -5, 0.4 → pipeline.intensity == 1

    **Validates: Requirements 11.5**
    """
    pipeline = HumanizationPipeline(
        intensity=intensity,
        api_key="fake-key",
        base_url="http://fake",
    )

    assert pipeline.intensity == 1, (
        f"Expected intensity clamped to 1 for input {intensity}, "
        f"got {pipeline.intensity}"
    )


# ---------------------------------------------------------------------------
# Sub-property 2: Above 5 clamps to 5
# ---------------------------------------------------------------------------


@given(
    intensity=st.one_of(
        st.floats(min_value=5.5, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.integers(min_value=6, max_value=10),
    ),
)
@settings(max_examples=100)
def test_above_5_clamps_to_5(intensity) -> None:
    """Property 20.2: Above 5 clamps to 5.

    intensity=6, 10, 5.9 → pipeline.intensity == 5

    **Validates: Requirements 11.5**
    """
    pipeline = HumanizationPipeline(
        intensity=intensity,
        api_key="fake-key",
        base_url="http://fake",
    )

    assert pipeline.intensity == 5, (
        f"Expected intensity clamped to 5 for input {intensity}, "
        f"got {pipeline.intensity}"
    )


# ---------------------------------------------------------------------------
# Sub-property 3: Non-integers round with halves up
# ---------------------------------------------------------------------------


@given(
    intensity=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_non_integers_round_halves_up(intensity: float) -> None:
    """Property 20.3: Non-integers round with halves up.

    2.5→3, 3.4→3, 3.6→4, 1.5→2

    **Validates: Requirements 11.6**
    """
    pipeline = HumanizationPipeline(
        intensity=intensity,
        api_key="fake-key",
        base_url="http://fake",
    )

    expected = _expected_resolved_intensity(intensity)

    assert pipeline.intensity == expected, (
        f"For intensity={intensity}, expected resolved level {expected}, "
        f"got {pipeline.intensity}. "
        f"(math.floor({intensity} + 0.5) = {int(math.floor(intensity + 0.5))})"
    )


# ---------------------------------------------------------------------------
# Sub-property 4: Resolved profile matches INTENSITY_PROFILES[resolved_level]
# ---------------------------------------------------------------------------


@given(
    intensity=st.one_of(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.integers(min_value=-10, max_value=10),
    ),
)
@settings(max_examples=100)
def test_resolved_profile_matches_intensity_profiles(intensity) -> None:
    """Property 20.4: Resolved profile matches INTENSITY_PROFILES[resolved_level].

    After rounding+clamping, the pipeline stage_config must start from the
    correct INTENSITY_PROFILES entry for the resolved level.

    **Validates: Requirements 11.2**
    """
    pipeline = HumanizationPipeline(
        intensity=intensity,
        api_key="fake-key",
        base_url="http://fake",
    )

    resolved_level = pipeline.intensity
    expected_profile = INTENSITY_PROFILES[resolved_level]

    # Every key in the expected profile should appear in stage_config with
    # the same value (stage_overrides=None so no overrides applied)
    for key, expected_value in expected_profile.items():
        actual_value = pipeline.stage_config.get(key)
        assert actual_value == expected_value, (
            f"For intensity={intensity} (resolved to level {resolved_level}), "
            f"stage_config[{key!r}] = {actual_value}, "
            f"expected {expected_value} from INTENSITY_PROFILES[{resolved_level}]"
        )
