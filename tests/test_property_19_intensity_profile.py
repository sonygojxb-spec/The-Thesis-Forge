# Feature: ultimate-humanizer, Property 19: Intensity profile structure and monotonicity
"""
Property 19: Intensity profile structure and monotonicity —
For all levels 1-5 and all new stages, an enabled boolean and aggression float
in [0,1] exist, and aggression at L+1 >= aggression at L for L in 1..4.

Validates: Requirements 11.1, 11.3
"""

import pytest

from humanizer.config import INTENSITY_PROFILES

# The new stages added as part of the Ultimate Humanizer
NEW_STAGES = [
    "semantic_transform",
    "iterative_paraphrase",
    "retrieval_augmented",
    "stylometric",
    "perplexity_optimize",
    "adversarial",
    "error_injection",
    "detector_optimize",
]

LEVELS = [1, 2, 3, 4, 5]


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("stage", NEW_STAGES)
def test_enabled_is_boolean(level, stage):
    """Each new stage at every level has an enabled key that is a boolean."""
    profile = INTENSITY_PROFILES[level]
    key = f"{stage}_enabled"
    assert key in profile, f"Missing key '{key}' at level {level}"
    assert isinstance(profile[key], bool), (
        f"'{key}' at level {level} should be bool, got {type(profile[key])}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("stage", NEW_STAGES)
def test_aggression_is_float_in_range(level, stage):
    """Each new stage at every level has an aggression float in [0, 1]."""
    profile = INTENSITY_PROFILES[level]
    key = f"{stage}_aggression"
    assert key in profile, f"Missing key '{key}' at level {level}"
    value = profile[key]
    assert isinstance(value, (int, float)), (
        f"'{key}' at level {level} should be numeric, got {type(value)}"
    )
    assert 0.0 <= value <= 1.0, (
        f"'{key}' at level {level} = {value}, expected in [0, 1]"
    )


@pytest.mark.parametrize("stage", NEW_STAGES)
def test_aggression_monotonically_non_decreasing(stage):
    """Aggression at L+1 >= aggression at L for L in 1..4 for each new stage."""
    key = f"{stage}_aggression"
    for level in range(1, 5):
        current = INTENSITY_PROFILES[level][key]
        next_val = INTENSITY_PROFILES[level + 1][key]
        assert next_val >= current, (
            f"'{key}' not monotonically non-decreasing: "
            f"level {level} = {current}, level {level + 1} = {next_val}"
        )
