# Feature: ultimate-humanizer, Property 23: Config invalid-field rejection
"""
Property 23: Config invalid-field rejection — for all representations with a missing field
or out-of-range value, `deserialize` raises an error naming the invalid field.

**Validates: Requirements 13.4**
"""

import json

import pytest
from hypothesis import given, settings, strategies as st, assume
from hypothesis.strategies import SearchStrategy

from humanizer.config_serializer import ConfigError, ConfigSerializer, PipelineConfig


# ---------------------------------------------------------------------------
# Composite strategy: valid config as a dict (for corruption)
# ---------------------------------------------------------------------------

@st.composite
def valid_config_dict(draw: st.DrawFn) -> dict:
    """Generate a valid PipelineConfig as a raw dict suitable for corruption."""

    intensity = draw(st.integers(min_value=1, max_value=5))

    # Stage toggles (9 booleans)
    semantic_transform_enabled = draw(st.booleans())
    iterative_paraphrase_enabled = draw(st.booleans())
    retrieval_augmented_enabled = draw(st.booleans())
    stylometric_enabled = draw(st.booleans())
    perplexity_optimize_enabled = draw(st.booleans())
    adversarial_enabled = draw(st.booleans())
    error_injection_enabled = draw(st.booleans())
    detector_optimize_enabled = draw(st.booleans())
    classifier_enabled = draw(st.booleans())

    # Per-stage aggression values (8 floats in [0.0, 1.0])
    aggression_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    semantic_transform_aggression = draw(aggression_strategy)
    iterative_paraphrase_aggression = draw(aggression_strategy)
    retrieval_augmented_aggression = draw(aggression_strategy)
    stylometric_aggression = draw(aggression_strategy)
    perplexity_optimize_aggression = draw(aggression_strategy)
    adversarial_aggression = draw(aggression_strategy)
    error_injection_aggression = draw(aggression_strategy)
    detector_optimize_aggression = draw(aggression_strategy)

    # Target perplexity profile
    target_perplexity_mean = draw(
        st.floats(min_value=0.01, max_value=200.0, allow_nan=False, allow_infinity=False)
    )
    target_perplexity_variance = draw(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )

    return {
        "intensity": intensity,
        "semantic_transform_enabled": semantic_transform_enabled,
        "iterative_paraphrase_enabled": iterative_paraphrase_enabled,
        "retrieval_augmented_enabled": retrieval_augmented_enabled,
        "stylometric_enabled": stylometric_enabled,
        "perplexity_optimize_enabled": perplexity_optimize_enabled,
        "adversarial_enabled": adversarial_enabled,
        "error_injection_enabled": error_injection_enabled,
        "detector_optimize_enabled": detector_optimize_enabled,
        "classifier_enabled": classifier_enabled,
        "semantic_transform_aggression": semantic_transform_aggression,
        "iterative_paraphrase_aggression": iterative_paraphrase_aggression,
        "retrieval_augmented_aggression": retrieval_augmented_aggression,
        "stylometric_aggression": stylometric_aggression,
        "perplexity_optimize_aggression": perplexity_optimize_aggression,
        "adversarial_aggression": adversarial_aggression,
        "error_injection_aggression": error_injection_aggression,
        "detector_optimize_aggression": detector_optimize_aggression,
        "target_perplexity_mean": target_perplexity_mean,
        "target_perplexity_variance": target_perplexity_variance,
    }


# ---------------------------------------------------------------------------
# Field classification for corruption strategies
# ---------------------------------------------------------------------------

_ALL_FIELDS = list({
    "intensity",
    "semantic_transform_enabled",
    "iterative_paraphrase_enabled",
    "retrieval_augmented_enabled",
    "stylometric_enabled",
    "perplexity_optimize_enabled",
    "adversarial_enabled",
    "error_injection_enabled",
    "detector_optimize_enabled",
    "classifier_enabled",
    "semantic_transform_aggression",
    "iterative_paraphrase_aggression",
    "retrieval_augmented_aggression",
    "stylometric_aggression",
    "perplexity_optimize_aggression",
    "adversarial_aggression",
    "error_injection_aggression",
    "detector_optimize_aggression",
    "target_perplexity_mean",
    "target_perplexity_variance",
})

# Out-of-range corruption values by field type
_INTENSITY_INVALID = [0, 6, "string"]
_TOGGLE_INVALID = [1, "true", None]
_AGGRESSION_INVALID = [-0.1, 1.5, "string"]
_PERPLEXITY_MEAN_INVALID = [0, -1]
_PERPLEXITY_VARIANCE_INVALID = [-1]

_TOGGLE_FIELDS = {
    "semantic_transform_enabled",
    "iterative_paraphrase_enabled",
    "retrieval_augmented_enabled",
    "stylometric_enabled",
    "perplexity_optimize_enabled",
    "adversarial_enabled",
    "error_injection_enabled",
    "detector_optimize_enabled",
    "classifier_enabled",
}

_AGGRESSION_FIELDS = {
    "semantic_transform_aggression",
    "iterative_paraphrase_aggression",
    "retrieval_augmented_aggression",
    "stylometric_aggression",
    "perplexity_optimize_aggression",
    "adversarial_aggression",
    "error_injection_aggression",
    "detector_optimize_aggression",
}


def _invalid_values_for_field(field_name: str) -> list:
    """Return a list of invalid values for the given field name."""
    if field_name == "intensity":
        return _INTENSITY_INVALID
    elif field_name in _TOGGLE_FIELDS:
        return _TOGGLE_INVALID
    elif field_name in _AGGRESSION_FIELDS:
        return _AGGRESSION_INVALID
    elif field_name == "target_perplexity_mean":
        return _PERPLEXITY_MEAN_INVALID
    elif field_name == "target_perplexity_variance":
        return _PERPLEXITY_VARIANCE_INVALID
    else:
        return [None]


# ---------------------------------------------------------------------------
# Property 23a: Missing field → ConfigError naming that field
# ---------------------------------------------------------------------------

@given(
    config_dict=valid_config_dict(),
    field_index=st.integers(min_value=0, max_value=len(_ALL_FIELDS) - 1),
)
@settings(max_examples=100)
def test_missing_field_raises_config_error(config_dict: dict, field_index: int) -> None:
    """Property 23a: For all valid configs, deleting a random field causes deserialize
    to raise ConfigError naming that field.

    **Validates: Requirements 13.4**
    """
    field_to_delete = _ALL_FIELDS[field_index]

    # Remove the field from the dict
    corrupted = dict(config_dict)
    del corrupted[field_to_delete]

    blob = json.dumps(corrupted)

    with pytest.raises(ConfigError) as exc_info:
        ConfigSerializer.deserialize(blob)

    assert exc_info.value.field == field_to_delete, (
        f"Expected ConfigError.field to be '{field_to_delete}', "
        f"got '{exc_info.value.field}'"
    )


# ---------------------------------------------------------------------------
# Property 23b: Out-of-range value → ConfigError naming that field
# ---------------------------------------------------------------------------

@given(
    config_dict=valid_config_dict(),
    field_index=st.integers(min_value=0, max_value=len(_ALL_FIELDS) - 1),
    invalid_value_index=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=100)
def test_out_of_range_value_raises_config_error(
    config_dict: dict, field_index: int, invalid_value_index: int
) -> None:
    """Property 23b: For all valid configs, setting a random field to an invalid value
    causes deserialize to raise ConfigError naming that field.

    **Validates: Requirements 13.4**
    """
    field_to_corrupt = _ALL_FIELDS[field_index]
    invalid_values = _invalid_values_for_field(field_to_corrupt)

    # Select an invalid value using modulo to stay in bounds
    invalid_value = invalid_values[invalid_value_index % len(invalid_values)]

    # Set the field to the invalid value
    corrupted = dict(config_dict)
    corrupted[field_to_corrupt] = invalid_value

    blob = json.dumps(corrupted)

    with pytest.raises(ConfigError) as exc_info:
        ConfigSerializer.deserialize(blob)

    assert exc_info.value.field == field_to_corrupt, (
        f"Expected ConfigError.field to be '{field_to_corrupt}', "
        f"got '{exc_info.value.field}'"
    )
