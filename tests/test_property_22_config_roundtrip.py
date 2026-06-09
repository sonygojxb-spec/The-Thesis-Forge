# Feature: ultimate-humanizer, Property 22: Config round-trip equivalence
"""
Property 22: Config round-trip equivalence — for all valid configs (hypothesis-generated),
deserialize(serialize(c)) equals c field-by-field across intensity, every toggle, both
profile values, and every per-stage aggression.

**Validates: Requirements 13.1, 13.2, 13.3**
"""

from hypothesis import given, settings, strategies as st
from hypothesis.strategies import SearchStrategy

from humanizer.config_serializer import ConfigSerializer, PipelineConfig


# ---------------------------------------------------------------------------
# Composite strategy: valid PipelineConfig
# ---------------------------------------------------------------------------

@st.composite
def valid_pipeline_config(draw: st.DrawFn) -> PipelineConfig:
    """Generate a valid PipelineConfig with all fields in their allowed ranges."""

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

    return PipelineConfig(
        intensity=intensity,
        semantic_transform_enabled=semantic_transform_enabled,
        iterative_paraphrase_enabled=iterative_paraphrase_enabled,
        retrieval_augmented_enabled=retrieval_augmented_enabled,
        stylometric_enabled=stylometric_enabled,
        perplexity_optimize_enabled=perplexity_optimize_enabled,
        adversarial_enabled=adversarial_enabled,
        error_injection_enabled=error_injection_enabled,
        detector_optimize_enabled=detector_optimize_enabled,
        classifier_enabled=classifier_enabled,
        semantic_transform_aggression=semantic_transform_aggression,
        iterative_paraphrase_aggression=iterative_paraphrase_aggression,
        retrieval_augmented_aggression=retrieval_augmented_aggression,
        stylometric_aggression=stylometric_aggression,
        perplexity_optimize_aggression=perplexity_optimize_aggression,
        adversarial_aggression=adversarial_aggression,
        error_injection_aggression=error_injection_aggression,
        detector_optimize_aggression=detector_optimize_aggression,
        target_perplexity_mean=target_perplexity_mean,
        target_perplexity_variance=target_perplexity_variance,
    )


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

@given(config=valid_pipeline_config())
@settings(max_examples=100)
def test_config_round_trip_equivalence(config: PipelineConfig) -> None:
    """Property 22: serialize then deserialize produces an equivalent config.

    For all valid PipelineConfig instances, deserialize(serialize(config)) must
    equal config field-by-field: intensity, every toggle, both profile values,
    and every per-stage aggression.

    **Validates: Requirements 13.1, 13.2, 13.3**
    """
    serialized = ConfigSerializer.serialize(config)
    restored = ConfigSerializer.deserialize(serialized)

    # Intensity
    assert restored.intensity == config.intensity, (
        f"intensity mismatch: {restored.intensity} != {config.intensity}"
    )

    # Stage toggles
    assert restored.semantic_transform_enabled == config.semantic_transform_enabled
    assert restored.iterative_paraphrase_enabled == config.iterative_paraphrase_enabled
    assert restored.retrieval_augmented_enabled == config.retrieval_augmented_enabled
    assert restored.stylometric_enabled == config.stylometric_enabled
    assert restored.perplexity_optimize_enabled == config.perplexity_optimize_enabled
    assert restored.adversarial_enabled == config.adversarial_enabled
    assert restored.error_injection_enabled == config.error_injection_enabled
    assert restored.detector_optimize_enabled == config.detector_optimize_enabled
    assert restored.classifier_enabled == config.classifier_enabled

    # Per-stage aggression values
    assert restored.semantic_transform_aggression == config.semantic_transform_aggression, (
        f"semantic_transform_aggression mismatch: {restored.semantic_transform_aggression} != {config.semantic_transform_aggression}"
    )
    assert restored.iterative_paraphrase_aggression == config.iterative_paraphrase_aggression, (
        f"iterative_paraphrase_aggression mismatch: {restored.iterative_paraphrase_aggression} != {config.iterative_paraphrase_aggression}"
    )
    assert restored.retrieval_augmented_aggression == config.retrieval_augmented_aggression, (
        f"retrieval_augmented_aggression mismatch: {restored.retrieval_augmented_aggression} != {config.retrieval_augmented_aggression}"
    )
    assert restored.stylometric_aggression == config.stylometric_aggression, (
        f"stylometric_aggression mismatch: {restored.stylometric_aggression} != {config.stylometric_aggression}"
    )
    assert restored.perplexity_optimize_aggression == config.perplexity_optimize_aggression, (
        f"perplexity_optimize_aggression mismatch: {restored.perplexity_optimize_aggression} != {config.perplexity_optimize_aggression}"
    )
    assert restored.adversarial_aggression == config.adversarial_aggression, (
        f"adversarial_aggression mismatch: {restored.adversarial_aggression} != {config.adversarial_aggression}"
    )
    assert restored.error_injection_aggression == config.error_injection_aggression, (
        f"error_injection_aggression mismatch: {restored.error_injection_aggression} != {config.error_injection_aggression}"
    )
    assert restored.detector_optimize_aggression == config.detector_optimize_aggression, (
        f"detector_optimize_aggression mismatch: {restored.detector_optimize_aggression} != {config.detector_optimize_aggression}"
    )

    # Target perplexity profile
    assert restored.target_perplexity_mean == config.target_perplexity_mean, (
        f"target_perplexity_mean mismatch: {restored.target_perplexity_mean} != {config.target_perplexity_mean}"
    )
    assert restored.target_perplexity_variance == config.target_perplexity_variance, (
        f"target_perplexity_variance mismatch: {restored.target_perplexity_variance} != {config.target_perplexity_variance}"
    )
