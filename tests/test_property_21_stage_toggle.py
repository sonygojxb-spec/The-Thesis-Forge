# Feature: ultimate-humanizer, Property 21: Stage-toggle override semantics
"""
Property 21: Stage-toggle override semantics.

For all configurations with a Stage_Toggle override for a new stage, the override
decides execution while the profile aggression still applies.

Test cases:
1. A stage enabled by profile but disabled by override → NOT executed
2. A stage disabled by profile but enabled by override → IS executed
3. In both cases: the aggression value from the profile still applies
   (stage_config has the profile's aggression even when override changes enabled state)

**Validates: Requirements 11.4**
"""

from __future__ import annotations

from typing import List, Tuple
from unittest.mock import patch, MagicMock

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from humanizer.pipeline import HumanizationPipeline
from humanizer.config import INTENSITY_PROFILES
from tests.strategies import multi_sentence_text


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# New stages (those that use <key>_enabled pattern) — excludes existing stages
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

# All stages in canonical order (mirrors HumanizationPipeline.STAGE_ORDER)
ALL_STAGES = [
    "structural",
    "lexical",
    "semantic_transform",
    "iterative_paraphrase",
    "llm_rewrite",
    "retrieval_augmented",
    "stylometric",
    "perplexity",
    "perplexity_optimize",
    "adversarial",
    "error_injection",
    "postprocess",
    "detector_optimize",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_stage_factory(stage_key, stream_callback=None):
    """Return a simple identity stage that passes text through unchanged."""

    class _IdentityStage:
        def process(self, text: str) -> str:
            return text

    return _IdentityStage()


def _collect_executed_stages(pipeline: HumanizationPipeline, text: str) -> List[str]:
    """Run the pipeline with identity stages and return list of stage keys that executed.

    Uses progress_callback to track which stages received a "running" event.
    """
    executed: List[str] = []

    # Map display names back to keys for comparison
    display_to_key = {v: k for k, v in HumanizationPipeline.STAGE_NAMES.items()}

    def _cb(stage_name: str, status: str) -> None:
        if status == "running":
            key = display_to_key.get(stage_name)
            if key:
                executed.append(key)

    pipeline.progress_callback = _cb

    with patch.object(
        HumanizationPipeline, "_build_stage", side_effect=_identity_stage_factory
    ):
        pipeline.process(text)

    return executed


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def intensity_and_stage_enabled_by_profile(draw: st.DrawFn) -> Tuple[int, str]:
    """Draw an intensity level and a new stage that IS enabled at that level."""
    intensity = draw(st.integers(min_value=1, max_value=5))
    profile = INTENSITY_PROFILES[intensity]

    # Find new stages enabled at this intensity
    enabled_stages = [
        s for s in NEW_STAGES
        if profile.get(f"{s}_enabled", False)
    ]
    assume(len(enabled_stages) > 0)
    stage = draw(st.sampled_from(enabled_stages))
    return intensity, stage


@st.composite
def intensity_and_stage_disabled_by_profile(draw: st.DrawFn) -> Tuple[int, str]:
    """Draw an intensity level and a new stage that is NOT enabled at that level."""
    intensity = draw(st.integers(min_value=1, max_value=5))
    profile = INTENSITY_PROFILES[intensity]

    # Find new stages disabled at this intensity
    disabled_stages = [
        s for s in NEW_STAGES
        if not profile.get(f"{s}_enabled", False)
    ]
    assume(len(disabled_stages) > 0)
    stage = draw(st.sampled_from(disabled_stages))
    return intensity, stage


# ---------------------------------------------------------------------------
# Sub-property 1: Stage enabled by profile but disabled by override → NOT executed
# ---------------------------------------------------------------------------


@given(
    text=multi_sentence_text(min_sentences=2, max_sentences=4),
    config=intensity_and_stage_enabled_by_profile(),
)
@settings(max_examples=100)
def test_override_disable_prevents_execution(text: str, config: Tuple[int, str]) -> None:
    """Property 21.1: A stage enabled by profile but disabled by override is NOT executed.

    When the profile enables a stage but the user provides stage_overrides={stage: False},
    the pipeline must skip that stage.

    **Validates: Requirements 11.4**
    """
    intensity, stage = config

    # Override disables the stage
    pipeline = HumanizationPipeline(
        intensity=intensity,
        stage_overrides={stage: False},
        api_key="fake-key",
        base_url="http://fake",
    )

    # Verify stage is NOT in enabled set
    enabled = pipeline.get_enabled_stages()
    assert stage not in enabled, (
        f"Stage '{stage}' should be disabled by override but is in enabled set.\n"
        f"Intensity: {intensity}, Enabled: {enabled}"
    )

    # Verify stage is NOT executed via progress callbacks
    executed = _collect_executed_stages(pipeline, text)
    assert stage not in executed, (
        f"Stage '{stage}' was executed despite being disabled by override.\n"
        f"Intensity: {intensity}, Executed: {executed}"
    )


# ---------------------------------------------------------------------------
# Sub-property 2: Stage disabled by profile but enabled by override → IS executed
# ---------------------------------------------------------------------------


@given(
    text=multi_sentence_text(min_sentences=2, max_sentences=4),
    config=intensity_and_stage_disabled_by_profile(),
)
@settings(max_examples=100)
def test_override_enable_forces_execution(text: str, config: Tuple[int, str]) -> None:
    """Property 21.2: A stage disabled by profile but enabled by override IS executed.

    When the profile disables a stage but the user provides stage_overrides={stage: True},
    the pipeline must execute that stage.

    **Validates: Requirements 11.4**
    """
    intensity, stage = config

    # Override enables the stage
    pipeline = HumanizationPipeline(
        intensity=intensity,
        stage_overrides={stage: True},
        api_key="fake-key",
        base_url="http://fake",
    )

    # Verify stage IS in enabled set
    enabled = pipeline.get_enabled_stages()
    assert stage in enabled, (
        f"Stage '{stage}' should be enabled by override but is NOT in enabled set.\n"
        f"Intensity: {intensity}, Enabled: {enabled}"
    )

    # Verify stage IS executed via progress callbacks
    executed = _collect_executed_stages(pipeline, text)
    assert stage in executed, (
        f"Stage '{stage}' was NOT executed despite being enabled by override.\n"
        f"Intensity: {intensity}, Executed: {executed}"
    )


# ---------------------------------------------------------------------------
# Sub-property 3: Profile aggression still applies when override changes enabled
# ---------------------------------------------------------------------------


@given(
    text=multi_sentence_text(min_sentences=2, max_sentences=4),
    config=intensity_and_stage_disabled_by_profile(),
)
@settings(max_examples=100)
def test_override_preserves_profile_aggression(text: str, config: Tuple[int, str]) -> None:
    """Property 21.3: When override enables a stage, aggression still comes from profile.

    Even when a stage_override changes the enabled state, the aggression value
    in stage_config must match the profile's aggression for that stage.

    **Validates: Requirements 11.4**
    """
    intensity, stage = config
    profile = INTENSITY_PROFILES[intensity]
    expected_aggression = profile.get(f"{stage}_aggression", 0.5)

    # Override enables the stage
    pipeline = HumanizationPipeline(
        intensity=intensity,
        stage_overrides={stage: True},
        api_key="fake-key",
        base_url="http://fake",
    )

    # Verify the aggression in stage_config matches the profile
    actual_aggression = pipeline.stage_config.get(f"{stage}_aggression")
    assert actual_aggression == expected_aggression, (
        f"Aggression mismatch for '{stage}' at intensity {intensity}.\n"
        f"Expected (from profile): {expected_aggression}\n"
        f"Actual (in stage_config): {actual_aggression}\n"
        f"The override should only affect enabled state, not aggression."
    )


@given(
    text=multi_sentence_text(min_sentences=2, max_sentences=4),
    config=intensity_and_stage_enabled_by_profile(),
)
@settings(max_examples=100)
def test_override_disable_preserves_profile_aggression(text: str, config: Tuple[int, str]) -> None:
    """Property 21.4: When override disables a stage, aggression still comes from profile.

    Even when a stage_override disables a stage, the aggression value in
    stage_config must match the profile's aggression for that stage (it just
    won't be used since the stage doesn't execute).

    **Validates: Requirements 11.4**
    """
    intensity, stage = config
    profile = INTENSITY_PROFILES[intensity]
    expected_aggression = profile.get(f"{stage}_aggression", 0.5)

    # Override disables the stage
    pipeline = HumanizationPipeline(
        intensity=intensity,
        stage_overrides={stage: False},
        api_key="fake-key",
        base_url="http://fake",
    )

    # Verify the aggression in stage_config is unchanged from profile
    actual_aggression = pipeline.stage_config.get(f"{stage}_aggression")
    assert actual_aggression == expected_aggression, (
        f"Aggression mismatch for '{stage}' at intensity {intensity}.\n"
        f"Expected (from profile): {expected_aggression}\n"
        f"Actual (in stage_config): {actual_aggression}\n"
        f"Disabling via override should NOT change the aggression value."
    )
