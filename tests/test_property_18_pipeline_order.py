"""
Property 18: Pipeline executed-set and deterministic order.

For all configs and inputs, executed stages equal the enabled set, appear in
canonical order, are identical across runs, emit no callback for skipped stages,
and return input unchanged when all stages are disabled (fakes injected for
model-backed stages).

Requirements: 10.1, 10.2, 10.3, 10.6, 10.9

# Feature: ultimate-humanizer, Property 18: Pipeline executed-set and deterministic order
"""

from __future__ import annotations

from typing import List, Tuple
from unittest.mock import patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from humanizer.pipeline import HumanizationPipeline
from tests.strategies import multi_sentence_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _identity_stage_factory(stage_key, stream_callback=None):
    """Return a simple identity stage that passes text through unchanged.

    This avoids needing real NLP models or LLM calls — we only care about
    execution order, not output quality.
    """

    class _IdentityStage:
        """A no-op stage that returns input text unchanged."""

        def process(self, text: str) -> str:
            return text

    return _IdentityStage()


def _make_stage_overrides(enabled_set: set) -> dict:
    """Build a stage_overrides dict that enables only the stages in enabled_set."""
    return {stage: (stage in enabled_set) for stage in ALL_STAGES}


def _collect_callbacks(pipeline: HumanizationPipeline, text: str) -> List[Tuple[str, str]]:
    """Run the pipeline and collect all progress callback invocations.

    Returns a list of (stage_display_name, status) tuples in call order.
    """
    callbacks: List[Tuple[str, str]] = []

    def _cb(stage_name: str, status: str) -> None:
        callbacks.append((stage_name, status))

    pipeline.progress_callback = _cb

    with patch.object(
        HumanizationPipeline, "_build_stage", side_effect=_identity_stage_factory
    ):
        pipeline.process(text)

    return callbacks


def _get_executed_stages_from_callbacks(
    callbacks: List[Tuple[str, str]],
) -> List[str]:
    """Extract the ordered list of stage display names that were actually executed.

    A stage is considered executed if it received a "running" callback.
    """
    return [name for name, status in callbacks if status == "running"]


# Strategy to generate a random subset of stages to enable
@st.composite
def stage_subset(draw: st.DrawFn) -> set:
    """Draw a non-empty subset of pipeline stages."""
    flags = draw(st.lists(st.booleans(), min_size=len(ALL_STAGES), max_size=len(ALL_STAGES)))
    subset = {stage for stage, flag in zip(ALL_STAGES, flags) if flag}
    # Ensure at least one stage is enabled for non-trivial testing
    assume(len(subset) > 0)
    return subset


# ---------------------------------------------------------------------------
# Sub-property 1: Executed stages equal the enabled set
# ---------------------------------------------------------------------------


@given(
    text=multi_sentence_text(min_sentences=2, max_sentences=4),
    enabled=stage_subset(),
)
@settings(max_examples=100)
def test_executed_stages_equal_enabled_set(text: str, enabled: set) -> None:
    """Property 18.1: Executed stages equal the enabled set.

    Use progress_callback to record which stages ran, verify they match
    get_enabled_stages().

    Validates: Requirements 10.1, 10.2, 10.3
    """
    overrides = _make_stage_overrides(enabled)
    pipeline = HumanizationPipeline(
        intensity=3,
        stage_overrides=overrides,
        api_key="fake-key",
        base_url="http://fake",
    )

    # Verify get_enabled_stages returns exactly the enabled set
    expected_enabled = pipeline.get_enabled_stages()
    assert set(expected_enabled) == enabled, (
        f"get_enabled_stages mismatch.\n"
        f"Expected: {enabled}\n"
        f"Got: {set(expected_enabled)}"
    )

    # Run the pipeline and collect callback stage names
    callbacks = _collect_callbacks(pipeline, text)
    executed_display_names = _get_executed_stages_from_callbacks(callbacks)

    # Map enabled stage keys to their display names
    expected_display_names = [
        HumanizationPipeline.STAGE_NAMES[s] for s in expected_enabled
    ]

    assert set(executed_display_names) == set(expected_display_names), (
        f"Executed stages do not match enabled set.\n"
        f"Enabled (display): {expected_display_names}\n"
        f"Executed (display): {executed_display_names}"
    )


# ---------------------------------------------------------------------------
# Sub-property 2: Stages appear in canonical STAGE_ORDER
# ---------------------------------------------------------------------------


@given(
    text=multi_sentence_text(min_sentences=2, max_sentences=4),
    enabled=stage_subset(),
)
@settings(max_examples=100)
def test_stages_in_canonical_order(text: str, enabled: set) -> None:
    """Property 18.2: Stages appear in canonical STAGE_ORDER.

    Verify the recorded execution order matches STAGE_ORDER filtered to
    enabled stages.

    Validates: Requirements 10.1
    """
    overrides = _make_stage_overrides(enabled)
    pipeline = HumanizationPipeline(
        intensity=3,
        stage_overrides=overrides,
        api_key="fake-key",
        base_url="http://fake",
    )

    callbacks = _collect_callbacks(pipeline, text)
    executed_display_names = _get_executed_stages_from_callbacks(callbacks)

    # Expected order: filter STAGE_ORDER to enabled and map to display names
    expected_order = [
        HumanizationPipeline.STAGE_NAMES[s]
        for s in HumanizationPipeline.STAGE_ORDER
        if s in enabled
    ]

    assert executed_display_names == expected_order, (
        f"Execution order does not match canonical STAGE_ORDER.\n"
        f"Expected: {expected_order}\n"
        f"Got: {executed_display_names}"
    )


# ---------------------------------------------------------------------------
# Sub-property 3: Identical across runs (deterministic)
# ---------------------------------------------------------------------------


@given(
    text=multi_sentence_text(min_sentences=2, max_sentences=4),
    enabled=stage_subset(),
)
@settings(max_examples=100)
def test_deterministic_across_runs(text: str, enabled: set) -> None:
    """Property 18.3: Identical across runs.

    Run process() twice with same config, verify same stages executed in
    same order.

    Validates: Requirements 10.1
    """
    overrides = _make_stage_overrides(enabled)

    # First run
    pipeline1 = HumanizationPipeline(
        intensity=3,
        stage_overrides=overrides,
        api_key="fake-key",
        base_url="http://fake",
        seed=42,
    )
    callbacks1 = _collect_callbacks(pipeline1, text)

    # Second run (same config)
    pipeline2 = HumanizationPipeline(
        intensity=3,
        stage_overrides=overrides,
        api_key="fake-key",
        base_url="http://fake",
        seed=42,
    )
    callbacks2 = _collect_callbacks(pipeline2, text)

    executed1 = _get_executed_stages_from_callbacks(callbacks1)
    executed2 = _get_executed_stages_from_callbacks(callbacks2)

    assert executed1 == executed2, (
        f"Non-deterministic execution order!\n"
        f"Run 1: {executed1}\n"
        f"Run 2: {executed2}"
    )


# ---------------------------------------------------------------------------
# Sub-property 4: No callback for skipped stages
# ---------------------------------------------------------------------------


@given(
    text=multi_sentence_text(min_sentences=2, max_sentences=4),
    enabled=stage_subset(),
)
@settings(max_examples=100)
def test_no_callback_for_skipped_stages(text: str, enabled: set) -> None:
    """Property 18.4: No callback for skipped stages.

    Verify disabled stages never appear in progress callbacks.

    Validates: Requirements 10.6
    """
    overrides = _make_stage_overrides(enabled)
    pipeline = HumanizationPipeline(
        intensity=3,
        stage_overrides=overrides,
        api_key="fake-key",
        base_url="http://fake",
    )

    callbacks = _collect_callbacks(pipeline, text)

    # Get display names of disabled stages
    disabled_stages = set(ALL_STAGES) - enabled
    disabled_display_names = {
        HumanizationPipeline.STAGE_NAMES[s] for s in disabled_stages
    }

    # Check that no disabled stage appears in any callback
    all_callback_names = {name for name, _status in callbacks}
    unexpected = all_callback_names & disabled_display_names

    assert not unexpected, (
        f"Disabled stages received callbacks!\n"
        f"Unexpected callbacks for: {unexpected}\n"
        f"All callbacks: {callbacks}"
    )


# ---------------------------------------------------------------------------
# Sub-property 5: All stages disabled → input unchanged
# ---------------------------------------------------------------------------


@given(text=multi_sentence_text(min_sentences=2, max_sentences=4))
@settings(max_examples=100)
def test_all_stages_disabled_returns_input_unchanged(text: str) -> None:
    """Property 18.5: All stages disabled → input unchanged.

    Create a pipeline with all stage_overrides=False, verify output == input.

    Validates: Requirements 10.9
    """
    # Disable every stage
    overrides = {stage: False for stage in ALL_STAGES}
    pipeline = HumanizationPipeline(
        intensity=3,
        stage_overrides=overrides,
        api_key="fake-key",
        base_url="http://fake",
    )

    # No need to mock _build_stage since no stages will execute
    result = pipeline.process(text)

    assert result == text, (
        f"Pipeline did not return input unchanged when all stages disabled.\n"
        f"Input:  {text!r}\n"
        f"Output: {result!r}"
    )
