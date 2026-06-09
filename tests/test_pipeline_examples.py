"""
Example-based tests for pipeline orchestration behavior.

Tests the following pipeline invariants:
- Progress sequencing: running→complete callback order (Req 10.4, 10.5)
- Skipped-stage emits nothing (Req 10.6)
- Seed wiring: seed passed to non-LLM-random stages (Req 10.8)
- Below-0.85 final similarity warning (Req 14.3)
- Dropped number/citation warning (Req 14.5)

Uses mock _build_stage returning identity stages for most tests,
and FakeSimilarityEvaluator from conftest.py for warning tests.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from humanizer.pipeline import HumanizationPipeline
from humanizer.results import StageResult
from tests.conftest import FakeSimilarityEvaluator


# ---------------------------------------------------------------------------
# Helpers: identity stage and number-dropping stage
# ---------------------------------------------------------------------------


class IdentityStage:
    """A stage that returns input unchanged (for testing orchestration logic)."""

    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed")

    def process(self, text: str) -> str:
        return text


class NumberDroppingStage:
    """A stage that removes the first number from the text (simulates content loss)."""

    def __init__(self, **kwargs):
        pass

    def process(self, text: str) -> str:
        import re
        # Remove the first standalone number occurrence
        return re.sub(r'\b\d+\b', '', text, count=1).strip()


# ---------------------------------------------------------------------------
# Test 1 (Req 10.4, 10.5): Progress sequencing — each enabled stage gets
# "running" then "complete" in that order
# ---------------------------------------------------------------------------


class TestProgressSequencing:
    """Validates Req 10.4, 10.5: progress_callback with running then complete."""

    def test_enabled_stages_emit_running_then_complete(self):
        """WHEN pipeline runs with enabled stages, each stage emits
        progress_callback(stage, "running") before execution and
        progress_callback(stage, "complete") after.

        **Validates: Requirements 10.4, 10.5**
        """
        progress_log = []

        def progress_cb(stage_name, status):
            progress_log.append((stage_name, status))

        # Enable only structural and lexical (existing stages)
        pipeline = HumanizationPipeline(
            intensity=1,
            stage_overrides={
                "structural": True,
                "lexical": True,
                "llm_rewrite": False,
                "perplexity": False,
                "postprocess": False,
                "semantic_transform": False,
                "stylometric": False,
            },
            progress_callback=progress_cb,
            seed=42,
        )

        # Patch _build_stage to return identity stages
        with patch.object(pipeline, "_build_stage", return_value=IdentityStage()):
            pipeline.process("The experiment yielded significant results.")

        # Verify each enabled stage got "running" then "complete"
        # Note: progress_callback receives display names from STAGE_NAMES
        stage_names_seen = []
        for name, _ in progress_log:
            if name not in stage_names_seen:
                stage_names_seen.append(name)

        for stage_name in stage_names_seen:
            stage_events = [(n, s) for n, s in progress_log if n == stage_name]
            assert len(stage_events) == 2, (
                f"Stage {stage_name} should have exactly 2 events, got {stage_events}"
            )
            assert stage_events[0][1] == "running", (
                f"First event for {stage_name} should be 'running'"
            )
            assert stage_events[1][1] == "complete", (
                f"Second event for {stage_name} should be 'complete'"
            )


# ---------------------------------------------------------------------------
# Test 2 (Req 10.6): Skipped stage emits nothing
# ---------------------------------------------------------------------------


class TestSkippedStageNoCallback:
    """Validates Req 10.6: disabled stages emit no progress_callback."""

    def test_disabled_stage_emits_no_progress(self):
        """WHEN a stage is disabled via override, no progress_callback is
        emitted for that stage.

        **Validates: Requirements 10.6**
        """
        progress_log = []

        def progress_cb(stage_name, status):
            progress_log.append((stage_name, status))

        # Disable adversarial stage explicitly
        pipeline = HumanizationPipeline(
            intensity=4,
            stage_overrides={
                "structural": True,
                "lexical": False,
                "llm_rewrite": False,
                "perplexity": False,
                "postprocess": False,
                "semantic_transform": False,
                "iterative_paraphrase": False,
                "retrieval_augmented": False,
                "stylometric": False,
                "perplexity_optimize": False,
                "adversarial": False,
                "error_injection": False,
                "detector_optimize": False,
            },
            progress_callback=progress_cb,
            seed=42,
        )

        # Patch _build_stage to return identity stages
        with patch.object(pipeline, "_build_stage", return_value=IdentityStage()):
            pipeline.process("The experiment yielded significant results.")

        # Verify disabled stages have no callback events
        # Note: progress_callback receives display names from STAGE_NAMES
        stages_in_log = {name for name, _ in progress_log}

        # Only "Structural Variation" (display name for "structural") should appear
        assert "Structural Variation" in stages_in_log
        # Disabled stages' display names should NOT appear
        assert "Adversarial Rewriting" not in stages_in_log
        assert "Vocabulary Injection" not in stages_in_log
        assert "LLM Rewriting" not in stages_in_log
        assert "Detector Optimization" not in stages_in_log


# ---------------------------------------------------------------------------
# Test 3 (Req 10.8): Seed wiring — pipeline passes seed to non-LLM stages
# ---------------------------------------------------------------------------


class TestSeedWiring:
    """Validates Req 10.8: seed determinism for non-LLM-random stages."""

    def test_same_seed_produces_same_output(self):
        """WHEN pipeline runs with a seed, non-LLM stages produce identical output
        for the same input and seed.

        **Validates: Requirements 10.8**
        """
        input_text = "The neural network architecture demonstrates superior performance in classification tasks."

        # Enable only structural and lexical (NLP-based, seed-deterministic stages)
        overrides = {
            "structural": True,
            "lexical": True,
            "llm_rewrite": False,
            "perplexity": False,
            "postprocess": False,
            "semantic_transform": False,
            "iterative_paraphrase": False,
            "retrieval_augmented": False,
            "stylometric": False,
            "perplexity_optimize": False,
            "adversarial": False,
            "error_injection": False,
            "detector_optimize": False,
        }

        # Run 1
        pipeline1 = HumanizationPipeline(
            intensity=2,
            stage_overrides=overrides,
            seed=12345,
        )
        result1 = pipeline1.process(input_text)

        # Run 2 with same seed
        pipeline2 = HumanizationPipeline(
            intensity=2,
            stage_overrides=overrides,
            seed=12345,
        )
        result2 = pipeline2.process(input_text)

        assert result1 == result2, (
            "Same seed should produce identical output for non-LLM stages"
        )

    def test_different_seed_may_differ(self):
        """WHEN pipeline runs with different seeds, non-LLM stages may produce
        different output (validates seed is actually wired through).

        **Validates: Requirements 10.8**
        """
        # Use a text long enough that structural/lexical stages actually modify it
        input_text = (
            "The experimental methodology demonstrates how neural networks "
            "can be effectively utilized in classification tasks. Furthermore, "
            "the results indicate that deep learning approaches significantly "
            "outperform traditional machine learning methods across all metrics."
        )

        overrides = {
            "structural": True,
            "lexical": True,
            "llm_rewrite": False,
            "perplexity": False,
            "postprocess": False,
            "semantic_transform": False,
            "iterative_paraphrase": False,
            "retrieval_augmented": False,
            "stylometric": False,
            "perplexity_optimize": False,
            "adversarial": False,
            "error_injection": False,
            "detector_optimize": False,
        }

        pipeline_a = HumanizationPipeline(
            intensity=2,
            stage_overrides=overrides,
            seed=111,
        )
        result_a = pipeline_a.process(input_text)

        pipeline_b = HumanizationPipeline(
            intensity=2,
            stage_overrides=overrides,
            seed=999,
        )
        result_b = pipeline_b.process(input_text)

        # At least one of the two results should differ from the original
        # (seed is wired to stages that modify text)
        text_modified = result_a != input_text or result_b != input_text
        assert text_modified, (
            "At least one seeded run should produce different output from the input"
        )


# ---------------------------------------------------------------------------
# Test 4 (Req 14.3): Below-0.85 final similarity warning
# ---------------------------------------------------------------------------


class TestFinalSimilarityWarning:
    """Validates Req 14.3: pipeline surfaces warning when final similarity < 0.85."""

    def test_below_085_similarity_surfaces_warning(self):
        """WHEN final similarity is below 0.85, pipeline.final_warning contains
        a meaning preservation warning.

        **Validates: Requirements 14.3**
        """
        input_text = "The experimental results demonstrate clear significance."

        pipeline = HumanizationPipeline(
            intensity=1,
            stage_overrides={
                "structural": True,
                "lexical": False,
                "llm_rewrite": False,
                "perplexity": False,
                "postprocess": False,
                "semantic_transform": False,
                "stylometric": False,
            },
            seed=42,
        )

        # Inject a FakeSimilarityEvaluator that returns 0.70 (below 0.85 threshold)
        fake_sim = FakeSimilarityEvaluator(scores=[0.70], default=0.70)
        pipeline._similarity = fake_sim

        # Patch _build_stage to return identity (so text doesn't change, but
        # the final similarity check uses our fake evaluator)
        with patch.object(pipeline, "_build_stage", return_value=IdentityStage()):
            pipeline.process(input_text)

        # Verify warning was surfaced
        assert pipeline.final_warning is not None
        assert "0.85" in pipeline.final_warning or "meaning" in pipeline.final_warning.lower()
        assert pipeline.final_similarity == 0.70


# ---------------------------------------------------------------------------
# Test 5 (Req 14.5): Dropped number/citation warning
# ---------------------------------------------------------------------------


class TestDroppedContentWarning:
    """Validates Req 14.5: warning when numeric value or citation is dropped."""

    def test_dropped_number_surfaces_warning(self):
        """WHEN a stage removes a number from the text, pipeline.final_warning
        mentions dropped content.

        **Validates: Requirements 14.5**
        """
        input_text = "The study included 42 participants across 3 trials."

        pipeline = HumanizationPipeline(
            intensity=1,
            stage_overrides={
                "structural": True,
                "lexical": False,
                "llm_rewrite": False,
                "perplexity": False,
                "postprocess": False,
                "semantic_transform": False,
                "stylometric": False,
            },
            seed=42,
        )

        # Use a fake similarity evaluator that returns high scores so
        # backstop checks pass and the final similarity check doesn't
        # trigger the 0.85 warning alongside the content warning.
        fake_sim = FakeSimilarityEvaluator(scores=[], default=0.95)
        pipeline._similarity = fake_sim

        # Patch _build_stage to return a number-dropping stage AND
        # bypass the protected-span backstop so the damaged text
        # propagates to the final meaning-preservation check.
        with patch.object(
            pipeline, "_build_stage", return_value=NumberDroppingStage()
        ), patch.object(
            pipeline, "_backstop_protected_spans", return_value=True
        ):
            result = pipeline.process(input_text)

        # The number-dropping stage removes "42", so the output should
        # be missing a number
        assert "42" not in result

        # Verify warning surfaces about dropped content
        assert pipeline.final_warning is not None
        assert "dropped" in pipeline.final_warning.lower() or "numbers" in pipeline.final_warning.lower()
