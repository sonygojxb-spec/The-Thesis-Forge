"""
End-to-end integration tests for the full humanization pipeline.

Tests:
1. Full pipeline run with mocked stages (identity stages for LLM-backed ones)
   asserting protected-term, numeric, and citation preservation plus final
   similarity reporting.
2. Profile smoke-check: all five intensity levels have well-formed profiles.
3. Disclaimer text presence and content.

Requirements validated: 10.1, 12.5, 14.1, 14.2, 14.4
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from humanizer.config import INTENSITY_PROFILES, PROTECTED_TERMS
from humanizer.pipeline import HumanizationPipeline
from humanizer.ui_helpers import DISCLAIMER_TEXT
from tests.conftest import FakeSimilarityEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class IdentityStage:
    """Stage that returns input unchanged — used to mock all stages."""

    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed")

    def process(self, text: str) -> str:
        return text


# Test input that contains protected terms, numbers, and citations
SAMPLE_INPUT = (
    "The hypothesis was tested using a regression algorithm with a coefficient "
    "of 0.95 across 42 participants. The correlation was significant (p < 0.01). "
    "As noted by Smith (Johnson, 2021), the methodology follows the paradigm "
    "described in [3] and [7, 12]. The parameter optimization achieved convergence "
    "at 1500 iterations with entropy reduction of 23.7%."
)


# ---------------------------------------------------------------------------
# Test 1: Full pipeline run — preservation and similarity reporting
# (Req 10.1, 14.1, 14.2, 14.4)
# ---------------------------------------------------------------------------


class TestFullPipelineE2E:
    """End-to-end test: full pipeline with all stages enabled (mocked),
    verifying preservation of protected terms, numbers, citations,
    and final similarity reporting."""

    def test_full_pipeline_preserves_protected_terms(self):
        """WHEN pipeline runs at level 5 (all stages) with identity mocks,
        protected terms in the output match those in the input.

        **Validates: Requirements 14.2**
        """
        pipeline = HumanizationPipeline(
            intensity=5,
            seed=42,
        )

        # Inject fake similarity that always returns a passing score
        fake_sim = FakeSimilarityEvaluator(scores=[], default=0.97)
        pipeline._similarity = fake_sim

        # Mock all stages to identity — tests pipeline orchestration
        with patch.object(pipeline, "_build_stage", return_value=IdentityStage()):
            result = pipeline.process(SAMPLE_INPUT)

        # Count protected terms in input and output
        for term in PROTECTED_TERMS:
            pattern = r'\b' + re.escape(term) + r'\b'
            input_count = len(re.findall(pattern, SAMPLE_INPUT))
            output_count = len(re.findall(pattern, result))
            if input_count > 0:
                assert output_count == input_count, (
                    f"Protected term '{term}' count changed: "
                    f"input={input_count}, output={output_count}"
                )

    def test_full_pipeline_preserves_numeric_values(self):
        """WHEN pipeline runs at level 5 with identity mocks,
        all numeric values from the input appear in the output.

        **Validates: Requirements 14.4**
        """
        pipeline = HumanizationPipeline(
            intensity=5,
            seed=42,
        )

        fake_sim = FakeSimilarityEvaluator(scores=[], default=0.97)
        pipeline._similarity = fake_sim

        with patch.object(pipeline, "_build_stage", return_value=IdentityStage()):
            result = pipeline.process(SAMPLE_INPUT)

        # Extract numbers from input and output
        number_pattern = r'-?\b\d+(?:\.\d+)?%?'
        input_numbers = re.findall(number_pattern, SAMPLE_INPUT)
        output_numbers = re.findall(number_pattern, result)

        assert len(input_numbers) > 0, "Test input should contain numbers"
        assert len(output_numbers) == len(input_numbers), (
            f"Numeric value count changed: "
            f"input={len(input_numbers)}, output={len(output_numbers)}"
        )

    def test_full_pipeline_preserves_citations(self):
        """WHEN pipeline runs at level 5 with identity mocks,
        all citation markers from the input appear in the output.

        **Validates: Requirements 14.4**
        """
        pipeline = HumanizationPipeline(
            intensity=5,
            seed=42,
        )

        fake_sim = FakeSimilarityEvaluator(scores=[], default=0.97)
        pipeline._similarity = fake_sim

        with patch.object(pipeline, "_build_stage", return_value=IdentityStage()):
            result = pipeline.process(SAMPLE_INPUT)

        # Check parenthetical citations: (Author, YEAR)
        paren_pattern = r'\([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?,\s*\d{4}\)'
        input_paren = re.findall(paren_pattern, SAMPLE_INPUT)
        output_paren = re.findall(paren_pattern, result)
        assert len(output_paren) == len(input_paren), (
            f"Parenthetical citations changed: "
            f"input={len(input_paren)}, output={len(output_paren)}"
        )

        # Check bracketed citations: [n] or [n, m]
        bracket_pattern = r'\[\d+(?:\s*[-,]\s*\d+)*\]'
        input_bracket = re.findall(bracket_pattern, SAMPLE_INPUT)
        output_bracket = re.findall(bracket_pattern, result)
        assert len(output_bracket) == len(input_bracket), (
            f"Bracketed citations changed: "
            f"input={len(input_bracket)}, output={len(output_bracket)}"
        )

    def test_full_pipeline_reports_final_similarity(self):
        """WHEN pipeline completes, final_similarity is set to a float in [0.0, 1.0].

        **Validates: Requirements 14.1**
        """
        pipeline = HumanizationPipeline(
            intensity=5,
            seed=42,
        )

        fake_sim = FakeSimilarityEvaluator(scores=[], default=0.92)
        pipeline._similarity = fake_sim

        with patch.object(pipeline, "_build_stage", return_value=IdentityStage()):
            pipeline.process(SAMPLE_INPUT)

        assert pipeline.final_similarity is not None, (
            "final_similarity should be set after pipeline completes"
        )
        assert 0.0 <= pipeline.final_similarity <= 1.0, (
            f"final_similarity should be in [0.0, 1.0], got {pipeline.final_similarity}"
        )

    def test_full_pipeline_populates_stage_results(self):
        """WHEN pipeline runs with stages enabled, stage_results is populated.

        **Validates: Requirements 10.1**
        """
        pipeline = HumanizationPipeline(
            intensity=5,
            seed=42,
        )

        fake_sim = FakeSimilarityEvaluator(scores=[], default=0.97)
        pipeline._similarity = fake_sim

        with patch.object(pipeline, "_build_stage", return_value=IdentityStage()):
            pipeline.process(SAMPLE_INPUT)

        # With all stages enabled at level 5, stage_results should be non-empty
        assert len(pipeline.stage_results) > 0, (
            "stage_results should be populated after pipeline run"
        )

    def test_full_pipeline_deterministic_stage_order(self):
        """WHEN pipeline runs at level 5, enabled stages execute in
        the canonical deterministic order.

        **Validates: Requirements 10.1**
        """
        pipeline = HumanizationPipeline(
            intensity=5,
            seed=42,
        )

        enabled = pipeline.get_enabled_stages()

        # All 13 stages should be enabled at level 5
        assert len(enabled) == 13, (
            f"Level 5 should enable all 13 stages, got {len(enabled)}"
        )

        # Verify the order matches the canonical STAGE_ORDER
        assert enabled == HumanizationPipeline.STAGE_ORDER, (
            "Enabled stages should be in canonical STAGE_ORDER"
        )


# ---------------------------------------------------------------------------
# Test 2: Profile smoke-check — all levels well-formed
# (Req 10.1)
# ---------------------------------------------------------------------------


class TestIntensityProfileSmoke:
    """Smoke test: all five intensity profiles are well-formed dicts
    with expected keys."""

    # Expected keys that every profile must contain
    REQUIRED_EXISTING_STAGE_KEYS = [
        "structural", "lexical", "llm_rewrite", "perplexity", "postprocess",
    ]

    REQUIRED_AGGRESSION_KEYS = [
        "structural_aggression", "lexical_aggression", "llm_aggression",
        "perplexity_aggression", "postprocess_aggression",
    ]

    REQUIRED_NEW_STAGE_ENABLED_KEYS = [
        "semantic_transform_enabled",
        "iterative_paraphrase_enabled",
        "retrieval_augmented_enabled",
        "stylometric_enabled",
        "perplexity_optimize_enabled",
        "adversarial_enabled",
        "error_injection_enabled",
        "detector_optimize_enabled",
    ]

    REQUIRED_NEW_STAGE_AGGRESSION_KEYS = [
        "semantic_transform_aggression",
        "iterative_paraphrase_aggression",
        "retrieval_augmented_aggression",
        "stylometric_aggression",
        "perplexity_optimize_aggression",
        "adversarial_aggression",
        "error_injection_aggression",
        "detector_optimize_aggression",
    ]

    REQUIRED_PERPLEXITY_KEYS = [
        "target_perplexity_mean",
        "target_perplexity_variance",
    ]

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
    def test_profile_is_dict_with_required_keys(self, level):
        """WHEN accessing INTENSITY_PROFILES[level], it is a dict
        containing all expected configuration keys.

        **Validates: Requirements 10.1**
        """
        profile = INTENSITY_PROFILES[level]
        assert isinstance(profile, dict), (
            f"Profile at level {level} should be a dict"
        )

        all_required_keys = (
            self.REQUIRED_EXISTING_STAGE_KEYS
            + self.REQUIRED_AGGRESSION_KEYS
            + self.REQUIRED_NEW_STAGE_ENABLED_KEYS
            + self.REQUIRED_NEW_STAGE_AGGRESSION_KEYS
            + self.REQUIRED_PERPLEXITY_KEYS
        )

        for key in all_required_keys:
            assert key in profile, (
                f"Profile at level {level} is missing key '{key}'"
            )

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
    def test_profile_aggression_values_in_range(self, level):
        """WHEN accessing aggression values in profiles, they are floats in [0.0, 1.0]."""
        profile = INTENSITY_PROFILES[level]

        aggression_keys = (
            self.REQUIRED_AGGRESSION_KEYS
            + self.REQUIRED_NEW_STAGE_AGGRESSION_KEYS
        )

        for key in aggression_keys:
            val = profile[key]
            assert isinstance(val, (int, float)), (
                f"Profile level {level}, key '{key}' should be numeric, got {type(val)}"
            )
            assert 0.0 <= val <= 1.0, (
                f"Profile level {level}, key '{key}' should be in [0.0, 1.0], got {val}"
            )

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
    def test_profile_enabled_flags_are_booleans(self, level):
        """WHEN accessing enabled flags in profiles, they are booleans."""
        profile = INTENSITY_PROFILES[level]

        enabled_keys = (
            self.REQUIRED_EXISTING_STAGE_KEYS
            + self.REQUIRED_NEW_STAGE_ENABLED_KEYS
        )

        for key in enabled_keys:
            val = profile[key]
            assert isinstance(val, bool), (
                f"Profile level {level}, key '{key}' should be bool, got {type(val)}"
            )


# ---------------------------------------------------------------------------
# Test 3: Disclaimer text presence and content
# (Req 12.5)
# ---------------------------------------------------------------------------


class TestDisclaimerText:
    """Validates Req 12.5: disclaimer text is present and has expected content."""

    def test_disclaimer_text_is_non_empty_string(self):
        """WHEN importing DISCLAIMER_TEXT, it is a non-empty string.

        **Validates: Requirements 12.5**
        """
        assert isinstance(DISCLAIMER_TEXT, str)
        assert len(DISCLAIMER_TEXT) > 0

    def test_disclaimer_mentions_estimates(self):
        """WHEN reading DISCLAIMER_TEXT, it mentions that scores are estimates.

        **Validates: Requirements 12.5**
        """
        assert "estimate" in DISCLAIMER_TEXT.lower(), (
            "Disclaimer should mention that scores are estimates"
        )

    def test_disclaimer_mentions_detection_tools(self):
        """WHEN reading DISCLAIMER_TEXT, it clarifies results may not match
        specific detection tools.

        **Validates: Requirements 12.5**
        """
        assert "detection" in DISCLAIMER_TEXT.lower() or "detector" in DISCLAIMER_TEXT.lower(), (
            "Disclaimer should reference detection tools or detectors"
        )
