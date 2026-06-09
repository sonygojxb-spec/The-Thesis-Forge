"""
Property 1: Protected-span invariance — parameterized across stages.

For all input texts and for every transformation stage T, the occurrence count of
every PROTECTED_TERMS entry (case-sensitive, whole-word) in the output equals its
occurrence count in the input.

This suite is parameterized: each stage is added as a pytest.param fixture so the
same property test runs against every stage independently.

Additionally includes the full HumanizationPipeline (with identity stages) and
Property 5 (similarity portion): final_similarity in [0.0, 1.0].

Requirements: 1.4, 2.3, 3.4, 4.2, 5.3, 6.4, 7.5, 8.7, 14.1, 14.2
"""

from __future__ import annotations

import re
from typing import Callable
from unittest.mock import patch

import pytest
from hypothesis import given, settings

from humanizer.config import PROTECTED_TERMS
from humanizer.pipeline import HumanizationPipeline
from humanizer.retrieval import ReferenceEntry
from humanizer.results import StageResult, TargetPerplexityProfile
from humanizer.stage_iterative import IterativeParaphraser
from humanizer.stage_perplexity_optimize import PerplexityOptimizer
from humanizer.stage_retrieval_augmented import RetrievalAugmentedRewriter
from humanizer.stage_adversarial import AdversarialRewriter
from humanizer.stage_detector_optimizer import DetectorOptimizer
from humanizer.stage_error_injector import ErrorInjector
from humanizer.stage_semantic import SemanticTransformer
from humanizer.stage_stylometric import StylometricObfuscator
from tests.conftest import FakeClassifier, FakeSimilarityEvaluator
from tests.strategies import academic_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_protected_terms(text: str) -> dict[str, int]:
    """Count occurrences of each PROTECTED_TERMS entry in text (whole-word, case-sensitive)."""
    counts: dict[str, int] = {}
    for term in PROTECTED_TERMS:
        pattern = r"\b" + re.escape(term) + r"\b"
        counts[term] = len(re.findall(pattern, text))
    return counts


# ---------------------------------------------------------------------------
# Stage factories — each returns a callable (text) -> str for the property test
# ---------------------------------------------------------------------------


def make_semantic_transformer() -> Callable[[str], str]:
    """Create a SemanticTransformer with a FakeSimilarityEvaluator returning high scores."""
    evaluator = FakeSimilarityEvaluator(default=0.95)
    transformer = SemanticTransformer(
        aggression=0.7,
        seed=42,
        similarity=evaluator,
        floor=0.90,
    )
    return transformer.process


def make_iterative_paraphraser() -> Callable[[str], str]:
    """Create an IterativeParaphraser with mocked _llm_pass (identity) and FakeSimilarityEvaluator (>=0.80).

    The mock _llm_pass returns the input text unchanged (identity function) to ensure
    protected spans are preserved through the mask/unmask cycle.
    """
    evaluator = FakeSimilarityEvaluator(default=0.95)
    paraphraser = IterativeParaphraser(
        aggression=0.5,
        seed=42,
        similarity=evaluator,
        timeout_s=30,
    )

    # Monkey-patch _llm_pass to be an identity function that preserves placeholders
    def identity_llm_pass(text: str, pass_index: int) -> str:
        return text

    paraphraser._llm_pass = identity_llm_pass
    return paraphraser.process


class _FakeRetrievalServiceForSpans:
    """A fake retrieval service returning passages that don't trigger the verbatim guard.

    The passages are short enough and unrelated enough to the generated academic
    text that an identity rewrite won't contain >8 consecutive word spans from them.
    """

    def __init__(self) -> None:
        self._passages = [
            ReferenceEntry(
                id="span_p1",
                text="Elegant prose flows like water through ancient riverbed channels carved by time.",
                source="test",
                embedding=None,
            ),
            ReferenceEntry(
                id="span_p2",
                text="Creative writing workshops often encourage freeform exploration of narrative voice.",
                source="test",
                embedding=None,
            ),
        ]

    @property
    def corpus(self):
        return list(self._passages)

    def retrieve(self, query_text: str):
        return list(self._passages)


def make_retrieval_augmented_rewriter() -> Callable[[str], str]:
    """Create a RetrievalAugmentedRewriter with identity _llm_rewrite and FakeSimilarityEvaluator (>=0.85).

    - FakeRetrievalService returns passages unrelated to academic text (no verbatim guard trigger).
    - _llm_rewrite is monkey-patched to return the input text unchanged (identity), so protected
      spans are preserved through the mask/unmask cycle.
    - FakeSimilarityEvaluator returns 0.95 (>= 0.85 floor) so the rewrite is accepted.
    """
    evaluator = FakeSimilarityEvaluator(default=0.95)
    retrieval = _FakeRetrievalServiceForSpans()

    rewriter = RetrievalAugmentedRewriter(
        aggression=0.5,
        seed=42,
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:9999",
        retrieval_service=retrieval,
        similarity=evaluator,
        floor=0.85,
        timeout_s=5,
    )

    # Monkey-patch _llm_rewrite to return text unchanged (identity)
    def identity_llm_rewrite(text: str, passages) -> str:
        return text

    rewriter._llm_rewrite = identity_llm_rewrite
    return rewriter.process


def make_stylometric_obfuscator() -> Callable[[str], str]:
    """Create a StylometricObfuscator with FakeSimilarityEvaluator (>=0.85).

    The StylometricObfuscator is an NLP-only stage — no LLM mocking needed.
    Uses aggression=0.5 and seed=42 for reproducibility. The similarity evaluator
    returns 0.95 (>= 0.85 floor) so transformations are accepted.

    Note: The stage requires >=2 sentences to apply transformations. The
    academic_text strategy generates multi-sentence text by default.
    """
    evaluator = FakeSimilarityEvaluator(default=0.95)
    obfuscator = StylometricObfuscator(
        aggression=0.5,
        seed=42,
        similarity=evaluator,
        floor=0.85,
    )
    return obfuscator.process


def make_perplexity_optimizer() -> Callable[[str], str]:
    """Create a PerplexityOptimizer with FakeSimilarityEvaluator (>=0.85).

    The PerplexityOptimizer is an NLP-only stage — no LLM mocking needed beyond
    similarity. Uses a TargetPerplexityProfile with target_mean=80.0 (far from
    typical input) to force optimization edits, and seed=42 for reproducibility.
    The similarity evaluator returns 0.95 (>= 0.85 floor) so candidates are accepted.
    """
    evaluator = FakeSimilarityEvaluator(default=0.95)
    target_profile = TargetPerplexityProfile(target_mean=80.0, target_variance=15.0)
    optimizer = PerplexityOptimizer(
        aggression=0.5,
        seed=42,
        similarity=evaluator,
        floor=0.85,
        target_profile=target_profile,
    )
    return optimizer.process


# ---------------------------------------------------------------------------
# AdversarialRewriter factory
# ---------------------------------------------------------------------------


def make_adversarial_rewriter() -> Callable[[str], str]:
    """Create an AdversarialRewriter with identity _llm_rewrite and FakeClassifier/FakeSimilarityEvaluator.

    - Mock _llm_rewrite returns input text unchanged (identity) so protected spans are preserved.
    - FakeClassifier returns scores [75.0, 50.0]: first call (input risk) = 75.0,
      second call (candidate risk) = 50.0 → candidate risk <= input risk → accepted.
    - FakeSimilarityEvaluator returns 0.95 (>= 0.85 floor) so the candidate passes the floor check.

    Validates: Requirements 4.2, 4.3, 4.5
    """
    evaluator = FakeSimilarityEvaluator(default=0.95)
    classifier = FakeClassifier(scores=[75.0, 50.0], default=50.0)

    rewriter = AdversarialRewriter(
        aggression=0.5,
        seed=42,
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:9999",
        similarity=evaluator,
        classifier=classifier,
        floor=0.85,
        timeout_s=5,
    )

    # Monkey-patch _llm_rewrite to return text unchanged (identity)
    def identity_llm_rewrite(text: str) -> str:
        return text

    rewriter._llm_rewrite = identity_llm_rewrite
    return rewriter.process


# ---------------------------------------------------------------------------
# ErrorInjector factory
# ---------------------------------------------------------------------------


def make_error_injector() -> Callable[[str], str]:
    """Create an ErrorInjector with aggression=0.5, seed=42.

    No similarity evaluator needed — ErrorInjector is an NLP-only stage that
    uses ProtectedSpanGuard internally to mask protected spans before injection.

    Validates: Requirements 5.3
    """
    injector = ErrorInjector(
        aggression=0.5,
        seed=42,
    )
    return injector.process


# ---------------------------------------------------------------------------
# DetectorOptimizer factory
# ---------------------------------------------------------------------------


def make_detector_optimizer() -> Callable[[str], str]:
    """Create a DetectorOptimizer with identity _generate_candidate, FakeClassifier, and FakeSimilarityEvaluator.

    - FakeClassifier returns scores [80.0, 50.0]: first call (input risk) = 80.0 (above target_threshold=30),
      second call (candidate risk) = 50.0 → candidate accepted as improvement.
    - FakeSimilarityEvaluator returns 0.95 (>= 0.85 floor) so the candidate passes similarity gate.
    - _generate_candidate is monkey-patched to return the input text unchanged (identity), so protected
      spans are preserved through the full mask/unmask cycle.
    - target_threshold=30, max_iterations=3.

    Validates: Requirements 8.7
    """
    evaluator = FakeSimilarityEvaluator(default=0.95)
    classifier = FakeClassifier(scores=[80.0, 50.0], default=50.0)

    optimizer = DetectorOptimizer(
        aggression=0.5,
        seed=42,
        classifier=classifier,
        similarity=evaluator,
        target_threshold=30,
        max_iterations=3,
    )

    # Monkey-patch _generate_candidate to return masked text unchanged (identity)
    # so protected spans are preserved through the full mask/unmask cycle
    def identity_generate_candidate(masked_text: str, seed: int, iteration: int) -> str:
        return masked_text

    optimizer._generate_candidate = identity_generate_candidate
    return optimizer.process


# ---------------------------------------------------------------------------
# Full Pipeline factory (identity stages via _build_stage mock)
# ---------------------------------------------------------------------------


class _IdentityStage:
    """A minimal identity stage that returns input text unchanged.

    Conforms to the stage contract: has process() and process_measured().
    """

    def process(self, text: str) -> str:
        return text

    def process_measured(self, text: str) -> StageResult:
        return StageResult(
            text=text,
            similarity=1.0,
            risk_before=None,
            risk_after=None,
            changed=False,
            fell_back=False,
            error=None,
        )


def make_pipeline() -> Callable[[str], str]:
    """Create a HumanizationPipeline with _build_stage returning identity stages.

    This tests that the pipeline-level backstop correctly preserves protected spans
    even when all stages are enabled. The identity stages pass text through unchanged,
    so the pipeline's own ProtectedSpanGuard.verify() backstop is the only thing being
    tested.

    The pipeline uses a FakeSimilarityEvaluator (returns 0.95) injected via
    _build_similarity_evaluator patch so that the final meaning check produces a
    score in [0.0, 1.0] without loading the real model.

    Validates: Requirements 14.1, 14.2
    """
    fake_similarity = FakeSimilarityEvaluator(default=0.95)

    # Enable a representative set of NLP-only stages
    stage_overrides = {
        "structural": True,
        "lexical": True,
        "semantic_transform": True,
        "stylometric": True,
        "error_injection": True,
        "postprocess": True,
    }

    pipeline = HumanizationPipeline(
        intensity=3,
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:9999",
        stage_overrides=stage_overrides,
        seed=42,
    )

    # Replace _build_stage to always return identity stages
    def identity_build_stage(stage_key, stream_callback=None):
        return _IdentityStage()

    pipeline._build_stage = identity_build_stage

    # Replace the shared similarity evaluator with a fake
    pipeline._similarity = fake_similarity

    return pipeline.process


# ---------------------------------------------------------------------------
# Parameterized test suite
# ---------------------------------------------------------------------------

# Feature: ultimate-humanizer, Property 1: Protected-span invariance

STAGE_PARAMS = [
    pytest.param(make_semantic_transformer, id="SemanticTransformer"),
    pytest.param(make_iterative_paraphraser, id="IterativeParaphraser"),
    pytest.param(make_retrieval_augmented_rewriter, id="RetrievalAugmentedRewriter"),
    pytest.param(make_stylometric_obfuscator, id="StylometricObfuscator"),
    pytest.param(make_perplexity_optimizer, id="PerplexityOptimizer"),
    pytest.param(make_adversarial_rewriter, id="AdversarialRewriter"),
    pytest.param(make_error_injector, id="ErrorInjector"),
    pytest.param(make_detector_optimizer, id="DetectorOptimizer"),
    pytest.param(make_pipeline, id="FullPipeline"),
]


@pytest.mark.parametrize("stage_factory", STAGE_PARAMS)
@given(text=academic_text(min_protected_terms=1, max_protected_terms=3))
@settings(max_examples=100)
def test_protected_span_invariance(stage_factory: Callable[[], Callable[[str], str]], text: str) -> None:
    """Property 1: Protected-span invariance.

    For all inputs containing PROTECTED_TERMS, after processing through the stage,
    every PROTECTED_TERMS occurrence count is unchanged.

    Validates: Requirements 1.4, 2.3, 6.4, 7.5, 14.2
    """
    process = stage_factory()

    # Count protected terms before processing
    counts_before = count_protected_terms(text)

    # Process through the stage
    output = process(text)

    # Count protected terms after processing
    counts_after = count_protected_terms(output)

    # Assert: every protected term count is preserved
    for term in PROTECTED_TERMS:
        before = counts_before[term]
        after = counts_after[term]
        assert after == before, (
            f"Protected term '{term}' count changed: before={before}, after={after}\n"
            f"Input:  {text!r}\n"
            f"Output: {output!r}"
        )


# ---------------------------------------------------------------------------
# Property 5 (similarity portion): final_similarity in [0.0, 1.0]
# ---------------------------------------------------------------------------

# Feature: ultimate-humanizer, Property 5: Final similarity score range


@given(text=academic_text(min_protected_terms=1, max_protected_terms=3))
@settings(max_examples=100)
def test_final_similarity_score_range(text: str) -> None:
    """Property 5 (similarity portion): final_similarity in [0.0, 1.0].

    After full pipeline processing, the final_similarity score computed by the
    pipeline must be a float in the range [0.0, 1.0] inclusive.

    Validates: Requirements 14.1, 14.2
    """
    fake_similarity = FakeSimilarityEvaluator(default=0.95)

    # Enable NLP-only stages
    stage_overrides = {
        "structural": True,
        "lexical": True,
        "semantic_transform": True,
        "stylometric": True,
        "error_injection": True,
        "postprocess": True,
    }

    pipeline = HumanizationPipeline(
        intensity=3,
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:9999",
        stage_overrides=stage_overrides,
        seed=42,
    )

    # Replace _build_stage to always return identity stages
    def identity_build_stage(stage_key, stream_callback=None):
        return _IdentityStage()

    pipeline._build_stage = identity_build_stage

    # Replace the shared similarity evaluator with a fake
    pipeline._similarity = fake_similarity

    # Run the pipeline
    pipeline.process(text)

    # Assert: final_similarity is computed and in [0.0, 1.0]
    assert pipeline.final_similarity is not None, (
        "final_similarity was not computed after pipeline.process()"
    )
    assert 0.0 <= pipeline.final_similarity <= 1.0, (
        f"final_similarity={pipeline.final_similarity} is outside [0.0, 1.0]\n"
        f"Input: {text!r}"
    )
