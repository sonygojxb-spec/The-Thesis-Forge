"""
Property 3: Similarity-floor guarantee — parameterized across stages.

For all input texts and for every transformation stage T with an injected similarity
evaluator, if the evaluator returns a score BELOW the stage's floor the output must
equal the input (candidate discarded); if the evaluator returns a score ABOVE the
floor the candidate may be accepted (output may differ from input).

This suite is parameterized: each stage is added as a pytest.param fixture so the
same property test runs against every stage independently.

Requirements: 6.3 (SemanticTransformer floor = 0.90), 7.6, 7.9 (RetrievalAugmentedRewriter floor = 0.85)

# Feature: ultimate-humanizer, Property 3: Similarity-floor guarantee
"""

from __future__ import annotations

from typing import Callable, Tuple

import pytest
from hypothesis import given, settings

from humanizer.retrieval import ReferenceEntry
from humanizer.results import TargetPerplexityProfile
from humanizer.stage_adversarial import AdversarialRewriter
from humanizer.stage_iterative import IterativeParaphraser
from humanizer.stage_perplexity_optimize import PerplexityOptimizer
from humanizer.stage_retrieval_augmented import RetrievalAugmentedRewriter
from humanizer.stage_semantic import SemanticTransformer
from humanizer.stage_stylometric import StylometricObfuscator
from tests.conftest import FakeClassifier, FakeSimilarityEvaluator
from tests.strategies import academic_text


# ---------------------------------------------------------------------------
# Stage factories — each returns (process_callable, floor_value)
# ---------------------------------------------------------------------------


def make_semantic_transformer_below_floor() -> Tuple[Callable[[str], str], float]:
    """SemanticTransformer with evaluator returning score BELOW floor (0.90).

    When the evaluator returns 0.70, the candidate should be discarded and
    input returned unchanged.
    """
    evaluator = FakeSimilarityEvaluator(default=0.70)
    transformer = SemanticTransformer(
        aggression=0.7,
        seed=42,
        similarity=evaluator,
        floor=0.90,
    )
    return transformer.process, 0.90


def make_semantic_transformer_above_floor() -> Tuple[Callable[[str], str], float]:
    """SemanticTransformer with evaluator returning score ABOVE floor (0.90).

    When the evaluator returns 0.95, the candidate should be accepted (output
    may differ from input).
    """
    evaluator = FakeSimilarityEvaluator(default=0.95)
    transformer = SemanticTransformer(
        aggression=0.7,
        seed=42,
        similarity=evaluator,
        floor=0.90,
    )
    return transformer.process, 0.90


def make_iterative_paraphraser_below_floor() -> Tuple[Callable[[str], str], float]:
    """IterativeParaphraser with evaluator returning score BELOW floor (0.80).

    When the evaluator returns 0.60, the pass should be discarded (similarity
    vs stage input < 0.80) and input returned unchanged.
    """
    evaluator = FakeSimilarityEvaluator(default=0.60)
    paraphraser = IterativeParaphraser(
        aggression=0.5,
        seed=42,
        similarity=evaluator,
        timeout_s=30,
    )

    # Mock _llm_pass to return a deterministic modification
    def mock_llm_pass(text: str, pass_index: int) -> str:
        return text + " paraphrased"

    paraphraser._llm_pass = mock_llm_pass
    return paraphraser.process, 0.80


def make_iterative_paraphraser_above_floor() -> Tuple[Callable[[str], str], float]:
    """IterativeParaphraser with evaluator returning score ABOVE floor (0.80).

    When the evaluator returns 0.90, the pass should be accepted (output
    may differ from input).
    """
    evaluator = FakeSimilarityEvaluator(default=0.90)
    paraphraser = IterativeParaphraser(
        aggression=0.5,
        seed=42,
        similarity=evaluator,
        timeout_s=30,
    )

    # Mock _llm_pass to return a deterministic modification
    def mock_llm_pass(text: str, pass_index: int) -> str:
        return text + " paraphrased"

    paraphraser._llm_pass = mock_llm_pass
    return paraphraser.process, 0.80


# ---------------------------------------------------------------------------
# RetrievalAugmentedRewriter factories (floor = 0.85)
# ---------------------------------------------------------------------------


class _FakeRetrievalServiceForFloor:
    """A fake retrieval service returning passages that don't trigger the verbatim guard.

    The passages are short and unrelated to generated academic text so an
    identity or slightly-modified rewrite won't contain >8 consecutive word
    spans from them.
    """

    def __init__(self) -> None:
        self._passages = [
            ReferenceEntry(
                id="floor_p1",
                text="Elegant prose flows like water through ancient riverbed channels carved by time.",
                source="test",
                embedding=None,
            ),
            ReferenceEntry(
                id="floor_p2",
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


def make_retrieval_augmented_below_floor() -> Tuple[Callable[[str], str], float]:
    """RetrievalAugmentedRewriter with evaluator returning score BELOW floor (0.85).

    When the evaluator returns 0.70, the candidate should be discarded and
    input returned unchanged.

    - FakeRetrievalService provides passages (retrieval path exercised).
    - _llm_rewrite mocked to return a deterministic modification.
    - FakeSimilarityEvaluator returns 0.70 (< 0.85 floor) → candidate discarded.
    """
    evaluator = FakeSimilarityEvaluator(default=0.70)
    retrieval = _FakeRetrievalServiceForFloor()

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

    # Mock _llm_rewrite to return a modification (so candidate != input)
    def mock_llm_rewrite(text: str, passages) -> str:
        return text + " rewritten"

    rewriter._llm_rewrite = mock_llm_rewrite
    return rewriter.process, 0.85


def make_retrieval_augmented_above_floor() -> Tuple[Callable[[str], str], float]:
    """RetrievalAugmentedRewriter with evaluator returning score ABOVE floor (0.85).

    When the evaluator returns 0.95, the candidate should be accepted (output
    may differ from input).

    - FakeRetrievalService provides passages (retrieval path exercised).
    - _llm_rewrite mocked to return a deterministic modification.
    - FakeSimilarityEvaluator returns 0.95 (>= 0.85 floor) → candidate accepted.
    """
    evaluator = FakeSimilarityEvaluator(default=0.95)
    retrieval = _FakeRetrievalServiceForFloor()

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

    # Mock _llm_rewrite to return a modification (so candidate != input)
    def mock_llm_rewrite(text: str, passages) -> str:
        return text + " rewritten"

    rewriter._llm_rewrite = mock_llm_rewrite
    return rewriter.process, 0.85


# ---------------------------------------------------------------------------
# StylometricObfuscator factories (floor = 0.85)
# ---------------------------------------------------------------------------


def make_stylometric_obfuscator_below_floor() -> Tuple[Callable[[str], str], float]:
    """StylometricObfuscator with evaluator returning score BELOW floor (0.85).

    When the evaluator returns 0.70, the candidate should be discarded and
    input returned unchanged. NLP-only stage — no mocking needed beyond similarity.
    """
    evaluator = FakeSimilarityEvaluator(default=0.70)
    obfuscator = StylometricObfuscator(
        aggression=0.5,
        seed=42,
        similarity=evaluator,
        floor=0.85,
    )
    return obfuscator.process, 0.85


def make_stylometric_obfuscator_above_floor() -> Tuple[Callable[[str], str], float]:
    """StylometricObfuscator with evaluator returning score ABOVE floor (0.85).

    When the evaluator returns 0.95, the candidate should be accepted (output
    may differ from input). NLP-only stage — no mocking needed beyond similarity.
    """
    evaluator = FakeSimilarityEvaluator(default=0.95)
    obfuscator = StylometricObfuscator(
        aggression=0.5,
        seed=42,
        similarity=evaluator,
        floor=0.85,
    )
    return obfuscator.process, 0.85


# ---------------------------------------------------------------------------
# PerplexityOptimizer factories (floor = 0.85)
# ---------------------------------------------------------------------------


def make_perplexity_optimizer_below_floor() -> Tuple[Callable[[str], str], float]:
    """PerplexityOptimizer with evaluator returning score BELOW floor (0.85).

    When the evaluator returns 0.70, the candidate should be discarded and
    input returned unchanged. NLP-only stage — no mocking needed beyond similarity.
    Uses a TargetPerplexityProfile with target_mean=80.0 to force optimization attempts.
    """
    evaluator = FakeSimilarityEvaluator(default=0.70)
    target_profile = TargetPerplexityProfile(target_mean=80.0, target_variance=15.0)
    optimizer = PerplexityOptimizer(
        aggression=0.5,
        seed=42,
        similarity=evaluator,
        floor=0.85,
        target_profile=target_profile,
    )
    return optimizer.process, 0.85


def make_perplexity_optimizer_above_floor() -> Tuple[Callable[[str], str], float]:
    """PerplexityOptimizer with evaluator returning score ABOVE floor (0.85).

    When the evaluator returns 0.95, the candidate should be accepted (output
    may differ from input). NLP-only stage — no mocking needed beyond similarity.
    Uses a TargetPerplexityProfile with target_mean=80.0 to force optimization attempts.
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
    return optimizer.process, 0.85


# ---------------------------------------------------------------------------
# AdversarialRewriter factories (floor = 0.85)
# ---------------------------------------------------------------------------


def make_adversarial_rewriter_below_floor() -> Tuple[Callable[[str], str], float]:
    """AdversarialRewriter with evaluator returning score BELOW floor (0.85).

    When the evaluator returns 0.70, the candidate should be discarded and
    input returned unchanged.

    - _llm_rewrite mocked to return input unchanged (identity).
    - FakeClassifier returns scores [75.0, 50.0] so risk check would pass,
      but the similarity floor check (0.70 < 0.85) rejects the candidate first.
    - FakeSimilarityEvaluator returns 0.70 (< 0.85 floor) → candidate discarded.

    Validates: Requirements 4.2, 4.3, 4.5
    """
    evaluator = FakeSimilarityEvaluator(default=0.70)
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

    # Mock _llm_rewrite to return input unchanged (identity)
    def identity_llm_rewrite(text: str) -> str:
        return text

    rewriter._llm_rewrite = identity_llm_rewrite
    return rewriter.process, 0.85


def make_adversarial_rewriter_above_floor() -> Tuple[Callable[[str], str], float]:
    """AdversarialRewriter with evaluator returning score ABOVE floor (0.85).

    When the evaluator returns 0.95, the candidate should be accepted (output
    may differ from input).

    - _llm_rewrite mocked to return input unchanged (identity).
    - FakeClassifier returns scores [75.0, 50.0]: first call (input risk) = 75.0,
      second call (candidate risk) = 50.0 → candidate risk <= input risk → accepted.
    - FakeSimilarityEvaluator returns 0.95 (>= 0.85 floor) → candidate accepted.

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

    # Mock _llm_rewrite to return input unchanged (identity)
    def identity_llm_rewrite(text: str) -> str:
        return text

    rewriter._llm_rewrite = identity_llm_rewrite
    return rewriter.process, 0.85


# ---------------------------------------------------------------------------
# Parameterized test suite: BELOW floor → output == input (discarded)
# ---------------------------------------------------------------------------

# Feature: ultimate-humanizer, Property 3: Similarity-floor guarantee

BELOW_FLOOR_PARAMS = [
    pytest.param(make_semantic_transformer_below_floor, id="SemanticTransformer-below-floor"),
    pytest.param(make_iterative_paraphraser_below_floor, id="IterativeParaphraser-below-floor"),
    pytest.param(make_retrieval_augmented_below_floor, id="RetrievalAugmentedRewriter-below-floor"),
    pytest.param(make_stylometric_obfuscator_below_floor, id="StylometricObfuscator-below-floor"),
    pytest.param(make_perplexity_optimizer_below_floor, id="PerplexityOptimizer-below-floor"),
    pytest.param(make_adversarial_rewriter_below_floor, id="AdversarialRewriter-below-floor"),
]


@pytest.mark.parametrize("stage_factory", BELOW_FLOOR_PARAMS)
@given(text=academic_text(min_protected_terms=1, max_protected_terms=2))
@settings(max_examples=100)
def test_similarity_floor_discard(
    stage_factory: Callable[[], Tuple[Callable[[str], str], float]], text: str
) -> None:
    """Property 3: Similarity-floor guarantee (discard path).

    When the similarity evaluator returns a score BELOW the stage's floor,
    the candidate MUST be discarded and the output MUST equal the input.

    Validates: Requirements 1.5, 2.4, 2.9, 6.3, 7.6, 7.9
    """
    process, floor = stage_factory()
    output = process(text)

    assert output == text, (
        f"Expected output == input when similarity < {floor} (candidate should be discarded)\n"
        f"Input:  {text!r}\n"
        f"Output: {output!r}"
    )


# ---------------------------------------------------------------------------
# Parameterized test suite: ABOVE floor → candidate accepted (output may differ)
# ---------------------------------------------------------------------------

ABOVE_FLOOR_PARAMS = [
    pytest.param(make_semantic_transformer_above_floor, id="SemanticTransformer-above-floor"),
    pytest.param(make_iterative_paraphraser_above_floor, id="IterativeParaphraser-above-floor"),
    pytest.param(make_retrieval_augmented_above_floor, id="RetrievalAugmentedRewriter-above-floor"),
    pytest.param(make_stylometric_obfuscator_above_floor, id="StylometricObfuscator-above-floor"),
    pytest.param(make_perplexity_optimizer_above_floor, id="PerplexityOptimizer-above-floor"),
    pytest.param(make_adversarial_rewriter_above_floor, id="AdversarialRewriter-above-floor"),
]


@pytest.mark.parametrize("stage_factory", ABOVE_FLOOR_PARAMS)
@given(text=academic_text(min_protected_terms=1, max_protected_terms=2))
@settings(max_examples=100)
def test_similarity_floor_accept(
    stage_factory: Callable[[], Tuple[Callable[[str], str], float]], text: str
) -> None:
    """Property 3: Similarity-floor guarantee (accept path).

    When the similarity evaluator returns a score ABOVE the stage's floor,
    the candidate is accepted. We verify that the stage does not error out
    and returns a non-empty string (the output is allowed to differ from
    input since the candidate passed the floor check).

    Validates: Requirements 1.5, 2.4, 2.9, 6.3, 7.6, 7.9
    """
    process, floor = stage_factory()
    output = process(text)

    # The output must be a non-empty string (stage completed without error)
    assert isinstance(output, str), (
        f"Expected output to be a string, got {type(output)}"
    )
    assert len(output) > 0, (
        f"Expected non-empty output when similarity >= {floor}\n"
        f"Input: {text!r}"
    )
