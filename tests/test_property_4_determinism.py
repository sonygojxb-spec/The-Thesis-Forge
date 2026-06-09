"""
Property 4: Seed determinism — parameterized across stages.

For all input texts and for every stage T that accepts a seed, providing the same
seed and the same input text MUST produce identical output across multiple calls.

This suite is parameterized: each stage is added as a pytest.param fixture so the
same property test runs against every stage independently.

Requirements: 6.6 (SemanticTransformer seed determinism)

# Feature: ultimate-humanizer, Property 4: Seed determinism
"""

from __future__ import annotations

from typing import Callable

import pytest
from hypothesis import given, settings

from humanizer.results import TargetPerplexityProfile
from humanizer.stage_error_injector import ErrorInjector
from humanizer.stage_iterative import IterativeParaphraser
from humanizer.stage_perplexity_optimize import PerplexityOptimizer
from humanizer.stage_semantic import SemanticTransformer
from humanizer.stage_stylometric import StylometricObfuscator
from tests.conftest import FakeSimilarityEvaluator
from tests.strategies import academic_text


# ---------------------------------------------------------------------------
# Stage factories — each returns a factory that creates a fresh stage instance
# with a fixed seed. We create fresh instances per call to ensure the RNG is
# reset to the same initial state each time.
# ---------------------------------------------------------------------------


def make_semantic_transformer_factory() -> Callable[[], Callable[[str], str]]:
    """Return a factory that creates a fresh SemanticTransformer with seed=42.

    Each call to the returned factory produces a new instance with the RNG
    reset to seed=42, ensuring deterministic behaviour.
    """
    def factory() -> Callable[[str], str]:
        evaluator = FakeSimilarityEvaluator(default=0.95)
        transformer = SemanticTransformer(
            aggression=0.7,
            seed=42,
            similarity=evaluator,
            floor=0.90,
        )
        return transformer.process
    return factory


def make_iterative_paraphraser_factory() -> Callable[[], Callable[[str], str]]:
    """Return a factory that creates a fresh IterativeParaphraser with seed=42.

    Each call to the returned factory produces a new instance with the RNG
    reset to seed=42, ensuring deterministic behaviour. The _llm_pass method
    is mocked with a deterministic modification (appends " rewritten") so the
    pass succeeds and non-LLM randomized selection is exercised deterministically.
    """
    def factory() -> Callable[[str], str]:
        evaluator = FakeSimilarityEvaluator(default=0.95)
        paraphraser = IterativeParaphraser(
            aggression=0.5,
            seed=42,
            similarity=evaluator,
            timeout_s=30,
        )

        # Mock _llm_pass with a deterministic modification
        def deterministic_llm_pass(text: str, pass_index: int) -> str:
            return text + " rewritten"

        paraphraser._llm_pass = deterministic_llm_pass
        return paraphraser.process
    return factory


def make_stylometric_obfuscator_factory() -> Callable[[], Callable[[str], str]]:
    """Return a factory that creates a fresh StylometricObfuscator with seed=42.

    Each call to the returned factory produces a new instance with the RNG
    reset to seed=42, ensuring deterministic behaviour. The StylometricObfuscator
    is NLP-only, so no LLM mocking is needed — only a FakeSimilarityEvaluator
    returning scores above the 0.85 floor.
    """
    def factory() -> Callable[[str], str]:
        evaluator = FakeSimilarityEvaluator(default=0.95)
        obfuscator = StylometricObfuscator(
            aggression=0.5,
            seed=42,
            similarity=evaluator,
            floor=0.85,
        )
        return obfuscator.process
    return factory


def make_perplexity_optimizer_factory() -> Callable[[], Callable[[str], str]]:
    """Return a factory that creates a fresh PerplexityOptimizer with seed=42.

    Each call to the returned factory produces a new instance with the RNG
    reset to seed=42, ensuring deterministic behaviour. The PerplexityOptimizer
    is NLP-only, so no LLM mocking is needed — only a FakeSimilarityEvaluator
    returning scores above the 0.85 floor. Uses a TargetPerplexityProfile with
    target_mean=80.0 to force optimization edits.
    """
    def factory() -> Callable[[str], str]:
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
    return factory


def make_error_injector_factory() -> Callable[[], Callable[[str], str]]:
    """Return a factory that creates a fresh ErrorInjector with seed=42 and aggression=0.5.

    Each call to the returned factory produces a new instance with the RNG
    reset to seed=42, ensuring deterministic behaviour. ErrorInjector is an
    NLP-only stage — no similarity evaluator or LLM mocking needed.

    Validates: Requirements 5.6
    """
    def factory() -> Callable[[str], str]:
        injector = ErrorInjector(
            aggression=0.5,
            seed=42,
        )
        return injector.process
    return factory


# ---------------------------------------------------------------------------
# Parameterized test suite
# ---------------------------------------------------------------------------

# Feature: ultimate-humanizer, Property 4: Seed determinism

STAGE_PARAMS = [
    pytest.param(make_semantic_transformer_factory, id="SemanticTransformer"),
    pytest.param(make_iterative_paraphraser_factory, id="IterativeParaphraser"),
    pytest.param(make_stylometric_obfuscator_factory, id="StylometricObfuscator"),
    pytest.param(make_perplexity_optimizer_factory, id="PerplexityOptimizer"),
    pytest.param(make_error_injector_factory, id="ErrorInjector"),
]


@pytest.mark.parametrize("stage_factory_maker", STAGE_PARAMS)
@given(text=academic_text(min_protected_terms=1, max_protected_terms=2))
@settings(max_examples=100)
def test_seed_determinism(
    stage_factory_maker: Callable[[], Callable[[], Callable[[str], str]]], text: str
) -> None:
    """Property 4: Seed determinism.

    For identical seed and identical input, the stage MUST produce identical
    output across multiple independent calls (fresh instances with same seed).

    Validates: Requirements 1.7, 6.6, 2.6
    """
    factory = stage_factory_maker()

    # Create two fresh instances with the same seed and process the same input
    process_1 = factory()
    output_1 = process_1(text)

    process_2 = factory()
    output_2 = process_2(text)

    assert output_1 == output_2, (
        f"Seed determinism violated: same seed + same input produced different outputs\n"
        f"Input:   {text!r}\n"
        f"Output1: {output_1!r}\n"
        f"Output2: {output_2!r}"
    )
