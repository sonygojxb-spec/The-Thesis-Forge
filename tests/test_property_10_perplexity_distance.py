"""
Property 10: Perplexity distance non-increase.

For all non-empty inputs and target profiles, the output's mean-perplexity
distance to the target is <= the input's mean-perplexity distance to the target.
For inputs with >=2 sentences, the output's variance distance to the target
variance is <= the input's variance distance to the target variance.

The property only applies when the stage actually changed the text
(result.changed == True). If changed is False, distance guarantees are
trivially satisfied (equal).

Requirements: 3.2, 3.3

# Feature: ultimate-humanizer, Property 10: Perplexity distance non-increase
"""

from __future__ import annotations

from hypothesis import assume, given, settings

from humanizer.results import TargetPerplexityProfile
from humanizer.stage_perplexity_optimize import PerplexityOptimizer
from humanizer.text_analysis import estimate_perplexity_score, split_sentences
from tests.conftest import FakeSimilarityEvaluator
from tests.strategies import multi_sentence_text


def _compute_variance(scores: list[float]) -> float:
    """Compute population variance of a list of scores."""
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    return sum((s - mean) ** 2 for s in scores) / len(scores)


@given(text=multi_sentence_text(min_sentences=2, max_sentences=6))
@settings(max_examples=100)
def test_perplexity_mean_distance_non_increase(text: str) -> None:
    """Property 10: Mean perplexity distance to target does not increase.

    For all non-empty inputs, the output's absolute distance between its
    measured mean perplexity and the target mean is <= the input's absolute
    distance between its measured mean perplexity and the target mean.

    Validates: Requirements 3.2
    """
    # Use a FakeSimilarityEvaluator returning high scores so candidates are accepted
    evaluator = FakeSimilarityEvaluator(default=0.95)

    target_profile = TargetPerplexityProfile(target_mean=60.0, target_variance=100.0)

    optimizer = PerplexityOptimizer(
        aggression=0.7,
        seed=42,
        similarity=evaluator,
        floor=0.85,
        target_profile=target_profile,
    )

    # Measure input perplexity
    input_sentences = split_sentences(text)
    input_scores = [estimate_perplexity_score(s) for s in input_sentences]

    # Filter inputs where perplexity is unmeasurable (all scores == 50.0)
    assume(not all(s == 50.0 for s in input_scores))

    input_mean = sum(input_scores) / len(input_scores)
    input_mean_distance = abs(input_mean - target_profile.target_mean)

    # Process through the optimizer
    result = optimizer.process_measured(text)

    if not result.changed:
        # If unchanged, distance is trivially equal (satisfied)
        return

    # Measure output perplexity
    output_sentences = split_sentences(result.text)
    output_scores = [estimate_perplexity_score(s) for s in output_sentences]
    output_mean = sum(output_scores) / len(output_scores)
    output_mean_distance = abs(output_mean - target_profile.target_mean)

    assert output_mean_distance <= input_mean_distance + 1e-9, (
        f"Mean perplexity distance to target increased!\n"
        f"Target mean: {target_profile.target_mean}\n"
        f"Input mean: {input_mean:.2f}, distance: {input_mean_distance:.4f}\n"
        f"Output mean: {output_mean:.2f}, distance: {output_mean_distance:.4f}\n"
        f"Input:  {text!r}\n"
        f"Output: {result.text!r}"
    )


@given(text=multi_sentence_text(min_sentences=2, max_sentences=6))
@settings(max_examples=100)
def test_perplexity_variance_distance_non_increase(text: str) -> None:
    """Property 10: Variance distance to target does not increase (>=2 sentences).

    For inputs with >=2 sentences, the output's absolute distance between its
    measured perplexity variance and the target variance is <= the input's
    absolute distance between its measured perplexity variance and the target
    variance.

    Validates: Requirements 3.3
    """
    # Use a FakeSimilarityEvaluator returning high scores so candidates are accepted
    evaluator = FakeSimilarityEvaluator(default=0.95)

    target_profile = TargetPerplexityProfile(target_mean=60.0, target_variance=100.0)

    optimizer = PerplexityOptimizer(
        aggression=0.7,
        seed=42,
        similarity=evaluator,
        floor=0.85,
        target_profile=target_profile,
    )

    # Measure input perplexity
    input_sentences = split_sentences(text)
    assume(len(input_sentences) >= 2)

    input_scores = [estimate_perplexity_score(s) for s in input_sentences]

    # Filter inputs where perplexity is unmeasurable (all scores == 50.0)
    assume(not all(s == 50.0 for s in input_scores))

    input_variance = _compute_variance(input_scores)
    input_variance_distance = abs(input_variance - target_profile.target_variance)

    # Process through the optimizer
    result = optimizer.process_measured(text)

    if not result.changed:
        # If unchanged, distance is trivially equal (satisfied)
        return

    # Measure output perplexity variance
    output_sentences = split_sentences(result.text)
    output_scores = [estimate_perplexity_score(s) for s in output_sentences]

    assume(len(output_scores) >= 2)

    output_variance = _compute_variance(output_scores)
    output_variance_distance = abs(output_variance - target_profile.target_variance)

    assert output_variance_distance <= input_variance_distance + 1e-9, (
        f"Variance distance to target increased!\n"
        f"Target variance: {target_profile.target_variance}\n"
        f"Input variance: {input_variance:.4f}, distance: {input_variance_distance:.4f}\n"
        f"Output variance: {output_variance:.4f}, distance: {output_variance_distance:.4f}\n"
        f"Input:  {text!r}\n"
        f"Output: {result.text!r}"
    )
