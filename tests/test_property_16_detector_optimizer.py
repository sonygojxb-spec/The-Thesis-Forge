"""
Property 16: Detector-optimizer selection and iteration bound.

For all inputs and generated candidates, the DetectorOptimizer:
1. Returns the lowest-risk candidate with similarity >= 0.85
2. Returns input when no candidate has similarity >= 0.85
3. Performs no more than max_iterations iterations (track call count)
4. Stops early when a candidate reaches target_threshold

Requirements: 8.2, 8.3, 8.4, 8.5, 8.6

# Feature: ultimate-humanizer, Property 16: Detector-optimizer selection and iteration bound
"""

from __future__ import annotations

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from humanizer.stage_detector_optimizer import DetectorOptimizer
from tests.conftest import FakeClassifier, FakeSimilarityEvaluator
from tests.strategies import academic_text


# ---------------------------------------------------------------------------
# Sub-property 1: Returns lowest-risk candidate when multiple valid candidates
# ---------------------------------------------------------------------------


@given(
    text=academic_text(min_protected_terms=1, max_protected_terms=2),
    max_iterations=st.integers(min_value=3, max_value=5),
)
@settings(max_examples=100)
def test_returns_lowest_risk_valid_candidate(text: str, max_iterations: int) -> None:
    """Property 16.1: Returns lowest-risk candidate among those with similarity >= 0.85.

    We generate multiple candidates with known risk scores. The optimizer should
    return the candidate with the lowest risk among those that pass the similarity
    floor.

    **Validates: Requirements 8.2, 8.3, 8.4**
    """
    # Setup: input risk = 80, candidates have risks [70, 40, 60]
    # All candidates have similarity >= 0.85 (default=0.90)
    # The lowest risk candidate is the one with score 40 (iteration 2)
    input_risk = 80.0
    candidate_risks = [70.0, 40.0, 60.0]

    # Classifier scores: first call scores input, then each candidate
    classifier_scores = [input_risk] + candidate_risks[:max_iterations]
    classifier = FakeClassifier(scores=classifier_scores, default=60.0)

    # All candidates pass similarity floor
    similarity = FakeSimilarityEvaluator(default=0.90)

    optimizer = DetectorOptimizer(
        aggression=0.5,
        seed=42,
        classifier=classifier,
        similarity=similarity,
        target_threshold=10,  # Set low so we don't early-stop
        max_iterations=max_iterations,
    )

    # Track generated candidates so we can identify which one is returned
    candidates_generated: list[str] = []

    def mock_generate_candidate(masked_text: str, seed: int, iteration: int) -> str:
        candidate = f"candidate_{iteration}_{text[:20]}"
        candidates_generated.append(candidate)
        return candidate

    optimizer._generate_candidate = mock_generate_candidate

    result = optimizer.process_measured(text)

    # The optimizer should have selected the lowest-risk candidate
    # Among candidates with similarity >= 0.85, the one with risk 40.0 is best
    assert result.risk_after is not None, "risk_after should be set"
    assert result.risk_after <= input_risk, (
        f"Result risk ({result.risk_after}) should be <= input risk ({input_risk})"
    )
    # Specifically: best candidate has risk 40.0 (second candidate)
    assert result.risk_after == 40.0, (
        f"Expected lowest risk 40.0, got {result.risk_after}"
    )
    assert result.changed is True, "Output should differ from input"


# ---------------------------------------------------------------------------
# Sub-property 2: Returns input when no candidate has similarity >= 0.85
# ---------------------------------------------------------------------------


@given(text=academic_text(min_protected_terms=1, max_protected_terms=2))
@settings(max_examples=100)
def test_returns_input_when_no_candidate_meets_similarity(text: str) -> None:
    """Property 16.2: Returns input unchanged when no candidate has similarity >= 0.85.

    When ALL candidates fail the similarity floor, the optimizer must return
    the original input text.

    **Validates: Requirements 8.4, 8.5**
    """
    # Input risk is high (above target) so optimization loop runs
    input_risk = 80.0
    # Candidates have low risk but will fail similarity gate
    classifier = FakeClassifier(scores=[input_risk], default=20.0)

    # All candidates fail the 0.85 similarity floor
    similarity = FakeSimilarityEvaluator(default=0.70)

    optimizer = DetectorOptimizer(
        aggression=0.5,
        seed=42,
        classifier=classifier,
        similarity=similarity,
        target_threshold=30,
        max_iterations=5,
    )

    def mock_generate_candidate(masked_text: str, seed: int, iteration: int) -> str:
        return f"low_sim_candidate_{iteration}"

    optimizer._generate_candidate = mock_generate_candidate

    result = optimizer.process_measured(text)

    # No candidate met similarity >= 0.85, so input must be returned unchanged
    assert result.text == text, (
        f"Expected input returned unchanged when no candidate meets similarity floor\n"
        f"Input:  {text!r}\n"
        f"Output: {result.text!r}"
    )
    assert result.changed is False, (
        "Expected changed == False when no valid candidate exists"
    )
    assert result.fell_back is True, (
        "Expected fell_back == True when no valid candidate exists"
    )


# ---------------------------------------------------------------------------
# Sub-property 3: Performs no more than max_iterations iterations
# ---------------------------------------------------------------------------


@given(
    text=academic_text(min_protected_terms=1, max_protected_terms=2),
    max_iterations=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_iteration_count_bounded_by_max_iterations(
    text: str, max_iterations: int
) -> None:
    """Property 16.3: Performs no more than max_iterations iterations.

    The optimizer must never call _generate_candidate more than max_iterations
    times, regardless of candidate quality.

    **Validates: Requirements 8.2, 8.6**
    """
    # Input risk is high so the loop always runs
    input_risk = 90.0
    # Candidate risk is always slightly better but never reaches target
    # This ensures the loop runs all iterations without early stopping
    classifier = FakeClassifier(scores=[input_risk], default=85.0)

    # All candidates pass similarity floor
    similarity = FakeSimilarityEvaluator(default=0.90)

    optimizer = DetectorOptimizer(
        aggression=0.5,
        seed=42,
        classifier=classifier,
        similarity=similarity,
        target_threshold=10,  # Very low target, never reached
        max_iterations=max_iterations,
    )

    call_count = 0

    def mock_generate_candidate(masked_text: str, seed: int, iteration: int) -> str:
        nonlocal call_count
        call_count += 1
        return f"candidate_{iteration}_{text[:10]}"

    optimizer._generate_candidate = mock_generate_candidate

    optimizer.process_measured(text)

    assert call_count <= max_iterations, (
        f"Iteration count ({call_count}) exceeded max_iterations ({max_iterations})"
    )
    # Should use all iterations when target is never reached
    assert call_count == max_iterations, (
        f"Expected exactly {max_iterations} iterations when target not reached, "
        f"got {call_count}"
    )


# ---------------------------------------------------------------------------
# Sub-property 4: Stops early when a candidate reaches target_threshold
# ---------------------------------------------------------------------------


@given(
    text=academic_text(min_protected_terms=1, max_protected_terms=2),
    early_stop_iteration=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100)
def test_stops_early_at_target_threshold(
    text: str, early_stop_iteration: int
) -> None:
    """Property 16.4: Stops early when a candidate reaches target_threshold.

    When a candidate achieves a risk score at or below the target threshold,
    the optimizer should stop and return that candidate without exhausting
    all iterations.

    **Validates: Requirements 8.2, 8.3**
    """
    target_threshold = 30
    max_iterations = 10  # Much more than early_stop_iteration
    input_risk = 80.0

    # Build classifier scores: input risk first, then candidate scores.
    # Candidates before early_stop_iteration have risk > target.
    # The candidate at early_stop_iteration has risk <= target.
    candidate_scores: list[float] = []
    for i in range(1, max_iterations + 1):
        if i < early_stop_iteration:
            candidate_scores.append(60.0)  # Above target, better than input
        elif i == early_stop_iteration:
            candidate_scores.append(25.0)  # At or below target → early stop
        else:
            candidate_scores.append(50.0)  # Should never be reached

    classifier_scores = [input_risk] + candidate_scores
    classifier = FakeClassifier(scores=classifier_scores, default=50.0)

    # All candidates pass similarity floor
    similarity = FakeSimilarityEvaluator(default=0.92)

    optimizer = DetectorOptimizer(
        aggression=0.5,
        seed=42,
        classifier=classifier,
        similarity=similarity,
        target_threshold=target_threshold,
        max_iterations=max_iterations,
    )

    call_count = 0

    def mock_generate_candidate(masked_text: str, seed: int, iteration: int) -> str:
        nonlocal call_count
        call_count += 1
        return f"candidate_{iteration}_{text[:10]}"

    optimizer._generate_candidate = mock_generate_candidate

    result = optimizer.process_measured(text)

    # Should have stopped at early_stop_iteration, not continued to max_iterations
    assert call_count == early_stop_iteration, (
        f"Expected early stop at iteration {early_stop_iteration}, "
        f"but {call_count} candidates were generated"
    )

    # The returned candidate should have risk <= target_threshold
    assert result.risk_after is not None, "risk_after must be set"
    assert result.risk_after <= target_threshold, (
        f"Expected risk_after ({result.risk_after}) <= target ({target_threshold}) "
        f"after early stop"
    )
    assert result.changed is True, "Output should differ from input after early stop"
