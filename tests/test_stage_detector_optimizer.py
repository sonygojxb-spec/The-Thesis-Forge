"""
Example-based tests for DetectorOptimizer — mid-loop classifier failure handling.

Verifies Requirement 8.8: when the classifier fails mid-loop, the optimization
loop stops early, surfaces the error via StageResult.error, and returns either
the best valid candidate found so far or the original input.

Requirements: 8.8
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from humanizer.results import StageResult
from humanizer.stage_detector_optimizer import DetectorOptimizer
from tests.conftest import FakeClassifier, FakeSimilarityEvaluator


def _make_failing_detection_risk_score(fail_after: int, scores: list[float]):
    """Create a detection_risk_score replacement that raises after N calls.

    This simulates Req 8.8: the classifier (or the risk scoring pathway)
    fails mid-loop, causing the optimizer to break out with an error.

    Parameters
    ----------
    fail_after : int
        Number of successful calls before raising RuntimeError.
    scores : list[float]
        Scores to return for each successful call (consumed in order).

    Returns
    -------
    callable
        A function with the same signature as detection_risk_score.
    """
    state = {"call_count": 0, "score_idx": 0}

    def fake_detection_risk_score(text, classifier=None):
        state["call_count"] += 1
        if state["call_count"] > fail_after:
            raise RuntimeError(
                f"Simulated classifier failure after {fail_after} calls"
            )
        idx = state["score_idx"]
        state["score_idx"] += 1
        score = scores[idx] if idx < len(scores) else scores[-1]
        return (float(score), "classifier")

    return fake_detection_risk_score


class TestMidLoopClassifierFailure:
    """Tests for mid-loop classifier failure handling (Req 8.8)."""

    def test_fail_after_2_returns_input_with_error(self):
        """Classifier fails after 2 calls: first call scores input (high risk),
        second call fails during candidate scoring → loop stops, returns
        input (no valid candidate yet), error is set.

        Call sequence:
          1. detection_risk_score(input) → 80.0 (input risk, above threshold)
          2. detection_risk_score(candidate_1) → RuntimeError (failure)

        Since no valid candidate was scored before the failure, the result
        should fall back to the original input text with an error message.
        """
        # Similarity evaluator returns high scores (>= 0.85)
        similarity = FakeSimilarityEvaluator(default=0.90)

        optimizer = DetectorOptimizer(
            aggression=0.5,
            seed=100,
            classifier=None,  # Not used directly; detection_risk_score is patched
            similarity=similarity,
            target_threshold=30,
            max_iterations=5,
        )

        input_text = (
            "The neural network model demonstrates significant improvements "
            "in accuracy over baseline methods."
        )

        # Mock _generate_candidate to return deterministic candidates
        candidates = [
            "A neural network shows notable accuracy gains compared to baseline approaches.",
            "The deep learning system achieves better accuracy than prior methods.",
            "Neural architectures outperform traditional baseline techniques significantly.",
        ]
        call_idx = {"i": 0}

        def fake_generate(masked_text, seed, iteration):
            idx = call_idx["i"]
            call_idx["i"] += 1
            if idx < len(candidates):
                return candidates[idx]
            return candidates[-1]

        # Patch detection_risk_score: succeeds once (input scoring), fails on 2nd call
        fake_drs = _make_failing_detection_risk_score(
            fail_after=1, scores=[80.0]
        )

        with patch(
            "humanizer.stage_detector_optimizer.detection_risk_score",
            side_effect=fake_drs,
        ), patch.object(optimizer, "_generate_candidate", side_effect=fake_generate):
            result = optimizer.process_measured(input_text)

        # Loop stopped due to classifier failure — no valid candidate scored
        assert result.error is not None, "Expected error to be set on classifier failure"
        assert "failure" in result.error.lower()
        # Since no valid candidate was scored before failure, returns input
        assert result.text == input_text
        assert result.fell_back is True

    def test_fail_after_3_returns_best_valid_candidate_with_error(self):
        """Classifier fails after 3 calls: first call scores input (risk=80),
        second call scores candidate 1 (risk=60, valid), third call fails →
        loop stops, returns best valid candidate from iteration 1, error is set.

        Call sequence:
          1. detection_risk_score(input) → 80.0 (input risk, above threshold)
          2. detection_risk_score(candidate_1) → 60.0 (valid, risk < input)
          3. detection_risk_score(candidate_2) → RuntimeError (failure)

        Since candidate 1 was valid (risk=60 < input_risk=80, sim >= 0.85),
        it should be returned as the best candidate, with error set.
        """
        # Similarity evaluator always returns high scores (>= 0.85)
        similarity = FakeSimilarityEvaluator(default=0.92)

        optimizer = DetectorOptimizer(
            aggression=0.5,
            seed=200,
            classifier=None,  # Not used directly; detection_risk_score is patched
            similarity=similarity,
            target_threshold=30,
            max_iterations=5,
        )

        input_text = (
            "The neural network model demonstrates significant improvements "
            "in accuracy over baseline methods."
        )

        candidate_1 = (
            "A neural network shows notable accuracy gains compared to "
            "baseline approaches."
        )
        candidate_2 = (
            "The deep learning system achieves better accuracy than prior methods."
        )
        candidates = [candidate_1, candidate_2, "Another candidate text here."]
        call_idx = {"i": 0}

        def fake_generate(masked_text, seed, iteration):
            idx = call_idx["i"]
            call_idx["i"] += 1
            if idx < len(candidates):
                return candidates[idx]
            return candidates[-1]

        # Patch detection_risk_score: succeeds twice (input + candidate 1), fails on 3rd
        fake_drs = _make_failing_detection_risk_score(
            fail_after=2, scores=[80.0, 60.0]
        )

        with patch(
            "humanizer.stage_detector_optimizer.detection_risk_score",
            side_effect=fake_drs,
        ), patch.object(optimizer, "_generate_candidate", side_effect=fake_generate):
            result = optimizer.process_measured(input_text)

        # Loop stopped due to classifier failure on iteration 2
        assert result.error is not None, "Expected error to be set on classifier failure"
        assert "failure" in result.error.lower()
        # Best valid candidate from iteration 1 should be returned
        assert result.text == candidate_1
        assert result.risk_after == 60.0
        assert result.similarity == 0.92
        assert result.changed is True
        assert result.fell_back is False
