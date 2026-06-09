"""
Property tests for the Classifier component.

Tests input validation behaviour for the Classifier (and FakeClassifier)
to confirm that invalid inputs are rejected with InvalidInput rather than
returning a numeric score.

Also tests score-range validity of detection_risk_score for both the
classifier path and heuristic fallback path.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from humanizer.classifier import Classifier, InvalidInput, detection_risk_score
from tests.conftest import FakeClassifier


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Empty or whitespace-only strings (always invalid)
_empty_strings = st.one_of(
    st.just(""),
    st.text(alphabet=" \t\n\r", min_size=1, max_size=50),
)

# Strings that exceed the 10,000-character limit.
# Hypothesis st.text() has internal limits on min_size, so we build oversized
# strings by repeating a base character a random number of times above the limit.
_oversized_strings = st.integers(min_value=10_001, max_value=15_000).map(
    lambda n: "a" * n
)


# ---------------------------------------------------------------------------
# Property 17: Classifier invalid-input rejection
# ---------------------------------------------------------------------------

# Feature: ultimate-humanizer, Property 17: Classifier invalid-input rejection


@settings(max_examples=100)
@given(text=_empty_strings)
def test_classifier_rejects_empty_input(text: str) -> None:
    """Classifier.score raises InvalidInput for empty/whitespace-only text.

    **Validates: Requirements 9.6**
    """
    clf = Classifier()
    with pytest.raises(InvalidInput):
        clf.score(text)


@settings(max_examples=100)
@given(text=_oversized_strings)
def test_classifier_rejects_oversized_input(text: str) -> None:
    """Classifier.score raises InvalidInput for text exceeding 10,000 characters.

    **Validates: Requirements 9.6**
    """
    clf = Classifier()
    with pytest.raises(InvalidInput):
        clf.score(text)


@settings(max_examples=100)
@given(text=_empty_strings)
def test_fake_classifier_rejects_empty_input(text: str) -> None:
    """FakeClassifier.score raises InvalidInput for empty/whitespace-only text.

    **Validates: Requirements 9.6**
    """
    clf = FakeClassifier()
    with pytest.raises(FakeClassifier.InvalidInput):
        clf.score(text)


@settings(max_examples=100)
@given(text=_oversized_strings)
def test_fake_classifier_rejects_oversized_input(text: str) -> None:
    """FakeClassifier.score raises InvalidInput for text exceeding 10,000 characters.

    **Validates: Requirements 9.6**
    """
    clf = FakeClassifier()
    with pytest.raises(FakeClassifier.InvalidInput):
        clf.score(text)


# ---------------------------------------------------------------------------
# Strategies for Property 5
# ---------------------------------------------------------------------------

# Valid-length text: 1 to 10,000 characters, non-whitespace-only.
# We use printable text with at least one non-whitespace character.
_valid_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=1,
    max_size=200,
).filter(lambda t: t.strip())

# Scores that a FakeClassifier can return (any float in [0, 100]).
_fake_scores = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)


# ---------------------------------------------------------------------------
# Property 5: Score-range validity (risk portion)
# ---------------------------------------------------------------------------

# Feature: ultimate-humanizer, Property 5: Score-range validity (risk portion)


@settings(max_examples=100)
@given(text=_valid_text, scripted_score=_fake_scores)
def test_detection_risk_score_classifier_path_range(text: str, scripted_score: float) -> None:
    """detection_risk_score with a FakeClassifier returns score in [0, 100] and source == "classifier".

    **Validates: Requirements 8.1, 9.1**
    """
    clf = FakeClassifier(scores=[scripted_score])
    score, source = detection_risk_score(text, classifier=clf)

    assert 0.0 <= score <= 100.0, f"Score {score} out of [0, 100] range"
    assert source == "classifier"


@settings(max_examples=100)
@given(text=_valid_text)
def test_detection_risk_score_heuristic_fallback_range(text: str) -> None:
    """detection_risk_score with classifier=None returns score in [0, 100] and source == "heuristic".

    **Validates: Requirements 8.1, 9.1**
    """
    score, source = detection_risk_score(text, classifier=None)

    assert 0.0 <= score <= 100.0, f"Score {score} out of [0, 100] range"
    assert source == "heuristic"


@settings(max_examples=100)
@given(text=_valid_text)
def test_detection_risk_score_classifier_failure_fallback_range(text: str) -> None:
    """detection_risk_score with a failing classifier falls back to heuristic with score in [0, 100].

    **Validates: Requirements 8.1, 9.1**
    """
    # FakeClassifier with fail_after=0 will raise on first call
    clf = FakeClassifier(fail_after=0)
    score, source = detection_risk_score(text, classifier=clf)

    assert 0.0 <= score <= 100.0, f"Score {score} out of [0, 100] range"
    assert source == "heuristic"


# ---------------------------------------------------------------------------
# Example tests: Classifier determinism and fallback indication (Task 4.4)
# ---------------------------------------------------------------------------


class TestClassifierDeterminism:
    """Example tests verifying that the same loaded model on identical input yields identical score.

    **Validates: Requirements 9.2**
    """

    def test_fake_classifier_returns_identical_score_on_repeated_calls(self) -> None:
        """FakeClassifier with deterministic scores returns the same result for the same input."""
        clf = FakeClassifier(scores=[72.5, 72.5])
        input_text = "The experiment demonstrates a significant correlation between variables."

        score_first = clf.score(input_text)
        score_second = clf.score(input_text)

        assert score_first == score_second
        assert score_first == 72.5

    def test_fake_classifier_default_score_is_deterministic(self) -> None:
        """FakeClassifier with empty queue returns identical default on repeated calls."""
        clf = FakeClassifier(default=45.0)
        input_text = "Linguistic patterns reveal underlying cognitive processes."

        score_first = clf.score(input_text)
        score_second = clf.score(input_text)

        assert score_first == score_second
        assert score_first == 45.0

    def test_detection_risk_score_deterministic_with_fake_classifier(self) -> None:
        """detection_risk_score returns identical (score, source) for identical input and classifier."""
        clf1 = FakeClassifier(scores=[88.0])
        clf2 = FakeClassifier(scores=[88.0])
        input_text = "Machine learning models require careful hyperparameter tuning."

        result1 = detection_risk_score(input_text, classifier=clf1)
        result2 = detection_risk_score(input_text, classifier=clf2)

        assert result1 == result2
        assert result1 == (88.0, "classifier")


class TestClassifierFallbackIndication:
    """Example tests verifying that forced load failure routes to heuristic and reports source.

    **Validates: Requirements 9.4**
    """

    def test_classifier_failure_falls_back_to_heuristic(self) -> None:
        """FakeClassifier that fails immediately causes fallback to heuristic source."""
        clf = FakeClassifier(fail_after=0)
        input_text = "Neural networks approximate complex non-linear functions."

        score, source = detection_risk_score(input_text, classifier=clf)

        assert source == "heuristic"
        assert 0.0 <= score <= 100.0

    def test_classifier_none_uses_heuristic(self) -> None:
        """When classifier is None, detection_risk_score uses heuristic directly."""
        input_text = "The results suggest a strong positive correlation."

        score, source = detection_risk_score(input_text, classifier=None)

        assert source == "heuristic"
        assert 0.0 <= score <= 100.0

    def test_fallback_score_is_numeric_and_bounded(self) -> None:
        """Heuristic fallback produces a numeric score within [0, 100]."""
        clf = FakeClassifier(fail_after=0)
        input_text = "Statistical significance was determined using a two-tailed t-test."

        score, source = detection_risk_score(input_text, classifier=clf)

        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0
        assert source == "heuristic"
