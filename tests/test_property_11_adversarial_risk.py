"""
Property 11: Adversarial risk non-increase.

For all inputs with >=1 non-whitespace character, the output risk (scored with
the same fake classifier on both input and output) is <= the input risk. This
validates that the AdversarialRewriter either reduces detection risk or returns
the input unchanged — it never increases risk.

Also tests the rejection path: when the fake classifier returns a higher score
for the candidate, the output must equal the input (candidate rejected).

Requirements: 4.1, 4.6

# Feature: ultimate-humanizer, Property 11: Adversarial risk non-increase
"""

from __future__ import annotations

from hypothesis import given, settings

from humanizer.stage_adversarial import AdversarialRewriter
from tests.conftest import FakeClassifier, FakeSimilarityEvaluator
from tests.strategies import academic_text


# ---------------------------------------------------------------------------
# Test: acceptance path — risk_after <= risk_before when changed
# ---------------------------------------------------------------------------


@given(text=academic_text(min_protected_terms=1, max_protected_terms=2))
@settings(max_examples=100)
def test_adversarial_risk_non_increase_acceptance(text: str) -> None:
    """Property 11: Adversarial risk non-increase (acceptance path).

    When the fake classifier returns a higher score for the input (75.0) and
    a lower score for the candidate (50.0), the candidate is accepted and the
    result's risk_after <= risk_before.

    **Validates: Requirements 4.1, 4.6**
    """
    # FakeClassifier returns scores in FIFO order:
    # 1st call: score input → 75.0 (high risk)
    # 2nd call: score candidate → 50.0 (lower risk, so candidate accepted)
    classifier = FakeClassifier(scores=[75.0, 50.0])

    # FakeSimilarityEvaluator returns high score so similarity floor doesn't interfere
    similarity = FakeSimilarityEvaluator(default=0.95)

    rewriter = AdversarialRewriter(
        aggression=0.5,
        seed=42,
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:9999",
        similarity=similarity,
        classifier=classifier,
        floor=0.85,
        timeout_s=5,
    )

    # Mock _llm_rewrite to return a deterministic modification
    def mock_llm_rewrite(text_input: str) -> str:
        return text_input + " adversarially rewritten"

    rewriter._llm_rewrite = mock_llm_rewrite

    result = rewriter.process_measured(text)

    # The property: when result.changed == True, risk_after <= risk_before
    if result.changed:
        assert result.risk_after is not None, (
            "risk_after must be set when changed == True"
        )
        assert result.risk_before is not None, (
            "risk_before must be set when changed == True"
        )
        assert result.risk_after <= result.risk_before, (
            f"Adversarial risk non-increase violated: "
            f"risk_after={result.risk_after} > risk_before={result.risk_before}\n"
            f"Input:  {text!r}\n"
            f"Output: {result.text!r}"
        )


# ---------------------------------------------------------------------------
# Test: rejection path — candidate risk > input risk → output == input
# ---------------------------------------------------------------------------


@given(text=academic_text(min_protected_terms=1, max_protected_terms=2))
@settings(max_examples=100)
def test_adversarial_risk_non_increase_rejection(text: str) -> None:
    """Property 11: Adversarial risk non-increase (rejection path).

    When the fake classifier returns a lower score for the input (40.0) and
    a higher score for the candidate (80.0), the candidate must be rejected
    and the output must equal the input.

    **Validates: Requirements 4.1, 4.6**
    """
    # FakeClassifier returns scores in FIFO order:
    # 1st call: score input → 40.0 (lower risk)
    # 2nd call: score candidate → 80.0 (higher risk, so candidate rejected)
    classifier = FakeClassifier(scores=[40.0, 80.0])

    # FakeSimilarityEvaluator returns high score so similarity floor doesn't interfere
    similarity = FakeSimilarityEvaluator(default=0.95)

    rewriter = AdversarialRewriter(
        aggression=0.5,
        seed=42,
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:9999",
        similarity=similarity,
        classifier=classifier,
        floor=0.85,
        timeout_s=5,
    )

    # Mock _llm_rewrite to return a deterministic modification
    def mock_llm_rewrite(text_input: str) -> str:
        return text_input + " adversarially rewritten"

    rewriter._llm_rewrite = mock_llm_rewrite

    result = rewriter.process_measured(text)

    # When candidate risk > input risk, output must equal input (Req 4.6)
    assert result.text == text, (
        f"Expected output == input when candidate risk > input risk\n"
        f"Input:  {text!r}\n"
        f"Output: {result.text!r}"
    )
    assert result.changed is False, (
        "Expected changed == False when candidate is rejected due to higher risk"
    )
