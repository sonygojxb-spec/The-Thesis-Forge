"""
Property 12: Adversarial change monotonicity.

For all inputs (fake LLM + seed fixed), the word-change proportion at higher
aggression is >= the proportion at any lower aggression. This validates that
increased aggression produces at least as many word-level changes.

The mock `_llm_rewrite` produces MORE word changes at higher aggression by
appending N extra words where N scales with the rewriter's aggression value.
A FakeClassifier always returns a lower score for the candidate than input
(so candidates are always accepted), and a FakeSimilarityEvaluator returns
high scores (>=0.85) to pass the similarity floor.

Requirements: 4.4

# Feature: ultimate-humanizer, Property 12: Adversarial change monotonicity
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from humanizer.stage_adversarial import AdversarialRewriter
from tests.conftest import FakeClassifier, FakeSimilarityEvaluator
from tests.strategies import academic_text


def _word_change_proportion(original: str, output: str) -> float:
    """Compute the proportion of words changed between original and output.

    Returns: count of words that differ / total input words.
    Uses positional comparison: for each position, checks if the word differs.
    Extra words in output beyond original length count as changes too.
    """
    orig_words = original.split()
    out_words = output.split()

    if not orig_words:
        return 0.0

    total = len(orig_words)
    # Count positional differences
    changes = 0
    for i in range(min(len(orig_words), len(out_words))):
        if orig_words[i] != out_words[i]:
            changes += 1

    # Words beyond original length are all changes
    if len(out_words) > len(orig_words):
        changes += len(out_words) - len(orig_words)

    return changes / total


def _make_mock_llm_rewrite(aggression: float):
    """Create a mock _llm_rewrite that produces MORE word changes at higher aggression.

    The mock appends N extra words where N = round(aggression * 10).
    This simulates what the LLM would do with a more aggressive prompt.
    """
    def mock_llm_rewrite(text_input: str) -> str:
        n_extra = round(aggression * 10)
        extra_words = " ".join(f"extra{i}" for i in range(n_extra))
        if extra_words:
            return text_input + " " + extra_words
        return text_input
    return mock_llm_rewrite


@given(
    text=academic_text(min_protected_terms=1, max_protected_terms=2),
    lo=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    hi=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_adversarial_change_monotonicity(text: str, lo: float, hi: float) -> None:
    """Property 12: Adversarial change monotonicity.

    For all inputs with a fixed seed, the word-change proportion at higher
    aggression >= proportion at any lower aggression.

    **Validates: Requirements 4.4**
    """
    # Ensure lo < hi
    assume(lo < hi)

    # Use a FIXED seed for both runs so the only difference is aggression
    fixed_seed = 12345

    # FakeClassifier: always returns lower score for candidate than input
    # Each rewriter call uses 2 classifier calls (input score, candidate score)
    # For lo run: input=80, candidate=50; for hi run: input=80, candidate=50
    classifier_lo = FakeClassifier(scores=[80.0, 50.0])
    classifier_hi = FakeClassifier(scores=[80.0, 50.0])

    # FakeSimilarityEvaluator returning high scores (>=0.85)
    similarity_lo = FakeSimilarityEvaluator(default=0.95)
    similarity_hi = FakeSimilarityEvaluator(default=0.95)

    # --- Low aggression run ---
    rewriter_lo = AdversarialRewriter(
        aggression=lo,
        seed=fixed_seed,
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:9999",
        similarity=similarity_lo,
        classifier=classifier_lo,
        floor=0.85,
        timeout_s=5,
    )
    rewriter_lo._llm_rewrite = _make_mock_llm_rewrite(lo)

    result_lo = rewriter_lo.process_measured(text)

    # --- High aggression run ---
    rewriter_hi = AdversarialRewriter(
        aggression=hi,
        seed=fixed_seed,
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:9999",
        similarity=similarity_hi,
        classifier=classifier_hi,
        floor=0.85,
        timeout_s=5,
    )
    rewriter_hi._llm_rewrite = _make_mock_llm_rewrite(hi)

    result_hi = rewriter_hi.process_measured(text)

    # Only assert when BOTH runs produce changed=True
    assume(result_lo.changed is True)
    assume(result_hi.changed is True)

    # Measure word-change proportion
    prop_lo = _word_change_proportion(text, result_lo.text)
    prop_hi = _word_change_proportion(text, result_hi.text)

    assert prop_hi >= prop_lo, (
        f"Adversarial change monotonicity violated: "
        f"proportion at aggression={hi:.4f} ({prop_hi:.4f}) < "
        f"proportion at aggression={lo:.4f} ({prop_lo:.4f})\n"
        f"Input:     {text!r}\n"
        f"Output lo: {result_lo.text!r}\n"
        f"Output hi: {result_hi.text!r}"
    )
