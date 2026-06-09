"""
Property 6: Iterative paraphrase produces divergence.

For all non-empty inputs, with at least one successful (fake) pass, the output
Lexical_Divergence from the input is > 0 — i.e., the output differs from the input
at the word/character level.

Requirements: 1.1
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from hypothesis import given, settings

from humanizer.stage_iterative import IterativeParaphraser
from tests.conftest import FakeSimilarityEvaluator
from tests.strategies import academic_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _modify_text(text: str) -> str:
    """Produce a deterministic modification of the input text.

    Substitutes common academic words with synonyms and appends a unique marker
    word to guarantee lexical divergence from the original at the word-set level.
    """
    substitutions = [
        ("the study", "this research"),
        ("results indicate", "findings suggest"),
        ("we observe", "it is observed"),
        ("the data", "these data"),
        ("the framework", "our framework"),
        ("the evidence", "this evidence"),
        ("the model", "the proposed model"),
        ("these results", "the outcomes"),
        ("the proposed method", "the suggested approach"),
    ]
    modified = text
    for old, new in substitutions:
        modified = modified.replace(old, new, 1)

    # Always append a unique marker word that cannot appear in the original
    # generated text to guarantee word-set divergence
    modified = modified + " DIVERGENCE_MARKER_XYZ"

    return modified


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

# Feature: ultimate-humanizer, Property 6: Iterative paraphrase produces divergence


@given(text=academic_text(min_protected_terms=1, max_protected_terms=2))
@settings(max_examples=100)
def test_iterative_paraphrase_produces_divergence(text: str) -> None:
    """Property 6: Iterative paraphrase produces divergence.

    For all non-empty inputs, with at least one successful (fake) pass,
    the output Lexical_Divergence from the input is > 0.

    Validates: Requirements 1.1
    """
    # Use a FakeSimilarityEvaluator that returns high scores (>=0.80)
    # so passes are accepted rather than discarded
    evaluator = FakeSimilarityEvaluator(default=0.90)

    paraphraser = IterativeParaphraser(
        aggression=0.5,
        seed=42,
        model="test-model",
        api_key="test-key",
        base_url="http://localhost",
        similarity=evaluator,
        timeout_s=30,
    )

    # Mock _llm_pass to return a modified version of the input
    # so at least one pass succeeds and produces divergence
    def fake_llm_pass(input_text: str, pass_index: int) -> str:
        return _modify_text(input_text)

    with patch.object(paraphraser, "_llm_pass", side_effect=fake_llm_pass):
        output = paraphraser.process(text)

    # Verify lexical divergence > 0: output must differ from input
    assert output != text, (
        f"Output is identical to input — no lexical divergence.\n"
        f"Input:  {text!r}\n"
        f"Output: {output!r}"
    )

    # Additional check: at least one word differs
    input_words = set(text.lower().split())
    output_words = set(output.lower().split())
    symmetric_diff = input_words.symmetric_difference(output_words)
    assert len(symmetric_diff) > 0, (
        f"No word-level divergence detected between input and output.\n"
        f"Input words:  {sorted(input_words)[:10]}...\n"
        f"Output words: {sorted(output_words)[:10]}..."
    )
