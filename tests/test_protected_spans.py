"""
Tests for ProtectedSpanGuard: property-based and unit tests.

Property tests validate that masking, arbitrary editing of non-placeholder content,
and unmasking preserves all numeric values and citation markers with identical counts.

Unit tests cover guard edge cases: quoted-content protection, overlapping spans,
citation/number adjacency, empty input, verify delta detection, and case-sensitivity.

Requirements: 5.4, 14.2, 14.4
"""

from __future__ import annotations

import re
import random

from hypothesis import given, settings
from hypothesis import strategies as st

from humanizer.protected_spans import ProtectedSpanGuard, _PLACEHOLDER_PREFIX, _PLACEHOLDER_SUFFIX
from humanizer.stage_error_injector import ErrorInjector
from tests.strategies import academic_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Regex patterns matching the placeholder tokens used by ProtectedSpanGuard
_PLACEHOLDER_RE = re.compile(
    re.escape(_PLACEHOLDER_PREFIX) + r".+?" + re.escape(_PLACEHOLDER_SUFFIX)
)

# Patterns for counting numeric values and citation markers (same as ProtectedSpanGuard._count_protected)
_NUMBER_RE = re.compile(r'-?\b\d+(?:\.\d+)?%?')
_CITATION_PAREN_RE = re.compile(r'\([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?,\s*\d{4}\)')
_CITATION_BRACKET_RE = re.compile(r'\[\d+(?:\s*[-,]\s*\d+)*\]')


def count_numbers(text: str) -> int:
    """Count numeric values in text."""
    return len(_NUMBER_RE.findall(text))


def count_citations(text: str) -> int:
    """Count citation markers (parenthetical + bracketed) in text."""
    return (
        len(_CITATION_PAREN_RE.findall(text))
        + len(_CITATION_BRACKET_RE.findall(text))
    )


def arbitrary_edit(masked_text: str, seed: int) -> str:
    """Apply an arbitrary edit to masked text without touching placeholders.

    Tokenizes the masked text into words (whitespace-separated), identifies which
    tokens are placeholders vs normal words, then applies random transformations
    (shuffle, insert filler) ONLY to the non-placeholder tokens while keeping
    placeholders in their original positions relative to the structure.
    """
    rng = random.Random(seed)

    # Tokenize by whitespace, preserving token boundaries
    tokens = masked_text.split()
    if not tokens:
        return masked_text

    # Classify tokens: placeholder or normal word
    placeholder_indices: list[int] = []
    normal_indices: list[int] = []

    for i, token in enumerate(tokens):
        if _PLACEHOLDER_PREFIX in token:
            placeholder_indices.append(i)
        else:
            normal_indices.append(i)

    # Extract normal words for editing
    normal_words = [tokens[i] for i in normal_indices]

    if normal_words:
        filler_words = ["indeed", "moreover", "clearly", "notably", "however", "thus"]

        # Choose a random edit operation
        op = rng.choice(["shuffle", "insert", "both"])

        if op in ("shuffle", "both"):
            rng.shuffle(normal_words)

        if op in ("insert", "both"):
            # Insert 1-2 random filler words at random positions
            for _ in range(rng.randint(1, 2)):
                pos = rng.randint(0, len(normal_words))
                normal_words.insert(pos, rng.choice(filler_words))

    # Rebuild: place placeholders at their original relative positions,
    # fill remaining slots with (possibly shuffled/extended) normal words
    result_tokens: list[str] = []
    normal_iter = iter(normal_words)
    placeholder_iter = iter(placeholder_indices)

    # We rebuild by iterating through original positions, but since
    # normal_words may have grown (inserts), we interleave differently:
    # Keep placeholders in order, fill everything else with normal_words.
    ph_set = set(placeholder_indices)
    normal_word_list = list(normal_words)
    nw_idx = 0

    for i in range(len(tokens)):
        if i in ph_set:
            result_tokens.append(tokens[i])
        else:
            if nw_idx < len(normal_word_list):
                result_tokens.append(normal_word_list[nw_idx])
                nw_idx += 1
            # else: skip (fewer normal slots than words — shouldn't happen with inserts)

    # Append any remaining inserted normal words at the end
    while nw_idx < len(normal_word_list):
        result_tokens.append(normal_word_list[nw_idx])
        nw_idx += 1

    return " ".join(result_tokens)


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------

# Feature: ultimate-humanizer, Property 2: Numeric and citation preservation
# Validates: Requirements 5.4, 14.4


@given(
    text=academic_text(include_numbers=True, include_citations=True),
    edit_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=100)
def test_numeric_and_citation_preservation(text: str, edit_seed: int) -> None:
    """Property 2: For all inputs, masking then unmasking around an arbitrary
    edit preserves every numeric value and citation marker with identical counts.

    Steps:
    1. Count numbers and citations in the original text
    2. Mask the text with ProtectedSpanGuard
    3. Apply an arbitrary edit to the masked text (shuffle/insert non-placeholder words)
    4. Unmask the edited text
    5. Count numbers and citations in the result
    6. Assert counts are identical
    """
    # Count before
    numbers_before = count_numbers(text)
    citations_before = count_citations(text)

    # Mask → edit → unmask
    guard = ProtectedSpanGuard()
    masked = guard.mask(text)
    edited = arbitrary_edit(masked, seed=edit_seed)
    restored = guard.unmask(edited)

    # Count after
    numbers_after = count_numbers(restored)
    citations_after = count_citations(restored)

    # Assertion: counts must be identical
    assert numbers_after == numbers_before, (
        f"Numeric count changed: before={numbers_before}, after={numbers_after}\n"
        f"Original: {text!r}\n"
        f"Restored: {restored!r}"
    )
    assert citations_after == citations_before, (
        f"Citation count changed: before={citations_before}, after={citations_after}\n"
        f"Original: {text!r}\n"
        f"Restored: {restored!r}"
    )


# Feature: ultimate-humanizer, Property 2: Numeric and citation preservation (ErrorInjector)
# Validates: Requirements 5.4


@given(
    text=academic_text(include_numbers=True, include_citations=True),
)
@settings(max_examples=100)
def test_error_injector_numeric_and_citation_preservation(text: str) -> None:
    """Property 2 (ErrorInjector): After error injection, every numeric value and
    citation marker is preserved with identical counts.

    The ErrorInjector uses ProtectedSpanGuard internally to mask numbers and
    citations before injection, so they must never be altered.

    Validates: Requirements 5.4
    """
    # Count before
    numbers_before = count_numbers(text)
    citations_before = count_citations(text)

    # Process through ErrorInjector
    injector = ErrorInjector(aggression=0.5, seed=42)
    output = injector.process(text)

    # Count after
    numbers_after = count_numbers(output)
    citations_after = count_citations(output)

    # Assertion: counts must be identical
    assert numbers_after == numbers_before, (
        f"Numeric count changed after ErrorInjector: before={numbers_before}, after={numbers_after}\n"
        f"Original: {text!r}\n"
        f"Output:   {output!r}"
    )
    assert citations_after == citations_before, (
        f"Citation count changed after ErrorInjector: before={citations_before}, after={citations_after}\n"
        f"Original: {text!r}\n"
        f"Output:   {output!r}"
    )


# ---------------------------------------------------------------------------
# Unit Tests: Guard Edge Cases (Task 2.3)
# Requirements: 5.4, 14.2, 14.4
# ---------------------------------------------------------------------------


class TestQuotedContentProtection:
    """Quoted content is preserved through the mask/unmask cycle."""

    def test_quoted_content_preserved(self) -> None:
        """Quoted content like "the central hypothesis" remains intact after mask/unmask."""
        text = 'The paper argues "the central hypothesis" is valid.'
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        restored = guard.unmask(masked)
        assert restored == text

    def test_multiple_quoted_spans_preserved(self) -> None:
        """Multiple quoted spans are all preserved."""
        text = 'He said "hello world" and she replied "goodbye world" today.'
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        # The quoted content should be replaced with placeholders in masked text
        assert '"hello world"' not in masked
        assert '"goodbye world"' not in masked
        restored = guard.unmask(masked)
        assert restored == text

    def test_quoted_content_with_protected_term_inside(self) -> None:
        """Quoted text containing a protected term: the quote takes priority (no double masking)."""
        text = 'The phrase "algorithm optimization" is commonly used.'
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        restored = guard.unmask(masked)
        assert restored == text


class TestOverlappingSpans:
    """Overlapping spans: numbers inside citations, quoted text with protected terms."""

    def test_number_inside_bracketed_citation(self) -> None:
        """A number inside a citation (e.g., [12]) — citation takes priority, number not double-masked."""
        text = "As shown in [12], the results are clear."
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        restored = guard.unmask(masked)
        assert restored == text
        assert "[12]" in restored

    def test_year_inside_parenthetical_citation(self) -> None:
        """The year (number) inside (Author, YEAR) — citation takes priority."""
        text = "Previous work (Smith, 2020) established the baseline."
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        restored = guard.unmask(masked)
        assert restored == text
        assert "(Smith, 2020)" in restored

    def test_quoted_text_containing_protected_term(self) -> None:
        """Quoted text containing a protected term — quote takes priority, both preserved."""
        text = 'They defined it as "a novel algorithm for convergence".'
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        restored = guard.unmask(masked)
        assert restored == text
        assert '"a novel algorithm for convergence"' in restored

    def test_multiple_numbers_in_bracket_citation_range(self) -> None:
        """A citation like [3-5] contains numbers — citation wins."""
        text = "See references [3-5] for details."
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        restored = guard.unmask(masked)
        assert restored == text
        assert "[3-5]" in restored


class TestCitationNumberAdjacency:
    """Citation/number adjacency: both preserved correctly when side by side."""

    def test_parenthetical_citation_adjacent_to_number(self) -> None:
        """(Smith, 2020) showed 3.14 — both the citation and the number preserved."""
        text = "(Smith, 2020) showed 3.14 was the threshold."
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        restored = guard.unmask(masked)
        assert restored == text
        assert "(Smith, 2020)" in restored
        assert "3.14" in restored

    def test_bracket_citation_adjacent_to_percentage(self) -> None:
        """[7] reports 42% accuracy — both preserved."""
        text = "[7] reports 42% accuracy in the study."
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        restored = guard.unmask(masked)
        assert restored == text
        assert "[7]" in restored
        assert "42%" in restored

    def test_multiple_citations_and_numbers_mixed(self) -> None:
        """Multiple citations and numbers interspersed — all preserved."""
        text = "(Johnson, 2019) found that [3] and [4] measured 2.71 and 6.28."
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        restored = guard.unmask(masked)
        assert restored == text


class TestEmptyInput:
    """Empty input: mask, unmask, and verify all handle gracefully."""

    def test_mask_empty_returns_empty(self) -> None:
        """mask("") returns ""."""
        guard = ProtectedSpanGuard()
        assert guard.mask("") == ""

    def test_unmask_empty_returns_empty(self) -> None:
        """unmask("") returns ""."""
        guard = ProtectedSpanGuard()
        guard.mask("some text")  # populate placeholders
        assert guard.unmask("") == ""

    def test_unmask_empty_no_prior_mask_returns_empty(self) -> None:
        """unmask("") with no prior mask returns ""."""
        guard = ProtectedSpanGuard()
        assert guard.unmask("") == ""

    def test_verify_empty_empty_returns_all_zeros(self) -> None:
        """verify("", "") returns all zero deltas."""
        result = ProtectedSpanGuard.verify("", "")
        assert result == {
            "protected_terms": 0,
            "numbers": 0,
            "citations": 0,
            "quotes": 0,
        }


class TestVerifyDetectsDeltas:
    """verify() correctly detects deltas when a protected item is lost or gained."""

    def test_verify_detects_lost_number(self) -> None:
        """When a number is removed from output, verify reports a negative delta."""
        original = "The value is 3.14 and 42."
        output = "The value is and ."
        result = ProtectedSpanGuard.verify(original, output)
        assert result["numbers"] < 0

    def test_verify_detects_lost_citation(self) -> None:
        """When a citation is removed from output, verify reports a negative delta."""
        original = "As shown by (Smith, 2020), the result holds."
        output = "As shown by the result holds."
        result = ProtectedSpanGuard.verify(original, output)
        assert result["citations"] < 0

    def test_verify_detects_lost_protected_term(self) -> None:
        """When a protected term is removed, verify reports a negative delta."""
        original = "The algorithm uses a heuristic approach."
        output = "The method uses a different approach."
        result = ProtectedSpanGuard.verify(original, output)
        assert result["protected_terms"] < 0

    def test_verify_no_delta_when_preserved(self) -> None:
        """When text is identical, all deltas are zero."""
        text = "The algorithm found 3.14 via (Smith, 2020)."
        result = ProtectedSpanGuard.verify(text, text)
        assert result == {
            "protected_terms": 0,
            "numbers": 0,
            "citations": 0,
            "quotes": 0,
        }


class TestCaseSensitiveProtection:
    """Protected terms are case-sensitive: only exact case matches are protected."""

    def test_exact_case_protected_term_is_masked(self) -> None:
        """A protected term in its exact case (e.g., 'algorithm') is masked."""
        text = "The algorithm converged quickly."
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        # The word 'algorithm' should be replaced by a placeholder
        assert "algorithm" not in masked
        assert _PLACEHOLDER_PREFIX in masked
        restored = guard.unmask(masked)
        assert restored == text

    def test_different_case_not_protected(self) -> None:
        """A term with different case (e.g., 'Algorithm' or 'ALGORITHM') is NOT masked."""
        text = "The Algorithm converged. ALGORITHM failed."
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        # 'Algorithm' and 'ALGORITHM' are not in PROTECTED_TERMS (which has 'algorithm')
        # so they should remain as-is in the masked text
        assert "Algorithm" in masked
        assert "ALGORITHM" in masked

    def test_mixed_case_in_sentence(self) -> None:
        """Only the exact-case term is protected; others remain unmasked."""
        text = "The algorithm outperformed Algorithm and ALGORITHM."
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        # Only lowercase 'algorithm' should be masked
        assert "Algorithm" in masked
        assert "ALGORITHM" in masked
        restored = guard.unmask(masked)
        assert restored == text
