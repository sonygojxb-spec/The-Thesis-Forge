"""
Protected Span Guard for the humanization pipeline.

Provides mask/unmask operations to protect specific text spans (academic terms,
numbers, citations, quoted content) from being altered during transformations,
and a verify function to check that protected content survived the pipeline.
"""

import re
import uuid
from typing import Dict, List, Tuple

from humanizer.config import PROTECTED_TERMS


# Unique prefix/suffix for placeholders that won't appear in normal text
_PLACEHOLDER_PREFIX = "\x00PSG_"
_PLACEHOLDER_SUFFIX = "_GSP\x00"


class ProtectedSpanGuard:
    """
    Guards protected spans (terms, numbers, citations, quotes) by masking them
    with unique placeholder tokens before a transformation and restoring them after.

    Usage:
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)
        # ... apply transformation to masked ...
        restored = guard.unmask(masked)

    The verify() classmethod checks whether protected content was preserved
    across a full pipeline pass (comparing original input to final output).
    """

    def __init__(self):
        self._placeholders: List[Tuple[str, str, str]] = []
        # Each entry: (placeholder, original_text, category)

    def mask(self, text: str) -> str:
        """
        Replace protected spans with unique placeholder tokens.

        Protected span categories (in priority order for overlapping spans):
        1. Quoted content (text within double quotes)
        2. Citation markers: (Author, YEAR) and [n]
        3. Numeric values (integers, decimals, percentages)
        4. Protected terms (whole-word, case-sensitive)

        Returns the masked text with placeholders substituted.
        """
        self._placeholders = []

        if not text:
            return text

        # Phase 1: Collect all candidate spans from the ORIGINAL text,
        # resolving overlaps by priority (earlier category wins).
        spans: List[Tuple[int, int, str, str]] = []
        # Each entry: (start, end, original_text, category)

        # 1. Quoted content
        self._collect_spans(text, r'"[^"]*"', "quotes", spans)

        # 2. Citation markers
        # Parenthetical: (Author, YEAR) or (Author et al., YEAR)
        self._collect_spans(
            text,
            r'\([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?,\s*\d{4}\)',
            "citations", spans
        )
        # Bracketed: [n] or [n, m] or [n-m]
        self._collect_spans(
            text,
            r'\[\d+(?:\s*[-,]\s*\d+)*\]',
            "citations", spans
        )

        # 3. Numeric values
        self._collect_spans(text, r'-?\b\d+(?:\.\d+)?%?', "numbers", spans)

        # 4. Protected terms (whole-word, case-sensitive)
        for term in sorted(PROTECTED_TERMS, key=len, reverse=True):
            pattern = r'\b' + re.escape(term) + r'\b'
            self._collect_spans(text, pattern, "protected_terms", spans)

        # Phase 2: Replace spans from end to start (preserves offsets).
        # Sort by start position descending for safe replacement.
        spans.sort(key=lambda s: s[0], reverse=True)

        for start, end, original, category in spans:
            placeholder = self._make_placeholder(original, category)
            text = text[:start] + placeholder + text[end:]

        return text

    def _collect_spans(
        self, text: str, pattern: str, category: str,
        spans: List[Tuple[int, int, str, str]]
    ) -> None:
        """
        Find all matches of pattern in text and add non-overlapping spans
        to the spans list. Skips any match that overlaps with an already-claimed span.
        """
        # Build a set of already-claimed positions from existing spans
        claimed = set()
        for start, end, _, _ in spans:
            claimed.update(range(start, end))

        for m in re.finditer(pattern, text):
            span_positions = set(range(m.start(), m.end()))
            if not span_positions & claimed:
                spans.append((m.start(), m.end(), m.group(), category))
                claimed.update(span_positions)

    def _make_placeholder(self, original: str, category: str) -> str:
        """Generate a unique placeholder token for the given original text."""
        uid = uuid.uuid4().hex[:12]
        placeholder = f"{_PLACEHOLDER_PREFIX}{category}_{uid}{_PLACEHOLDER_SUFFIX}"
        self._placeholders.append((placeholder, original, category))
        return placeholder

    def unmask(self, text: str) -> str:
        """
        Restore all placeholder tokens back to their original text.

        Processes in reverse order to handle any nested cases correctly.
        """
        if not text or not self._placeholders:
            return text

        # Restore in reverse order (last masked = first restored)
        for placeholder, original, _category in reversed(self._placeholders):
            text = text.replace(placeholder, original)

        return text

    @staticmethod
    def verify(original: str, output: str) -> Dict[str, int]:
        """
        Compare original and output texts to check that protected content
        was preserved through the pipeline.

        Returns a dict with per-category occurrence-count deltas:
            {"protected_terms": 0, "numbers": -1, "citations": 0, "quotes": 0}

        A delta of 0 means the count is unchanged. Negative values mean
        a protected item was lost. Positive values mean one was gained
        (rare but possible with some transformations).
        """
        if not original and not output:
            return {
                "protected_terms": 0,
                "numbers": 0,
                "citations": 0,
                "quotes": 0,
            }

        original_counts = ProtectedSpanGuard._count_protected(original or "")
        output_counts = ProtectedSpanGuard._count_protected(output or "")

        return {
            category: output_counts[category] - original_counts[category]
            for category in original_counts
        }

    @staticmethod
    def _count_protected(text: str) -> Dict[str, int]:
        """Count occurrences of each protected span category in text."""
        counts = {
            "protected_terms": 0,
            "numbers": 0,
            "citations": 0,
            "quotes": 0,
        }

        # Quotes
        counts["quotes"] = len(re.findall(r'"[^"]*"', text))

        # Citations: parenthetical
        paren_citations = len(re.findall(
            r'\([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?,\s*\d{4}\)', text
        ))
        # Citations: bracketed
        bracket_citations = len(re.findall(
            r'\[\d+(?:\s*[-,]\s*\d+)*\]', text
        ))
        counts["citations"] = paren_citations + bracket_citations

        # Numbers (standalone numeric values)
        counts["numbers"] = len(re.findall(r'-?\b\d+(?:\.\d+)?%?', text))

        # Protected terms (whole-word, case-sensitive)
        for term in PROTECTED_TERMS:
            pattern = r'\b' + re.escape(term) + r'\b'
            counts["protected_terms"] += len(re.findall(pattern, text))

        return counts
