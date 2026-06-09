"""
Reusable Hypothesis strategies for property-based testing.

Provides composite strategies that generate realistic academic text inputs
to meaningfully exercise invariance properties across the Ultimate Humanizer
pipeline stages.

Strategies:
- academic_text: text containing PROTECTED_TERMS, numbers, citations, quotes
- multi_sentence_text: text guaranteed to have >=2 or >=3 sentences
- aggression: floats in [0.0, 1.0]

Requirements: 1.4, 2.3, 5.3, 5.4 (strategies that meaningfully exercise
invariance properties — protected terms, numeric values, citations, quotes)
"""

from __future__ import annotations

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from humanizer.config import PROTECTED_TERMS

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

# Convert to a sorted list for deterministic sampling
_PROTECTED_TERMS_LIST = sorted(PROTECTED_TERMS)

# Sample filler words/phrases that form natural-sounding academic sentence parts
_FILLER_PHRASES = [
    "the study demonstrates",
    "results indicate that",
    "we observe a trend in",
    "this approach relies on",
    "the data suggest",
    "as shown in the analysis",
    "the framework addresses",
    "a key observation is",
    "the evidence supports",
    "our findings confirm",
    "it is apparent from the data",
    "the model accounts for",
    "these results align with",
    "the proposed method uses",
    "based on prior work",
]

# Numeric values that must be preserved through transformations
_NUMERIC_VALUES = [
    "3.14",
    "42%",
    "0.05",
    "99.7%",
    "2.718",
    "1,000",
    "p < 0.001",
    "n = 150",
    "95%",
    "10.5",
    "3.0",
    "7.2%",
    "0.95",
    "100",
    "12.8",
]

# Citation markers in parenthetical and bracket styles
_CITATIONS_PARENTHETICAL = [
    "(Smith, 2020)",
    "(Jones & Lee, 2019)",
    "(Garcia et al., 2021)",
    "(Brown, 2018)",
    "(Williams & Chen, 2022)",
    "(Davis, 2017)",
    "(Martinez et al., 2023)",
    "(Anderson & Taylor, 2020)",
]

_CITATIONS_BRACKET = [
    "[1]",
    "[2]",
    "[3]",
    "[12]",
    "[15]",
    "[7]",
    "[23]",
    "[42]",
]

# Quoted spans that must be preserved verbatim
_QUOTED_SPANS = [
    '"the central hypothesis"',
    '"statistical significance"',
    '"paradigm shift in methodology"',
    '"a marked improvement"',
    '"robust and scalable"',
    '"no significant difference"',
]


# ---------------------------------------------------------------------------
# Strategy: academic_text
# ---------------------------------------------------------------------------


@st.composite
def academic_text(
    draw: st.DrawFn,
    *,
    min_protected_terms: int = 1,
    max_protected_terms: int = 3,
    include_numbers: bool = True,
    include_citations: bool = True,
    include_quotes: bool = True,
) -> str:
    """Generate realistic academic text containing protected elements.

    The generated text includes:
    - At least `min_protected_terms` PROTECTED_TERMS (whole words)
    - Numeric values like '3.14' or '42%'
    - Citation markers in parenthetical '(Smith, 2020)' or bracket '[12]' form
    - Quoted spans like '"the central hypothesis"'

    These elements exercise the invariance properties specified in Requirements
    1.4 (protected terms), 2.3 (protected terms), 5.3 (protected terms), and
    5.4 (numeric/citation/quote preservation).
    """
    # Select protected terms to embed
    num_terms = draw(
        st.integers(min_value=min_protected_terms, max_value=max_protected_terms)
    )
    terms = draw(
        st.lists(
            st.sampled_from(_PROTECTED_TERMS_LIST),
            min_size=num_terms,
            max_size=num_terms,
        )
    )

    # Build sentence fragments around the protected terms
    fragments: list[str] = []

    for term in terms:
        filler = draw(st.sampled_from(_FILLER_PHRASES))
        fragments.append(f"{filler} the {term}")

    # Optionally inject numeric values
    if include_numbers:
        num_count = draw(st.integers(min_value=1, max_value=2))
        for _ in range(num_count):
            num_val = draw(st.sampled_from(_NUMERIC_VALUES))
            filler = draw(st.sampled_from(_FILLER_PHRASES))
            fragments.append(f"{filler} with a value of {num_val}")

    # Optionally inject citations
    if include_citations:
        citation_count = draw(st.integers(min_value=1, max_value=2))
        for _ in range(citation_count):
            use_bracket = draw(st.booleans())
            if use_bracket:
                cite = draw(st.sampled_from(_CITATIONS_BRACKET))
            else:
                cite = draw(st.sampled_from(_CITATIONS_PARENTHETICAL))
            filler = draw(st.sampled_from(_FILLER_PHRASES))
            fragments.append(f"{filler} {cite}")

    # Optionally inject quoted spans
    if include_quotes:
        quote_count = draw(st.integers(min_value=0, max_value=1))
        for _ in range(quote_count):
            quote = draw(st.sampled_from(_QUOTED_SPANS))
            filler = draw(st.sampled_from(_FILLER_PHRASES))
            fragments.append(f"{filler} described as {quote}")

    # Join fragments into sentences (order is already varied by draws)
    sentences = [f.capitalize() + "." for f in fragments]

    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Strategy: multi_sentence_text
# ---------------------------------------------------------------------------


@st.composite
def multi_sentence_text(
    draw: st.DrawFn,
    *,
    min_sentences: int = 2,
    max_sentences: int = 6,
) -> str:
    """Generate text with a guaranteed minimum number of sentences.

    Each sentence is a distinct, well-formed academic-style statement ending
    with a period. Useful for testing stages that require multi-sentence input
    (e.g., StylometricObfuscator requires >=2 sentences, PerplexityOptimizer
    benefits from >=3 for variance measurement).

    Parameters:
        min_sentences: minimum sentence count (default 2)
        max_sentences: maximum sentence count (default 6)
    """
    num_sentences = draw(
        st.integers(min_value=min_sentences, max_value=max_sentences)
    )

    sentences: list[str] = []
    for _ in range(num_sentences):
        # Each sentence gets a filler + optional protected term
        filler = draw(st.sampled_from(_FILLER_PHRASES))
        include_term = draw(st.booleans())
        if include_term:
            term = draw(st.sampled_from(_PROTECTED_TERMS_LIST))
            sentence = f"{filler} the {term}".capitalize() + "."
        else:
            sentence = filler.capitalize() + "."
        sentences.append(sentence)

    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Strategy: aggression
# ---------------------------------------------------------------------------


def aggression() -> SearchStrategy[float]:
    """Generate aggression values uniformly over [0.0, 1.0].

    The aggression parameter controls how strongly a stage transforms text.
    All stages accept aggression in [0.0, 1.0] inclusive.
    """
    return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Convenience: combined academic + multi-sentence strategy
# ---------------------------------------------------------------------------


@st.composite
def academic_multi_sentence_text(
    draw: st.DrawFn,
    *,
    min_sentences: int = 3,
    max_sentences: int = 6,
    min_protected_terms: int = 1,
    max_protected_terms: int = 3,
    include_numbers: bool = True,
    include_citations: bool = True,
    include_quotes: bool = True,
) -> str:
    """Generate multi-sentence academic text with all protected element types.

    Combines the guarantees of both academic_text and multi_sentence_text:
    - At least min_sentences distinct sentences
    - PROTECTED_TERMS, numbers, citations, and quotes embedded throughout

    Ideal for exercising both sentence-count-dependent logic and invariance
    properties simultaneously.
    """
    # Ensure we have enough sentences for protected elements
    num_sentences = draw(
        st.integers(min_value=min_sentences, max_value=max_sentences)
    )

    # Select protected terms
    num_terms = draw(
        st.integers(min_value=min_protected_terms, max_value=max_protected_terms)
    )
    terms = draw(
        st.lists(
            st.sampled_from(_PROTECTED_TERMS_LIST),
            min_size=num_terms,
            max_size=num_terms,
        )
    )

    sentences: list[str] = []

    # First: sentences with protected terms
    for term in terms:
        filler = draw(st.sampled_from(_FILLER_PHRASES))
        sentences.append(f"{filler} the {term}".capitalize() + ".")

    # Add numeric value sentence
    if include_numbers:
        num_val = draw(st.sampled_from(_NUMERIC_VALUES))
        filler = draw(st.sampled_from(_FILLER_PHRASES))
        sentences.append(f"{filler} with a value of {num_val}".capitalize() + ".")

    # Add citation sentence
    if include_citations:
        use_bracket = draw(st.booleans())
        if use_bracket:
            cite = draw(st.sampled_from(_CITATIONS_BRACKET))
        else:
            cite = draw(st.sampled_from(_CITATIONS_PARENTHETICAL))
        filler = draw(st.sampled_from(_FILLER_PHRASES))
        sentences.append(f"{filler} {cite}".capitalize() + ".")

    # Add quoted span sentence
    if include_quotes:
        quote = draw(st.sampled_from(_QUOTED_SPANS))
        filler = draw(st.sampled_from(_FILLER_PHRASES))
        sentences.append(f"{filler} described as {quote}".capitalize() + ".")

    # Pad to reach minimum sentence count
    while len(sentences) < num_sentences:
        filler = draw(st.sampled_from(_FILLER_PHRASES))
        sentences.append(filler.capitalize() + ".")

    return " ".join(sentences)
