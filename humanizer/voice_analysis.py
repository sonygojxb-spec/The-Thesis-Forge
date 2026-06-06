"""
Voice Analysis Module

Provides detailed voice profiling metrics beyond basic text analysis,
including hedge/booster counts, pronoun usage, engagement markers,
enhanced type-token ratio, and idiosyncrasy detection.
"""

import re

from humanizer.config import VOICE_HEDGES, VOICE_BOOSTERS, VOICE_ENGAGEMENT_MARKERS
from humanizer.text_analysis import compute_type_token_ratio, split_sentences


def measure_hedges(text):
    """
    Count hedging words/phrases in text.

    Hedging indicates uncertainty and is characteristic of human academic writing.

    Returns:
        int: Count of hedging expressions found.
    """
    text_lower = text.lower()
    count = 0
    for hedge in VOICE_HEDGES:
        count += len(re.findall(r'\b' + re.escape(hedge) + r'\b', text_lower))
    return count


def measure_boosters(text):
    """
    Count certainty/booster words in text.

    High booster density can signal AI-generated text.

    Returns:
        int: Count of booster expressions found.
    """
    text_lower = text.lower()
    count = 0
    for booster in VOICE_BOOSTERS:
        count += len(re.findall(r'\b' + re.escape(booster) + r'\b', text_lower))
    return count


def measure_pronouns(text):
    """
    Count first-person and inclusive pronouns.

    Human academic writing uses personal pronouns more than AI text.

    Returns:
        int: Count of first-person/inclusive pronouns found.
    """
    pronouns = ["i", "we", "our", "us", "my"]
    text_lower = text.lower()
    count = 0
    for pronoun in pronouns:
        count += len(re.findall(r'\b' + re.escape(pronoun) + r'\b', text_lower))
    return count


def measure_engagement(text):
    """
    Count engagement/counter-voice markers in text.

    These markers indicate the writer is engaging with opposing viewpoints,
    which is characteristic of authentic academic discourse.

    Returns:
        int: Count of engagement markers found.
    """
    text_lower = text.lower()
    count = 0
    for marker in VOICE_ENGAGEMENT_MARKERS:
        count += len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
    return count


def compute_enhanced_ttr(text):
    """
    Compute enhanced type-token ratio with hapax legomena ratio
    and moving-average TTR.

    Returns:
        dict: Contains 'base_ttr', 'hapax_ratio', 'moving_avg_ttr'.
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return {"base_ttr": 0.0, "hapax_ratio": 0.0, "moving_avg_ttr": 0.0}

    base_ttr = compute_type_token_ratio(text)

    # Hapax legomena ratio: words appearing exactly once / total words
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    hapax_count = sum(1 for count in word_counts.values() if count == 1)
    hapax_ratio = round(hapax_count / len(words), 4) if words else 0.0

    # Moving-average TTR: compute TTR over windows of 50 words
    window_size = 50
    if len(words) < window_size:
        moving_avg_ttr = base_ttr
    else:
        window_ttrs = []
        for i in range(0, len(words) - window_size + 1, window_size):
            window = words[i:i + window_size]
            unique = set(window)
            window_ttrs.append(len(unique) / len(window))
        moving_avg_ttr = round(sum(window_ttrs) / len(window_ttrs), 4) if window_ttrs else base_ttr

    return {
        "base_ttr": base_ttr,
        "hapax_ratio": hapax_ratio,
        "moving_avg_ttr": moving_avg_ttr,
    }


def detect_idiosyncrasies(text):
    """
    Detect unusual writing patterns that signal human authorship.

    Looks for parenthetical asides, em-dashes, sentence fragments, and questions.

    Returns:
        dict: Counts of each idiosyncrasy type.
    """
    sentences = split_sentences(text)

    # Parenthetical asides: text within parentheses
    parentheticals = len(re.findall(r'\([^)]+\)', text))

    # Em-dashes (various forms)
    em_dashes = len(re.findall(r'--|\u2014|\u2013', text))

    # Sentence fragments: sentences with fewer than 5 words and no main verb indicator
    fragments = 0
    for sentence in sentences:
        words = sentence.split()
        if len(words) < 5 and not sentence.endswith('?'):
            fragments += 1

    # Questions
    questions = sum(1 for s in sentences if s.strip().endswith('?'))

    return {
        "parentheticals": parentheticals,
        "em_dashes": em_dashes,
        "fragments": fragments,
        "questions": questions,
    }


def get_voice_profile(text):
    """
    Get a comprehensive voice profile for the given text.

    Returns:
        dict: Aggregated voice metrics with keys: hedges, boosters,
              pronouns, engagement, ttr, enhanced_ttr, idiosyncrasies.
    """
    if not text or not text.strip():
        return {
            "hedges": 0,
            "boosters": 0,
            "pronouns": 0,
            "engagement": 0,
            "ttr": 0.0,
            "enhanced_ttr": {"base_ttr": 0.0, "hapax_ratio": 0.0, "moving_avg_ttr": 0.0},
            "idiosyncrasies": {"parentheticals": 0, "em_dashes": 0, "fragments": 0, "questions": 0},
        }

    return {
        "hedges": measure_hedges(text),
        "boosters": measure_boosters(text),
        "pronouns": measure_pronouns(text),
        "engagement": measure_engagement(text),
        "ttr": compute_type_token_ratio(text),
        "enhanced_ttr": compute_enhanced_ttr(text),
        "idiosyncrasies": detect_idiosyncrasies(text),
    }
