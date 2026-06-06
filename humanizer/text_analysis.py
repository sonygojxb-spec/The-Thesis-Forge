"""
Text analysis utilities for measuring AI detection risk indicators.

Provides sentence splitting, readability scoring, perplexity estimation,
and metrics for vocabulary richness and AI fingerprint detection.
"""

import re
import math
import textstat

try:
    from wordfreq import word_frequency
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False


def split_sentences(text):
    """Split text into sentences using regex-based approach."""
    # Handle common abbreviations to avoid false splits
    text_clean = text.replace("e.g.", "e<DOT>g<DOT>")
    text_clean = text_clean.replace("i.e.", "i<DOT>e<DOT>")
    text_clean = text_clean.replace("et al.", "et al<DOT>")
    text_clean = text_clean.replace("Dr.", "Dr<DOT>")
    text_clean = text_clean.replace("Mr.", "Mr<DOT>")
    text_clean = text_clean.replace("Mrs.", "Mrs<DOT>")
    text_clean = text_clean.replace("vs.", "vs<DOT>")
    text_clean = text_clean.replace("Fig.", "Fig<DOT>")
    text_clean = text_clean.replace("Eq.", "Eq<DOT>")

    sentences = re.split(r'(?<=[.!?])\s+', text_clean)
    sentences = [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]
    return sentences


def compute_readability(text):
    """Compute Flesch-Kincaid grade level."""
    if not text.strip():
        return 0.0
    try:
        return round(textstat.flesch_kincaid_grade(text), 1)
    except Exception:
        return 0.0


def compute_type_token_ratio(text):
    """Compute vocabulary diversity as type-token ratio."""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return 0.0
    unique_words = set(words)
    return round(len(unique_words) / len(words), 4)


def compute_sentence_length_variance(text):
    """Compute variance in sentence lengths (word count)."""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    return round(variance, 2)


def compute_transition_word_density(text):
    """Compute density of AI-typical transition words."""
    from humanizer.config import AI_TRANSITION_WORDS

    text_lower = text.lower()
    word_count = len(text.split())
    if word_count == 0:
        return 0.0

    transition_count = 0
    for phrase in AI_TRANSITION_WORDS:
        transition_count += text_lower.count(phrase)

    return round(transition_count / word_count * 100, 3)


def estimate_perplexity_score(text):
    """
    Estimate text perplexity using word frequency data.
    Lower scores indicate more predictable (AI-like) text.
    Higher scores indicate more varied (human-like) text.
    """
    if not HAS_WORDFREQ:
        return 50.0  # neutral default

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return 50.0

    log_probs = []
    for word in words:
        freq = word_frequency(word, 'en')
        if freq > 0:
            log_probs.append(-math.log2(freq))
        else:
            log_probs.append(20.0)  # rare word

    if not log_probs:
        return 50.0

    avg_surprise = sum(log_probs) / len(log_probs)
    # Normalize to a 0-100 scale
    score = min(100, max(0, (avg_surprise - 5) * 10))
    return round(score, 1)


def compute_ai_risk_score(text):
    """
    Compute an estimated AI detection risk score (0-100).
    Higher = more likely to be flagged as AI.

    Factors:
    - Low sentence length variance = AI-like
    - High transition word density = AI-like
    - Low vocabulary diversity = AI-like
    - Uniform perplexity = AI-like
    """
    if not text.strip():
        return 0

    sentences = split_sentences(text)

    # Factor 1: Sentence length variance (low = AI)
    variance = compute_sentence_length_variance(text)
    # Human writing typically has variance > 50
    variance_score = max(0, 30 - min(30, variance * 0.3))

    # Factor 2: Transition word density (high = AI)
    transition_density = compute_transition_word_density(text)
    transition_score = min(25, transition_density * 50)

    # Factor 3: Vocabulary diversity (low TTR = AI)
    ttr = compute_type_token_ratio(text)
    # Human text: TTR > 0.5, AI text: TTR around 0.3-0.4
    diversity_score = max(0, 25 - min(25, (ttr - 0.3) * 100))

    # Factor 4: Sentence perplexity uniformity
    if len(sentences) >= 3:
        perplexities = [estimate_perplexity_score(s) for s in sentences]
        if perplexities:
            mean_p = sum(perplexities) / len(perplexities)
            p_variance = sum((p - mean_p) ** 2 for p in perplexities) / len(perplexities)
            # Low variance in perplexity = AI
            uniformity_score = max(0, 20 - min(20, p_variance * 0.1))
        else:
            uniformity_score = 10
    else:
        uniformity_score = 10

    total_risk = variance_score + transition_score + diversity_score + uniformity_score
    return min(100, max(0, round(total_risk)))


def get_text_analytics(text):
    """Get comprehensive text analytics for display."""
    if not text.strip():
        return {
            "sentences": 0,
            "avg_len": 0,
            "grade": "N/A",
            "vocabulary_diversity": 0,
            "ai_risk": 0,
        }

    sentences = split_sentences(text)
    word_counts = [len(s.split()) for s in sentences]
    avg_len = round(sum(word_counts) / len(word_counts), 1) if word_counts else 0

    return {
        "sentences": len(sentences),
        "avg_len": avg_len,
        "grade": compute_readability(text),
        "vocabulary_diversity": compute_type_token_ratio(text),
        "ai_risk": compute_ai_risk_score(text),
    }
