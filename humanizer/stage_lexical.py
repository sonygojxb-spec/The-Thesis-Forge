"""
Stage 2: Controlled Vocabulary Injection

Replaces high-frequency/high-probability words with less predictable
synonyms to disrupt token probability distributions that AI detectors flag.

Uses WordNet for synonym lookup and wordfreq for frequency data to prefer
lower-frequency replacements. Preserves domain-specific terminology.
"""

import re
import random

try:
    from wordfreq import word_frequency
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False

try:
    import nltk
    from nltk.corpus import wordnet
    try:
        wordnet.synsets('test')
        HAS_WORDNET = True
    except LookupError:
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            wordnet.synsets('test')
            HAS_WORDNET = True
        except Exception:
            HAS_WORDNET = False
except ImportError:
    HAS_WORDNET = False

from humanizer.config import PROTECTED_TERMS


class LexicalInjection:
    """Replaces predictable words with less common synonyms."""

    def __init__(self, aggression=0.5, seed=None):
        """
        Args:
            aggression: Float 0-1 controlling replacement aggressiveness.
            seed: Optional int seed for reproducible results.
        """
        self.aggression = aggression
        self.seed = seed
        # Common academic words that are frequently flagged
        self.target_words = {
            "utilize": ["employ", "use", "apply", "leverage"],
            "demonstrate": ["show", "reveal", "illustrate", "indicate"],
            "significant": ["notable", "substantial", "considerable", "marked"],
            "implement": ["apply", "execute", "carry out", "put into practice"],
            "facilitate": ["enable", "support", "help", "assist"],
            "establish": ["set up", "create", "build", "form"],
            "comprehensive": ["thorough", "extensive", "complete", "broad"],
            "fundamental": ["basic", "core", "essential", "primary"],
            "particularly": ["especially", "notably", "specifically", "in particular"],
            "effectively": ["well", "successfully", "capably", "competently"],
            "subsequently": ["then", "afterward", "later", "next"],
            "primarily": ["mainly", "chiefly", "mostly", "largely"],
            "approximately": ["about", "roughly", "around", "close to"],
            "indicate": ["suggest", "show", "point to", "signal"],
            "numerous": ["many", "several", "various", "a range of"],
            "sufficient": ["enough", "adequate", "ample", "satisfactory"],
            "enhance": ["improve", "boost", "strengthen", "elevate"],
            "crucial": ["key", "vital", "critical", "important"],
            "inherent": ["built-in", "intrinsic", "natural", "innate"],
            "optimal": ["best", "ideal", "most effective", "top"],
        }

    def process(self, text):
        """Apply lexical injection to the input text."""
        if not text.strip():
            return text

        if self.seed is not None:
            random.seed(self.seed)

        # Direct replacement of known AI-favorite words
        text = self._replace_target_words(text)

        # WordNet-based synonym replacement for adjectives/adverbs
        if HAS_WORDNET and HAS_WORDFREQ:
            text = self._wordnet_replace(text)

        return text

    def _replace_target_words(self, text):
        """Replace known AI-preferred vocabulary with alternatives."""
        # Collect all replacements as (start, end, replacement) tuples
        replacements = []
        for word, candidates in self.target_words.items():
            if random.random() > self.aggression:
                continue
            pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                if random.random() < self.aggression * 0.7:
                    replacement = random.choice(candidates)
                    # Preserve capitalization
                    if match.group()[0].isupper():
                        replacement = replacement[0].upper() + replacement[1:]
                    replacements.append((match.start(), match.end(), replacement))

        # Apply replacements in reverse order to maintain position validity
        replacements.sort(key=lambda r: r[0], reverse=True)
        for start, end, replacement in replacements:
            text = text[:start] + replacement + text[end:]

        return text

    def _wordnet_replace(self, text):
        """Replace high-frequency adjectives/adverbs with lower-frequency synonyms."""
        words = text.split()
        replacement_count = 0
        max_replacements = int(len(words) * self.aggression * 0.08)

        result_words = []
        for word in words:
            clean_word = re.sub(r'[^a-zA-Z]', '', word)

            if (clean_word.lower() in PROTECTED_TERMS or
                    len(clean_word) < 4 or
                    replacement_count >= max_replacements):
                result_words.append(word)
                continue

            # Check if word is high-frequency (predictable)
            freq = word_frequency(clean_word.lower(), 'en')
            if freq < 1e-5:  # Skip rare words
                result_words.append(word)
                continue

            # Only target words that are somewhat common
            if freq > 5e-5 and random.random() < self.aggression * 0.3:
                synonym = self._get_lower_freq_synonym(clean_word)
                if synonym and synonym.lower() != clean_word.lower():
                    # Preserve punctuation around the word
                    prefix = ''
                    suffix = ''
                    for ch in word:
                        if ch.isalpha():
                            break
                        prefix += ch
                    for ch in reversed(word):
                        if ch.isalpha():
                            break
                        suffix = ch + suffix

                    # Preserve case
                    if clean_word[0].isupper():
                        synonym = synonym[0].upper() + synonym[1:]

                    result_words.append(f"{prefix}{synonym}{suffix}")
                    replacement_count += 1
                else:
                    result_words.append(word)
            else:
                result_words.append(word)

        return ' '.join(result_words)

    def _get_lower_freq_synonym(self, word):
        """Find a synonym with lower frequency than the original."""
        synsets = wordnet.synsets(word.lower())
        if not synsets:
            return None

        candidates = []
        original_freq = word_frequency(word.lower(), 'en')

        for syn in synsets[:3]:  # Limit to first 3 synsets for relevance
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() == word.lower():
                    continue
                if name.lower() in PROTECTED_TERMS:
                    continue
                if len(name.split()) > 2:  # Skip multi-word phrases
                    continue

                freq = word_frequency(name.lower(), 'en')
                # Prefer words that are less frequent but still known
                if 1e-7 < freq < original_freq * 0.8:
                    candidates.append((name, freq))

        if not candidates:
            return None

        # Sort by frequency (prefer moderately rare words)
        candidates.sort(key=lambda x: x[1], reverse=True)
        # Pick from top candidates with some randomness
        top = candidates[:min(3, len(candidates))]
        return random.choice(top)[0]
