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
        self.rng = random.Random(seed) if seed is not None else random.Random()
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
            "achieve": ["attain", "reach", "accomplish", "secure"],
            "approach": ["method", "technique", "way", "strategy"],
            "impact": ["effect", "influence", "bearing", "consequence"],
            "framework": ["structure", "model", "scheme", "arrangement"],
            "landscape": ["field", "domain", "area", "sphere"],
            "paradigm": ["model", "framework", "pattern", "standard"],
            "robust": ["strong", "solid", "sturdy", "resilient"],
            "leveraging": ["using", "employing", "drawing on", "making use of"],
            "innovative": ["novel", "original", "fresh", "inventive"],
            "streamline": ["simplify", "improve", "refine", "tighten"],
            "pivotal": ["key", "central", "important", "decisive"],
            "foster": ["encourage", "promote", "cultivate", "nurture"],
            "underscore": ["highlight", "stress", "emphasize", "point up"],
            "delve": ["examine", "explore", "look into", "investigate"],
            "realm": ["area", "field", "domain", "sphere"],
            "plethora": ["many", "abundance", "wealth", "range"],
            "myriad": ["many", "numerous", "a range of", "countless"],
            "catalyst": ["trigger", "driver", "stimulus", "spark"],
            "holistic": ["comprehensive", "complete", "all-round", "integrated"],
            "synergy": ["cooperation", "collaboration", "combined effect", "partnership"],
            "ecosystem": ["environment", "system", "network", "setting"],
            "transformative": ["significant", "major", "far-reaching", "game-changing"],
            "spearhead": ["lead", "drive", "champion", "head"],
            "groundbreaking": ["pioneering", "original", "innovative", "path-breaking"],
            "cutting-edge": ["advanced", "latest", "modern", "state-of-the-art"],
            "moreover": ["also", "and", "besides", "in addition"],
            "furthermore": ["also", "and", "what is more", "besides"],
        }

    def process(self, text):
        """Apply lexical injection to the input text."""
        if not text.strip():
            return text

        # Direct replacement of known AI-favorite words
        text = self._replace_target_words(text)

        # WordNet-based synonym replacement for adjectives/adverbs
        if HAS_WORDNET and HAS_WORDFREQ:
            text = self._wordnet_replace(text)

        # Convert American spellings to British/Indian English
        text = self._apply_british_spellings(text)

        return text

    def _replace_target_words(self, text):
        """Replace known AI-preferred vocabulary with alternatives."""
        # Collect all replacements as (start, end, replacement) tuples
        replacements = []
        for word, candidates in self.target_words.items():
            pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                if self.rng.random() < self.aggression:
                    replacement = self.rng.choice(candidates)
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
        max_replacements = int(len(words) * self.aggression * 0.15)

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
            if freq > 5e-5 and self.rng.random() < self.aggression * 0.5:
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
        return self.rng.choice(top)[0]

    def _apply_british_spellings(self, text):
        """Convert American English spellings to British/Indian English equivalents."""
        # Specific word replacements
        specific_replacements = {
            "color": "colour", "Color": "Colour",
            "behavior": "behaviour", "Behavior": "Behaviour",
            "favor": "favour", "Favor": "Favour",
            "honor": "honour", "Honor": "Honour",
            "labor": "labour", "Labor": "Labour",
            "neighbor": "neighbour", "Neighbor": "Neighbour",
            "center": "centre", "Center": "Centre",
            "fiber": "fibre", "Fiber": "Fibre",
            "meter": "metre", "Meter": "Metre",
            "defense": "defence", "Defense": "Defence",
            "license": "licence", "License": "Licence",
            "offense": "offence", "Offense": "Offence",
        }

        for american, british in specific_replacements.items():
            pattern = re.compile(r'\b' + re.escape(american) + r'\b')
            text = pattern.sub(british, text)

        # Pattern-based replacements: -ization -> -isation
        text = re.sub(r'\b(\w+)ization\b', r'\1isation', text)
        text = re.sub(r'\b(\w+)izations\b', r'\1isations', text)

        # Handle -yze -> -yse (analyze -> analyse, paralyze -> paralyse)
        text = re.sub(r'\b(\w+)yze\b', r'\1yse', text)
        text = re.sub(r'\b(\w+)yzed\b', r'\1ysed', text)
        text = re.sub(r'\b(\w+)yzes\b', r'\1yses', text)
        text = re.sub(r'\b(\w+)yzing\b', r'\1ysing', text)

        # Handle -ize -> -ise (but not words where -ize is part of the root)
        exceptions = {'size', 'sized', 'sizes', 'sizing',
                      'prize', 'prized', 'prizes', 'prizing',
                      'seize', 'seized', 'seizes', 'seizing',
                      'capsize', 'capsized', 'capsizes', 'capsizing',
                      'frozen', 'horizon', 'horizons',
                      'citizen', 'citizens', 'citizenship'}

        def ize_replacer(match):
            word = match.group(0)
            # Skip proper nouns (capitalized words)
            if word[0].isupper():
                return word
            if word.lower() in exceptions:
                return word
            if word.lower().rstrip('dse') in PROTECTED_TERMS or word.lower() in PROTECTED_TERMS:
                return word
            return word[:-3] + 'ise'

        def ized_replacer(match):
            word = match.group(0)
            if word[0].isupper():
                return word
            if word.lower() in exceptions:
                return word
            if word.lower().rstrip('d') in PROTECTED_TERMS or word.lower() in PROTECTED_TERMS:
                return word
            return word[:-4] + 'ised'

        def izes_replacer(match):
            word = match.group(0)
            if word[0].isupper():
                return word
            if word.lower() in exceptions:
                return word
            if word.lower().rstrip('s') in PROTECTED_TERMS or word.lower() in PROTECTED_TERMS:
                return word
            return word[:-4] + 'ises'

        def izing_replacer(match):
            word = match.group(0)
            if word[0].isupper():
                return word
            if word.lower() in exceptions:
                return word
            base = word.lower()[:-4] + 'e'
            if base in PROTECTED_TERMS or word.lower() in PROTECTED_TERMS:
                return word
            return word[:-5] + 'ising'

        text = re.sub(r'\b\w+ize\b', ize_replacer, text)
        text = re.sub(r'\b\w+ized\b', ized_replacer, text)
        text = re.sub(r'\b\w+izes\b', izes_replacer, text)
        text = re.sub(r'\b\w+izing\b', izing_replacer, text)

        return text
