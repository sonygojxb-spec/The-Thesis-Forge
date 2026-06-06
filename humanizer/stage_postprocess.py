"""
Stage 5: Final NLP Post-processing

Applies finishing touches to remove remaining AI fingerprints:
- Removes repeated sentence starters
- Eliminates AI transition words (Moreover, Furthermore, etc.)
- Injects natural imperfections (varied comma usage, paragraph variation)
- Ensures no uniform punctuation patterns
- Final readability check and adjustment
"""

import re
import random

from humanizer.config import AI_TRANSITION_WORDS, TRANSITION_REPLACEMENTS


class PostProcessor:
    """Final cleanup to remove AI fingerprints and add natural imperfections."""

    def __init__(self, aggression=0.5, seed=None):
        """
        Args:
            aggression: Float 0-1 controlling post-processing intensity.
            seed: Optional int seed for reproducible results.
        """
        self.aggression = aggression
        self.seed = seed

    def process(self, text):
        """Apply all post-processing steps."""
        if not text.strip():
            return text

        if self.seed is not None:
            random.seed(self.seed)

        text = self._remove_ai_transitions(text)
        text = self._fix_repeated_starters(text)
        text = self._vary_punctuation(text)
        text = self._inject_natural_imperfections(text)
        text = self._final_cleanup(text)

        return text

    def _remove_ai_transitions(self, text):
        """Remove or replace AI-typical transition words and phrases."""
        for phrase in AI_TRANSITION_WORDS:
            pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
            matches = list(pattern.finditer(text))

            for match in reversed(matches):  # Reverse to maintain positions
                if random.random() > self.aggression * 0.8:
                    continue

                original = match.group()
                phrase_lower = phrase.lower()

                if phrase_lower in TRANSITION_REPLACEMENTS:
                    replacements = TRANSITION_REPLACEMENTS[phrase_lower]
                    replacement = random.choice(replacements)
                else:
                    replacement = ""

                # Preserve capitalization
                if original[0].isupper() and replacement:
                    replacement = replacement[0].upper() + replacement[1:]

                # Handle empty replacement (just remove the word)
                if not replacement:
                    # Remove the word and fix spacing/punctuation
                    start = match.start()
                    end = match.end()
                    # Check if followed by comma
                    if end < len(text) and text[end] == ',':
                        end += 1
                    # Check if preceded by space
                    if start > 0 and text[start - 1] == ' ':
                        start -= 1
                    text = text[:start] + text[end:]
                    # Capitalize next word if at sentence start
                    if start < len(text) and start > 0 and text[start - 1] in '.!?\n':
                        while start < len(text) and text[start] == ' ':
                            start += 1
                        if start < len(text):
                            text = text[:start] + text[start].upper() + text[start + 1:]
                else:
                    text = text[:match.start()] + replacement + text[match.end():]

        return text

    def _fix_repeated_starters(self, text):
        """Fix sentences that start with the same word/phrase repeatedly."""
        paragraphs = text.split('\n\n')
        result = []

        for para in paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) < 3:
                result.append(para)
                continue

            # Check for repeated starters
            starters = [s.split()[0].lower() if s.split() else '' for s in sentences]
            modified = list(sentences)

            for i in range(1, len(starters)):
                if starters[i] == starters[i - 1] and random.random() < self.aggression:
                    modified[i] = self._rephrase_starter(modified[i])

            # Also check for "The" starting too many sentences
            the_count = sum(1 for s in starters if s == 'the')
            if the_count > len(sentences) * 0.4:
                for i in range(len(modified)):
                    if (starters[i] == 'the' and
                            random.random() < self.aggression * 0.5):
                        modified[i] = self._rephrase_starter(modified[i])

            result.append(' '.join(modified))

        return '\n\n'.join(result)

    def _rephrase_starter(self, sentence):
        """Rephrase the beginning of a sentence to avoid repetition."""
        starters_to_add = [
            "In practice, ",
            "Here, ",
            "At this point, ",
            "Looking at this differently, ",
            "From the data, ",
            "Given this, ",
            "As such, ",
            "To clarify, ",
        ]

        words = sentence.split()
        if len(words) < 3:
            return sentence

        # Option 1: Add a starter phrase
        if random.random() < 0.5:
            prefix = random.choice(starters_to_add)
            # Lowercase the first word of original
            words[0] = words[0][0].lower() + words[0][1:] if words[0][0].isupper() else words[0]
            return prefix + ' '.join(words)

        # Option 2: Swap first two elements if possible
        if len(words) > 4 and ',' in sentence[:30]:
            comma_idx = sentence.index(',')
            before = sentence[:comma_idx]
            after = sentence[comma_idx + 1:].strip()
            if after:
                return after[0].upper() + after[1:].rstrip('.') + ', ' + before.lower() + '.'

        return sentence

    def _vary_punctuation(self, text):
        """Slightly vary punctuation patterns for natural feel."""
        if self.aggression < 0.3:
            return text

        # Occasionally replace semicolons with periods or dashes
        if ';' in text and random.random() < self.aggression * 0.4:
            semicolons = [m.start() for m in re.finditer(';', text)]
            if semicolons:
                idx = random.choice(semicolons)
                replacement = random.choice(['. ', ' - '])
                text = text[:idx] + replacement + text[idx + 1:]
                # Capitalize after period
                next_char_idx = idx + len(replacement)
                if replacement == '. ' and next_char_idx < len(text):
                    text = (text[:next_char_idx] +
                            text[next_char_idx].upper() +
                            text[next_char_idx + 1:])

        return text

    def _inject_natural_imperfections(self, text):
        """Add subtle natural imperfections that humans typically produce."""
        if self.aggression < 0.4:
            return text

        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []

        for sent in sentences:
            # Occasionally add a slightly informal connector
            if (random.random() < self.aggression * 0.1 and
                    len(sent.split()) > 10):
                informal_connectors = [
                    " - and this is key - ",
                    " (though not always) ",
                    " - in theory at least - ",
                ]
                words = sent.split()
                if len(words) > 6:
                    pos = random.randint(3, min(7, len(words) - 3))
                    connector = random.choice(informal_connectors)
                    words.insert(pos, connector)
                    sent = ' '.join(words)

            result.append(sent)

        return ' '.join(result)

    def _final_cleanup(self, text):
        """Clean up any artifacts from processing."""
        # Fix double spaces
        text = re.sub(r'  +', ' ', text)
        # Fix space before punctuation
        text = re.sub(r' ([.,;:!?])', r'\1', text)
        # Fix missing space after punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        # Fix double periods
        text = text.replace('..', '.')
        # Fix orphaned punctuation
        text = re.sub(r'^\s*[,;]\s*', '', text, flags=re.MULTILINE)
        # Ensure proper paragraph spacing
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip trailing whitespace
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        return text.strip()
