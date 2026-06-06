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

from humanizer.config import AI_TRANSITION_WORDS, TRANSITION_REPLACEMENTS, INDIAN_DISCOURSE_MARKERS


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
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def process(self, text):
        """Apply all post-processing steps."""
        if not text.strip():
            return text

        text = self._remove_ai_transitions(text)
        text = self._disrupt_ngram_patterns(text)
        text = self._inject_contractions(text)
        text = self._inject_discourse_markers(text)
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
                if self.rng.random() > self.aggression * 0.8:
                    continue

                original = match.group()
                phrase_lower = phrase.lower()

                if phrase_lower in TRANSITION_REPLACEMENTS:
                    replacements = TRANSITION_REPLACEMENTS[phrase_lower]
                    replacement = self.rng.choice(replacements)
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

    def _disrupt_ngram_patterns(self, text):
        """Identify and disrupt common n-gram patterns."""
        # Swap adjective-noun pairs occasionally
        adjective_noun_patterns = [
            (r'\b(important|significant|critical|key|major|primary|essential)\s+(finding|result|factor|aspect|issue|point|role)\b',
             lambda m: f"{m.group(2)} of {m.group(1).rstrip('al')}ce" if self.rng.random() < self.aggression * 0.3
             else m.group(0)),
        ]

        for pattern, repl in adjective_noun_patterns:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        # Insert adverbs into verb phrases
        adverb_insertions = [
            (r'\bhas shown\b', ['has consistently shown', 'has, in fact, shown', 'has clearly shown']),
            (r'\bhas been\b', ['has long been', 'has, in effect, been', 'has indeed been']),
            (r'\bcan be\b', ['can certainly be', 'can, in principle, be', 'can reasonably be']),
            (r'\bwill be\b', ['will likely be', 'will, in all likelihood, be', 'will undoubtedly be']),
            (r'\bmay be\b', ['may well be', 'may, arguably, be', 'may indeed be']),
            (r'\bhave been\b', ['have often been', 'have, by and large, been', 'have consistently been']),
        ]

        for pattern, replacements in adverb_insertions:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in reversed(matches):
                if self.rng.random() < self.aggression * 0.3:
                    replacement = self.rng.choice(replacements)
                    # Preserve case
                    if match.group()[0].isupper():
                        replacement = replacement[0].upper() + replacement[1:]
                    text = text[:match.start()] + replacement + text[match.end():]

        return text

    def _inject_contractions(self, text):
        """Convert formal constructions to contractions in non-first sentences."""
        contraction_map = {
            'it is': "it's",
            'do not': "don't",
            'does not': "doesn't",
            'cannot': "can't",
            'will not': "won't",
            'would not': "wouldn't",
            'should not': "shouldn't",
            'is not': "isn't",
            'are not': "aren't",
            'has not': "hasn't",
            'have not': "haven't",
            'that is': "that's",
            'there is': "there's",
        }

        paragraphs = text.split('\n\n')
        result_paragraphs = []

        for para in paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            result_sentences = []

            for i, sent in enumerate(sentences):
                if i == 0:
                    # Keep first sentence formal
                    result_sentences.append(sent)
                    continue

                for formal, contraction in contraction_map.items():
                    if self.rng.random() < self.aggression * 0.3:
                        pattern = re.compile(r'\b' + re.escape(formal) + r'\b', re.IGNORECASE)
                        matches = list(pattern.finditer(sent))
                        for match in reversed(matches):
                            original = match.group()
                            replacement = contraction
                            # Preserve capitalization
                            if original[0].isupper():
                                replacement = replacement[0].upper() + replacement[1:]
                            sent = sent[:match.start()] + replacement + sent[match.end():]

                result_sentences.append(sent)

            result_paragraphs.append(' '.join(result_sentences))

        return '\n\n'.join(result_paragraphs)

    def _inject_discourse_markers(self, text):
        """Insert Indian English discourse markers at sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []

        # Words that indicate the sentence already starts with a connector
        connector_starts = {
            'however', 'moreover', 'furthermore', 'additionally', 'nevertheless',
            'consequently', 'therefore', 'hence', 'thus', 'meanwhile',
            'nonetheless', 'accordingly', 'similarly', 'conversely', 'alternatively',
            'in', 'on', 'as', 'to', 'by', 'for', 'given', 'that',
        }

        for i, sent in enumerate(sentences):
            if not sent.strip():
                result.append(sent)
                continue

            first_word = sent.split()[0].lower().rstrip(',') if sent.split() else ''

            if (i > 0 and first_word not in connector_starts and
                    self.rng.random() < self.aggression * 0.08):
                marker = self.rng.choice(INDIAN_DISCOURSE_MARKERS)
                # Capitalize marker and prepend
                marker_capitalized = marker[0].upper() + marker[1:]
                # Add comma after marker if not already ending with one
                if not marker_capitalized.endswith(','):
                    marker_capitalized += ','
                sent = f"{marker_capitalized} {sent[0].lower()}{sent[1:]}"

            result.append(sent)

        return ' '.join(result)

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
                if starters[i] == starters[i - 1] and self.rng.random() < self.aggression:
                    modified[i] = self._rephrase_starter(modified[i])

            # Also check for "The" starting too many sentences
            the_count = sum(1 for s in starters if s == 'the')
            if the_count > len(sentences) * 0.4:
                for i in range(len(modified)):
                    if (starters[i] == 'the' and
                            self.rng.random() < self.aggression * 0.5):
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
            "One may observe that ",
            "It is worth considering that ",
            "Interestingly, ",
            "On closer examination, ",
            "Viewed differently, ",
            "That said, ",
            "To be precise, ",
            "Broadly, ",
            "In point of fact, ",
        ]

        words = sentence.split()
        if len(words) < 3:
            return sentence

        # Option 1: Add a starter phrase
        if self.rng.random() < 0.5:
            prefix = self.rng.choice(starters_to_add)
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
        if ';' in text and self.rng.random() < self.aggression * 0.4:
            semicolons = [m.start() for m in re.finditer(';', text)]
            if semicolons:
                idx = self.rng.choice(semicolons)
                replacement = '. '
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
            if (self.rng.random() < self.aggression * 0.08 and
                    len(sent.split()) > 10):
                informal_connectors = [
                    " (though not always) ",
                    " (to some extent) ",
                    " (admittedly) ",
                    " (in a sense) ",
                    " (one might argue) ",
                    " (to some degree) ",
                ]
                words = sent.split()
                if len(words) > 6:
                    pos = self.rng.randint(3, min(7, len(words) - 3))
                    connector = self.rng.choice(informal_connectors)
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
