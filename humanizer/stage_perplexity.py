"""
Stage 4: Perplexity Variance Injection

Post-LLM processing that deliberately varies sentence complexity to mimic
natural human writing patterns. AI text tends to have uniform perplexity
across sentences, while human text varies wildly - some ideas stated simply,
others elaborated in complex structures.

This stage uses NLP techniques (not LLM calls) to:
- Simplify some sentences (shorter, common words)
- Make others more complex (embedded clauses, longer structures)
"""

import re
import random

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        HAS_SPACY = True
    except OSError:
        HAS_SPACY = False
        nlp = None
except ImportError:
    HAS_SPACY = False
    nlp = None


class PerplexityVariance:
    """Injects variance in sentence complexity for natural perplexity patterns."""

    def __init__(self, aggression=0.5, seed=None):
        """
        Args:
            aggression: Float 0-1 controlling how much variance to inject.
            seed: Optional int seed for reproducible results.
        """
        self.aggression = aggression
        self.seed = seed
        # Parenthetical insertions for complexity
        self.parentheticals = [
            "at least in part",
            "as one might expect",
            "broadly speaking",
            "to some extent",
            "in a practical sense",
            "under typical conditions",
            "as noted earlier",
            "from this perspective",
            "in most cases",
            "given these factors",
        ]
        # Simplification connectors
        self.simple_starters = [
            "Put simply,",
            "In short,",
            "The point is",
            "This means",
            "Basically,",
        ]

    def process(self, text):
        """Apply perplexity variance to the input text."""
        if not text.strip():
            return text

        if self.seed is not None:
            random.seed(self.seed)

        paragraphs = text.split('\n\n')
        processed = []

        for para in paragraphs:
            sentences = self._split_sentences(para)
            if len(sentences) < 3:
                processed.append(para)
                continue

            modified = self._inject_variance(sentences)
            processed.append(' '.join(modified))

        return '\n\n'.join(processed)

    def _split_sentences(self, text):
        """Split into sentences."""
        text_clean = text.replace("e.g.", "e<DOT>g<DOT>")
        text_clean = text_clean.replace("i.e.", "i<DOT>e<DOT>")
        text_clean = text_clean.replace("et al.", "et al<DOT>")

        sentences = re.split(r'(?<=[.!?])\s+', text_clean)
        return [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]

    def _inject_variance(self, sentences):
        """Inject complexity variance across sentences."""
        result = []
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 15

        for i, sent in enumerate(sentences):
            words = sent.split()
            word_count = len(words)

            # Randomly decide: simplify, complexify, or leave alone
            action = random.random()
            modification_threshold = 1.0 - (self.aggression * 0.5)

            if action < self.aggression * 0.25 and word_count > 12:
                # Simplify this sentence
                result.append(self._simplify_sentence(sent))
            elif action < self.aggression * 0.5 and word_count > 8 and word_count < 30:
                # Add complexity
                result.append(self._complexify_sentence(sent))
            elif (action < self.aggression * 0.3 and word_count > 20 and
                  i > 0 and i < len(sentences) - 1):
                # Break into two with different complexity
                parts = self._create_length_contrast(sent)
                result.extend(parts)
            else:
                result.append(sent)

        return result

    def _simplify_sentence(self, sentence):
        """Make a sentence simpler and shorter."""
        words = sentence.split()

        # Remove hedge phrases
        hedges_to_remove = [
            "it is important to note that",
            "it should be noted that",
            "it can be observed that",
            "in this context,",
            "in this regard,",
        ]
        sent_lower = sentence.lower()
        for hedge in hedges_to_remove:
            if hedge in sent_lower:
                idx = sent_lower.find(hedge)
                sentence = sentence[:idx] + sentence[idx + len(hedge):]
                sentence = sentence.strip()
                if sentence and sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                break

        # If sentence is still long, try to truncate at a natural break
        words = sentence.split()
        if len(words) > 20:
            # Find a comma after the midpoint and cut there
            mid = len(words) // 2
            for j in range(mid, min(mid + 8, len(words))):
                if words[j].endswith(','):
                    sentence = ' '.join(words[:j + 1]).rstrip(',') + '.'
                    break

        return sentence

    def _complexify_sentence(self, sentence):
        """Add complexity to a sentence via parenthetical or embedding."""
        words = sentence.split()

        if len(words) < 8:
            return sentence

        # Insert a parenthetical phrase
        if random.random() < 0.6:
            insertion = random.choice(self.parentheticals)
            # Find a good insertion point (after subject, around mid-sentence)
            insert_pos = random.randint(3, min(8, len(words) - 2))

            # Wrap in appropriate punctuation
            if random.random() < 0.5:
                # Use dashes
                words.insert(insert_pos, f"- {insertion} -")
            else:
                # Use commas
                if not words[insert_pos - 1].endswith(','):
                    words[insert_pos - 1] = words[insert_pos - 1] + ','
                words.insert(insert_pos, f"{insertion},")

            return ' '.join(words)

        return sentence

    def _create_length_contrast(self, sentence):
        """Split a long sentence into two with contrasting lengths."""
        # Find a split point
        comma_positions = [i for i, ch in enumerate(sentence) if ch == ',']
        if not comma_positions:
            return [sentence]

        # Split near middle
        mid = len(sentence) // 2
        best_comma = min(comma_positions, key=lambda x: abs(x - mid))

        if best_comma < 10 or best_comma > len(sentence) - 10:
            return [sentence]

        first = sentence[:best_comma].strip() + '.'
        second = sentence[best_comma + 1:].strip()
        if second:
            second = second[0].upper() + second[1:]
            if not second.endswith('.') and not second.endswith('!') and not second.endswith('?'):
                second += '.'

        return [first, second]
