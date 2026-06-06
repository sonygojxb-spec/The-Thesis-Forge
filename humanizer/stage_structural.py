"""
Stage 1: Structural Variation

Deterministic structural transformations that alter the text's sentence
and paragraph structure without changing meaning. This breaks the uniform
rhythm that AI detectors flag.

Operations:
- Sentence splitting (break long compound sentences)
- Sentence merging (combine short adjacent sentences)
- Clause reordering within sentences
- Paragraph restructuring (vary paragraph lengths)
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


class StructuralVariation:
    """Applies structural transformations to text."""

    def __init__(self, aggression=0.5, seed=None):
        """
        Args:
            aggression: Float 0-1 controlling how aggressively to restructure.
            seed: Optional int seed for reproducible results.
        """
        self.aggression = aggression
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def process(self, text):
        """Apply structural variation to the input text."""
        if not text.strip():
            return text

        paragraphs = text.split('\n\n')
        processed_paragraphs = []

        for para in paragraphs:
            if not para.strip():
                continue
            sentences = self._split_into_sentences(para)
            sentences = self._split_long_sentences(sentences)
            sentences = self._merge_short_sentences(sentences)
            if HAS_SPACY and self.aggression > 0.4:
                sentences = self._reorder_clauses(sentences)
            sentences = self._invert_sentence_order(sentences)
            sentences = self._inject_rhetorical_questions(sentences)
            processed_paragraphs.append(' '.join(sentences))

        # Restructure paragraphs at higher aggression
        if self.aggression > 0.5 and len(processed_paragraphs) > 2:
            processed_paragraphs = self._restructure_paragraphs(processed_paragraphs)

        return '\n\n'.join(processed_paragraphs)

    def _split_into_sentences(self, text):
        """Split text into sentences."""
        text_clean = text.replace("e.g.", "e<DOT>g<DOT>")
        text_clean = text_clean.replace("i.e.", "i<DOT>e<DOT>")
        text_clean = text_clean.replace("et al.", "et al<DOT>")

        sentences = re.split(r'(?<=[.!?])\s+', text_clean)
        return [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]

    def _split_long_sentences(self, sentences):
        """Split sentences that are too long at conjunction points."""
        result = []
        threshold = int(30 - (self.aggression * 10))  # 20-30 words

        for sent in sentences:
            words = sent.split()
            if len(words) > threshold:
                # Try to split at conjunctions or semicolons
                split_points = self._find_split_points(sent)
                if split_points:
                    parts = self._split_at_point(sent, split_points[0])
                    result.extend(parts)
                else:
                    result.append(sent)
            else:
                result.append(sent)

        return result

    def _find_split_points(self, sentence):
        """Find safe points to split a sentence."""
        conjunctions = [
            ', and ', ', but ', ', yet ', '; ', ', while ',
            ', whereas ', ', although ', ', however ',
        ]
        points = []
        for conj in conjunctions:
            idx = sentence.find(conj)
            if idx > 10 and idx < len(sentence) - 10:
                points.append((idx, conj))
        return sorted(points, key=lambda x: abs(x[0] - len(sentence) // 2))

    def _split_at_point(self, sentence, split_info):
        """Split sentence at a conjunction point."""
        idx, conj = split_info
        first = sentence[:idx].strip()
        second = sentence[idx + len(conj):].strip()

        # Ensure proper sentence endings
        if first and not first[-1] in '.!?':
            first += '.'
        if second:
            second = second[0].upper() + second[1:]
            if not second[-1] in '.!?':
                second += '.'

        return [first, second]

    def _merge_short_sentences(self, sentences):
        """Merge adjacent short sentences occasionally."""
        if len(sentences) < 2:
            return sentences

        merge_threshold = int(8 + (1 - self.aggression) * 5)  # 8-13 words
        result = []
        i = 0

        while i < len(sentences):
            current = sentences[i]
            words_current = len(current.split())

            if (words_current < merge_threshold and
                    i + 1 < len(sentences) and
                    len(sentences[i + 1].split()) < merge_threshold and
                    self.rng.random() < self.aggression * 0.6):
                # Merge with next sentence
                next_sent = sentences[i + 1]
                # Remove period from first, combine
                if current.endswith('.'):
                    current = current[:-1]
                merged = f"{current}, and {next_sent[0].lower()}{next_sent[1:]}"
                result.append(merged)
                i += 2
            else:
                result.append(current)
                i += 1

        return result

    def _reorder_clauses(self, sentences):
        """Reorder clauses within selected sentences using spaCy."""
        if not HAS_SPACY or not nlp:
            return sentences

        result = []
        for sent in sentences:
            if self.rng.random() < self.aggression * 0.6 and ',' in sent:
                reordered = self._try_reorder(sent)
                result.append(reordered)
            else:
                result.append(sent)
        return result

    def _try_reorder(self, sentence):
        """Try to reorder a sentence by moving a dependent clause."""
        doc = nlp(sentence)

        # Find adverbial clauses or prepositional phrases at start
        for token in doc:
            if token.dep_ == 'advcl' and token.i > 3:
                # Move an initial clause to the end
                clause_tokens = [t for t in token.subtree]
                if len(clause_tokens) > 2 and len(clause_tokens) < len(doc) - 3:
                    clause_start = min(t.i for t in clause_tokens)
                    clause_end = max(t.i for t in clause_tokens)

                    if clause_start == 0 or clause_start == 1:
                        # Clause is at the beginning, leave it
                        return sentence
                    else:
                        # Move clause to front
                        clause_text = doc[clause_start:clause_end + 1].text
                        main_text = (doc[:clause_start].text + " " +
                                     doc[clause_end + 1:].text).strip()
                        if main_text and clause_text:
                            main_text = main_text.rstrip(' ,')
                            if not main_text[-1] in '.!?':
                                result = f"{clause_text}, {main_text[0].lower()}{main_text[1:]}."
                            else:
                                result = f"{clause_text}, {main_text[0].lower()}{main_text[1:]}"
                            return result
                break

        return sentence

    def _invert_sentence_order(self, sentences):
        """Occasionally reverse the order of presenting information within a group."""
        if self.aggression <= 0.5 or len(sentences) < 3:
            return sentences

        result = list(sentences)
        # Look at groups of 3-4 sentences and occasionally present conclusion first
        i = 0
        while i < len(result) - 2:
            group_size = min(self.rng.choice([3, 4]), len(result) - i)
            if self.rng.random() < self.aggression * 0.25:
                # Move the last sentence of the group to the front
                group = result[i:i + group_size]
                last = group[-1]
                group = [last] + group[:-1]
                result[i:i + group_size] = group
                i += group_size
            else:
                i += 1

        return result

    def _inject_rhetorical_questions(self, sentences):
        """Occasionally convert a statement into a rhetorical question form."""
        result = []
        for i, sent in enumerate(sentences):
            words = sent.split()
            if (len(words) > 15 and self.rng.random() < self.aggression * 0.1):
                # Try to convert to rhetorical question
                question = self._to_rhetorical_question(sent)
                if question:
                    result.append(question)
                    result.append(sent)
                    continue
            result.append(sent)
        return result

    def _to_rhetorical_question(self, sentence):
        """Convert a statement into a rhetorical question."""
        sent_lower = sentence.lower().strip()
        # Pattern: "X is important/significant/crucial"
        match = re.match(r'^(.+?)\s+(?:is|are)\s+(?:important|significant|crucial|essential|vital|necessary)',
                         sent_lower)
        if match:
            subject = match.group(1).strip()
            # Remove leading articles for the question
            subject = re.sub(r'^(?:the|this|these|a|an)\s+', '', subject)
            return f"Why is {subject} important?"

        # Pattern: general statement - ask "But why does this matter?"
        if len(sentence.split()) > 18:
            questions = [
                "But why does this matter?",
                "What makes this significant?",
                "Why is this relevant?",
            ]
            return self.rng.choice(questions)

        return None

    def _restructure_paragraphs(self, paragraphs):
        """Vary paragraph lengths by occasionally splitting or merging."""
        result = []
        i = 0

        while i < len(paragraphs):
            para = paragraphs[i]
            sentences = self._split_into_sentences(para)

            # Split long paragraphs
            if len(sentences) > 5 and self.rng.random() < self.aggression * 0.4:
                mid = len(sentences) // 2
                result.append(' '.join(sentences[:mid]))
                result.append(' '.join(sentences[mid:]))
            # Merge short paragraphs
            elif (len(sentences) <= 2 and i + 1 < len(paragraphs) and
                  self.rng.random() < self.aggression * 0.3):
                next_para = paragraphs[i + 1]
                result.append(f"{para} {next_para}")
                i += 1
            else:
                result.append(para)
            i += 1

        return result
