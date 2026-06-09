"""
Stage 3 (new): Semantic-Preserving Transformations

Applies surface-form NLP transformations (synonym substitution, passive/active
voice toggling, clause reordering) that measurably preserve meaning. Uses a
strict 0.90 similarity floor — any candidate that drops below this threshold
is discarded and the input returned unchanged.

This stage runs early in the pipeline so later stages build on a meaning-safe
base. No network or model dependency: transformations are pure NLP driven by
`aggression` level and `seed` for determinism.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.6, 6.7, 6.8
"""

from __future__ import annotations

import random
import re
from typing import Optional

from humanizer.protected_spans import ProtectedSpanGuard
from humanizer.results import StageResult

# Try to import NLTK wordnet for richer synonym replacement
try:
    import nltk
    from nltk.corpus import wordnet

    try:
        wordnet.synsets("test")
        HAS_WORDNET = True
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            wordnet.synsets("test")
            HAS_WORDNET = True
        except Exception:
            HAS_WORDNET = False
except ImportError:
    HAS_WORDNET = False

# Simple built-in synonym map as a fallback when NLTK is unavailable
_SYNONYM_MAP = {
    "important": ["significant", "crucial", "vital", "essential"],
    "large": ["substantial", "considerable", "extensive", "sizable"],
    "small": ["minor", "modest", "limited", "slight"],
    "show": ["demonstrate", "illustrate", "reveal", "indicate"],
    "help": ["assist", "support", "facilitate", "aid"],
    "use": ["employ", "apply", "utilize", "adopt"],
    "make": ["create", "produce", "generate", "construct"],
    "good": ["effective", "beneficial", "favourable", "positive"],
    "bad": ["poor", "unfavourable", "detrimental", "negative"],
    "big": ["substantial", "major", "considerable", "significant"],
    "get": ["obtain", "acquire", "gain", "secure"],
    "find": ["discover", "identify", "detect", "locate"],
    "give": ["provide", "supply", "offer", "deliver"],
    "increase": ["enhance", "elevate", "amplify", "boost"],
    "decrease": ["reduce", "diminish", "lower", "lessen"],
    "change": ["modify", "alter", "adjust", "transform"],
    "start": ["begin", "commence", "initiate", "launch"],
    "end": ["conclude", "terminate", "finish", "complete"],
    "need": ["require", "demand", "necessitate", "call for"],
    "different": ["distinct", "varied", "diverse", "dissimilar"],
    "similar": ["comparable", "analogous", "akin", "related"],
    "clear": ["evident", "apparent", "obvious", "transparent"],
    "hard": ["difficult", "challenging", "demanding", "arduous"],
    "fast": ["rapid", "swift", "quick", "expeditious"],
    "new": ["novel", "recent", "fresh", "modern"],
    "old": ["established", "longstanding", "prior", "earlier"],
    "main": ["primary", "principal", "central", "chief"],
    "part": ["component", "element", "segment", "portion"],
    "problem": ["issue", "challenge", "difficulty", "concern"],
    "result": ["outcome", "finding", "consequence", "effect"],
    "study": ["investigation", "research", "examination", "analysis"],
    "work": ["function", "operate", "perform", "serve"],
    "think": ["consider", "believe", "regard", "view"],
    "provide": ["supply", "offer", "furnish", "deliver"],
    "suggest": ["indicate", "imply", "propose", "recommend"],
    "develop": ["create", "establish", "formulate", "devise"],
    "include": ["encompass", "comprise", "incorporate", "contain"],
    "consider": ["examine", "evaluate", "assess", "contemplate"],
    "describe": ["characterize", "depict", "outline", "portray"],
    "report": ["document", "present", "note", "detail"],
    "require": ["necessitate", "demand", "need", "call for"],
    "reduce": ["diminish", "decrease", "lower", "minimize"],
    "produce": ["generate", "create", "yield", "manufacture"],
    "maintain": ["preserve", "sustain", "retain", "uphold"],
    "determine": ["establish", "ascertain", "identify", "assess"],
    "however": ["nonetheless", "yet", "still", "though"],
    "therefore": ["thus", "hence", "accordingly", "consequently"],
    "although": ["though", "while", "even though", "despite the fact that"],
    "because": ["since", "as", "given that", "due to the fact that"],
}

# Passive voice conversion patterns (active -> passive-ish restructuring)
_PASSIVE_STARTERS = [
    "It was found that",
    "It has been observed that",
    "It can be noted that",
    "It is seen that",
    "It was noted that",
]


class SemanticTransformer:
    """Applies surface-form NLP transformations preserving meaning.

    Produces a candidate whose character sequence differs from the input,
    computes a [0,1] similarity score, and discards the candidate (returning
    input unchanged) when similarity < floor (default 0.90).

    Parameters
    ----------
    aggression : float
        Controls transformation intensity (0.0-1.0). Higher values apply
        more synonym substitutions and restructuring.
    seed : int or None
        Optional seed for deterministic behaviour via random.Random(seed).
    similarity : object or None
        A SimilarityEvaluator (or compatible fake) with a `score(a, b)` method.
        If None, uses a built-in lexical proxy (token Jaccard).
    floor : float
        Minimum similarity threshold. Candidates below this are discarded.
    """

    def __init__(
        self,
        aggression: float = 0.5,
        seed: Optional[int] = None,
        similarity=None,
        floor: float = 0.90,
    ) -> None:
        self.aggression = aggression
        self.seed = seed
        self.similarity = similarity
        self.floor = floor
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def process(self, text: str) -> str:
        """Apply semantic transformation and return the result text.

        Delegates to process_measured() and returns only the text.
        """
        result = self.process_measured(text)
        return result.text

    def process_measured(self, text: str) -> StageResult:
        """Apply semantic transformation with measurement metadata.

        Returns a StageResult carrying the transformed (or fallback) text
        plus the computed similarity score.
        """
        # Empty input → unchanged, no score (Req 6.7)
        if not text or not text.strip():
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=False,
                error=None,
            )

        # Attempt transformation
        try:
            candidate = self._transform(text)
        except Exception as e:
            # Source error → unchanged (Req 6.8)
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error=str(e),
            )

        # If transformation produced empty → unchanged (Req 6.8)
        if not candidate or not candidate.strip():
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error="Transformation produced empty result",
            )

        # Compute similarity (Req 6.2)
        score = self._compute_similarity(text, candidate)

        # If candidate is identical to input, try harder — we must produce
        # a differing character sequence for non-empty input (Req 6.1)
        if candidate == text:
            # Force at least a minor transformation
            candidate = self._force_minimal_change(text)
            if candidate == text:
                # Truly unable to change — return input, no score
                return StageResult(
                    text=text,
                    similarity=None,
                    risk_before=None,
                    risk_after=None,
                    changed=False,
                    fell_back=True,
                    error="Unable to produce differing candidate",
                )
            score = self._compute_similarity(text, candidate)

        # Discard if below floor (Req 6.3)
        if score < self.floor:
            return StageResult(
                text=text,
                similarity=score,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error=None,
            )

        # Accept candidate
        return StageResult(
            text=candidate,
            similarity=score,
            risk_before=None,
            risk_after=None,
            changed=True,
            fell_back=False,
            error=None,
        )

    def _transform(self, text: str) -> str:
        """Apply NLP transformations with protected span masking.

        Uses ProtectedSpanGuard to mask/unmask protected spans (Req 6.4).
        """
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)

        # Apply transformations on masked text
        transformed = self._apply_transformations(masked)

        # Unmask protected spans
        result = guard.unmask(transformed)
        return result

    def _apply_transformations(self, text: str) -> str:
        """Apply the suite of semantic transformations.

        The set of transformations applied scales with aggression:
        - Low (0.0-0.3): light synonym substitution only
        - Medium (0.3-0.7): synonym substitution + clause reordering
        - High (0.7-1.0): all of the above + passive/active restructuring

        Transformations are conservative to stay above the 0.90 floor.
        """
        # Synonym substitution (always applied)
        text = self._synonym_substitution(text)

        # Sentence-level restructuring at medium+ aggression
        if self.aggression > 0.5:
            text = self._reorder_clauses(text)

        return text

    def _synonym_substitution(self, text: str) -> str:
        """Replace words with synonyms based on aggression level.

        Uses NLTK WordNet if available, otherwise falls back to the built-in
        synonym map. Conservatively limits replacements to maintain high
        similarity (the 0.90 floor is strict).
        """
        words = text.split()
        if not words:
            return text

        # Very conservative replacement to stay above 0.90 Jaccard.
        # At aggression 0.0 → ~2% of words; at 1.0 → ~8%.
        # Max 1 replacement per ~12 words to stay safely above 0.90 floor.
        replacement_rate = 0.02 + self.aggression * 0.06
        max_replacements = max(1, int(len(words) * replacement_rate))

        result_words = []
        replacements_made = 0

        for word in words:
            # Skip placeholder tokens (ProtectedSpanGuard)
            if "\x00" in word:
                result_words.append(word)
                continue

            # Extract the clean alphabetic core
            clean = re.sub(r"[^a-zA-Z]", "", word)
            if len(clean) < 4 or replacements_made >= max_replacements:
                result_words.append(word)
                continue

            # Probabilistic selection
            if self.rng.random() > replacement_rate:
                result_words.append(word)
                continue

            # Find a synonym
            synonym = self._find_synonym(clean.lower())
            if synonym and synonym.lower() != clean.lower():
                # Preserve surrounding punctuation
                prefix = ""
                suffix = ""
                for ch in word:
                    if ch.isalpha():
                        break
                    prefix += ch
                for ch in reversed(word):
                    if ch.isalpha():
                        break
                    suffix = ch + suffix

                # Preserve capitalization
                if clean[0].isupper():
                    synonym = synonym[0].upper() + synonym[1:]

                result_words.append(f"{prefix}{synonym}{suffix}")
                replacements_made += 1
            else:
                result_words.append(word)

        return " ".join(result_words)

    def _find_synonym(self, word: str) -> Optional[str]:
        """Find a synonym for the given word.

        Prefers the built-in map (curated, single-word synonyms) and only
        falls back to WordNet for words not in the map, keeping only
        single-word lemmas from the first synset for relevance.
        """
        # Prefer built-in map (curated, safe synonyms)
        if word in _SYNONYM_MAP:
            return self.rng.choice(_SYNONYM_MAP[word])

        # Fall back to WordNet for broader coverage
        if HAS_WORDNET:
            synsets = wordnet.synsets(word)
            if synsets:
                candidates = []
                for syn in synsets[:2]:  # First 2 synsets for relevance
                    for lemma in syn.lemmas():
                        name = lemma.name().replace("_", " ")
                        if (
                            name.lower() != word.lower()
                            and len(name.split()) == 1
                            and len(name) >= 3
                        ):
                            candidates.append(name)
                if candidates:
                    return self.rng.choice(candidates[:4])

        return None

    def _reorder_clauses(self, text: str) -> str:
        """Reorder clauses within sentences for variety.

        Moves prepositional phrases or adverbial clauses to different
        positions within sentences.
        """
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return text

        result = []
        for sent in sentences:
            if self.rng.random() < self.aggression * 0.4:
                reordered = self._try_clause_reorder(sent)
                result.append(reordered)
            else:
                result.append(sent)

        return " ".join(result)

    def _try_clause_reorder(self, sentence: str) -> str:
        """Try to reorder clauses in a sentence.

        Moves a comma-separated initial clause to the end, or vice versa.
        """
        # Skip short sentences
        if len(sentence.split()) < 8:
            return sentence

        # Pattern: "Clause, rest of sentence." → "Rest of sentence, clause."
        match = re.match(
            r"^([A-Z][^,]{5,40}),\s+(.+)$", sentence, re.DOTALL
        )
        if match and self.rng.random() < 0.5:
            clause = match.group(1)
            rest = match.group(2)
            # Move initial clause to end
            if rest and rest[-1] in ".!?":
                end_punct = rest[-1]
                rest = rest[:-1].strip()
                # Capitalize rest, lowercase clause
                new_sent = (
                    rest[0].upper() + rest[1:] + ", " + clause[0].lower() + clause[1:] + end_punct
                )
                return new_sent

        # Pattern: "Main clause, trailing clause." → "Trailing clause, main clause."
        parts = sentence.rsplit(",", 1)
        if len(parts) == 2 and len(parts[1].split()) >= 3 and self.rng.random() < 0.3:
            main = parts[0].strip()
            trailing = parts[1].strip()
            if trailing and trailing[-1] in ".!?":
                end_punct = trailing[-1]
                trailing = trailing[:-1].strip()
                new_sent = (
                    trailing[0].upper() + trailing[1:] + ", " + main[0].lower() + main[1:] + end_punct
                )
                return new_sent

        return sentence

    def _voice_restructuring(self, text: str) -> str:
        """Apply passive/active voice restructuring to select sentences.

        Prepends passive-voice starters to some sentences for variety.
        """
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return text

        result = []
        for sent in sentences:
            # Only restructure some sentences
            if (
                self.rng.random() < self.aggression * 0.2
                and len(sent.split()) > 6
                and not sent.startswith("It ")
                and not any(sent.startswith(p) for p in _PASSIVE_STARTERS)
            ):
                starter = self.rng.choice(_PASSIVE_STARTERS)
                # Lowercase the first letter of the original sentence
                lower_sent = sent[0].lower() + sent[1:]
                # Remove trailing period to append after starter
                if lower_sent.endswith("."):
                    lower_sent = lower_sent[:-1]
                result.append(f"{starter} {lower_sent}.")
            else:
                result.append(sent)

        return " ".join(result)

    def _force_minimal_change(self, text: str) -> str:
        """Force at least a minimal change when other transforms failed.

        Applies a single comma insertion or word reorder to guarantee the
        output differs from the input while preserving meaning.
        """
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)

        words = masked.split()
        if len(words) < 2:
            return text

        # Strategy 1: Insert a comma after a suitable word
        for i in range(min(len(words) - 1, 10)):
            idx = self.rng.randint(1, len(words) - 1)
            word = words[idx]
            if "\x00" not in word and not word.endswith(",") and len(word) > 2:
                words[idx] = word + ","
                result = guard.unmask(" ".join(words))
                if result != text:
                    return result
                # Revert
                words[idx] = word

        # Strategy 2: Swap two adjacent non-protected words
        for _ in range(10):
            idx = self.rng.randint(0, len(words) - 2)
            w1, w2 = words[idx], words[idx + 1]
            if "\x00" not in w1 and "\x00" not in w2:
                words[idx], words[idx + 1] = w2, w1
                result = guard.unmask(" ".join(words))
                if result != text:
                    return result
                # Revert
                words[idx], words[idx + 1] = w1, w2

        return text

    def _compute_similarity(self, original: str, candidate: str) -> float:
        """Compute similarity between original and candidate.

        Uses the injected similarity evaluator if available, otherwise
        falls back to lexical proxy (token Jaccard).
        """
        if self.similarity is not None:
            return self.similarity.score(original, candidate)
        return self._lexical_similarity(original, candidate)

    @staticmethod
    def _lexical_similarity(a: str, b: str) -> float:
        """Compute token Jaccard similarity as a lexical proxy.

        Returns 1.0 for identical token sets, 0.0 for completely disjoint.
        """
        tokens_a = set(re.findall(r"\b\w+\b", a.lower()))
        tokens_b = set(re.findall(r"\b\w+\b", b.lower()))

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    @staticmethod
    def _split_sentences(text: str) -> list:
        """Split text into sentences using regex."""
        # Handle common abbreviations
        text_clean = text.replace("e.g.", "e<DOT>g<DOT>")
        text_clean = text_clean.replace("i.e.", "i<DOT>e<DOT>")
        text_clean = text_clean.replace("et al.", "et al<DOT>")

        sentences = re.split(r"(?<=[.!?])\s+", text_clean)
        return [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]
