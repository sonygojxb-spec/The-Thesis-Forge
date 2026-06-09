"""
Stage 7: Stylometric Obfuscation

Disrupts stylometric fingerprints by adjusting sentence-length distribution,
function-word frequency, punctuation patterns, and type-token ratio so that
authorship and AI-signature analysis cannot reliably identify the text as
machine-generated.

This is an NLP-only stage (no LLM dependency). It uses deterministic
random.Random(seed) for reproducibility and ProtectedSpanGuard for term
protection.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9
"""

from __future__ import annotations

import random
import re
from typing import Optional

from humanizer.protected_spans import ProtectedSpanGuard
from humanizer.results import StageResult
from humanizer.text_analysis import (
    compute_sentence_length_variance,
    compute_type_token_ratio,
    split_sentences,
)

# Human-writing variance threshold: if the input already meets this,
# the stage only needs to stay within +-2% rather than increase by 10%.
STYLO_VARIANCE_THRESHOLD = 50.0

# Function words commonly used in English academic writing
_FUNCTION_WORDS = [
    "the", "a", "an", "of", "in", "to", "for", "on", "with", "at",
    "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "this", "that", "these", "those", "it", "its", "they", "their",
    "which", "who", "whom", "what", "where", "when", "how",
    "not", "no", "nor", "but", "or", "and", "if", "then", "than",
    "so", "yet", "both", "either", "neither", "each", "every",
    "all", "any", "few", "more", "most", "some", "such",
    "very", "quite", "rather", "also", "just", "only", "still",
]

# Synonym substitutions to adjust type-token ratio (replace repeated words)
_TTR_SYNONYMS = {
    "important": ["significant", "crucial", "vital", "essential", "key"],
    "show": ["demonstrate", "reveal", "indicate", "illustrate"],
    "use": ["employ", "utilize", "apply", "leverage"],
    "make": ["create", "produce", "generate", "form"],
    "good": ["effective", "beneficial", "positive", "favorable"],
    "increase": ["enhance", "elevate", "boost", "raise"],
    "decrease": ["reduce", "diminish", "lower", "lessen"],
    "large": ["substantial", "considerable", "extensive", "vast"],
    "small": ["minor", "modest", "limited", "slight"],
    "change": ["modify", "alter", "adjust", "shift"],
    "result": ["outcome", "finding", "consequence", "effect"],
    "study": ["investigation", "research", "examination", "analysis"],
    "provide": ["supply", "offer", "furnish", "deliver"],
    "suggest": ["indicate", "imply", "propose", "hint"],
    "help": ["assist", "support", "facilitate", "aid"],
    "find": ["discover", "identify", "detect", "locate"],
    "different": ["distinct", "varied", "diverse", "dissimilar"],
    "similar": ["comparable", "analogous", "akin", "related"],
    "problem": ["issue", "challenge", "difficulty", "concern"],
    "method": ["approach", "technique", "procedure", "strategy"],
}


class StylometricObfuscator:
    """Disrupts stylometric fingerprints via NLP-only transformations.

    Adjusts sentence-length distribution, function-word frequency,
    punctuation patterns, and type-token ratio. At least one attribute
    shifts by >= 5% when aggression > 0 and input has >= 2 sentences.

    Parameters
    ----------
    aggression : float
        Controls transformation intensity (0.0-1.0). Higher values apply
        more aggressive distributional adjustments.
    seed : int or None
        Optional seed for deterministic behaviour via random.Random(seed).
    similarity : object or None
        A SimilarityEvaluator (or compatible fake) with a `score(a, b)` method.
        If None, uses a built-in lexical proxy (token Jaccard).
    floor : float
        Minimum similarity threshold. Candidates below this are discarded.
    variance_threshold : float
        The human-writing variance threshold. When input variance already
        meets or exceeds this, the stage only needs to stay within +-2%.
    """

    def __init__(
        self,
        aggression: float = 0.5,
        seed: Optional[int] = None,
        similarity=None,
        floor: float = 0.85,
        variance_threshold: float = STYLO_VARIANCE_THRESHOLD,
    ) -> None:
        self.aggression = aggression
        self.seed = seed
        self.similarity = similarity
        self.floor = floor
        self.variance_threshold = variance_threshold
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def process(self, text: str) -> str:
        """Apply stylometric obfuscation and return the result text."""
        result = self.process_measured(text)
        return result.text

    def process_measured(self, text: str) -> StageResult:
        """Apply stylometric obfuscation with measurement metadata."""
        # Empty/whitespace input → unchanged (Req 2.8 implied)
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

        # Aggression 0.0 → return unchanged (Req 2.7)
        if self.aggression <= 0.0:
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=False,
                error=None,
            )

        # < 2 sentences → return unchanged (Req 2.8)
        sentences = split_sentences(text)
        if len(sentences) < 2:
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
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error=str(e),
            )

        # If candidate is identical, return unchanged
        if candidate == text:
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error="No transformation applied",
            )

        # Compute similarity and enforce floor (Req 2.4, 2.9)
        score = self._compute_similarity(text, candidate)
        if score < self.floor:
            # Discard candidate, return input (which has similarity 1.0 >= 0.85)
            return StageResult(
                text=text,
                similarity=score,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error=None,
            )

        # Verify at least one attribute shifted by >=5% (Req 2.1)
        # If not, the transformation is insufficient — discard.
        if not self._meets_attribute_shift(text, candidate):
            return StageResult(
                text=text,
                similarity=score,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error="Insufficient attribute shift",
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
        """Apply stylometric transformations with protected span masking."""
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)

        # Apply transformations on masked text
        transformed = self._apply_transformations(masked)

        # Unmask protected spans (Req 2.3)
        result = guard.unmask(transformed)
        return result

    def _meets_attribute_shift(self, original: str, candidate: str) -> bool:
        """Check if stylometric requirements 2.1 and 2.2 are both met.

        Req 2.1: at least one attribute (variance, TTR, function-word freq,
        punctuation density) shifted by >=5%.
        Req 2.2: sentence-length variance increased by >=10%, OR input variance
        already at threshold and output within +-2%.

        Returns True only if both requirements are satisfied.
        """
        var_before = compute_sentence_length_variance(original)
        var_after = compute_sentence_length_variance(candidate)

        ttr_before = compute_type_token_ratio(original)
        ttr_after = compute_type_token_ratio(candidate)

        fw_before = self._function_word_freq(original)
        fw_after = self._function_word_freq(candidate)

        punct_before = self._punctuation_density(original)
        punct_after = self._punctuation_density(candidate)

        # Req 2.1: at least one attribute shifted by >=5%
        has_shift = False
        for before, after in [
            (var_before, var_after),
            (ttr_before, ttr_after),
            (fw_before, fw_after),
            (punct_before, punct_after),
        ]:
            if before == 0.0:
                if abs(after) > 0.0:
                    has_shift = True
                    break
            elif abs(after - before) / abs(before) >= 0.05:
                has_shift = True
                break

        if not has_shift:
            return False

        # Req 2.2: variance check
        already_at_threshold = var_before >= self.variance_threshold
        if already_at_threshold:
            # Within +-2% is acceptable
            if var_before > 0:
                rel_diff = abs(var_after - var_before) / var_before
                if rel_diff > 0.02:
                    return False
        else:
            # Must increase by >=10%
            if var_before > 0:
                increase = (var_after - var_before) / var_before
                if increase < 0.10:
                    return False
            # var_before == 0: any non-negative output is acceptable

        return True

    @staticmethod
    def _function_word_freq(text: str) -> float:
        """Compute proportion of function words in text."""
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words:
            return 0.0
        count = sum(1 for w in words if w in _FUNCTION_WORDS)
        return count / len(words)

    @staticmethod
    def _punctuation_density(text: str) -> float:
        """Compute punctuation characters per word."""
        words = text.split()
        if not words:
            return 0.0
        punct_count = sum(1 for ch in text if ch in ".,;:!?\u2014-\"'()")
        return punct_count / len(words)

    def _apply_transformations(self, text: str) -> str:
        """Apply the suite of stylometric transformations.

        MONOTONICITY GUARANTEE (Req 2.5): The pipeline computes ALL possible
        candidate changes at maximum intensity (aggression=1.0), ranks them
        by a deterministic priority, and then applies the top K where
        K = ceil(total_candidates * aggression). This guarantees that higher
        aggression applies a strict SUPERSET of lower aggression changes.

        The transformation strategy:
        1. Compute all candidate sentence splits (deterministic by length)
        2. Compute all candidate function-word insertions (deterministic by RNG priority)
        3. Compute all candidate punctuation additions (deterministic by RNG priority)
        4. Compute all candidate TTR synonym replacements (deterministic by frequency)
        5. Rank all candidates in a unified priority list
        6. Apply top-K where K scales with aggression
        """
        base_seed = self.seed if self.seed is not None else 42
        original_variance = compute_sentence_length_variance(text)
        already_at_threshold = original_variance >= self.variance_threshold

        # --- Phase 1: Sentence splitting (deterministic, no RNG) ---
        # Split ratio gets more asymmetric with aggression (monotonic variance)
        if not already_at_threshold:
            text = self._vary_sentence_lengths(text)
            # Post-check for variance requirement
            final_variance = compute_sentence_length_variance(text)
            target_variance = original_variance * 1.10 if original_variance > 0 else 0.01
            if final_variance < target_variance:
                text = self._vary_sentence_lengths_force(text, original_variance)

        # --- Phase 2: Word-level additive changes ---
        # All word-level operations use a single unified priority system.
        # We identify ALL possible insertions/additions, assign priority,
        # then apply top-K where K = f(aggression).
        self.rng = random.Random(base_seed)
        text = self._apply_word_level_changes(text)

        return text

    def _apply_word_level_changes(self, text: str) -> str:
        """Apply word-level changes (punctuation, TTR) using a unified priority
        system that guarantees monotonicity.

        MONOTONICITY: Uses only two types of additive changes that DON'T
        interact with each other:
        - Comma additions: increase punctuation density (punct_count/word_count).
          Since word count is unchanged, density strictly increases with more commas.
        - TTR synonym replacements: increase type-token ratio by replacing
          repeated words with unique synonyms. Word count is unchanged.

        Function-word INSERTIONS are NOT used because they increase word count,
        which DILUTES punctuation density and can counteract comma additions.

        All candidates are ranked by RNG priority. Top-K are applied where
        K scales with aggression → strict superset at higher aggression.
        """
        words = text.split()
        if len(words) < 5:
            return text

        # --- Identify ALL candidate changes ---
        candidates = []

        # 1. Punctuation (comma) addition candidates
        # Find words in positions where a comma can be added (after ~1/3 of
        # each sentence segment, for sentences with 6+ words)
        sentence_starts = [0]
        for i, word in enumerate(words):
            if word.endswith(('.', '!', '?', ';')):
                if i + 1 < len(words):
                    sentence_starts.append(i + 1)

        for start in sentence_starts:
            # Find end of this sentence segment
            end = len(words)
            for j in range(start + 1, len(words)):
                if words[j - 1].endswith(('.', '!', '?', ';')):
                    end = j
                    break
            segment_len = end - start
            if segment_len >= 6:
                pos = start + segment_len // 3
                if pos < end and not words[pos].endswith(",") and "\x00" not in words[pos]:
                    priority = self.rng.random()
                    candidates.append((priority, pos, "add_comma", None))
                # Also consider a second comma at 2/3 point for longer segments
                if segment_len >= 10:
                    pos2 = start + (2 * segment_len) // 3
                    if pos2 < end and pos2 != pos and not words[pos2].endswith(",") and "\x00" not in words[pos2]:
                        priority = self.rng.random()
                        candidates.append((priority, pos2, "add_comma", None))

        # 2. TTR synonym replacement candidates
        word_counts = {}
        for i, w in enumerate(words):
            if "\x00" in w:
                continue
            clean = re.sub(r"[^a-zA-Z]", "", w).lower()
            if len(clean) >= 4 and clean not in _FUNCTION_WORDS:
                if clean not in word_counts:
                    word_counts[clean] = []
                word_counts[clean].append(i)

        repeated = {k: v for k, v in word_counts.items() if len(v) > 1}
        sorted_repeated = sorted(repeated.items(), key=lambda x: (-len(x[1]), x[0]))

        for word, positions in sorted_repeated:
            if word not in _TTR_SYNONYMS:
                continue
            synonyms = _TTR_SYNONYMS[word]
            for pos in positions[1:]:
                if "\x00" in words[pos]:
                    continue
                priority = self.rng.random()
                synonym = synonyms[pos % len(synonyms)]
                candidates.append((priority, pos, "ttr_replace", synonym))

        # 3. Function-word SUBSTITUTION candidates — DISABLED
        # Substitutions can reduce TTR if the replacement word already exists
        # in the text, which would violate monotonicity. Since comma additions
        # and TTR replacements are sufficient for the 5% shift requirement,
        # we skip function-word substitutions.

        if not candidates:
            return text

        # --- Sort all candidates by priority ---
        candidates.sort(key=lambda x: x[0])

        # --- Apply top-K where K scales with aggression ---
        k = max(1, int(len(candidates) * self.aggression))
        to_apply = candidates[:k]

        # --- Build modification map ---
        comma_adds = set()
        ttr_replacements = {}

        for _, pos, change_type, data in to_apply:
            if change_type == "add_comma":
                comma_adds.add(pos)
            elif change_type == "ttr_replace":
                if pos not in ttr_replacements:  # first wins if conflict
                    ttr_replacements[pos] = data

        # --- Apply all changes in a single pass ---
        result_words = []
        for i, word in enumerate(words):
            if i in ttr_replacements:
                synonym = ttr_replacements[i]
                original_word = word
                # Preserve casing and trailing punctuation
                suffix = ""
                for ch in reversed(original_word):
                    if ch in ".,;:!?\"'":
                        suffix = ch + suffix
                    else:
                        break
                prefix = ""
                for ch in original_word:
                    if not ch.isalpha():
                        prefix += ch
                    else:
                        break
                if original_word and original_word[len(prefix):len(prefix) + 1].isupper():
                    synonym = synonym[0].upper() + synonym[1:]
                new_word = prefix + synonym + suffix
                # Also add comma if needed
                if i in comma_adds and not new_word.endswith(","):
                    new_word = new_word + ","
                result_words.append(new_word)
            elif i in comma_adds:
                if not word.endswith(","):
                    result_words.append(word + ",")
                else:
                    result_words.append(word)
            else:
                result_words.append(word)

        return " ".join(result_words)

    def _vary_sentence_lengths_force(self, text: str, original_variance: float) -> str:
        """Force sentence-length variance increase when standard splits aren't enough.

        This is a fallback that uses progressively more asymmetric splits
        until the 10% variance increase target is met.
        """
        sentences = self._split_into_sentences(text)
        if len(sentences) < 2:
            return text

        target = original_variance * 1.10 if original_variance > 0 else 0.01

        # Try splitting at 1/4 point for maximum asymmetry
        indexed = [(i, len(s.split()), s) for i, s in enumerate(sentences)]
        ranked = sorted(indexed, key=lambda x: (-x[1], x[0]))
        splittable = [(i, wc, s) for i, wc, s in ranked if wc >= 6]

        if not splittable:
            return text

        split_results = {}
        for orig_idx, wc, sent in splittable:
            words = sent.split()
            split_point = max(2, len(words) // 4)

            first_half = " ".join(words[:split_point])
            second_half = " ".join(words[split_point:])

            if not first_half.rstrip()[-1:] in ".!?;":
                first_half = first_half.rstrip(",") + "."
            if second_half and second_half[0].islower():
                second_half = second_half[0].upper() + second_half[1:]

            split_results[orig_idx] = [first_half, second_half]

            # Check if target is met after this split
            result_sentences = []
            for i, s in enumerate(sentences):
                if i in split_results:
                    result_sentences.extend(split_results[i])
                else:
                    result_sentences.append(s)

            new_text = " ".join(result_sentences)
            new_variance = compute_sentence_length_variance(new_text)
            if new_variance >= target:
                return new_text

        # Return whatever we got
        result_sentences = []
        for i, s in enumerate(sentences):
            if i in split_results:
                result_sentences.extend(split_results[i])
            else:
                result_sentences.append(s)
        return " ".join(result_sentences)

    def _vary_sentence_lengths(self, text: str) -> str:
        """Vary sentence lengths by splitting long sentences at a fixed point.

        Targets >= 10% increase in sentence-length variance (Req 2.2),
        or within +-2% when already at the human threshold.

        MONOTONICITY (Req 2.5): ALL splittable sentences (>= 8 words) are
        ALWAYS split at the 1/3 point, regardless of aggression. This means
        the sentence structure is identical across all aggression levels,
        ensuring that downstream word-level operations (which DO scale with
        aggression) operate on the same text and can provide monotonic
        magnitude increases independently.
        """
        sentences = self._split_into_sentences(text)
        if len(sentences) < 2:
            return text

        # Calculate current variance
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        current_variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)

        already_at_threshold = current_variance >= self.variance_threshold

        if already_at_threshold:
            # Already at human threshold: stay within +-2% (Req 2.2)
            return " ".join(sentences)

        # Fixed split ratio at 1/3 for all aggression levels
        split_ratio = 1.0 / 3.0

        # Split ALL sentences with >= 8 words (deterministic set)
        split_results = {}
        for i, sent in enumerate(sentences):
            words = sent.split()
            if len(words) < 8:
                continue

            split_point = max(3, int(len(words) * split_ratio))
            if split_point >= len(words) - 2:
                split_point = len(words) - 3

            first_half = " ".join(words[:split_point])
            second_half = " ".join(words[split_point:])

            # Ensure proper sentence endings
            if not first_half.rstrip()[-1:] in ".!?;":
                first_half = first_half.rstrip(",") + "."
            # Capitalize second half
            if second_half and second_half[0].islower():
                second_half = second_half[0].upper() + second_half[1:]

            split_results[i] = [first_half, second_half]

        if not split_results:
            return " ".join(sentences)

        # Reassemble in original order
        result_sentences = []
        for i, sent in enumerate(sentences):
            if i in split_results:
                result_sentences.extend(split_results[i])
            else:
                result_sentences.append(sent)

        return " ".join(result_sentences)

    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences, preserving placeholder tokens."""
        # Use the same approach as text_analysis.split_sentences but
        # handle placeholder tokens
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
        return [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]

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
        """Compute token Jaccard similarity as a lexical proxy."""
        tokens_a = set(re.findall(r"\b\w+\b", a.lower()))
        tokens_b = set(re.findall(r"\b\w+\b", b.lower()))

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)
