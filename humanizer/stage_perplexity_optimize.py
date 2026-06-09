"""
Stage 9: Perplexity Optimization

Tunes text toward a Target_Perplexity_Profile by greedily applying
candidate edits (word simplification / complexification) to individual
sentences. Only accepts edits that do not increase the absolute distance
to the target mean perplexity, and (when >=2 sentences) do not increase
the distance to the target cross-sentence variance.

This is an NLP-only stage (no LLM dependency). It uses deterministic
random.Random(seed) for reproducibility and ProtectedSpanGuard for term
protection.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9
"""

from __future__ import annotations

import random
import re
from typing import Optional

from humanizer.config import PROTECTED_TERMS
from humanizer.protected_spans import ProtectedSpanGuard
from humanizer.results import StageResult, TargetPerplexityProfile, DEFAULT_PERPLEXITY_PROFILE
from humanizer.text_analysis import estimate_perplexity_score, split_sentences

# Default tolerances: within 5% of target mean and 10% of target variance
# means text is already "close enough" and should be returned unchanged (Req 3.5).
PPX_MEAN_TOL = 0.05
PPX_VAR_TOL = 0.10

# Word-level simplification map: complex -> simpler (lower perplexity)
_SIMPLIFY_MAP = {
    "utilize": "use",
    "demonstrate": "show",
    "approximately": "about",
    "subsequently": "then",
    "furthermore": "also",
    "nevertheless": "still",
    "consequently": "so",
    "methodology": "method",
    "implementation": "setup",
    "facilitate": "help",
    "endeavor": "try",
    "ascertain": "find",
    "comprehend": "grasp",
    "magnitude": "size",
    "numerous": "many",
    "commence": "start",
    "terminate": "end",
    "sufficient": "enough",
    "primarily": "mainly",
    "additional": "more",
    "substantial": "large",
    "significant": "big",
    "fundamental": "basic",
    "preliminary": "early",
    "subsequent": "next",
    "preceding": "prior",
    "regarding": "about",
    "concerning": "about",
    "aforementioned": "stated",
    "notwithstanding": "despite",
    "hitherto": "until now",
    "whereby": "where",
    "thereof": "of it",
    "therein": "in it",
    "whereas": "while",
    "albeit": "though",
    "inasmuch": "since",
    "heretofore": "before",
}

# Word-level complexification map: simple -> more complex (higher perplexity)
_COMPLEXIFY_MAP = {
    "use": "utilize",
    "show": "demonstrate",
    "about": "approximately",
    "then": "subsequently",
    "also": "furthermore",
    "still": "nevertheless",
    "so": "consequently",
    "method": "methodology",
    "help": "facilitate",
    "try": "endeavor",
    "find": "ascertain",
    "many": "numerous",
    "start": "commence",
    "end": "terminate",
    "enough": "sufficient",
    "mainly": "primarily",
    "more": "additional",
    "large": "substantial",
    "big": "significant",
    "basic": "fundamental",
    "early": "preliminary",
    "next": "subsequent",
    "prior": "preceding",
    "while": "whereas",
    "though": "albeit",
    "since": "inasmuch",
    "before": "heretofore",
    "get": "obtain",
    "give": "provide",
    "make": "construct",
    "keep": "maintain",
    "need": "require",
    "seem": "appear",
    "look": "examine",
    "work": "function",
    "fast": "expeditious",
    "hard": "arduous",
    "clear": "unambiguous",
    "good": "efficacious",
}


class PerplexityOptimizer:
    """Tunes text toward a target perplexity profile via greedy editing.

    Accepts a TargetPerplexityProfile and greedily applies candidate edits
    (simplify/complexify words) to individual sentences, accepting only
    edits that reduce distance to the target mean (and target variance
    for >=2 sentences).

    Parameters
    ----------
    aggression : float
        Controls transformation intensity (0.0-1.0). Higher values allow
        more candidate edits per sentence.
    seed : int or None
        Optional seed for deterministic behaviour via random.Random(seed).
    similarity : object or None
        A SimilarityEvaluator (or compatible fake) with a `score(a, b)` method.
        If None, uses a built-in lexical proxy (token Jaccard).
    floor : float
        Minimum similarity threshold (default 0.85). Candidates below this
        are discarded.
    target_profile : TargetPerplexityProfile or None
        Target perplexity profile. Uses DEFAULT_PERPLEXITY_PROFILE if None.
    mean_tol : float
        Fractional tolerance for target mean (default 0.05 = 5%).
    var_tol : float
        Fractional tolerance for target variance (default 0.10 = 10%).
    """

    def __init__(
        self,
        aggression: float = 0.5,
        seed: Optional[int] = None,
        similarity=None,
        floor: float = 0.85,
        target_profile: Optional[TargetPerplexityProfile] = None,
        mean_tol: float = PPX_MEAN_TOL,
        var_tol: float = PPX_VAR_TOL,
    ) -> None:
        self.aggression = aggression
        self.seed = seed
        self.similarity = similarity
        self.floor = floor
        self.target_profile = target_profile or DEFAULT_PERPLEXITY_PROFILE
        self.mean_tol = mean_tol
        self.var_tol = var_tol
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def process(self, text: str) -> str:
        """Apply perplexity optimization and return the result text."""
        result = self.process_measured(text)
        return result.text

    def process_measured(self, text: str) -> StageResult:
        """Apply perplexity optimization with measurement metadata."""
        # Empty/whitespace input → unchanged (Req 3.8)
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

        # Measure input perplexity
        sentences = split_sentences(text)
        if not sentences:
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=False,
                error=None,
            )

        # Compute per-sentence perplexity scores
        scores = [estimate_perplexity_score(s) for s in sentences]

        # Check if perplexity is unmeasurable (Req 3.9)
        # If all scores are the neutral default (50.0) and wordfreq isn't
        # providing real data, treat as unmeasurable.
        if not self._is_perplexity_measurable(sentences, scores):
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=False,
                error=None,
            )

        target_mean = self.target_profile.target_mean
        target_variance = self.target_profile.target_variance

        input_mean = sum(scores) / len(scores)
        input_variance = self._compute_variance(scores)

        # Check if already within tolerances (Req 3.5)
        mean_within = abs(input_mean - target_mean) <= target_mean * self.mean_tol
        var_within = (
            len(sentences) < 2
            or abs(input_variance - target_variance) <= target_variance * self.var_tol + 0.001
        )

        if mean_within and var_within:
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=False,
                error=None,
            )

        # Attempt greedy optimization
        try:
            candidate = self._optimize(text, sentences, scores)
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
                fell_back=False,
                error=None,
            )

        # Enforce similarity floor (Req 3.6)
        score = self._compute_similarity(text, candidate)
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

    def _optimize(self, text: str, sentences: list, scores: list) -> str:
        """Greedy optimization: try edits on each sentence, accept only if
        they reduce distance to target mean and target variance.

        Uses ProtectedSpanGuard to preserve protected spans (Req 3.4).
        """
        target_mean = self.target_profile.target_mean
        target_variance = self.target_profile.target_variance

        current_scores = list(scores)
        current_sentences = list(sentences)

        # Determine how many edit attempts to make based on aggression
        max_edits_per_sentence = max(1, int(1 + self.aggression * 4))

        # Iterate over sentences, attempting edits
        for sent_idx in range(len(current_sentences)):
            for _ in range(max_edits_per_sentence):
                current_mean = sum(current_scores) / len(current_scores)
                current_mean_dist = abs(current_mean - target_mean)

                if len(current_scores) >= 2:
                    current_var = self._compute_variance(current_scores)
                    current_var_dist = abs(current_var - target_variance)
                else:
                    current_var = 0.0
                    current_var_dist = 0.0

                # Determine direction: should this sentence's perplexity go up or down?
                sent_score = current_scores[sent_idx]
                if current_mean < target_mean:
                    # Need higher perplexity overall → try complexifying
                    candidate_sent = self._try_complexify(current_sentences[sent_idx])
                elif current_mean > target_mean:
                    # Need lower perplexity overall → try simplifying
                    candidate_sent = self._try_simplify(current_sentences[sent_idx])
                else:
                    # Mean is at target; try adjusting variance if needed
                    if len(current_scores) >= 2 and current_var < target_variance:
                        # Need more variance → push this sentence away from mean
                        if sent_score >= current_mean:
                            candidate_sent = self._try_complexify(current_sentences[sent_idx])
                        else:
                            candidate_sent = self._try_simplify(current_sentences[sent_idx])
                    elif len(current_scores) >= 2 and current_var > target_variance:
                        # Need less variance → push this sentence toward mean
                        if sent_score > current_mean:
                            candidate_sent = self._try_simplify(current_sentences[sent_idx])
                        else:
                            candidate_sent = self._try_complexify(current_sentences[sent_idx])
                    else:
                        break  # Already optimal for this sentence

                # If no change was made, skip
                if candidate_sent == current_sentences[sent_idx]:
                    break

                # Score the candidate sentence
                candidate_score = estimate_perplexity_score(candidate_sent)

                # Build candidate score list
                candidate_scores = list(current_scores)
                candidate_scores[sent_idx] = candidate_score

                candidate_mean = sum(candidate_scores) / len(candidate_scores)
                candidate_mean_dist = abs(candidate_mean - target_mean)

                # Check mean distance guarantee (Req 3.2): must not increase
                if candidate_mean_dist > current_mean_dist + 1e-9:
                    break  # Reject this edit

                # Check variance distance guarantee (Req 3.3) for >=2 sentences
                if len(candidate_scores) >= 2:
                    candidate_var = self._compute_variance(candidate_scores)
                    candidate_var_dist = abs(candidate_var - target_variance)
                    if candidate_var_dist > current_var_dist + 1e-9:
                        break  # Reject this edit

                # Accept the edit
                current_sentences[sent_idx] = candidate_sent
                current_scores[sent_idx] = candidate_score

        # Reconstruct text from modified sentences
        return self._reconstruct_text(text, sentences, current_sentences)

    def _try_simplify(self, sentence: str) -> str:
        """Try to simplify a sentence by replacing complex words with simpler ones.

        Uses ProtectedSpanGuard to protect spans, then applies word substitution.
        """
        guard = ProtectedSpanGuard()
        masked = guard.mask(sentence)

        words = masked.split()
        if not words:
            return sentence

        # Find candidate words to simplify
        candidates = []
        for i, word in enumerate(words):
            if "\x00" in word:
                continue
            clean = re.sub(r"[^a-zA-Z]", "", word).lower()
            if clean in _SIMPLIFY_MAP:
                replacement = _SIMPLIFY_MAP[clean]
                # Skip if replacement would introduce a protected term (Req 3.4)
                if replacement.lower() in PROTECTED_TERMS:
                    continue
                candidates.append((i, clean))

        if not candidates:
            return sentence

        # Pick one candidate deterministically based on rng
        idx, clean = candidates[self.rng.randint(0, len(candidates) - 1)]
        replacement = _SIMPLIFY_MAP[clean]

        # Preserve casing and punctuation
        original_word = words[idx]
        new_word = self._preserve_format(original_word, replacement)
        words[idx] = new_word

        result = guard.unmask(" ".join(words))
        return result

    def _try_complexify(self, sentence: str) -> str:
        """Try to complexify a sentence by replacing simple words with complex ones.

        Uses ProtectedSpanGuard to protect spans, then applies word substitution.
        """
        guard = ProtectedSpanGuard()
        masked = guard.mask(sentence)

        words = masked.split()
        if not words:
            return sentence

        # Find candidate words to complexify
        candidates = []
        for i, word in enumerate(words):
            if "\x00" in word:
                continue
            clean = re.sub(r"[^a-zA-Z]", "", word).lower()
            if clean in _COMPLEXIFY_MAP:
                replacement = _COMPLEXIFY_MAP[clean]
                # Skip if replacement would introduce a protected term (Req 3.4)
                if replacement.lower() in PROTECTED_TERMS:
                    continue
                candidates.append((i, clean))

        if not candidates:
            return sentence

        # Pick one candidate deterministically based on rng
        idx, clean = candidates[self.rng.randint(0, len(candidates) - 1)]
        replacement = _COMPLEXIFY_MAP[clean]

        # Preserve casing and punctuation
        original_word = words[idx]
        new_word = self._preserve_format(original_word, replacement)
        words[idx] = new_word

        result = guard.unmask(" ".join(words))
        return result

    @staticmethod
    def _preserve_format(original_word: str, replacement: str) -> str:
        """Preserve the casing and surrounding punctuation of the original word."""
        # Extract leading non-alpha
        prefix = ""
        for ch in original_word:
            if ch.isalpha():
                break
            prefix += ch

        # Extract trailing non-alpha
        suffix = ""
        for ch in reversed(original_word):
            if ch.isalpha():
                break
            suffix = ch + suffix

        # Extract the alphabetic core for casing reference
        core = original_word[len(prefix): len(original_word) - len(suffix) if suffix else len(original_word)]

        # Match casing
        if core and core[0].isupper():
            if core.isupper() and len(core) > 1:
                replacement = replacement.upper()
            else:
                replacement = replacement[0].upper() + replacement[1:]

        return prefix + replacement + suffix

    def _reconstruct_text(self, original: str, original_sentences: list, modified_sentences: list) -> str:
        """Reconstruct text preserving paragraph structure and spacing.

        Replaces each modified sentence in the original text.
        """
        result = original
        for orig_sent, mod_sent in zip(original_sentences, modified_sentences):
            if orig_sent != mod_sent:
                result = result.replace(orig_sent, mod_sent, 1)
        return result

    def _is_perplexity_measurable(self, sentences: list, scores: list) -> bool:
        """Check if perplexity is actually measurable (Req 3.9).

        If all sentences return the neutral default (50.0), it means wordfreq
        is not installed or no words could be scored.
        """
        # If all scores are exactly 50.0 (the neutral fallback), perplexity
        # is unmeasurable
        if all(s == 50.0 for s in scores):
            # Double-check: if all sentences are non-empty but all return 50.0,
            # this means HAS_WORDFREQ is False
            return False
        return True

    @staticmethod
    def _compute_variance(scores: list) -> float:
        """Compute variance of a list of scores."""
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        return sum((s - mean) ** 2 for s in scores) / len(scores)

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
