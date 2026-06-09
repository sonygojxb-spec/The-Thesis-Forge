"""
Stage 11: Human-Like Error Injection

Injects controlled natural imperfections — minor punctuation variations,
whitespace variations, and informal word-form substitutions — so that the
output exhibits the minor irregularities characteristic of human writing.

The injection rate is monotonic in aggression and capped at
floor(0.05 * word_count) words. Numbers, citations, quoted content, and
Protected_Terms are never altered (masked via ProtectedSpanGuard).

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
"""

from __future__ import annotations

import math
import random
import re
from typing import Optional

from humanizer.protected_spans import ProtectedSpanGuard
from humanizer.results import StageResult


# Informal word-form substitution map (formal → informal)
_INFORMAL_SUBSTITUTIONS = {
    "which": "that",
    "cannot": "can't",
    "will not": "won't",
    "do not": "don't",
    "does not": "doesn't",
    "is not": "isn't",
    "are not": "aren't",
    "would not": "wouldn't",
    "should not": "shouldn't",
    "could not": "couldn't",
    "have not": "haven't",
    "has not": "hasn't",
    "had not": "hadn't",
    "it is": "it's",
    "that is": "that's",
    "there is": "there's",
    "what is": "what's",
    "who is": "who's",
    "they are": "they're",
    "we are": "we're",
    "you are": "you're",
    "i am": "I'm",
    "upon": "on",
    "utilize": "use",
    "utilise": "use",
    "commence": "start",
    "terminate": "end",
    "approximately": "about",
    "furthermore": "also",
    "nevertheless": "still",
    "however": "but",
    "therefore": "so",
    "subsequently": "then",
    "additionally": "also",
    "regarding": "about",
    "numerous": "many",
    "sufficient": "enough",
    "prior to": "before",
    "in order to": "to",
    "due to the fact that": "because",
}

# Single-word informal substitutions (for word-level replacement)
_SINGLE_WORD_SUBSTITUTIONS = {
    "which": "that",
    "cannot": "can't",
    "upon": "on",
    "utilize": "use",
    "utilise": "use",
    "commence": "start",
    "terminate": "end",
    "approximately": "about",
    "furthermore": "also",
    "nevertheless": "still",
    "however": "but",
    "therefore": "so",
    "subsequently": "then",
    "additionally": "also",
    "regarding": "about",
    "numerous": "many",
    "sufficient": "enough",
}


class ErrorInjector:
    """Injects controlled human-like imperfections into text.

    Produces minor punctuation variations, whitespace variations, and
    informal word-form substitutions at a rate monotonic in aggression,
    capped at floor(0.05 * word_count) words.

    Parameters
    ----------
    aggression : float
        Controls injection intensity (0.0-1.0). Higher values inject more
        imperfections. At 0.0, text is returned unchanged.
    seed : int or None
        Optional seed for deterministic behaviour via random.Random(seed).
    max_alter_ratio : float
        Maximum proportion of words that may be altered (default 0.05 = 5%).
    """

    def __init__(
        self,
        aggression: float = 0.5,
        seed: Optional[int] = None,
        max_alter_ratio: float = 0.05,
    ) -> None:
        self.aggression = aggression
        self.seed = seed
        self.max_alter_ratio = max_alter_ratio
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def process(self, text: str) -> str:
        """Apply error injection and return the result text.

        Delegates to process_measured() and returns only the text.
        """
        result = self.process_measured(text)
        return result.text

    def process_measured(self, text: str) -> StageResult:
        """Apply error injection with measurement metadata.

        Returns a StageResult carrying the transformed (or unchanged) text.
        No similarity check is needed — this is controlled injection.
        """
        # Empty/whitespace → unchanged (Req 5.7)
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

        # Aggression 0.0 → unchanged (Req 5.5)
        if self.aggression == 0.0:
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=False,
                error=None,
            )

        # Compute the maximum number of words to alter
        word_count = len(text.split())
        max_cap = math.floor(self.max_alter_ratio * word_count)

        # If max_cap < 1 → zero alterations → unchanged (Req 5.8)
        if max_cap < 1:
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=False,
                error=None,
            )

        # Actual number of alterations = floor(aggression * max_cap)
        # This is monotonic in aggression (Req 5.1)
        num_alterations = math.floor(self.aggression * max_cap)

        if num_alterations < 1:
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=False,
                error=None,
            )

        # Mask protected spans (Req 5.3, 5.4)
        guard = ProtectedSpanGuard()
        masked = guard.mask(text)

        # Apply injections
        injected = self._inject_errors(masked, num_alterations)

        # Unmask protected spans
        result = guard.unmask(injected)

        changed = result != text
        return StageResult(
            text=result,
            similarity=None,
            risk_before=None,
            risk_after=None,
            changed=changed,
            fell_back=False,
            error=None,
        )

    def _inject_errors(self, text: str, num_alterations: int) -> str:
        """Apply the specified number of error injections to the masked text.

        Injection types are randomly chosen from:
        1. Punctuation variations (extra/missing commas, semicolons→commas)
        2. Whitespace variations (double spaces)
        3. Informal word-form substitutions
        """
        alterations_made = 0

        # Split into words while preserving whitespace structure
        words = text.split(" ")

        # Identify eligible word indices (not protected placeholders)
        eligible_indices = []
        for i, word in enumerate(words):
            if "\x00" not in word and word.strip():
                eligible_indices.append(i)

        if not eligible_indices:
            return text

        # Shuffle eligible indices for random selection
        self.rng.shuffle(eligible_indices)

        for idx in eligible_indices:
            if alterations_made >= num_alterations:
                break

            word = words[idx]
            # Choose injection type randomly
            injection_type = self.rng.choice(["punctuation", "whitespace", "substitution"])

            new_word = self._apply_injection(word, injection_type)
            if new_word != word:
                words[idx] = new_word
                alterations_made += 1

        # If we haven't made enough alterations, do a second pass with
        # different injection types
        if alterations_made < num_alterations:
            self.rng.shuffle(eligible_indices)
            for idx in eligible_indices:
                if alterations_made >= num_alterations:
                    break

                word = words[idx]
                # Try a different injection type
                for injection_type in ["punctuation", "substitution", "whitespace"]:
                    new_word = self._apply_injection(word, injection_type)
                    if new_word != word:
                        words[idx] = new_word
                        alterations_made += 1
                        break

        return " ".join(words)

    def _apply_injection(self, word: str, injection_type: str) -> str:
        """Apply a single injection of the specified type to a word."""
        if injection_type == "punctuation":
            return self._punctuation_variation(word)
        elif injection_type == "whitespace":
            return self._whitespace_variation(word)
        elif injection_type == "substitution":
            return self._informal_substitution(word)
        return word

    def _punctuation_variation(self, word: str) -> str:
        """Apply a minor punctuation variation to a word.

        Variations include:
        - Removing a trailing comma
        - Adding a trailing comma
        - Converting semicolons to commas
        - Removing a trailing semicolon
        """
        # Convert semicolon to comma
        if word.endswith(";"):
            return word[:-1] + ","

        # Remove trailing comma (sometimes humans forget commas)
        if word.endswith(",") and len(word) > 1:
            choice = self.rng.random()
            if choice < 0.5:
                return word[:-1]

        # Add a trailing comma to a plain word (humans sometimes add extra commas)
        if (
            word
            and word[-1].isalpha()
            and len(word) > 2
            and not word.endswith(",")
        ):
            return word + ","

        return word

    def _whitespace_variation(self, word: str) -> str:
        """Apply a whitespace variation — insert a double space after the word.

        This simulates the common human typo of hitting the spacebar twice.
        The extra space is appended as part of the word token since we join
        with single spaces.
        """
        if word and word[-1].isalpha() and len(word) > 1:
            return word + " "
        return word

    def _informal_substitution(self, word: str) -> str:
        """Replace a formal word with its informal equivalent.

        Only substitutes single words from the substitution map.
        Preserves original capitalization.
        """
        # Extract the alphabetic core and surrounding punctuation
        prefix = ""
        suffix = ""
        core = word

        # Strip leading punctuation
        while core and not core[0].isalpha():
            prefix += core[0]
            core = core[1:]

        # Strip trailing punctuation
        while core and not core[-1].isalpha():
            suffix = core[-1] + suffix
            core = core[:-1]

        if not core:
            return word

        # Check for substitution match (case-insensitive lookup)
        lower_core = core.lower()
        if lower_core in _SINGLE_WORD_SUBSTITUTIONS:
            replacement = _SINGLE_WORD_SUBSTITUTIONS[lower_core]

            # Preserve capitalization
            if core[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]

            return prefix + replacement + suffix

        return word
