"""
Stage 4: Iterative Paraphrasing

Performs multiple controlled passes of LLM-based paraphrasing so that output
progressively diverges from the original AI phrasing while retaining meaning.
Each pass feeds its output into the next; passes whose similarity vs the
original stage input drops below 0.80 are discarded. A 30-second per-pass
timeout is enforced.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9
"""

from __future__ import annotations

import json
import random
import re
from typing import Optional

import requests

from humanizer.config import API_KEY, BASE_URL, DEFAULT_MODEL
from humanizer.protected_spans import ProtectedSpanGuard
from humanizer.results import StageResult


class IterativeParaphraser:
    """Multi-pass iterative paraphrasing with similarity gating.

    Each pass uses the LLM to paraphrase the text, feeding the previous
    pass's output as input to the next. Passes whose similarity to the
    original stage input falls below 0.80 are discarded and the previous
    good output is retained.

    Parameters
    ----------
    aggression : float
        Controls number of passes: pass_count = 1 + round(aggression * 4).
        Range 0.0-1.0.
    seed : int or None
        Optional seed for deterministic non-LLM randomized selection.
    model : str or None
        LLM model identifier.
    api_key : str or None
        API key for the LLM service.
    base_url : str or None
        Base URL for the LLM API.
    similarity : object or None
        A SimilarityEvaluator (or compatible) with a `score(a, b)` method.
        If None, uses a built-in lexical proxy (token Jaccard).
    timeout_s : int
        Per-pass timeout in seconds (default 30).
    """

    SIMILARITY_FLOOR = 0.80

    def __init__(
        self,
        aggression: float = 0.5,
        seed: Optional[int] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        similarity=None,
        timeout_s: int = 30,
    ) -> None:
        self.aggression = aggression
        self.seed = seed
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        self.similarity = similarity
        self.timeout_s = timeout_s
        self.rng = random.Random(seed) if seed is not None else random.Random()

    @property
    def pass_count(self) -> int:
        """Number of paraphrasing passes: 1 + round(aggression * 4)."""
        return 1 + round(self.aggression * 4)

    def process(self, text: str) -> str:
        """Apply iterative paraphrasing and return the result text.

        Delegates to process_measured() and returns only the text.
        """
        return self.process_measured(text).text

    def process_measured(self, text: str) -> StageResult:
        """Apply iterative paraphrasing with measurement metadata.

        Returns a StageResult carrying the transformed (or fallback) text
        plus the computed similarity score.
        """
        # Empty / whitespace input → return unchanged
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

        stage_input = text
        last_good_output: Optional[str] = None
        current_input = text
        last_error: Optional[str] = None

        guard = ProtectedSpanGuard()

        for pass_idx in range(self.pass_count):
            # Mask protected spans before sending to LLM
            masked_input = guard.mask(current_input)

            try:
                pass_result = self._llm_pass(masked_input, pass_idx)
            except Exception as e:
                last_error = str(e)
                # LLM error/timeout: if we have a prior success, use it (Req 1.6)
                if last_good_output is not None:
                    break
                # First-pass failure with no prior → return original (Req 1.8)
                if pass_idx == 0:
                    return StageResult(
                        text=stage_input,
                        similarity=None,
                        risk_before=None,
                        risk_after=None,
                        changed=False,
                        fell_back=True,
                        error=last_error,
                    )
                break

            # Empty result handling
            if not pass_result or not pass_result.strip():
                last_error = "LLM returned empty result"
                if last_good_output is not None:
                    break
                if pass_idx == 0:
                    return StageResult(
                        text=stage_input,
                        similarity=None,
                        risk_before=None,
                        risk_after=None,
                        changed=False,
                        fell_back=True,
                        error=last_error,
                    )
                break

            # Unmask protected spans
            unmasked_result = guard.unmask(pass_result)

            # Check similarity against the ORIGINAL stage input (Req 1.5)
            sim_score = self._compute_similarity(stage_input, unmasked_result)

            if sim_score < self.SIMILARITY_FLOOR:
                # Discard this pass, keep previous good output (Req 1.5)
                # Don't continue further passes since divergence is increasing
                break
            else:
                # Accept this pass
                last_good_output = unmasked_result
                # Feed this pass's output as input to the next pass (Req 1.3)
                current_input = unmasked_result
                # Re-create guard for next pass so placeholders are fresh
                guard = ProtectedSpanGuard()

        # Determine final output
        if last_good_output is not None:
            final_similarity = self._compute_similarity(stage_input, last_good_output)
            return StageResult(
                text=last_good_output,
                similarity=final_similarity,
                risk_before=None,
                risk_after=None,
                changed=(last_good_output != stage_input),
                fell_back=False,
                error=last_error,
            )
        else:
            # No successful pass at all
            return StageResult(
                text=stage_input,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error=last_error,
            )

    def _llm_pass(self, text: str, pass_index: int) -> str:
        """Execute a single LLM paraphrasing pass via SSE streaming.

        Uses the same HTTP/SSE pattern as LLMRewriter._llm_pass_stream
        with a configurable per-pass timeout.

        Parameters
        ----------
        text : str
            The (masked) text to paraphrase.
        pass_index : int
            Zero-based pass index, used to vary the prompt slightly.

        Returns
        -------
        str
            The paraphrased text.

        Raises
        ------
        RuntimeError
            On LLM API error, timeout, or empty response.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        endpoint = f"{self.base_url.rstrip('/')}/v1/chat/completions"

        system_prompt = self._get_system_prompt(pass_index)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Paraphrase the following academic text. Preserve all meaning, "
                        "facts, and technical terminology. Change sentence structures, "
                        "word choices, and phrasing while keeping the content identical. "
                        "Output ONLY the paraphrased text, no explanations:\n\n" + text
                    ),
                },
            ],
            "temperature": self._get_temperature(pass_index),
            "stream": True,
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                stream=True,
                timeout=self.timeout_s,
            )
            response.raise_for_status()

            chunks = []
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    chunks.append(delta["content"])
                        except json.JSONDecodeError:
                            continue

            result = "".join(chunks)
            if not result.strip():
                raise RuntimeError("LLM returned empty result")
            return result

        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"LLM pass {pass_index + 1} timed out after {self.timeout_s}s"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LLM API error: {e}")

    def _get_system_prompt(self, pass_index: int) -> str:
        """Generate the system prompt for a paraphrasing pass.

        Later passes emphasize more structural variation while maintaining
        meaning preservation.
        """
        base = (
            "You are an expert academic paraphraser. Your task is to rewrite "
            "academic text using different sentence structures, vocabulary, and "
            "phrasing while preserving all meaning, facts, data, and technical "
            "terminology exactly. Do not add or remove information. Do not change "
            "any placeholder tokens (sequences containing special characters). "
            "Output only the paraphrased text."
        )

        if pass_index == 0:
            return base + (
                " Focus on varying sentence structure and replacing generic "
                "vocabulary with precise alternatives."
            )
        elif pass_index == 1:
            return base + (
                " Focus on restructuring clause order and using different "
                "transitional approaches while maintaining academic rigor."
            )
        elif pass_index == 2:
            return base + (
                " Focus on varying sentence lengths — mix short direct sentences "
                "with longer complex ones. Change passive to active voice and "
                "vice versa where appropriate."
            )
        elif pass_index == 3:
            return base + (
                " Focus on replacing common academic phrasings with less "
                "formulaic alternatives. Use varied discourse markers."
            )
        else:
            return base + (
                " Apply comprehensive rephrasing: vary syntax, vocabulary, "
                "sentence rhythm, and discourse structure."
            )

    def _get_temperature(self, pass_index: int) -> float:
        """Calculate temperature for a given pass.

        Slightly increases with pass index to encourage more variation
        in later passes, capped at 1.2.
        """
        base_temp = 0.7 + self.aggression * 0.2
        return min(1.2, base_temp + pass_index * 0.05)

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
