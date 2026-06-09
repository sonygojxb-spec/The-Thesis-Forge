"""
Stage 10: Adversarial Rewriting

Rewrites text specifically to evade AI-detector signals using an LLM with a
detector-evasion prompt scaled by aggression. Scores risk on both input and
candidate with the same ``detection_risk_score`` helper, and returns the input
unchanged if the candidate doesn't improve detection risk, fails the similarity
floor, or on any LLM error/timeout.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8
"""

from __future__ import annotations

import json
import random
import re
from typing import Optional

import requests

from humanizer.classifier import detection_risk_score
from humanizer.config import API_KEY, BASE_URL, DEFAULT_MODEL
from humanizer.protected_spans import ProtectedSpanGuard
from humanizer.results import StageResult


class AdversarialRewriter:
    """LLM-backed adversarial rewriter that targets detector evasion.

    Produces rewritten text whose AI-detection risk score is less than or equal
    to the input's risk score, while maintaining semantic similarity above a
    configurable floor (default 0.85). Higher aggression scales the evasion
    prompt to produce more aggressive word-level changes.

    Parameters
    ----------
    aggression : float
        Controls evasion intensity (0.0-1.0). Higher values produce more
        aggressive detector-evasion prompts and accept larger word changes.
    seed : int or None
        Optional seed for deterministic non-LLM randomization.
    model : str or None
        LLM model identifier.
    api_key : str or None
        API key for the LLM service.
    base_url : str or None
        Base URL for the LLM API.
    similarity : object or None
        A SimilarityEvaluator (or compatible) with a ``score(a, b)`` method.
        If None, uses a built-in lexical proxy (token Jaccard).
    classifier : object or None
        A Classifier (or FakeClassifier) instance for detection risk scoring.
        Passed to ``detection_risk_score``.
    floor : float
        Minimum semantic similarity between input and output (default 0.85).
    timeout_s : int
        Per-call LLM timeout in seconds (default 30).
    """

    SIMILARITY_FLOOR = 0.85

    def __init__(
        self,
        aggression: float = 0.5,
        seed: Optional[int] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        similarity=None,
        classifier=None,
        floor: float = 0.85,
        timeout_s: int = 30,
    ) -> None:
        self.aggression = aggression
        self.seed = seed
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        self.similarity = similarity
        self.classifier = classifier
        self.floor = floor
        self.timeout_s = timeout_s
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def process(self, text: str) -> str:
        """Apply adversarial rewriting and return the result text.

        Delegates to process_measured() and returns only the text.
        """
        return self.process_measured(text).text

    def process_measured(self, text: str) -> StageResult:
        """Apply adversarial rewriting with measurement metadata.

        Returns a StageResult carrying the transformed (or fallback) text
        plus risk_before, risk_after, and similarity scores.
        """
        # Empty / whitespace input → return unchanged (Req 4.8)
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

        # Score risk on input (Req 4.1)
        input_risk, _source = detection_risk_score(text, self.classifier)

        # Mask protected spans before sending to LLM (Req 4.2)
        guard = ProtectedSpanGuard()
        masked_input = guard.mask(text)

        # Call LLM with detector-evasion prompt (Req 4.7 on error/empty/timeout)
        try:
            candidate_raw = self._llm_rewrite(masked_input)
        except Exception as e:
            return StageResult(
                text=text,
                similarity=None,
                risk_before=input_risk,
                risk_after=None,
                changed=False,
                fell_back=True,
                error=str(e),
            )

        # Empty result handling (Req 4.7)
        if not candidate_raw or not candidate_raw.strip():
            return StageResult(
                text=text,
                similarity=None,
                risk_before=input_risk,
                risk_after=None,
                changed=False,
                fell_back=True,
                error="LLM returned empty result",
            )

        # Unmask protected spans
        candidate = guard.unmask(candidate_raw)

        # Check similarity (Req 4.3, 4.5)
        sim_score = self._compute_similarity(text, candidate)
        if sim_score < self.floor:
            return StageResult(
                text=text,
                similarity=sim_score,
                risk_before=input_risk,
                risk_after=None,
                changed=False,
                fell_back=True,
                error=None,
            )

        # Score risk on candidate with the SAME scorer (Req 4.1, 4.6)
        candidate_risk, _source2 = detection_risk_score(candidate, self.classifier)

        # If candidate risk > input risk, return input unchanged (Req 4.6)
        if candidate_risk > input_risk:
            return StageResult(
                text=text,
                similarity=sim_score,
                risk_before=input_risk,
                risk_after=candidate_risk,
                changed=False,
                fell_back=True,
                error=None,
            )

        # Accept the candidate
        return StageResult(
            text=candidate,
            similarity=sim_score,
            risk_before=input_risk,
            risk_after=candidate_risk,
            changed=(candidate != text),
            fell_back=False,
            error=None,
        )

    def _llm_rewrite(self, text: str) -> str:
        """Execute the adversarial rewrite via SSE streaming with 30s timeout.

        Uses the same HTTP/SSE pattern as IterativeParaphraser._llm_pass
        with a detector-evasion prompt scaled by aggression.

        Parameters
        ----------
        text : str
            The (masked) text to rewrite for detector evasion.

        Returns
        -------
        str
            The rewritten text.

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

        system_prompt = self._get_system_prompt()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Rewrite the following academic text to evade AI detection "
                        "while preserving all meaning, facts, and technical terminology. "
                        "Do not alter placeholder tokens (sequences with special characters). "
                        "Output ONLY the rewritten text, no explanations:\n\n" + text
                    ),
                },
            ],
            "temperature": self._get_temperature(),
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
                f"LLM rewrite timed out after {self.timeout_s}s"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LLM API error: {e}")

    def _get_system_prompt(self) -> str:
        """Generate the detector-evasion system prompt scaled by aggression.

        Higher aggression produces a more aggressive evasion instruction,
        encouraging greater word-change proportion (Req 4.4).
        """
        base = (
            "You are an expert academic text rewriter specializing in evading "
            "AI-text detection systems. Your goal is to rewrite the provided text "
            "so that AI detectors cannot identify it as machine-generated, while "
            "preserving all meaning, facts, data, and technical terminology exactly. "
            "Do not add or remove information. Do not change any placeholder tokens "
            "(sequences containing special characters). Output only the rewritten text."
        )

        if self.aggression <= 0.2:
            return base + (
                " Make subtle changes: vary a few sentence openings, replace "
                "some generic transitions with natural alternatives, and adjust "
                "occasional word choices. Keep changes minimal but targeted at "
                "features AI detectors rely on (uniform sentence length, "
                "predictable transitions, low perplexity)."
            )
        elif self.aggression <= 0.4:
            return base + (
                " Make moderate changes: restructure several sentences, vary "
                "sentence lengths noticeably, replace formulaic academic phrases "
                "with more natural alternatives, and introduce some voice variation "
                "(mix passive and active). Target uniform perplexity patterns and "
                "repetitive transition usage that detectors flag."
            )
        elif self.aggression <= 0.6:
            return base + (
                " Make substantial changes: restructure most sentences, significantly "
                "vary sentence lengths and complexity, replace all formulaic phrases, "
                "introduce varied discourse markers, mix sentence types (declarative, "
                "interrogative where appropriate), and break up predictable paragraph "
                "rhythms. Aggressively target the statistical uniformity that AI "
                "detectors rely on."
            )
        elif self.aggression <= 0.8:
            return base + (
                " Make aggressive changes: completely restructure sentence flow, "
                "heavily vary sentence lengths from short punchy sentences to longer "
                "complex ones, replace nearly all generic vocabulary with precise "
                "domain-specific alternatives, introduce natural imperfections in "
                "rhythm, use varied clause structures, and disrupt all statistical "
                "patterns that AI detectors use. Change word choices extensively "
                "while keeping meaning identical."
            )
        else:
            return base + (
                " Make maximum changes: completely transform the writing style while "
                "keeping meaning identical. Restructure every sentence, replace all "
                "predictable word choices, heavily vary sentence lengths and complexity, "
                "introduce highly varied discourse patterns, break every statistical "
                "regularity that AI detectors target. Use unconventional but "
                "academically valid sentence structures. Maximize word-level changes "
                "to ensure the text is unrecognizable as AI-generated while preserving "
                "all factual content and technical terminology."
            )

    def _get_temperature(self) -> float:
        """Calculate LLM temperature based on aggression.

        Higher aggression → higher temperature for more varied output.
        Range: 0.7 (low aggression) to 1.1 (high aggression).
        """
        return 0.7 + self.aggression * 0.4

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
