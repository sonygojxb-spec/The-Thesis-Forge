"""
Stage 6: Retrieval-Augmented Humanization

Uses retrieved human-written reference passages as style guidance in the LLM
prompt to ground rewrites in authentic human phrasing. A post-generation guard
rejects any output containing a span of more than 8 consecutive words copied
verbatim from any retrieved passage.

Requirements: 7.3, 7.5, 7.6, 7.7, 7.8, 7.9
"""

from __future__ import annotations

import json
import random
import re
from typing import List, Optional

import requests

from humanizer.config import API_KEY, BASE_URL, DEFAULT_MODEL
from humanizer.protected_spans import ProtectedSpanGuard
from humanizer.results import StageResult


class RetrievalAugmentedRewriter:
    """Retrieval-augmented rewriting using human reference passages as style guidance.

    Retrieves relevant passages from a reference corpus, uses them as style
    examples in the LLM prompt, and applies a post-generation verbatim-span
    guard to ensure no more than 8 consecutive words are copied from any
    retrieved passage.

    Parameters
    ----------
    aggression : float
        Controls rewrite intensity (0.0-1.0).
    seed : int or None
        Optional seed for deterministic non-LLM randomization.
    model : str or None
        LLM model identifier.
    api_key : str or None
        API key for the LLM service.
    base_url : str or None
        Base URL for the LLM API.
    retrieval_service : object or None
        A RetrievalService (or compatible) with a `retrieve(query_text)` method
        and a `corpus` property. If None, stage returns input unchanged.
    similarity : object or None
        A SimilarityEvaluator (or compatible) with a `score(a, b)` method.
        If None, uses a built-in lexical proxy (token Jaccard).
    floor : float
        Minimum similarity score to accept a rewrite (default 0.85).
    timeout_s : int
        Timeout in seconds for the LLM call (default 30).
    """

    VERBATIM_WORD_LIMIT = 8

    def __init__(
        self,
        aggression: float = 0.5,
        seed: Optional[int] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retrieval_service=None,
        similarity=None,
        floor: float = 0.85,
        timeout_s: int = 30,
    ) -> None:
        self.aggression = aggression
        self.seed = seed
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        self.retrieval_service = retrieval_service
        self.similarity = similarity
        self.floor = floor
        self.timeout_s = timeout_s
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def process(self, text: str) -> str:
        """Apply retrieval-augmented rewriting and return the result text.

        Delegates to process_measured() and returns only the text.
        """
        return self.process_measured(text).text

    def process_measured(self, text: str) -> StageResult:
        """Apply retrieval-augmented rewriting with measurement metadata.

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

        # No retrieval service → return unchanged
        if self.retrieval_service is None:
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error="No retrieval service configured",
            )

        # Check if corpus is empty (Req 7.4)
        try:
            corpus = self.retrieval_service.corpus
            if not corpus:
                return StageResult(
                    text=text,
                    similarity=None,
                    risk_before=None,
                    risk_after=None,
                    changed=False,
                    fell_back=True,
                    error="Reference corpus is empty",
                )
        except Exception as e:
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error=f"Retrieval service error: {e}",
            )

        # Retrieve relevant passages
        try:
            passages = self.retrieval_service.retrieve(text)
        except Exception as e:
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error=f"Retrieval error: {e}",
            )

        # No results → return unchanged (Req 7.4)
        if not passages:
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error="No relevant passages retrieved",
            )

        # Mask protected spans before sending to LLM
        guard = ProtectedSpanGuard()
        masked_input = guard.mask(text)

        # Call LLM with retrieved passages as style guidance
        try:
            rewrite_result = self._llm_rewrite(masked_input, passages)
        except Exception as e:
            # LLM error/empty/timeout → return input unchanged (Req 7.8)
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error=str(e),
            )

        # Empty result handling (Req 7.8)
        if not rewrite_result or not rewrite_result.strip():
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error="LLM returned empty result",
            )

        # Unmask protected spans
        unmasked_result = guard.unmask(rewrite_result)

        # Post-generation guard: check for verbatim spans > 8 words (Req 7.7)
        passage_texts = [p.text for p in passages]
        if self._contains_verbatim_span(unmasked_result, passage_texts):
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error="Output contains verbatim span from retrieved passage",
            )

        # Check similarity (Req 7.6, 7.9)
        sim_score = self._compute_similarity(text, unmasked_result)
        if sim_score < self.floor:
            return StageResult(
                text=text,
                similarity=sim_score,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error=f"Similarity {sim_score:.3f} below floor {self.floor}",
            )

        # Accept the rewrite
        return StageResult(
            text=unmasked_result,
            similarity=sim_score,
            risk_before=None,
            risk_after=None,
            changed=(unmasked_result != text),
            fell_back=False,
            error=None,
        )

    def _llm_rewrite(self, text: str, passages) -> str:
        """Execute an LLM rewrite using retrieved passages as style guidance.

        Uses the same HTTP/SSE streaming pattern as LLMRewriter and
        IterativeParaphraser with a configurable timeout.

        Parameters
        ----------
        text : str
            The (masked) text to rewrite.
        passages : list
            Retrieved ReferenceEntry objects to use as style guidance.

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

        system_prompt = self._get_system_prompt(passages)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Rewrite the following academic text using the style guidance "
                        "from the reference passages above. Adopt their natural phrasing "
                        "patterns, sentence rhythms, and vocabulary choices — but do NOT "
                        "copy any phrases verbatim. Preserve all meaning, facts, and "
                        "technical terminology. Output ONLY the rewritten text, no "
                        "explanations:\n\n" + text
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

            chunks: List[str] = []
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

    def _get_system_prompt(self, passages) -> str:
        """Generate the system prompt with retrieved passages as style guidance.

        The passages are presented as style examples only — the prompt
        explicitly instructs the LLM not to copy verbatim.
        """
        # Build style guidance from retrieved passages
        passage_excerpts = []
        for i, passage in enumerate(passages[:5], 1):
            # Limit passage text to avoid overly long prompts
            excerpt = passage.text[:500] if len(passage.text) > 500 else passage.text
            passage_excerpts.append(f"Example {i}: \"{excerpt}\"")

        style_examples = "\n".join(passage_excerpts)

        base = (
            "You are an expert academic writer. Your task is to rewrite "
            "academic text using the natural style, phrasing patterns, and "
            "sentence rhythms demonstrated in the following reference passages. "
            "These passages are STYLE GUIDANCE ONLY — do NOT copy any phrase "
            "of more than a few words from them.\n\n"
            f"STYLE REFERENCE PASSAGES:\n{style_examples}\n\n"
            "RULES:\n"
            "1. Adopt the natural phrasing style and sentence rhythms from the "
            "reference passages.\n"
            "2. Do NOT copy any sequence of more than 4 consecutive words from "
            "any reference passage.\n"
            "3. Preserve ALL meaning, facts, data, and technical terminology "
            "from the input text exactly.\n"
            "4. Do not change any placeholder tokens (sequences containing "
            "special characters).\n"
            "5. Output only the rewritten text — no explanations or preambles."
        )

        if self.aggression >= 0.7:
            base += (
                "\n6. Apply significant stylistic transformation — vary "
                "sentence structures broadly and adopt the reference style "
                "aggressively while preserving meaning."
            )
        elif self.aggression >= 0.4:
            base += (
                "\n6. Apply moderate stylistic changes — adopt key phrasing "
                "patterns from the references while maintaining readability."
            )
        else:
            base += (
                "\n6. Apply light stylistic adjustments — subtly incorporate "
                "reference phrasing patterns with minimal structural change."
            )

        return base

    def _get_temperature(self) -> float:
        """Calculate temperature based on aggression level."""
        return 0.7 + self.aggression * 0.3

    def _contains_verbatim_span(
        self, output: str, passage_texts: List[str]
    ) -> bool:
        """Check if output contains a verbatim span of > 8 consecutive words
        from any retrieved passage.

        Parameters
        ----------
        output : str
            The rewritten output text to check.
        passage_texts : list of str
            The text content of each retrieved passage.

        Returns
        -------
        bool
            True if a verbatim span exceeding the limit is found.
        """
        # Tokenize the output into words (lowercased for comparison)
        output_words = re.findall(r"\b\w+\b", output.lower())

        if len(output_words) <= self.VERBATIM_WORD_LIMIT:
            return False

        for passage_text in passage_texts:
            passage_words = re.findall(r"\b\w+\b", passage_text.lower())

            if len(passage_words) <= self.VERBATIM_WORD_LIMIT:
                continue

            # Build set of all contiguous word spans of length (LIMIT + 1)
            # from the passage for fast lookup
            span_length = self.VERBATIM_WORD_LIMIT + 1
            passage_spans = set()
            for i in range(len(passage_words) - span_length + 1):
                span = " ".join(passage_words[i : i + span_length])
                passage_spans.add(span)

            # Check if any span of (LIMIT + 1) words in the output matches
            for i in range(len(output_words) - span_length + 1):
                output_span = " ".join(output_words[i : i + span_length])
                if output_span in passage_spans:
                    return True

        return False

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
