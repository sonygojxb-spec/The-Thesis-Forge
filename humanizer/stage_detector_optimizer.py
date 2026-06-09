"""
Stage 13: Detector-Aware Optimization

Implements a closed-loop optimization controller that iteratively generates
candidate rewrites and scores them against an AI-detection classifier, returning
the lowest-risk candidate that maintains semantic similarity above 0.85. Uses
AdversarialRewriter and IterativeParaphraser for candidate generation with
varied seeds per iteration.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8
"""

from __future__ import annotations

from typing import Optional

from humanizer.classifier import detection_risk_score
from humanizer.config import API_KEY, BASE_URL, DEFAULT_MODEL
from humanizer.protected_spans import ProtectedSpanGuard
from humanizer.results import StageResult
from humanizer.stage_adversarial import AdversarialRewriter
from humanizer.stage_iterative import IterativeParaphraser


class DetectorOptimizer:
    """Closed-loop detector-aware optimization stage.

    Iteratively generates candidate rewrites using AdversarialRewriter and
    IterativeParaphraser, scores each candidate with the classifier, and
    returns the lowest-risk candidate among those with similarity >= 0.85.

    Parameters
    ----------
    aggression : float
        Controls the aggression of internal rewriters (0.0-1.0).
    seed : int or None
        Base seed for deterministic candidate generation.
    classifier : object or None
        A Classifier (or FakeClassifier) instance for detection risk scoring.
    similarity : object or None
        A SimilarityEvaluator (or compatible) with a ``score(a, b)`` method.
    target_threshold : int
        Target detection risk score (0-100). Optimization stops when a
        candidate achieves this or lower. Default 30.
    max_iterations : int
        Maximum number of optimization iterations (1-20). Default 10.
    model : str or None
        LLM model identifier for internal rewriters.
    api_key : str or None
        API key for the LLM service.
    base_url : str or None
        Base URL for the LLM API.
    """

    SIMILARITY_FLOOR = 0.85

    def __init__(
        self,
        aggression: float = 0.5,
        seed: Optional[int] = None,
        classifier=None,
        similarity=None,
        target_threshold: int = 30,
        max_iterations: int = 10,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.aggression = aggression
        self.seed = seed
        self.classifier = classifier
        self.similarity = similarity
        self.target_threshold = max(0, min(100, target_threshold))
        self.max_iterations = max(1, min(20, max_iterations))
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL

    def process(self, text: str) -> str:
        """Apply detector-aware optimization and return the result text.

        Delegates to process_measured() and returns only the text.
        """
        return self.process_measured(text).text

    def process_measured(self, text: str) -> StageResult:
        """Apply detector-aware optimization with measurement metadata.

        Returns a StageResult carrying the optimized (or fallback) text
        plus risk_before, risk_after, similarity, and error fields.
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

        # Step 1: Score input risk (Req 8.1)
        try:
            input_risk, _source = detection_risk_score(text, self.classifier)
        except Exception as e:
            # Classifier failure on initial scoring — return input with error
            return StageResult(
                text=text,
                similarity=None,
                risk_before=None,
                risk_after=None,
                changed=False,
                fell_back=True,
                error=f"Classifier failure on input: {e}",
            )

        # Step 2: If input already meets target, return unchanged
        if input_risk <= self.target_threshold:
            return StageResult(
                text=text,
                similarity=None,
                risk_before=input_risk,
                risk_after=input_risk,
                changed=False,
                fell_back=False,
                error=None,
            )

        # Step 3: Optimization loop (Req 8.2, 8.6)
        best_candidate: Optional[str] = None
        best_risk: float = input_risk
        best_similarity: Optional[float] = None
        loop_error: Optional[str] = None

        base_seed = self.seed if self.seed is not None else 42

        for iteration in range(1, self.max_iterations + 1):
            iter_seed = base_seed + iteration

            # Generate candidate using protected spans guard
            guard = ProtectedSpanGuard()
            masked_text = guard.mask(text)

            try:
                candidate_raw = self._generate_candidate(
                    masked_text, iter_seed, iteration
                )
            except Exception:
                # Generation failure — continue to next iteration
                continue

            if not candidate_raw or not candidate_raw.strip():
                continue

            # Unmask protected spans (Req 8.7)
            candidate = guard.unmask(candidate_raw)

            # Compute similarity to original (Req 8.3, 8.4)
            sim_score = self._compute_similarity(text, candidate)

            if sim_score < self.SIMILARITY_FLOOR:
                # Candidate fails similarity gate — skip
                continue

            # Score candidate risk (Req 8.1)
            try:
                candidate_risk, _src = detection_risk_score(
                    candidate, self.classifier
                )
            except Exception as e:
                # Classifier failure mid-loop (Req 8.8)
                loop_error = f"Classifier failure at iteration {iteration}: {e}"
                break

            # Track valid candidate if it's the best so far
            if candidate_risk < best_risk:
                best_candidate = candidate
                best_risk = candidate_risk
                best_similarity = sim_score

            # Early stop if target reached (Req 8.2)
            if candidate_risk <= self.target_threshold:
                best_candidate = candidate
                best_risk = candidate_risk
                best_similarity = sim_score
                break

        # Step 4: Return best valid candidate or input (Req 8.3, 8.5)
        if best_candidate is not None:
            return StageResult(
                text=best_candidate,
                similarity=best_similarity,
                risk_before=input_risk,
                risk_after=best_risk,
                changed=(best_candidate != text),
                fell_back=False,
                error=loop_error,
            )
        else:
            # No valid candidate found (Req 8.5)
            return StageResult(
                text=text,
                similarity=None,
                risk_before=input_risk,
                risk_after=input_risk,
                changed=False,
                fell_back=True,
                error=loop_error,
            )

    def _generate_candidate(
        self, masked_text: str, seed: int, iteration: int
    ) -> str:
        """Generate a candidate rewrite using internal rewriters.

        Alternates between AdversarialRewriter (odd iterations) and
        IterativeParaphraser (even iterations) for variety.

        Parameters
        ----------
        masked_text : str
            The masked text (protected spans replaced with placeholders).
        seed : int
            Seed for this iteration (base_seed + iteration).
        iteration : int
            Current iteration number (1-based).

        Returns
        -------
        str
            The raw candidate text (still masked).
        """
        if iteration % 2 == 1:
            # Use AdversarialRewriter for odd iterations
            rewriter = AdversarialRewriter(
                aggression=self.aggression,
                seed=seed,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                similarity=self.similarity,
                classifier=self.classifier,
            )
            return rewriter.process(masked_text)
        else:
            # Use IterativeParaphraser for even iterations
            paraphraser = IterativeParaphraser(
                aggression=self.aggression,
                seed=seed,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                similarity=self.similarity,
            )
            return paraphraser.process(masked_text)

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
        import re

        tokens_a = set(re.findall(r"\b\w+\b", a.lower()))
        tokens_b = set(re.findall(r"\b\w+\b", b.lower()))

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)
