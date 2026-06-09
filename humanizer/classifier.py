"""
Transformer-based AI-text classification with heuristic fallback.

Provides ``Classifier`` — a lazy-loading RoBERTa-style AI-text detector that
scores text on a 0-100 scale representing the probability of AI authorship — and
``detection_risk_score`` — a centralized helper that attempts the Classifier
first and falls back to the heuristic ``compute_ai_risk_score`` from
``humanizer.text_analysis`` on any failure.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.6
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

# Default model identifier — a RoBERTa-style AI-text detection checkpoint.
DETECTOR_MODEL = "roberta-base-openai-detector"

# Maximum allowed input length (characters).
MAX_INPUT_CHARS = 10_000


class InvalidInput(Exception):
    """Raised when input text is empty or exceeds the character limit."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class Classifier:
    """Transformer-based AI-text classifier.

    Lazily loads a RoBERTa-style sequence-classification model on the first
    call to ``score()``. The model runs in ``torch.no_grad()`` eval mode for
    deterministic, fast inference.

    Parameters
    ----------
    model_name : str
        The Hugging Face model identifier to load.
    timeout_s : float
        Maximum seconds allowed for a single inference call.
    """

    # Re-export InvalidInput as a class attribute so consumers can catch it
    # via ``Classifier.InvalidInput`` (matches FakeClassifier interface).
    InvalidInput = InvalidInput

    MAX_CHARS = MAX_INPUT_CHARS

    def __init__(
        self,
        model_name: str = DETECTOR_MODEL,
        timeout_s: float = 5.0,
    ) -> None:
        self._model_name = model_name
        self._timeout_s = timeout_s
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None
        self._model_load_attempted: bool = False
        self._model_loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True only if the model loaded successfully.

        Triggers a lazy load on first call.
        """
        self._load_model()
        return self._model_loaded

    def score(self, text: str) -> float:
        """Score text for AI-detection risk.

        Parameters
        ----------
        text : str
            The text to classify.

        Returns
        -------
        float
            A score in [0, 100] where higher means more likely AI-generated.

        Raises
        ------
        InvalidInput
            If *text* is empty/whitespace-only or exceeds 10,000 characters.
        RuntimeError
            If the model cannot be loaded or inference fails.
        """
        self._validate_input(text)
        self._load_model()

        if not self._model_loaded:
            raise RuntimeError(
                f"Classifier model '{self._model_name}' could not be loaded"
            )

        return self._infer(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input(self, text: str) -> None:
        """Raise InvalidInput for empty text or text exceeding MAX_CHARS."""
        if not text or not text.strip():
            raise InvalidInput("Input text is empty")
        if len(text) > self.MAX_CHARS:
            raise InvalidInput(
                f"Input exceeds {self.MAX_CHARS:,} characters (got {len(text):,})"
            )

    def _load_model(self) -> None:
        """Attempt to load model and tokenizer once."""
        if self._model_load_attempted:
            return
        self._model_load_attempted = True

        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            model.eval()
            self._model = model
            self._model_loaded = True
        except Exception:
            # ImportError (torch/transformers missing), OSError (model not found),
            # or any other loading failure.
            self._model = None
            self._tokenizer = None
            self._model_loaded = False

    def _infer(self, text: str) -> float:
        """Run inference with a timeout guard.

        Returns a float in [0, 100].
        """
        import torch

        result_container: list = []
        error_container: list = []

        def _run_inference() -> None:
            try:
                inputs = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                with torch.no_grad():
                    outputs = self._model(**inputs)

                # Extract probability of the "AI/Fake" class.
                # Most AI-detector models have class 0=Real, class 1=Fake/AI.
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

                # Use the last class (typically "Fake"/"AI") as the AI probability.
                ai_prob = probs[0, -1].item()

                # Scale to 0-100 and clamp.
                score = max(0.0, min(100.0, ai_prob * 100.0))
                result_container.append(score)
            except Exception as exc:
                error_container.append(exc)

        thread = threading.Thread(target=_run_inference, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout_s)

        if thread.is_alive():
            # Timeout exceeded — thread is still running.
            raise RuntimeError(
                f"Classifier inference timed out after {self._timeout_s}s"
            )

        if error_container:
            raise RuntimeError(
                f"Classifier inference error: {error_container[0]}"
            )

        if not result_container:
            raise RuntimeError("Classifier inference produced no result")

        return result_container[0]


# ---------------------------------------------------------------------------
# Centralized risk scoring with fallback
# ---------------------------------------------------------------------------


def detection_risk_score(
    text: str,
    classifier: Optional[Classifier] = None,
) -> Tuple[float, str]:
    """Compute AI detection risk score with automatic heuristic fallback.

    Attempts ``classifier.score(text)`` first. On any exception (load failure,
    inference error, timeout, InvalidInput for valid-length text that somehow
    fails), falls back to the heuristic ``compute_ai_risk_score`` from
    ``humanizer.text_analysis``.

    Parameters
    ----------
    text : str
        The text to score.
    classifier : Classifier or compatible, optional
        A classifier instance (or FakeClassifier). If None or unavailable,
        the heuristic fallback is used directly.

    Returns
    -------
    tuple of (float, str)
        A ``(score, source)`` pair where *score* is in [0, 100] and *source*
        is ``"classifier"`` or ``"heuristic"``.
    """
    # Attempt classifier path
    if classifier is not None:
        try:
            score = classifier.score(text)
            return (float(score), "classifier")
        except InvalidInput:
            # Re-raise input validation errors — callers should handle these
            # explicitly if they want to distinguish bad input from failures.
            # However, per spec the fallback centralizes *all* failures, so
            # we fall through to heuristic.
            pass
        except Exception:
            # Load failure, inference error, timeout, etc.
            pass

    # Heuristic fallback
    from humanizer.text_analysis import compute_ai_risk_score

    heuristic_score = compute_ai_risk_score(text)
    # Ensure the score is within [0, 100]
    clamped = max(0.0, min(100.0, float(heuristic_score)))
    return (clamped, "heuristic")
