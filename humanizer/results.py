"""
Data models for stage results and perplexity profiling.

Provides the StageResult dataclass for per-stage analytics (similarity, risk,
fallback status) and the TargetPerplexityProfile for perplexity optimization
configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StageResult:
    """Result of a measured stage execution.

    Carries the transformed text plus measurement metadata so the pipeline
    orchestrator can enforce discard-on-violation semantics and the UI can
    display per-stage analytics, while the plain ``process()`` method continues
    to return only the text string.

    Attributes:
        text: Transformed (or fallback) text produced by the stage.
        similarity: Semantic similarity score (0.0-1.0) between the stage input
            and the output text. None if similarity was not computed.
        risk_before: Detection risk score (0-100) of the stage input text.
            None if risk was not measured.
        risk_after: Detection risk score (0-100) of the output text.
            None if risk was not measured.
        changed: True if the output text differs from the stage input.
        fell_back: True if a violation or error forced fallback to the
            stage input text.
        error: Error description if an error occurred during processing,
            otherwise None.
    """

    text: str
    similarity: Optional[float]
    risk_before: Optional[float]
    risk_after: Optional[float]
    changed: bool
    fell_back: bool
    error: Optional[str]


@dataclass
class TargetPerplexityProfile:
    """Target perplexity profile for the Perplexity Optimizer stage.

    Specifies the desired mean perplexity and cross-sentence perplexity
    variance that the optimizer should steer text toward.

    Attributes:
        target_mean: Target mean perplexity value. Must be greater than 0.
        target_variance: Target perplexity variance (burstiness). Must be
            greater than or equal to 0.
    """

    target_mean: float
    target_variance: float

    def __post_init__(self) -> None:
        """Validate constraints on target values."""
        if self.target_mean <= 0:
            raise ValueError(
                f"target_mean must be greater than 0, got {self.target_mean}"
            )
        if self.target_variance < 0:
            raise ValueError(
                f"target_variance must be >= 0, got {self.target_variance}"
            )


# Module-level default profile representing typical human academic writing.
# Mean perplexity ~60 and moderate variance ~15 are representative values
# for human-authored academic text.
DEFAULT_PERPLEXITY_PROFILE = TargetPerplexityProfile(
    target_mean=60.0,
    target_variance=15.0,
)
