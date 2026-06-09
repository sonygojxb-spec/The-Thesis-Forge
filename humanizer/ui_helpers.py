"""
UI helper functions extracted from app.py for testability.

These are pure functions that encapsulate logic used by the Streamlit frontend
for building pipeline configuration, computing analytics payloads, managing
export state, and displaying the disclaimer.

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8
"""

from typing import Dict, Optional


# --- Disclaimer constant (Req 12.5) ---

DISCLAIMER_TEXT = (
    "Scores are estimates based on heuristic analysis and may not "
    "match specific AI detection tools."
)


def build_stage_overrides(toggles: Dict[str, bool]) -> Dict[str, bool]:
    """Build the stage_overrides dict from UI toggle state.

    Maps UI toggle keys to their corresponding pipeline stage keys.
    Each toggle key corresponds directly to a pipeline stage name.

    Args:
        toggles: A dict mapping stage names to their enabled/disabled boolean.
            Expected keys include the original five stages (structural, lexical,
            llm_rewrite, perplexity, postprocess) and the nine new capability
            stages (semantic_transform, iterative_paraphrase, retrieval_augmented,
            stylometric, perplexity_optimize, adversarial, error_injection,
            detector_optimize, classifier).

    Returns:
        A dict suitable for passing as stage_overrides to HumanizationPipeline.

    Validates: Requirements 12.1
    """
    # Pass through the toggle dict as-is — each key maps 1:1 to a pipeline stage.
    # The pipeline's __init__ handles the distinction between existing and new stages.
    return dict(toggles)


def compute_analytics_payload(
    before_score: float,
    after_score: float,
    similarity: Optional[float],
) -> Dict[str, object]:
    """Structure analytics data for display in the UI.

    Computes the score change and packages before/after detection risk scores
    and the semantic similarity score into a single dict for rendering.

    Args:
        before_score: Detection risk score (0-100) of the original input text.
        after_score: Detection risk score (0-100) of the final output text.
        similarity: Semantic similarity score (0.0-1.0) between original and
            final text, or None if not available.

    Returns:
        A dict with keys:
            - "before_score": float (0-100)
            - "after_score": float (0-100)
            - "score_change": float (after - before, negative is improvement)
            - "similarity": float or None (0.0-1.0)

    Validates: Requirements 12.3, 12.4
    """
    return {
        "before_score": before_score,
        "after_score": after_score,
        "score_change": after_score - before_score,
        "similarity": similarity,
    }


def build_export_payload(text: str) -> str:
    """Return the text content for export as a downloadable file.

    The export payload is the complete final output text, unchanged.

    Args:
        text: The full final output text from the pipeline.

    Returns:
        The text content ready for export/download.

    Validates: Requirements 12.6
    """
    return text


def is_export_enabled(output: str) -> bool:
    """Determine whether the export control should be enabled.

    The export button is enabled only when a pipeline run has completed
    and produced non-empty output in the current session.

    Args:
        output: The current humanized output text (from session state).

    Returns:
        True if output is non-empty (export should be enabled),
        False otherwise (export should be disabled).

    Validates: Requirements 12.8
    """
    return bool(output)
