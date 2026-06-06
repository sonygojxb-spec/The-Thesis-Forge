"""
Critic Loop Module

Wraps the humanization pipeline externally to perform iterative
refinement. Re-runs the pipeline with increasing intensity if
the AI risk score remains above a threshold.
"""

from humanizer.config import CRITIC_DEFAULT_THRESHOLD, CRITIC_MAX_RETRIES
from humanizer.text_analysis import compute_ai_risk_score
from humanizer.voice_analysis import get_voice_profile


class CriticLoop:
    """
    Iterative critic that re-runs the pipeline if AI risk is too high.

    The critic takes a pipeline_factory callable that creates a pipeline
    with a given intensity. It runs the pipeline, evaluates the output,
    and retries with higher intensity if needed.
    """

    def __init__(self, pipeline_factory, max_retries=None, risk_threshold=None):
        """
        Args:
            pipeline_factory: Callable(intensity: int) -> pipeline object with process(text) method.
            max_retries: Maximum number of retries (default from config).
            risk_threshold: AI risk score threshold to trigger retry (default from config).
        """
        self.pipeline_factory = pipeline_factory
        self.max_retries = max_retries if max_retries is not None else CRITIC_MAX_RETRIES
        self.risk_threshold = risk_threshold if risk_threshold is not None else CRITIC_DEFAULT_THRESHOLD

    def run(self, text, initial_intensity=3):
        """
        Run the pipeline with critic evaluation and retry logic.

        Args:
            text: Input text to humanize.
            initial_intensity: Starting intensity level (1-5).

        Returns:
            dict with keys:
                - final_text: The best output text produced.
                - attempts: List of dicts, each with text, risk_score,
                  voice_profile, and intensity_used.
                - success: Boolean indicating if risk was brought below threshold.
        """
        attempts = []
        current_intensity = initial_intensity
        best_text = text

        for attempt_num in range(self.max_retries + 1):
            # Cap intensity at 5
            capped_intensity = min(5, current_intensity)

            # Create and run the pipeline
            pipeline = self.pipeline_factory(capped_intensity)
            result_text = pipeline.process(text)

            # Evaluate the output
            risk_score = compute_ai_risk_score(result_text)
            voice_profile = get_voice_profile(result_text)

            attempts.append({
                "text": result_text,
                "risk_score": risk_score,
                "voice_profile": voice_profile,
                "intensity_used": capped_intensity,
            })

            best_text = result_text

            # Check if risk is acceptable
            if risk_score <= self.risk_threshold:
                return {
                    "final_text": best_text,
                    "attempts": attempts,
                    "success": True,
                }

            # If we have retries left, increase intensity
            if attempt_num < self.max_retries:
                current_intensity += 1

        # Exhausted retries without meeting threshold
        return {
            "final_text": best_text,
            "attempts": attempts,
            "success": False,
        }
