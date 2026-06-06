"""
Pipeline Orchestrator

Chains all humanization stages together with configurable intensity
and stage toggles. Supports progress callbacks for UI integration.
"""

import random

from humanizer.config import INTENSITY_PROFILES, DEFAULT_MODEL, API_KEY, BASE_URL
from humanizer.stage_structural import StructuralVariation
from humanizer.stage_lexical import LexicalInjection
from humanizer.stage_llm_rewrite import LLMRewriter
from humanizer.stage_perplexity import PerplexityVariance
from humanizer.stage_postprocess import PostProcessor


class HumanizationPipeline:
    """
    Multi-stage humanization pipeline that transforms AI-generated text
    to bypass AI detection systems.

    Stages:
        1. Structural Variation (deterministic)
        2. Lexical Injection (NLP-based)
        3. LLM Rewrite (multi-pass, streaming)
        4. Perplexity Variance (NLP-based)
        5. Post-processing (cleanup)
    """

    STAGE_NAMES = {
        "structural": "Structural Variation",
        "lexical": "Vocabulary Injection",
        "llm_rewrite": "LLM Rewriting",
        "perplexity": "Perplexity Variance",
        "postprocess": "Post-processing",
    }

    def __init__(self, intensity=4, model=None, api_key=None, base_url=None,
                 stage_overrides=None, progress_callback=None, seed=None, identity=None):
        """
        Initialize the pipeline.

        Args:
            intensity: Int 1-5, controls aggressiveness.
            model: LLM model name.
            api_key: API key for LLM calls.
            base_url: API base URL.
            stage_overrides: Dict of {stage_name: bool} to enable/disable stages.
            progress_callback: Callable(stage_name, status) for UI updates.
            seed: Optional int seed for reproducible NLP stage outputs.
            identity: Optional AcademicIdentity instance for role conditioning.
        """
        self.intensity = max(1, min(5, intensity))
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        self.progress_callback = progress_callback
        self.seed = seed
        self.identity = identity

        # Get intensity profile
        profile = INTENSITY_PROFILES[self.intensity]

        # Apply stage overrides
        self.stage_config = dict(profile)
        if stage_overrides:
            for stage, enabled in stage_overrides.items():
                if stage in self.stage_config:
                    self.stage_config[stage] = enabled

    def get_enabled_stages(self):
        """Return list of enabled stage names."""
        stages = ["structural", "lexical", "llm_rewrite", "perplexity", "postprocess"]
        return [s for s in stages if self.stage_config.get(s, False)]

    def process(self, text, stream_callback=None):
        """
        Run the full pipeline on input text.

        Args:
            text: Input text to humanize.
            stream_callback: Optional callback for LLM streaming chunks.

        Returns:
            Processed text.
        """
        if not text.strip():
            return text

        current_text = text
        enabled_stages = self.get_enabled_stages()

        # Stage 1: Structural Variation
        if "structural" in enabled_stages:
            self._notify_progress("structural", "running")
            stage = StructuralVariation(
                aggression=self.stage_config.get("structural_aggression", 0.5),
                seed=self.seed,
            )
            current_text = stage.process(current_text)
            self._notify_progress("structural", "complete")

        # Stage 2: Lexical Injection
        if "lexical" in enabled_stages:
            self._notify_progress("lexical", "running")
            stage = LexicalInjection(
                aggression=self.stage_config.get("lexical_aggression", 0.5),
                seed=self.seed,
            )
            current_text = stage.process(current_text)
            self._notify_progress("lexical", "complete")

        # Stage 3: LLM Rewrite
        if "llm_rewrite" in enabled_stages:
            self._notify_progress("llm_rewrite", "running")
            stage = LLMRewriter(
                aggression=self.stage_config.get("llm_aggression", 0.5),
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                identity=self.identity,
            )
            try:
                current_text = stage.process(current_text, stream_callback=stream_callback)
            except RuntimeError:
                # LLM failed entirely; fall back to pre-LLM text
                pass
            self._notify_progress("llm_rewrite", "complete")

        # Stage 4: Perplexity Variance
        if "perplexity" in enabled_stages:
            self._notify_progress("perplexity", "running")
            stage = PerplexityVariance(
                aggression=self.stage_config.get("perplexity_aggression", 0.5),
                seed=self.seed,
            )
            current_text = stage.process(current_text)
            self._notify_progress("perplexity", "complete")

        # Stage 5: Post-processing
        if "postprocess" in enabled_stages:
            self._notify_progress("postprocess", "running")
            stage = PostProcessor(
                aggression=self.stage_config.get("postprocess_aggression", 0.5),
                seed=self.seed,
            )
            current_text = stage.process(current_text)
            self._notify_progress("postprocess", "complete")

        return current_text

    def process_stream(self, text):
        """
        Run the pipeline and yield streaming output from the LLM stage.

        Non-LLM stages run synchronously before and after the LLM stage.
        Pass 1 runs internally (not yielded). Only Pass 2 chunks are yielded
        for real-time display. Post-processing runs after streaming and the
        final text is yielded if it differs from what was streamed.

        Args:
            text: Input text to humanize.

        Yields:
            Text chunks during LLM Pass 2, then optionally a
            "__POSTPROCESSED__" sentinel followed by the final text if
            post-processing changed the output.
        """
        if not text.strip():
            yield text
            return

        current_text = text
        enabled_stages = self.get_enabled_stages()

        # Pre-LLM stages (synchronous)
        if "structural" in enabled_stages:
            self._notify_progress("structural", "running")
            stage = StructuralVariation(
                aggression=self.stage_config.get("structural_aggression", 0.5),
                seed=self.seed,
            )
            current_text = stage.process(current_text)
            self._notify_progress("structural", "complete")

        if "lexical" in enabled_stages:
            self._notify_progress("lexical", "running")
            stage = LexicalInjection(
                aggression=self.stage_config.get("lexical_aggression", 0.5),
                seed=self.seed,
            )
            current_text = stage.process(current_text)
            self._notify_progress("lexical", "complete")

        pre_llm_text = current_text

        # LLM stage (streaming)
        if "llm_rewrite" in enabled_stages:
            self._notify_progress("llm_rewrite", "running")
            stage = LLMRewriter(
                aggression=self.stage_config.get("llm_aggression", 0.5),
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                identity=self.identity,
            )

            try:
                # Run Pass 1 internally (collect but do not yield)
                pass1_chunks = []
                for chunk in stage.pass1_stream(current_text):
                    pass1_chunks.append(chunk)
                pass1_result = ''.join(pass1_chunks)

                if not pass1_result.strip():
                    pass1_result = current_text

                # Run Pass 2 and yield chunks for streaming display
                if stage.aggression >= 0.5:
                    pass2_chunks = []
                    try:
                        for chunk in stage.pass2_stream(pass1_result):
                            pass2_chunks.append(chunk)
                            yield chunk
                        pass2_result = ''.join(pass2_chunks)
                        current_text = pass2_result if pass2_result.strip() else pass1_result
                    except RuntimeError:
                        # Pass 2 failed; fall back to Pass 1 result
                        current_text = pass1_result
                        yield current_text
                else:
                    # No Pass 2 at low aggression; yield Pass 1 result
                    current_text = pass1_result
                    yield current_text
            except RuntimeError:
                # LLM failed entirely; fall back to pre-LLM text
                current_text = pre_llm_text
                yield current_text

            self._notify_progress("llm_rewrite", "complete")
        else:
            # If no LLM stage, yield the pre-processed text
            yield current_text

        # Post-LLM stages (synchronous, apply to collected LLM output)
        post_text = current_text

        if "perplexity" in enabled_stages:
            self._notify_progress("perplexity", "running")
            stage = PerplexityVariance(
                aggression=self.stage_config.get("perplexity_aggression", 0.5),
                seed=self.seed,
            )
            post_text = stage.process(post_text)
            self._notify_progress("perplexity", "complete")

        if "postprocess" in enabled_stages:
            self._notify_progress("postprocess", "running")
            stage = PostProcessor(
                aggression=self.stage_config.get("postprocess_aggression", 0.5),
                seed=self.seed,
            )
            post_text = stage.process(post_text)
            self._notify_progress("postprocess", "complete")

        # If post-processing changed the text, yield the final version
        if post_text != current_text:
            yield "\n\n__POSTPROCESSED__\n\n"
            yield post_text

    def _notify_progress(self, stage, status):
        """Send progress notification."""
        if self.progress_callback:
            stage_name = self.STAGE_NAMES.get(stage, stage)
            self.progress_callback(stage_name, status)
