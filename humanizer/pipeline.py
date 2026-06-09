"""
Pipeline Orchestrator

Chains all humanization stages together with configurable intensity
and stage toggles. Supports progress callbacks for UI integration.

The pipeline executes a canonical 13-stage order, constructing shared
evaluation services (SimilarityEvaluator, Classifier) once and injecting
them into stages that need them.

Each enabled stage is wrapped in a discard-on-violation backstop:
after the stage produces a candidate, the pipeline verifies that no
protected spans were lost (via ProtectedSpanGuard.verify()) and that
the candidate meets the stage's similarity floor. On violation, the
candidate is discarded and the previous text is retained. StageResult
analytics are captured for the UI.
"""

import math
import random
from typing import List, Optional

from humanizer.config import INTENSITY_PROFILES, DEFAULT_MODEL, API_KEY, BASE_URL
from humanizer.protected_spans import ProtectedSpanGuard
from humanizer.results import StageResult
from humanizer.stage_structural import StructuralVariation
from humanizer.stage_lexical import LexicalInjection
from humanizer.stage_semantic import SemanticTransformer
from humanizer.stage_iterative import IterativeParaphraser
from humanizer.stage_llm_rewrite import LLMRewriter
from humanizer.stage_retrieval_augmented import RetrievalAugmentedRewriter
from humanizer.stage_stylometric import StylometricObfuscator
from humanizer.stage_perplexity import PerplexityVariance
from humanizer.stage_perplexity_optimize import PerplexityOptimizer
from humanizer.stage_adversarial import AdversarialRewriter
from humanizer.stage_error_injector import ErrorInjector
from humanizer.stage_postprocess import PostProcessor
from humanizer.stage_detector_optimizer import DetectorOptimizer


class HumanizationPipeline:
    """
    Multi-stage humanization pipeline that transforms AI-generated text
    to bypass AI detection systems.

    Stages (canonical 13-stage order):
        1.  Structural Variation (existing, deterministic)
        2.  Lexical Injection (existing, NLP-based)
        3.  Semantic Transformer (NEW, NLP-based)
        4.  Iterative Paraphraser (NEW, LLM)
        5.  LLM Rewrite (existing, LLM)
        6.  Retrieval-Augmented Rewriting (NEW, LLM)
        7.  Stylometric Obfuscator (NEW, NLP)
        8.  Perplexity Variance (existing, NLP)
        9.  Perplexity Optimizer (NEW, NLP)
        10. Adversarial Rewriter (NEW, LLM)
        11. Error Injector (NEW, NLP)
        12. Post-processing (existing, cleanup)
        13. Detector Optimizer (NEW, closed-loop)
    """

    STAGE_NAMES = {
        "structural": "Structural Variation",
        "lexical": "Vocabulary Injection",
        "semantic_transform": "Semantic Transformation",
        "iterative_paraphrase": "Iterative Paraphrasing",
        "llm_rewrite": "LLM Rewriting",
        "retrieval_augmented": "Retrieval-Augmented Rewriting",
        "stylometric": "Stylometric Obfuscation",
        "perplexity": "Perplexity Variance",
        "perplexity_optimize": "Perplexity Optimization",
        "adversarial": "Adversarial Rewriting",
        "error_injection": "Error Injection",
        "postprocess": "Post-processing",
        "detector_optimize": "Detector Optimization",
    }

    # Canonical execution order — all 13 stages in the deterministic sequence
    # defined by the design document. Disabled stages are simply skipped.
    STAGE_ORDER = [
        "structural",
        "lexical",
        "semantic_transform",
        "iterative_paraphrase",
        "llm_rewrite",
        "retrieval_augmented",
        "stylometric",
        "perplexity",
        "perplexity_optimize",
        "adversarial",
        "error_injection",
        "postprocess",
        "detector_optimize",
    ]

    # Mapping from stage key to the profile key that controls its enabled state.
    # Existing stages use their key directly as a boolean in the profile;
    # new stages use "<key>_enabled".
    _EXISTING_STAGE_KEYS = {"structural", "lexical", "llm_rewrite", "perplexity", "postprocess"}

    # Per-stage similarity floors for the pipeline-level backstop check.
    # Stages that already enforce their own internal floor have the same value here
    # (defense-in-depth). Stages without an explicit floor use 0.0 (always accept).
    STAGE_SIMILARITY_FLOORS = {
        "structural": 0.0,
        "lexical": 0.0,
        "semantic_transform": 0.90,
        "iterative_paraphrase": 0.80,
        "llm_rewrite": 0.0,
        "retrieval_augmented": 0.85,
        "stylometric": 0.85,
        "perplexity": 0.0,
        "perplexity_optimize": 0.85,
        "adversarial": 0.85,
        "error_injection": 0.0,
        "postprocess": 0.0,
        "detector_optimize": 0.85,
    }

    def __init__(self, intensity=4, model=None, api_key=None, base_url=None,
                 stage_overrides=None, progress_callback=None, seed=None, identity=None,
                 style_instructions=None, target_perplexity_profile=None):
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
            style_instructions: Optional string with style preferences to append to LLM prompts.
            target_perplexity_profile: Optional dict with 'target_perplexity_mean' and
                'target_perplexity_variance' to override the intensity profile defaults.
        """
        # Round non-integer intensity with halves rounding up, then clamp to [1, 5]
        self.intensity = max(1, min(5, int(math.floor(intensity + 0.5))))
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        self.progress_callback = progress_callback
        self.seed = seed
        self.identity = identity
        self.style_instructions = style_instructions

        # Get intensity profile
        profile = INTENSITY_PROFILES[self.intensity]

        # Apply stage overrides
        self.stage_config = dict(profile)
        if stage_overrides:
            for stage, enabled in stage_overrides.items():
                # For existing stages, the key is used directly as a boolean
                if stage in self._EXISTING_STAGE_KEYS:
                    self.stage_config[stage] = enabled
                else:
                    # For new stages, override the <key>_enabled flag
                    self.stage_config[f"{stage}_enabled"] = enabled

        # Apply target perplexity profile overrides
        if target_perplexity_profile:
            if "target_perplexity_mean" in target_perplexity_profile:
                self.stage_config["target_perplexity_mean"] = target_perplexity_profile["target_perplexity_mean"]
            if "target_perplexity_variance" in target_perplexity_profile:
                self.stage_config["target_perplexity_variance"] = target_perplexity_profile["target_perplexity_variance"]

        # Construct shared evaluation services once
        self._similarity = self._build_similarity_evaluator()
        self._classifier = self._build_classifier()

        # Analytics capture: populated during process() with StageResult per stage
        self.stage_results: List[StageResult] = []

        # Final meaning-preservation check results (reset at start of each process() call)
        self.final_similarity: Optional[float] = None
        self.final_warning: Optional[str] = None

    def _build_similarity_evaluator(self):
        """Construct a shared SimilarityEvaluator instance."""
        from humanizer.similarity import SimilarityEvaluator
        return SimilarityEvaluator()

    def _build_classifier(self):
        """Construct a shared Classifier instance if classifier is enabled."""
        if self.stage_config.get("classifier_enabled", False):
            from humanizer.classifier import Classifier
            return Classifier()
        return None

    def get_enabled_stages(self):
        """Return list of enabled stage keys in canonical order.

        For existing stages (structural, lexical, llm_rewrite, perplexity,
        postprocess), checks the boolean value at their key in the profile.
        For new stages, checks the '<key>_enabled' flag.
        """
        enabled = []
        for stage in self.STAGE_ORDER:
            if stage in self._EXISTING_STAGE_KEYS:
                if self.stage_config.get(stage, False):
                    enabled.append(stage)
            else:
                if self.stage_config.get(f"{stage}_enabled", False):
                    enabled.append(stage)
        return enabled

    def process(self, text, stream_callback=None):
        """
        Run the full pipeline on input text.

        Each enabled stage is wrapped in a discard-on-violation backstop:
        1. Emit progress_callback(stage, "running") before execution.
        2. Execute the stage (preferring process_measured() when available).
        3. Verify candidate with ProtectedSpanGuard.verify() — if any
           protected span count dropped, discard the candidate.
        4. Verify candidate meets the stage's similarity floor — if below
           floor, discard the candidate.
        5. Emit progress_callback(stage, "complete") on success.
        6. On unhandled error: keep last good text, emit "error", continue.

        Skipped (disabled) stages emit no progress callback (Req 10.6).
        All stages disabled → input unchanged (Req 10.9).
        Empty/whitespace input → input unchanged (Req 10.10).

        Args:
            text: Input text to humanize.
            stream_callback: Optional callback for LLM streaming chunks.

        Returns:
            Processed text.
        """
        if not text.strip():
            return text

        original_input = text
        current_text = text
        enabled_stages = self.get_enabled_stages()

        # Reset analytics for this run
        self.stage_results = []
        self.final_similarity = None
        self.final_warning = None

        for stage_key in enabled_stages:
            try:
                self._notify_progress(stage_key, "running")
                stage_instance = self._build_stage(stage_key, stream_callback)
                if stage_instance is None:
                    self._notify_progress(stage_key, "complete")
                    continue

                # Prefer process_measured() for StageResult analytics capture
                stage_result = self._execute_stage_measured(
                    stage_instance, stage_key, current_text, stream_callback
                )

                if stage_result is not None:
                    # Backstop: verify protected spans survived
                    if not self._backstop_protected_spans(current_text, stage_result.text):
                        # Protected span violation — discard candidate
                        stage_result = StageResult(
                            text=current_text,
                            similarity=stage_result.similarity,
                            risk_before=stage_result.risk_before,
                            risk_after=stage_result.risk_after,
                            changed=False,
                            fell_back=True,
                            error="Protected span violation (pipeline backstop)",
                        )
                    # Backstop: verify similarity floor
                    elif not self._backstop_similarity(
                        stage_key, current_text, stage_result.text
                    ):
                        stage_result = StageResult(
                            text=current_text,
                            similarity=stage_result.similarity,
                            risk_before=stage_result.risk_before,
                            risk_after=stage_result.risk_after,
                            changed=False,
                            fell_back=True,
                            error="Similarity below floor (pipeline backstop)",
                        )

                    self.stage_results.append(stage_result)
                    current_text = stage_result.text
                else:
                    # Stage produced no result object — use plain text result
                    current_text = current_text  # Already handled in _execute_stage_measured

                self._notify_progress(stage_key, "complete")
            except Exception:
                # Unhandled stage error: retain last good text, notify error,
                # and continue with next stage (Req 10.7)
                self.stage_results.append(StageResult(
                    text=current_text,
                    similarity=None,
                    risk_before=None,
                    risk_after=None,
                    changed=False,
                    fell_back=True,
                    error=f"Unhandled error in stage {stage_key}",
                ))
                self._notify_progress(stage_key, "error")
                continue

        # Final meaning-preservation check (Req 14.1, 14.3, 14.5)
        self._run_final_meaning_check(original_input, current_text)

        return current_text

    def _run_final_meaning_check(self, original_input: str, final_output: str) -> None:
        """Run the final meaning-preservation check after all stages complete.

        Computes similarity between original input and final output; if below 0.85,
        surfaces a warning (but still returns the output). Also runs
        ProtectedSpanGuard.verify() and surfaces a warning if any protected term,
        numeric value, or citation marker count dropped.

        Results are stored in self.final_similarity and self.final_warning.
        """
        warnings = []

        # Compute final similarity (Req 14.1)
        self.final_similarity = self._similarity.score(original_input, final_output)

        # Surface warning if similarity < 0.85 (Req 14.3)
        if self.final_similarity < 0.85:
            warnings.append(
                f"Meaning preservation warning: final similarity "
                f"({self.final_similarity:.3f}) fell below 0.85 threshold."
            )

        # Run ProtectedSpanGuard.verify (Req 14.2, 14.4, 14.5)
        deltas = ProtectedSpanGuard.verify(original_input, final_output)
        dropped_categories = []
        for category, delta in deltas.items():
            if delta < 0:
                dropped_categories.append(f"{category} (lost {abs(delta)})")

        if dropped_categories:
            warnings.append(
                f"Protected content warning: the following were dropped: "
                f"{', '.join(dropped_categories)}."
            )

        self.final_warning = " ".join(warnings) if warnings else None

    def _execute_stage_measured(self, stage_instance, stage_key, current_text, stream_callback=None):
        """Execute a stage, preferring process_measured() for analytics.

        Returns a StageResult if available, or constructs one from plain process().
        """
        # Try process_measured() first for analytics
        if hasattr(stage_instance, "process_measured"):
            result = stage_instance.process_measured(current_text)
            return result

        # Fall back to plain process()
        if stage_key == "llm_rewrite":
            output = stage_instance.process(current_text, stream_callback=stream_callback)
        else:
            output = stage_instance.process(current_text)

        changed = output != current_text
        return StageResult(
            text=output,
            similarity=None,
            risk_before=None,
            risk_after=None,
            changed=changed,
            fell_back=False,
            error=None,
        )

    def _backstop_protected_spans(self, original: str, candidate: str) -> bool:
        """Verify that no protected spans were lost in the candidate.

        Returns True if the candidate passes (no spans lost), False on violation.
        """
        deltas = ProtectedSpanGuard.verify(original, candidate)
        # Any negative delta means a protected span was lost
        for delta in deltas.values():
            if delta < 0:
                return False
        return True

    def _backstop_similarity(self, stage_key: str, original: str, candidate: str) -> bool:
        """Verify that the candidate meets the stage's similarity floor.

        Returns True if the candidate passes, False on violation.
        Stages with floor 0.0 always pass.
        """
        floor = self.STAGE_SIMILARITY_FLOORS.get(stage_key, 0.0)
        if floor <= 0.0:
            return True
        # Skip check if candidate is unchanged
        if candidate == original:
            return True
        score = self._similarity.score(original, candidate)
        return score >= floor

    def _build_stage(self, stage_key, stream_callback=None):
        """Instantiate the stage object for the given key.

        Injects shared services (similarity, classifier, seed) as needed.
        """
        aggression_key = f"{stage_key}_aggression"
        aggression = self.stage_config.get(aggression_key, 0.5)

        if stage_key == "structural":
            return StructuralVariation(
                aggression=self.stage_config.get("structural_aggression", 0.5),
                seed=self.seed,
            )

        elif stage_key == "lexical":
            return LexicalInjection(
                aggression=self.stage_config.get("lexical_aggression", 0.5),
                seed=self.seed,
            )

        elif stage_key == "semantic_transform":
            return SemanticTransformer(
                aggression=self.stage_config.get("semantic_transform_aggression", 0.5),
                seed=self.seed,
                similarity=self._similarity,
            )

        elif stage_key == "iterative_paraphrase":
            return IterativeParaphraser(
                aggression=self.stage_config.get("iterative_paraphrase_aggression", 0.5),
                seed=self.seed,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                similarity=self._similarity,
            )

        elif stage_key == "llm_rewrite":
            return LLMRewriter(
                aggression=self.stage_config.get("llm_aggression", 0.5),
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                identity=self.identity,
                style_instructions=self.style_instructions,
            )

        elif stage_key == "retrieval_augmented":
            return RetrievalAugmentedRewriter(
                aggression=self.stage_config.get("retrieval_augmented_aggression", 0.5),
                seed=self.seed,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                similarity=self._similarity,
            )

        elif stage_key == "stylometric":
            return StylometricObfuscator(
                aggression=self.stage_config.get("stylometric_aggression", 0.5),
                seed=self.seed,
                similarity=self._similarity,
            )

        elif stage_key == "perplexity":
            return PerplexityVariance(
                aggression=self.stage_config.get("perplexity_aggression", 0.5),
                seed=self.seed,
            )

        elif stage_key == "perplexity_optimize":
            from humanizer.results import TargetPerplexityProfile
            target_profile = TargetPerplexityProfile(
                target_mean=self.stage_config.get("target_perplexity_mean", 60.0),
                target_variance=self.stage_config.get("target_perplexity_variance", 15.0),
            )
            return PerplexityOptimizer(
                aggression=self.stage_config.get("perplexity_optimize_aggression", 0.5),
                seed=self.seed,
                similarity=self._similarity,
                target_profile=target_profile,
            )

        elif stage_key == "adversarial":
            return AdversarialRewriter(
                aggression=self.stage_config.get("adversarial_aggression", 0.5),
                seed=self.seed,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                similarity=self._similarity,
                classifier=self._classifier,
            )

        elif stage_key == "error_injection":
            return ErrorInjector(
                aggression=self.stage_config.get("error_injection_aggression", 0.5),
                seed=self.seed,
            )

        elif stage_key == "postprocess":
            return PostProcessor(
                aggression=self.stage_config.get("postprocess_aggression", 0.5),
                seed=self.seed,
            )

        elif stage_key == "detector_optimize":
            return DetectorOptimizer(
                aggression=self.stage_config.get("detector_optimize_aggression", 0.5),
                seed=self.seed,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                classifier=self._classifier,
                similarity=self._similarity,
            )

        return None

    def process_stream(self, text):
        """
        Run the pipeline and yield streaming output from the LLM stage.

        Non-LLM stages run synchronously before and after the LLM stage.
        Pass 1 runs internally (not yielded). Only Pass 2 chunks are yielded
        for real-time display. Post-processing runs after streaming and the
        final text is yielded if it differs from what was streamed.

        Uses the same discard-on-violation backstop as process():
        - ProtectedSpanGuard.verify() checks that no protected spans were lost
        - Similarity floor check discards candidates below the stage's floor
        - Skipped stages emit no progress callback (Req 10.6)

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

        original_input = text
        current_text = text
        enabled_stages = self.get_enabled_stages()

        # Reset analytics for this run
        self.stage_results = []
        self.final_similarity = None
        self.final_warning = None

        # Pre-LLM stages (synchronous) — everything before llm_rewrite
        pre_llm_stages = []
        post_llm_stages = []
        found_llm = False
        for stage_key in enabled_stages:
            if stage_key == "llm_rewrite":
                found_llm = True
                continue
            if not found_llm:
                pre_llm_stages.append(stage_key)
            else:
                post_llm_stages.append(stage_key)

        # Run pre-LLM stages with backstop wrapper
        for stage_key in pre_llm_stages:
            try:
                self._notify_progress(stage_key, "running")
                stage_instance = self._build_stage(stage_key)
                if stage_instance is not None:
                    stage_result = self._execute_stage_measured(
                        stage_instance, stage_key, current_text
                    )
                    if stage_result is not None:
                        # Backstop checks
                        if not self._backstop_protected_spans(current_text, stage_result.text):
                            stage_result = StageResult(
                                text=current_text,
                                similarity=stage_result.similarity,
                                risk_before=stage_result.risk_before,
                                risk_after=stage_result.risk_after,
                                changed=False,
                                fell_back=True,
                                error="Protected span violation (pipeline backstop)",
                            )
                        elif not self._backstop_similarity(
                            stage_key, current_text, stage_result.text
                        ):
                            stage_result = StageResult(
                                text=current_text,
                                similarity=stage_result.similarity,
                                risk_before=stage_result.risk_before,
                                risk_after=stage_result.risk_after,
                                changed=False,
                                fell_back=True,
                                error="Similarity below floor (pipeline backstop)",
                            )
                        self.stage_results.append(stage_result)
                        current_text = stage_result.text
                self._notify_progress(stage_key, "complete")
            except Exception:
                self.stage_results.append(StageResult(
                    text=current_text,
                    similarity=None,
                    risk_before=None,
                    risk_after=None,
                    changed=False,
                    fell_back=True,
                    error=f"Unhandled error in stage {stage_key}",
                ))
                self._notify_progress(stage_key, "error")

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
                style_instructions=self.style_instructions,
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
                        llm_output = pass2_result if pass2_result.strip() else pass1_result
                    except RuntimeError:
                        # Pass 2 failed; fall back to Pass 1 result
                        llm_output = pass1_result
                        yield llm_output
                else:
                    # No Pass 2 at low aggression; yield Pass 1 result
                    llm_output = pass1_result
                    yield llm_output

                # Backstop the LLM result
                if not self._backstop_protected_spans(current_text, llm_output):
                    llm_output = current_text
                elif not self._backstop_similarity("llm_rewrite", current_text, llm_output):
                    llm_output = current_text

                current_text = llm_output
                self.stage_results.append(StageResult(
                    text=current_text,
                    similarity=None,
                    risk_before=None,
                    risk_after=None,
                    changed=current_text != pre_llm_text,
                    fell_back=current_text == pre_llm_text and llm_output != pre_llm_text,
                    error=None,
                ))
            except RuntimeError:
                # LLM failed entirely; fall back to pre-LLM text
                current_text = pre_llm_text
                yield current_text
                self.stage_results.append(StageResult(
                    text=current_text,
                    similarity=None,
                    risk_before=None,
                    risk_after=None,
                    changed=False,
                    fell_back=True,
                    error="LLM stage failed",
                ))

            self._notify_progress("llm_rewrite", "complete")
        else:
            # If no LLM stage, yield the pre-processed text
            yield current_text

        # Post-LLM stages (synchronous) with backstop wrapper
        post_text = current_text
        for stage_key in post_llm_stages:
            try:
                self._notify_progress(stage_key, "running")
                stage_instance = self._build_stage(stage_key)
                if stage_instance is not None:
                    stage_result = self._execute_stage_measured(
                        stage_instance, stage_key, post_text
                    )
                    if stage_result is not None:
                        # Backstop checks
                        if not self._backstop_protected_spans(post_text, stage_result.text):
                            stage_result = StageResult(
                                text=post_text,
                                similarity=stage_result.similarity,
                                risk_before=stage_result.risk_before,
                                risk_after=stage_result.risk_after,
                                changed=False,
                                fell_back=True,
                                error="Protected span violation (pipeline backstop)",
                            )
                        elif not self._backstop_similarity(
                            stage_key, post_text, stage_result.text
                        ):
                            stage_result = StageResult(
                                text=post_text,
                                similarity=stage_result.similarity,
                                risk_before=stage_result.risk_before,
                                risk_after=stage_result.risk_after,
                                changed=False,
                                fell_back=True,
                                error="Similarity below floor (pipeline backstop)",
                            )
                        self.stage_results.append(stage_result)
                        post_text = stage_result.text
                self._notify_progress(stage_key, "complete")
            except Exception:
                self.stage_results.append(StageResult(
                    text=post_text,
                    similarity=None,
                    risk_before=None,
                    risk_after=None,
                    changed=False,
                    fell_back=True,
                    error=f"Unhandled error in stage {stage_key}",
                ))
                self._notify_progress(stage_key, "error")

        # If post-processing changed the text, yield the final version
        if post_text != current_text:
            yield "\n\n__POSTPROCESSED__\n\n"
            yield post_text

        # Final meaning-preservation check (Req 14.1, 14.3, 14.5)
        self._run_final_meaning_check(original_input, post_text)

    def _notify_progress(self, stage, status):
        """Send progress notification."""
        if self.progress_callback:
            stage_name = self.STAGE_NAMES.get(stage, stage)
            self.progress_callback(stage_name, status)
