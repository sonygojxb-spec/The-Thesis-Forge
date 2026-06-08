# Requirements Document

## Introduction

The **Ultimate Humanizer** extends "The Thesis Forge" AI humanization pipeline from its current
five-stage architecture (Structural Variation, Lexical Injection, LLM Rewrite, Perplexity Variance,
Post-processing) by adding nine advanced capabilities. These capabilities deepen the system's ability
to transform AI-generated academic text so that it statistically resembles human writing while
preserving the original meaning and academic accuracy.

The nine new capabilities are:

1. Iterative Paraphrasing
2. Stylometric Obfuscation
3. Perplexity Optimization
4. Adversarial Rewriting
5. Human-Like Error Injection
6. Semantic-Preserving Transformations
7. Retrieval-Augmented Humanization
8. Detector-Aware Optimization
9. Transformer-Based Classification Model

Each capability is implemented as a modular stage (or supporting service) that integrates with the
existing `HumanizationPipeline` orchestrator. The orchestrator MUST continue to support configurable
intensity levels (1-5), per-stage enable/disable toggles, reproducible seeding, progress callbacks,
and streaming, and the Streamlit frontend (`app.py`) MUST expose controls and analytics for each new
capability. All new stages MUST conform to the existing stage contract: a constructor accepting an
`aggression` value (float 0.0-1.0) and an optional `seed`, and a `process(text) -> str` method that
returns transformed text.

This document captures functional and quality requirements for all nine capabilities and their
integration with the pipeline, intensity profiles, stage toggles, analytics, and UI.

## Glossary

- **Pipeline**: The `HumanizationPipeline` orchestrator in `humanizer/pipeline.py` that chains stages
  in a defined order with configurable intensity and per-stage toggles.
- **Stage**: A modular text transformation unit conforming to the contract
  `__init__(aggression: float, seed: Optional[int], ...)` and `process(text: str) -> str`.
- **Aggression**: A float between 0.0 and 1.0 derived from the intensity profile that controls how
  strongly a stage transforms text.
- **Intensity_Profile**: The per-level configuration (levels 1-5) in `humanizer/config.py` that
  determines which stages are enabled and each stage's aggression value.
- **Stage_Toggle**: A user-controllable boolean that enables or disables an individual stage,
  overriding the intensity profile default.
- **Seed**: An optional integer that makes all non-LLM stage outputs reproducible.
- **Progress_Callback**: A callable `(stage_name, status)` invoked by the Pipeline to report stage
  progress to the UI.
- **Iterative_Paraphraser**: The stage implementing repeated multi-pass paraphrasing.
- **Stylometric_Obfuscator**: The stage that disrupts stylometric fingerprints.
- **Perplexity_Optimizer**: The stage that tunes text toward a target perplexity and burstiness profile.
- **Adversarial_Rewriter**: The stage that rewrites text to evade detector signals.
- **Error_Injector**: The stage that injects controlled human-like imperfections.
- **Semantic_Transformer**: The stage that applies surface-form transformations while preserving meaning.
- **Retrieval_Service**: The component that retrieves human-written reference passages from a
  reference corpus to ground rewrites.
- **Reference_Corpus**: A stored collection of human-written text passages used by the Retrieval_Service.
- **Detector_Optimizer**: The closed-loop controller that iteratively adjusts output to reduce
  estimated detection probability.
- **Detection_Risk_Score**: A numeric estimate (0-100) of the likelihood that text is classified as
  AI-generated, produced by the Risk_Scorer or the Classifier.
- **Risk_Scorer**: The existing heuristic scorer in `humanizer/text_analysis.py`
  (`compute_ai_risk_score`).
- **Classifier**: The Transformer-Based Classification Model that estimates AI-detection risk and can
  act as an internal discriminator.
- **Semantic_Similarity_Score**: A numeric measure (0.0-1.0) of meaning preservation between two texts,
  computed by an embedding-based comparison.
- **Lexical_Divergence**: A normalized measure (0.0-1.0) of the proportion of word-level changes
  between two texts.
- **Protected_Terms**: The set of academic/domain terms in `humanizer/config.py` that MUST NOT be
  altered or removed by any transformation.
- **Target_Perplexity_Profile**: A configured pair of target values for mean perplexity and perplexity
  variance (burstiness) representative of human writing.
- **Config_Serializer**: The component that serializes and deserializes pipeline and capability
  configuration to and from a persisted representation.

## Requirements

### Requirement 1: Iterative Paraphrasing

**User Story:** As a user humanizing academic text, I want the system to paraphrase the text across
multiple controlled passes, so that the output progressively diverges from the original AI phrasing
while retaining the original meaning.

#### Acceptance Criteria

1. WHEN the Iterative_Paraphraser stage is enabled and receives non-empty input text, THE
   Iterative_Paraphraser SHALL produce output text whose Lexical_Divergence from the input text is
   greater than 0.
2. THE Iterative_Paraphraser SHALL accept a stage aggression value in the range 0.0 to 1.0 inclusive
   and SHALL perform a number of paraphrasing passes determined by a monotonic mapping of that
   aggression value, performing 1 pass at aggression 0.0 and up to a maximum of 5 passes at
   aggression 1.0.
3. WHEN each paraphrasing pass completes, THE Iterative_Paraphraser SHALL use the output of that pass
   as the input to the next pass.
4. THE Iterative_Paraphraser SHALL preserve every term in Protected_Terms unchanged in the output
   text, matching each Protected_Terms entry case-sensitively on whole-word boundaries.
5. WHEN a paraphrasing pass produces text with a Semantic_Similarity_Score below 0.80 relative to the
   stage input text, THE Iterative_Paraphraser SHALL discard that pass result and retain the previous
   pass output.
6. IF the paraphrasing source (LLM) returns an error or empty result during a pass when a prior
   successful pass output exists, THEN THE Iterative_Paraphraser SHALL return the most recent
   successful pass output.
7. WHEN a Seed is provided and the paraphrasing uses non-LLM randomized selection, THE
   Iterative_Paraphraser SHALL produce identical output for identical input and Seed.
8. IF the first paraphrasing pass fails with an LLM error or empty result and no prior successful pass
   output exists, THEN THE Iterative_Paraphraser SHALL return the original input text unchanged.
9. THE Iterative_Paraphraser SHALL apply a 30-second timeout to each paraphrasing pass, and IF a pass
   does not complete within 30 seconds, THEN THE Iterative_Paraphraser SHALL treat that pass as failed
   and retain the most recent successful pass output.

### Requirement 2: Stylometric Obfuscation

**User Story:** As a user, I want the system to disrupt stylometric fingerprints in the text, so that
authorship and AI-signature analysis cannot reliably identify the text as machine-generated.

#### Acceptance Criteria

1. WHEN the Stylometric_Obfuscator stage is enabled, the stage aggression value is greater than 0.0,
   and the stage receives non-empty input text, THE Stylometric_Obfuscator SHALL change at least one
   of the following measurable attributes by at least 5 percent: sentence length distribution,
   function-word frequency distribution, punctuation pattern distribution, or type-token ratio.
2. WHEN processing completes, THE Stylometric_Obfuscator SHALL increase the sentence length variance
   of the output relative to the input by at least 10 percent, OR leave the sentence length variance
   unchanged within a tolerance band of plus or minus 2 percent when the input variance already meets
   the configured human-writing threshold.
3. THE Stylometric_Obfuscator SHALL preserve every term in Protected_Terms unchanged in the output text.
4. THE Stylometric_Obfuscator SHALL produce output with a Semantic_Similarity_Score of at least 0.85
   relative to the input text.
5. WHERE the stage aggression value is higher, THE Stylometric_Obfuscator SHALL apply a greater degree
   of distributional adjustment to the targeted stylometric attributes.
6. WHEN a Seed is provided, THE Stylometric_Obfuscator SHALL produce identical output for identical
   input and Seed.
7. WHEN the stage aggression value is 0.0, THE Stylometric_Obfuscator SHALL return the input text
   unchanged.
8. WHEN the input text contains fewer than 2 sentences, THE Stylometric_Obfuscator SHALL return the
   input text unchanged.
9. IF an adjustment cannot meet the 0.85 Semantic_Similarity_Score floor relative to the input text,
   THEN THE Stylometric_Obfuscator SHALL discard that adjustment and return text with a
   Semantic_Similarity_Score of at least 0.85 relative to the input text.

### Requirement 3: Perplexity Optimization

**User Story:** As a user, I want the system to tune the text toward a human-like perplexity and
burstiness profile, so that the statistical predictability of the text matches genuine human writing.

#### Acceptance Criteria

1. THE Perplexity_Optimizer SHALL accept a Target_Perplexity_Profile specifying a target mean
   perplexity value greater than 0 and a target perplexity variance value greater than or equal to 0.
2. WHEN the Perplexity_Optimizer stage is enabled and receives non-empty input text, THE
   Perplexity_Optimizer SHALL produce output text whose absolute distance between its measured mean
   perplexity and the target mean perplexity value is less than or equal to the absolute distance
   between the input text's measured mean perplexity and the target mean perplexity value.
3. WHEN the Perplexity_Optimizer processes input text containing at least 2 sentences, THE
   Perplexity_Optimizer SHALL produce output text whose absolute distance between its measured
   perplexity variance across sentences and the target perplexity variance value is less than or equal
   to the absolute distance between the input text's measured perplexity variance and the target
   perplexity variance value.
4. THE Perplexity_Optimizer SHALL preserve every term in Protected_Terms unchanged in the output text.
5. WHEN the measured mean perplexity of the input text is within the configured mean-perplexity
   tolerance of the target mean perplexity value AND the measured perplexity variance of the input
   text is within the configured variance tolerance of the target perplexity variance value, THE
   Perplexity_Optimizer SHALL return the input text unchanged.
6. THE Perplexity_Optimizer SHALL produce output with a Semantic_Similarity_Score of at least 0.85
   relative to the input text.
7. WHEN a Seed is provided, THE Perplexity_Optimizer SHALL produce identical output for identical
   input and Seed.
8. IF the input text is empty or contains only whitespace characters, THEN THE Perplexity_Optimizer
   SHALL return the input text unchanged.
9. IF the perplexity profile of the input text cannot be measured, THEN THE Perplexity_Optimizer SHALL
   return the input text unchanged.

### Requirement 4: Adversarial Rewriting

**User Story:** As a user, I want the system to rewrite text specifically to evade detector signals,
so that the output reduces the features that AI detectors rely on for classification.

#### Acceptance Criteria

1. WHEN the Adversarial_Rewriter stage is enabled and receives input text containing at least one
   non-whitespace character, THE Adversarial_Rewriter SHALL produce output text whose
   Detection_Risk_Score, computed by the same scorer (Risk_Scorer or Classifier) applied to both the
   input text and the output text, is less than or equal to the Detection_Risk_Score of the input text.
2. THE Adversarial_Rewriter SHALL preserve every term in Protected_Terms unchanged in the output text.
3. THE Adversarial_Rewriter SHALL produce output with a Semantic_Similarity_Score of at least 0.85
   relative to the input text.
4. WHERE the stage aggression value is higher, THE Adversarial_Rewriter SHALL apply a degree of
   rewriting, measured as the proportion of input-text words changed in the output text, that is
   greater than or equal to the proportion changed at any lower aggression value for the same input
   text.
5. IF an adversarial rewrite produces text with a Semantic_Similarity_Score below 0.85 relative to the
   input text, THEN THE Adversarial_Rewriter SHALL return the input text unchanged.
6. IF an adversarial rewrite produces output text whose Detection_Risk_Score is greater than the
   Detection_Risk_Score of the input text, THEN THE Adversarial_Rewriter SHALL return the input text
   unchanged.
7. IF the rewriting source (LLM) returns an error, returns an empty result, or does not respond within
   30 seconds, THEN THE Adversarial_Rewriter SHALL return the input text unchanged.
8. WHEN the Adversarial_Rewriter stage is enabled and receives input text that is empty or contains
   only whitespace characters, THE Adversarial_Rewriter SHALL return the input text unchanged.

### Requirement 5: Human-Like Error Injection

**User Story:** As a user, I want the system to inject controlled natural imperfections into the text,
so that the output exhibits the minor irregularities characteristic of human writing.

#### Acceptance Criteria

1. WHEN the Error_Injector stage is enabled and receives non-empty input text, THE Error_Injector
   SHALL insert imperfections drawn from the following observable categories: minor punctuation
   variations, whitespace variations, and informal word-form substitutions, at a rate that increases
   monotonically with the stage aggression value and is bounded by the maximum defined in criterion 2
   at aggression 1.0.
2. THE Error_Injector SHALL limit the proportion of words altered by imperfection injection to a
   maximum of 5 percent of the total word count of the input text, rounded down to the nearest whole
   word.
3. THE Error_Injector SHALL preserve every term in Protected_Terms unchanged in the output text.
4. THE Error_Injector SHALL NOT alter numeric values, citations, or content within quotation marks.
5. WHEN the stage aggression value is 0.0, THE Error_Injector SHALL return the input text unchanged.
6. WHEN a Seed is provided, THE Error_Injector SHALL produce identical output for identical input and
   Seed.
7. WHEN the input text is empty or contains only whitespace characters, THE Error_Injector SHALL
   return the input text unchanged.
8. WHEN 5 percent of the total word count of the input text rounds down to less than one word, THE
   Error_Injector SHALL alter zero words.

### Requirement 6: Semantic-Preserving Transformations

**User Story:** As a user, I want surface-form transformations that measurably preserve the meaning of
the text, so that aggressive humanization does not distort the original academic content.

#### Acceptance Criteria

1. WHEN the Semantic_Transformer stage is enabled and receives non-empty input text, THE
   Semantic_Transformer SHALL produce a candidate transformed text whose character sequence differs
   from the input text.
2. WHEN the Semantic_Transformer produces a candidate transformed text, THE Semantic_Transformer SHALL
   compute a Semantic_Similarity_Score in the range 0.0 to 1.0 inclusive between the input text and
   the candidate transformed text.
3. IF the Semantic_Similarity_Score between the candidate transformed text and the input text is below
   0.90, THEN THE Semantic_Transformer SHALL discard the candidate transformed text and return the
   input text unchanged.
4. THE Semantic_Transformer SHALL preserve every term in Protected_Terms unchanged in the output text.
5. THE Semantic_Transformer SHALL make the computed Semantic_Similarity_Score available for display by
   the Pipeline and UI.
6. WHEN a Seed is provided and the transformation uses non-LLM randomized selection, THE
   Semantic_Transformer SHALL produce identical output for identical input and Seed.
7. WHEN the input text is empty, THE Semantic_Transformer SHALL return the input text unchanged and
   compute no Semantic_Similarity_Score.
8. IF the transformation source returns an error or an empty result, THEN THE Semantic_Transformer
   SHALL return the input text unchanged.

### Requirement 7: Retrieval-Augmented Humanization

**User Story:** As a user, I want the system to ground rewrites in retrieved human-written reference
passages, so that the output adopts authentic human style and phrasing.

#### Acceptance Criteria

1. THE Retrieval_Service SHALL maintain a Reference_Corpus of human-written text passages.
2. WHEN the Retrieval-Augmented Humanization stage receives non-empty input text, THE
   Retrieval_Service SHALL retrieve up to a configured maximum of 10 reference passages, ranked in
   descending order of relevance computed by an embedding-based similarity comparison to the input
   text.
3. WHEN reference passages are retrieved, THE Retrieval-Augmented Humanization stage SHALL use the
   retrieved passages as style guidance for rewriting the input text.
4. IF the Reference_Corpus contains no passages OR the Retrieval_Service returns no results, THEN THE
   Retrieval-Augmented Humanization stage SHALL return the input text unchanged.
5. THE Retrieval-Augmented Humanization stage SHALL preserve every term in Protected_Terms unchanged
   in the output text.
6. THE Retrieval-Augmented Humanization stage SHALL produce output with a Semantic_Similarity_Score of
   at least 0.85 relative to the input text.
7. THE Retrieval-Augmented Humanization stage SHALL NOT copy any span of more than 8 consecutive words
   verbatim from any retrieved reference passage into the output text.
8. IF the rewriting source returns an error or an empty result, THEN THE Retrieval-Augmented
   Humanization stage SHALL return the input text unchanged.
9. IF the rewrite produces text with a Semantic_Similarity_Score below 0.85 relative to the input text,
   THEN THE Retrieval-Augmented Humanization stage SHALL return the input text unchanged.

### Requirement 8: Detector-Aware Optimization

**User Story:** As a user, I want the system to optimize output against detection scoring in a closed
loop, so that the final text minimizes its estimated probability of being flagged as AI-generated.

#### Acceptance Criteria

1. WHEN the Detector_Optimizer stage is enabled and receives non-empty input text, THE
   Detector_Optimizer SHALL compute a Detection_Risk_Score in the range 0 to 100 inclusive for the
   input text and for each candidate produced at each optimization iteration.
2. THE Detector_Optimizer SHALL perform optimization iterations until the Detection_Risk_Score reaches
   the configured target threshold, which is an integer in the range 0 to 100 inclusive, OR the
   configured maximum iteration count, which is an integer in the range 1 to 20 inclusive, is reached,
   whichever occurs first.
3. THE Detector_Optimizer SHALL return the candidate text with the lowest observed Detection_Risk_Score
   among those candidates whose Semantic_Similarity_Score relative to the stage input text is at least
   0.85.
4. THE Detector_Optimizer SHALL only return a candidate text whose Semantic_Similarity_Score relative
   to the stage input text is at least 0.85.
5. IF no candidate text satisfies the minimum Semantic_Similarity_Score of 0.85, THEN THE
   Detector_Optimizer SHALL return the stage input text unchanged.
6. THE Detector_Optimizer SHALL limit the number of optimization iterations to the configured maximum
   iteration count.
7. THE Detector_Optimizer SHALL preserve every term in Protected_Terms unchanged in the returned text.
8. IF the Detection_Risk_Score source fails during the optimization loop, THEN THE Detector_Optimizer
   SHALL stop iterating, return the best valid candidate text whose Semantic_Similarity_Score is at
   least 0.85 or the stage input text unchanged when no valid candidate exists, and surface an error
   indication.

### Requirement 9: Transformer-Based Classification Model

**User Story:** As a user, I want a transformer-based classifier that estimates AI-detection risk, so
that the system can present an accurate risk estimate and guide detector-aware optimization.

#### Acceptance Criteria

1. WHEN the Classifier receives input text containing between 1 and 10,000 characters inclusive, THE
   Classifier SHALL return a numeric Detection_Risk_Score in the range 0 to 100 inclusive within 5
   seconds.
2. WHEN the Classifier receives byte-for-byte identical input text on repeated invocations using the
   same loaded model, THE Classifier SHALL return an identical Detection_Risk_Score.
3. WHERE the Detector_Optimizer is enabled, THE Detector_Optimizer SHALL use the Classifier as the
   source of the Detection_Risk_Score for optimization feedback.
4. IF the Classifier model cannot be loaded, fails during inference, or does not respond within 5
   seconds, THEN THE system SHALL fall back to the heuristic Risk_Scorer to produce the
   Detection_Risk_Score and SHALL surface an indication that the fallback Risk_Scorer was used.
5. WHEN the Detection_Risk_Score is produced by either the Classifier or the fallback Risk_Scorer, THE
   system SHALL make the score available to the UI for display in the before/after analytics.
6. IF the Classifier receives invalid input that is empty or exceeds 10,000 characters, THEN THE
   Classifier SHALL reject the input with an invalid-input indication.

### Requirement 10: Pipeline Integration and Orchestration

**User Story:** As a developer, I want the nine new capabilities integrated into the existing pipeline
orchestrator, so that they run in a defined order alongside the original five stages with consistent
configuration and control.

#### Acceptance Criteria

1. THE Pipeline SHALL execute each enabled new stage in a deterministic order relative to the existing
   five stages, producing an identical stage execution order across runs for identical configuration
   and input text.
2. WHEN a Stage_Toggle for a new stage is set to disabled, THE Pipeline SHALL skip that stage during
   processing.
3. WHEN a Stage_Toggle for a new stage is set to enabled, THE Pipeline SHALL execute that stage during
   processing.
4. WHEN an enabled new stage is about to execute, THE Pipeline SHALL invoke the Progress_Callback with
   the stage name and status "running" before the stage executes.
5. WHEN an enabled new stage completes successfully, THE Pipeline SHALL invoke the Progress_Callback
   with the stage name and status "complete".
6. WHEN a new stage is disabled or skipped, THE Pipeline SHALL emit no Progress_Callback notification
   for that stage.
7. IF a new stage raises an unhandled error during processing, THEN THE Pipeline SHALL retain the text
   produced by the most recently completed stage, or the original input text when no stage has
   completed, invoke the Progress_Callback with the stage name and status "error", and continue with
   the next enabled stage.
8. THE Pipeline SHALL pass the Seed to every new stage that performs non-LLM randomized operations.
9. WHEN all stages are disabled, THE Pipeline SHALL return the input text unchanged.
10. WHEN the input text is empty or contains only whitespace characters, THE Pipeline SHALL return the
    input text unchanged.

### Requirement 11: Intensity Profile Configuration

**User Story:** As a user, I want the new capabilities to respond to the intensity level, so that a
single intensity control governs how aggressively the entire pipeline transforms text.

#### Acceptance Criteria

1. THE Intensity_Profile for each integer level 1 through 5 inclusive SHALL define, for each of the
   nine new stages, an enabled flag as a boolean and an aggression value as a float in the range 0.0
   to 1.0 inclusive.
2. WHEN the user selects an intensity level in the range 1 to 5 inclusive, THE Pipeline SHALL apply
   the enabled flags and aggression values defined by the corresponding Intensity_Profile to each of
   the nine new stages.
3. FOR each of the nine new stages and for each adjacent level pair L and L+1 where L is 1 through 4,
   THE Intensity_Profile SHALL assign an aggression value at level L+1 that is greater than or equal to
   the aggression value assigned at level L.
4. WHEN a Stage_Toggle override is provided for a new stage, THE Pipeline SHALL use the Stage_Toggle
   value instead of the Intensity_Profile enabled flag to determine whether that stage executes, while
   continuing to apply the Intensity_Profile aggression value for that stage.
5. IF an intensity value below 1 is provided, THEN THE Pipeline SHALL clamp the value to level 1, and
   IF an intensity value above 5 is provided, THEN THE Pipeline SHALL clamp the value to level 5.
6. IF a non-integer intensity value within the range 1 to 5 inclusive is provided, THEN THE Pipeline
   SHALL round the value to the nearest integer level, rounding halves up to the next higher integer
   level.

### Requirement 12: User Interface Controls and Analytics

**User Story:** As a user of the Streamlit frontend, I want controls and analytics for the new
capabilities, so that I can configure them and see their effect on detection risk and meaning
preservation.

#### Acceptance Criteria

1. THE frontend SHALL present an individual Stage_Toggle control for each of the nine new capabilities,
   and each Stage_Toggle SHALL reflect the current enabled state of its stage and override the
   Intensity_Profile default enabled flag for that stage.
2. WHILE a pipeline run is in progress, THE frontend SHALL update the displayed status of each enabled
   new stage to "running" or "complete" within 1 second of each corresponding Progress_Callback
   notification.
3. WHEN a pipeline run completes, THE frontend SHALL display the before and after Detection_Risk_Score,
   each in the range 0 to 100 inclusive, for the processed text.
4. WHEN a pipeline run completes, THE frontend SHALL display the Semantic_Similarity_Score, in the
   range 0.0 to 1.0 inclusive, between the original input text and the final output text.
5. THE frontend SHALL display a disclaimer stating that the Detection_Risk_Score is an estimate and
   not a guarantee of evading any specific detector.
6. WHEN the user performs the export action, THE frontend SHALL produce a downloadable file containing
   the complete final output text.
7. IF a pipeline run terminates with an error, THEN THE frontend SHALL display an error indication and
   retain the last successful analytics.
8. WHILE no pipeline run has completed in the current session, THE frontend SHALL disable the export
   control.

### Requirement 13: Configuration Persistence

**User Story:** As a user, I want my pipeline and capability configuration to be saveable and
loadable, so that I can reuse a configuration across sessions.

#### Acceptance Criteria

1. THE Config_Serializer SHALL serialize the current intensity level as an integer in the range 1 to 5
   inclusive, every Stage_Toggle state, the Target_Perplexity_Profile target mean perplexity value and
   target perplexity variance value, and every per-stage aggression value as a float in the range 0.0
   to 1.0 inclusive, into a persisted representation.
2. WHEN the Config_Serializer deserializes a persisted representation, THE Config_Serializer SHALL
   produce a configuration that populates the intensity level, every Stage_Toggle state, both
   Target_Perplexity_Profile values, and every per-stage aggression value usable by the Pipeline.
3. FOR ALL valid configurations, serializing a configuration and then deserializing the result SHALL
   produce a configuration equivalent to the original, where equivalent means field-by-field identity
   of the intensity level, every Stage_Toggle state, both Target_Perplexity_Profile values, and every
   per-stage aggression value (round-trip property).
4. IF the Config_Serializer receives a persisted representation with a missing field or an
   out-of-range value, THEN THE Config_Serializer SHALL return an error indicating which field is
   invalid and THE system SHALL retain the current active configuration unchanged.

### Requirement 14: Meaning Preservation and Academic Accuracy

**User Story:** As a user humanizing academic work, I want the system to preserve meaning and academic
accuracy across the full pipeline, so that the humanized output remains factually faithful to the
original.

#### Acceptance Criteria

1. WHEN the Pipeline completes processing, THE system SHALL compute a Semantic_Similarity_Score, in
   the range 0.0 to 1.0 inclusive, between the original input text and the final output text.
2. THE system SHALL preserve every term in Protected_Terms with identical occurrence counts in the
   final output text relative to the original input text.
3. IF the Semantic_Similarity_Score between the final output text and the original input text is below
   0.85, THEN THE system SHALL return the final output text and surface a warning to the user, when
   processing completes, indicating that meaning preservation fell below 0.85.
4. THE system SHALL preserve all numeric values and citation markers from the input text in the final
   output text with identical values and identical occurrence counts.
5. IF any numeric value or citation marker present in the input text is dropped from the final output
   text, THEN THE system SHALL surface a warning to the user.
