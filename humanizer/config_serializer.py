"""
Configuration serialization and deserialization for the humanization pipeline.

Provides PipelineConfig (a dataclass capturing the full pipeline configuration),
ConfigError (raised on invalid/missing fields during deserialization), and
ConfigSerializer (serialize to JSON, deserialize from JSON with validation).

Requirements: 13.1, 13.2, 13.4
"""

import json
from dataclasses import dataclass, fields, asdict
from typing import Any


class ConfigError(Exception):
    """Raised when a configuration field is missing or has an invalid value.

    Attributes:
        field: The name of the invalid or missing configuration field.
    """

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"Invalid or missing configuration field: {field}")


# All stage toggle field names (booleans)
_STAGE_TOGGLE_FIELDS = (
    "semantic_transform_enabled",
    "iterative_paraphrase_enabled",
    "retrieval_augmented_enabled",
    "stylometric_enabled",
    "perplexity_optimize_enabled",
    "adversarial_enabled",
    "error_injection_enabled",
    "detector_optimize_enabled",
    "classifier_enabled",
)

# All per-stage aggression field names (floats in [0, 1])
_AGGRESSION_FIELDS = (
    "semantic_transform_aggression",
    "iterative_paraphrase_aggression",
    "retrieval_augmented_aggression",
    "stylometric_aggression",
    "perplexity_optimize_aggression",
    "adversarial_aggression",
    "error_injection_aggression",
    "detector_optimize_aggression",
)


@dataclass
class PipelineConfig:
    """Full pipeline configuration for serialization and deserialization.

    Captures intensity level, all stage toggles, per-stage aggression values,
    and the target perplexity profile.

    Attributes:
        intensity: Pipeline intensity level (1-5).
        semantic_transform_enabled: Whether the Semantic Transformer stage is on.
        iterative_paraphrase_enabled: Whether the Iterative Paraphraser stage is on.
        retrieval_augmented_enabled: Whether the Retrieval-Augmented Rewriter is on.
        stylometric_enabled: Whether the Stylometric Obfuscator stage is on.
        perplexity_optimize_enabled: Whether the Perplexity Optimizer stage is on.
        adversarial_enabled: Whether the Adversarial Rewriter stage is on.
        error_injection_enabled: Whether the Error Injector stage is on.
        detector_optimize_enabled: Whether the Detector Optimizer stage is on.
        classifier_enabled: Whether the Classifier stage is on.
        semantic_transform_aggression: Aggression for the Semantic Transformer [0,1].
        iterative_paraphrase_aggression: Aggression for the Iterative Paraphraser [0,1].
        retrieval_augmented_aggression: Aggression for Retrieval-Augmented [0,1].
        stylometric_aggression: Aggression for the Stylometric Obfuscator [0,1].
        perplexity_optimize_aggression: Aggression for the Perplexity Optimizer [0,1].
        adversarial_aggression: Aggression for the Adversarial Rewriter [0,1].
        error_injection_aggression: Aggression for the Error Injector [0,1].
        detector_optimize_aggression: Aggression for the Detector Optimizer [0,1].
        target_perplexity_mean: Target mean perplexity (> 0).
        target_perplexity_variance: Target perplexity variance (>= 0).
    """

    # Intensity level
    intensity: int

    # Stage toggles
    semantic_transform_enabled: bool
    iterative_paraphrase_enabled: bool
    retrieval_augmented_enabled: bool
    stylometric_enabled: bool
    perplexity_optimize_enabled: bool
    adversarial_enabled: bool
    error_injection_enabled: bool
    detector_optimize_enabled: bool
    classifier_enabled: bool

    # Per-stage aggression values
    semantic_transform_aggression: float
    iterative_paraphrase_aggression: float
    retrieval_augmented_aggression: float
    stylometric_aggression: float
    perplexity_optimize_aggression: float
    adversarial_aggression: float
    error_injection_aggression: float
    detector_optimize_aggression: float

    # Target perplexity profile
    target_perplexity_mean: float
    target_perplexity_variance: float


class ConfigSerializer:
    """Serializes and deserializes PipelineConfig to/from JSON strings.

    serialize() produces a JSON representation of the full pipeline config.
    deserialize() parses JSON and validates all fields, raising ConfigError
    on any missing or out-of-range field. The active configuration is never
    modified on error — a new PipelineConfig is returned only on success.
    """

    @staticmethod
    def serialize(config: PipelineConfig) -> str:
        """Serialize a PipelineConfig to a JSON string.

        Args:
            config: The pipeline configuration to serialize.

        Returns:
            A JSON string representation of the configuration.
        """
        return json.dumps(asdict(config), indent=2)

    @staticmethod
    def deserialize(blob: str) -> PipelineConfig:
        """Deserialize a JSON string into a PipelineConfig.

        Validates all fields for presence and correct range. On any invalid
        or missing field, raises ConfigError naming the offending field.
        The active configuration is left unchanged on error (this method
        constructs a new object only on success).

        Args:
            blob: A JSON string to parse.

        Returns:
            A validated PipelineConfig instance.

        Raises:
            ConfigError: If any field is missing or has an out-of-range value.
        """
        try:
            data: dict[str, Any] = json.loads(blob)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ConfigError("__root__") from exc

        if not isinstance(data, dict):
            raise ConfigError("__root__")

        # Collect all expected field names from the dataclass
        expected_fields = {f.name for f in fields(PipelineConfig)}

        # Check for missing fields first
        for field_name in expected_fields:
            if field_name not in data:
                raise ConfigError(field_name)

        # Validate intensity: must be int in [1, 5]
        intensity = data["intensity"]
        if not isinstance(intensity, int) or isinstance(intensity, bool):
            raise ConfigError("intensity")
        if intensity < 1 or intensity > 5:
            raise ConfigError("intensity")

        # Validate stage toggles: must be bool
        for toggle_field in _STAGE_TOGGLE_FIELDS:
            val = data[toggle_field]
            if not isinstance(val, bool):
                raise ConfigError(toggle_field)

        # Validate aggression values: must be float/int in [0.0, 1.0]
        for aggr_field in _AGGRESSION_FIELDS:
            val = data[aggr_field]
            if isinstance(val, bool):
                raise ConfigError(aggr_field)
            if not isinstance(val, (int, float)):
                raise ConfigError(aggr_field)
            if val < 0.0 or val > 1.0:
                raise ConfigError(aggr_field)

        # Validate target_perplexity_mean: must be > 0
        mean_val = data["target_perplexity_mean"]
        if isinstance(mean_val, bool):
            raise ConfigError("target_perplexity_mean")
        if not isinstance(mean_val, (int, float)):
            raise ConfigError("target_perplexity_mean")
        if mean_val <= 0:
            raise ConfigError("target_perplexity_mean")

        # Validate target_perplexity_variance: must be >= 0
        var_val = data["target_perplexity_variance"]
        if isinstance(var_val, bool):
            raise ConfigError("target_perplexity_variance")
        if not isinstance(var_val, (int, float)):
            raise ConfigError("target_perplexity_variance")
        if var_val < 0:
            raise ConfigError("target_perplexity_variance")

        # All validations passed — construct the config object
        return PipelineConfig(
            intensity=int(intensity),
            semantic_transform_enabled=data["semantic_transform_enabled"],
            iterative_paraphrase_enabled=data["iterative_paraphrase_enabled"],
            retrieval_augmented_enabled=data["retrieval_augmented_enabled"],
            stylometric_enabled=data["stylometric_enabled"],
            perplexity_optimize_enabled=data["perplexity_optimize_enabled"],
            adversarial_enabled=data["adversarial_enabled"],
            error_injection_enabled=data["error_injection_enabled"],
            detector_optimize_enabled=data["detector_optimize_enabled"],
            classifier_enabled=data["classifier_enabled"],
            semantic_transform_aggression=float(data["semantic_transform_aggression"]),
            iterative_paraphrase_aggression=float(data["iterative_paraphrase_aggression"]),
            retrieval_augmented_aggression=float(data["retrieval_augmented_aggression"]),
            stylometric_aggression=float(data["stylometric_aggression"]),
            perplexity_optimize_aggression=float(data["perplexity_optimize_aggression"]),
            adversarial_aggression=float(data["adversarial_aggression"]),
            error_injection_aggression=float(data["error_injection_aggression"]),
            detector_optimize_aggression=float(data["detector_optimize_aggression"]),
            target_perplexity_mean=float(mean_val),
            target_perplexity_variance=float(var_val),
        )
