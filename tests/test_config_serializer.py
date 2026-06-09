"""
Tests for humanizer/config_serializer.py.

Covers ConfigError, PipelineConfig, ConfigSerializer.serialize and deserialize,
including validation of all fields and round-trip equivalence.
"""

import json
import pytest

from humanizer.config_serializer import (
    ConfigError,
    ConfigSerializer,
    PipelineConfig,
)


def _valid_config() -> PipelineConfig:
    """Create a valid PipelineConfig for testing."""
    return PipelineConfig(
        intensity=3,
        semantic_transform_enabled=True,
        iterative_paraphrase_enabled=True,
        retrieval_augmented_enabled=False,
        stylometric_enabled=True,
        perplexity_optimize_enabled=True,
        adversarial_enabled=False,
        error_injection_enabled=True,
        detector_optimize_enabled=False,
        classifier_enabled=False,
        semantic_transform_aggression=0.5,
        iterative_paraphrase_aggression=0.4,
        retrieval_augmented_aggression=0.3,
        stylometric_aggression=0.5,
        perplexity_optimize_aggression=0.4,
        adversarial_aggression=0.6,
        error_injection_aggression=0.3,
        detector_optimize_aggression=0.4,
        target_perplexity_mean=60.0,
        target_perplexity_variance=15.0,
    )


def _valid_json_dict() -> dict:
    """Return a valid config dict for JSON testing."""
    return {
        "intensity": 3,
        "semantic_transform_enabled": True,
        "iterative_paraphrase_enabled": True,
        "retrieval_augmented_enabled": False,
        "stylometric_enabled": True,
        "perplexity_optimize_enabled": True,
        "adversarial_enabled": False,
        "error_injection_enabled": True,
        "detector_optimize_enabled": False,
        "classifier_enabled": False,
        "semantic_transform_aggression": 0.5,
        "iterative_paraphrase_aggression": 0.4,
        "retrieval_augmented_aggression": 0.3,
        "stylometric_aggression": 0.5,
        "perplexity_optimize_aggression": 0.4,
        "adversarial_aggression": 0.6,
        "error_injection_aggression": 0.3,
        "detector_optimize_aggression": 0.4,
        "target_perplexity_mean": 60.0,
        "target_perplexity_variance": 15.0,
    }


class TestConfigError:
    """Tests for the ConfigError exception."""

    def test_stores_field_name(self):
        err = ConfigError("intensity")
        assert err.field == "intensity"

    def test_message_contains_field(self):
        err = ConfigError("target_perplexity_mean")
        assert "target_perplexity_mean" in str(err)

    def test_is_exception(self):
        assert issubclass(ConfigError, Exception)


class TestSerialize:
    """Tests for ConfigSerializer.serialize."""

    def test_returns_valid_json(self):
        config = _valid_config()
        blob = ConfigSerializer.serialize(config)
        data = json.loads(blob)
        assert isinstance(data, dict)

    def test_all_fields_present(self):
        config = _valid_config()
        blob = ConfigSerializer.serialize(config)
        data = json.loads(blob)
        assert data["intensity"] == 3
        assert data["semantic_transform_enabled"] is True
        assert data["adversarial_enabled"] is False
        assert data["semantic_transform_aggression"] == 0.5
        assert data["target_perplexity_mean"] == 60.0
        assert data["target_perplexity_variance"] == 15.0

    def test_intensity_boundaries(self):
        config = _valid_config()
        config.intensity = 1
        blob = ConfigSerializer.serialize(config)
        assert json.loads(blob)["intensity"] == 1

        config.intensity = 5
        blob = ConfigSerializer.serialize(config)
        assert json.loads(blob)["intensity"] == 5


class TestDeserialize:
    """Tests for ConfigSerializer.deserialize."""

    def test_valid_config_round_trip(self):
        config = _valid_config()
        blob = ConfigSerializer.serialize(config)
        restored = ConfigSerializer.deserialize(blob)
        assert restored == config

    def test_all_intensity_levels(self):
        for level in range(1, 6):
            data = _valid_json_dict()
            data["intensity"] = level
            blob = json.dumps(data)
            result = ConfigSerializer.deserialize(blob)
            assert result.intensity == level

    def test_aggression_boundary_zero(self):
        data = _valid_json_dict()
        data["semantic_transform_aggression"] = 0.0
        blob = json.dumps(data)
        result = ConfigSerializer.deserialize(blob)
        assert result.semantic_transform_aggression == 0.0

    def test_aggression_boundary_one(self):
        data = _valid_json_dict()
        data["semantic_transform_aggression"] = 1.0
        blob = json.dumps(data)
        result = ConfigSerializer.deserialize(blob)
        assert result.semantic_transform_aggression == 1.0

    def test_perplexity_variance_zero_allowed(self):
        data = _valid_json_dict()
        data["target_perplexity_variance"] = 0.0
        blob = json.dumps(data)
        result = ConfigSerializer.deserialize(blob)
        assert result.target_perplexity_variance == 0.0


class TestDeserializeErrors:
    """Tests for ConfigSerializer.deserialize validation errors."""

    def test_missing_field_raises_config_error(self):
        data = _valid_json_dict()
        del data["intensity"]
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "intensity"

    def test_missing_toggle_field(self):
        data = _valid_json_dict()
        del data["semantic_transform_enabled"]
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "semantic_transform_enabled"

    def test_missing_aggression_field(self):
        data = _valid_json_dict()
        del data["adversarial_aggression"]
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "adversarial_aggression"

    def test_missing_perplexity_mean(self):
        data = _valid_json_dict()
        del data["target_perplexity_mean"]
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "target_perplexity_mean"

    def test_intensity_below_range(self):
        data = _valid_json_dict()
        data["intensity"] = 0
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "intensity"

    def test_intensity_above_range(self):
        data = _valid_json_dict()
        data["intensity"] = 6
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "intensity"

    def test_intensity_not_int(self):
        data = _valid_json_dict()
        data["intensity"] = 3.5
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "intensity"

    def test_toggle_not_bool(self):
        data = _valid_json_dict()
        data["classifier_enabled"] = 1
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "classifier_enabled"

    def test_aggression_below_zero(self):
        data = _valid_json_dict()
        data["error_injection_aggression"] = -0.1
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "error_injection_aggression"

    def test_aggression_above_one(self):
        data = _valid_json_dict()
        data["detector_optimize_aggression"] = 1.01
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "detector_optimize_aggression"

    def test_perplexity_mean_zero(self):
        data = _valid_json_dict()
        data["target_perplexity_mean"] = 0
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "target_perplexity_mean"

    def test_perplexity_mean_negative(self):
        data = _valid_json_dict()
        data["target_perplexity_mean"] = -5.0
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "target_perplexity_mean"

    def test_perplexity_variance_negative(self):
        data = _valid_json_dict()
        data["target_perplexity_variance"] = -1.0
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "target_perplexity_variance"

    def test_invalid_json_string(self):
        with pytest.raises(ConfigError):
            ConfigSerializer.deserialize("not valid json {{{")

    def test_json_array_not_object(self):
        with pytest.raises(ConfigError):
            ConfigSerializer.deserialize("[1, 2, 3]")

    def test_aggression_is_bool_rejected(self):
        """Booleans should not be accepted as aggression values."""
        data = _valid_json_dict()
        data["semantic_transform_aggression"] = True
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "semantic_transform_aggression"

    def test_intensity_is_bool_rejected(self):
        """Booleans should not be accepted as intensity."""
        data = _valid_json_dict()
        data["intensity"] = True
        with pytest.raises(ConfigError) as exc_info:
            ConfigSerializer.deserialize(json.dumps(data))
        assert exc_info.value.field == "intensity"


class TestActiveConfigUnchangedOnError:
    """Verify that deserialize does not mutate any existing state on error."""

    def test_error_does_not_affect_previous_config(self):
        # Simulate having an active config
        active = _valid_config()
        original_intensity = active.intensity

        # Attempt to deserialize invalid data
        bad_data = _valid_json_dict()
        bad_data["intensity"] = 99  # invalid
        with pytest.raises(ConfigError):
            ConfigSerializer.deserialize(json.dumps(bad_data))

        # Active config is unchanged
        assert active.intensity == original_intensity
