"""
Unit tests for humanizer/ui_helpers.py — pure helper functions extracted
from app.py for testability.

Tests cover:
- Toggle reflect/override (Req 12.1)
- Analytics computation (Req 12.3, 12.4)
- Export payload (Req 12.6)
- Export-disabled state (Req 12.8)
- Disclaimer presence (Req 12.5)
"""

import pytest

from humanizer.ui_helpers import (
    DISCLAIMER_TEXT,
    build_stage_overrides,
    compute_analytics_payload,
    build_export_payload,
    is_export_enabled,
)


# ---------------------------------------------------------------------------
# 12.1: Toggle reflect/override — build_stage_overrides
# ---------------------------------------------------------------------------


class TestBuildStageOverrides:
    """Validates: Requirements 12.1"""

    def test_all_stages_enabled(self):
        """All toggles enabled should produce an all-True override dict."""
        toggles = {
            "structural": True,
            "lexical": True,
            "llm_rewrite": True,
            "perplexity": True,
            "postprocess": True,
            "semantic_transform": True,
            "iterative_paraphrase": True,
            "retrieval_augmented": True,
            "stylometric": True,
            "perplexity_optimize": True,
            "adversarial": True,
            "error_injection": True,
            "detector_optimize": True,
            "classifier": True,
        }
        result = build_stage_overrides(toggles)
        assert result == toggles
        # Each key maps 1:1
        for key, val in toggles.items():
            assert result[key] is val

    def test_all_stages_disabled(self):
        """All toggles disabled should produce an all-False override dict."""
        toggles = {
            "structural": False,
            "lexical": False,
            "llm_rewrite": False,
            "perplexity": False,
            "postprocess": False,
            "semantic_transform": False,
            "iterative_paraphrase": False,
            "retrieval_augmented": False,
            "stylometric": False,
            "perplexity_optimize": False,
            "adversarial": False,
            "error_injection": False,
            "detector_optimize": False,
            "classifier": False,
        }
        result = build_stage_overrides(toggles)
        for key in toggles:
            assert result[key] is False

    def test_mixed_toggles_reflect_state(self):
        """Mixed toggles should reflect each individual state correctly."""
        toggles = {
            "structural": True,
            "lexical": False,
            "semantic_transform": True,
            "adversarial": False,
            "detector_optimize": True,
        }
        result = build_stage_overrides(toggles)
        assert result["structural"] is True
        assert result["lexical"] is False
        assert result["semantic_transform"] is True
        assert result["adversarial"] is False
        assert result["detector_optimize"] is True

    def test_empty_toggles(self):
        """Empty toggles dict should produce empty overrides."""
        result = build_stage_overrides({})
        assert result == {}

    def test_returns_new_dict(self):
        """build_stage_overrides returns a copy, not the original dict."""
        toggles = {"structural": True}
        result = build_stage_overrides(toggles)
        assert result is not toggles


# ---------------------------------------------------------------------------
# 12.3, 12.4: Analytics computation — compute_analytics_payload
# ---------------------------------------------------------------------------


class TestComputeAnalyticsPayload:
    """Validates: Requirements 12.3, 12.4"""

    def test_basic_improvement(self):
        """Score decrease (improvement) should show negative score_change."""
        payload = compute_analytics_payload(
            before_score=75.0, after_score=30.0, similarity=0.92
        )
        assert payload["before_score"] == 75.0
        assert payload["after_score"] == 30.0
        assert payload["score_change"] == -45.0
        assert payload["similarity"] == 0.92

    def test_no_change(self):
        """Equal scores should show zero score_change."""
        payload = compute_analytics_payload(
            before_score=50.0, after_score=50.0, similarity=0.99
        )
        assert payload["score_change"] == 0.0

    def test_score_worsened(self):
        """Score increase (worsened) should show positive score_change."""
        payload = compute_analytics_payload(
            before_score=20.0, after_score=65.0, similarity=0.88
        )
        assert payload["score_change"] == 45.0

    def test_similarity_none(self):
        """Similarity can be None when not available."""
        payload = compute_analytics_payload(
            before_score=80.0, after_score=40.0, similarity=None
        )
        assert payload["similarity"] is None

    def test_boundary_scores(self):
        """Boundary values at extremes (0 and 100) work correctly."""
        payload = compute_analytics_payload(
            before_score=100.0, after_score=0.0, similarity=1.0
        )
        assert payload["before_score"] == 100.0
        assert payload["after_score"] == 0.0
        assert payload["score_change"] == -100.0
        assert payload["similarity"] == 1.0


# ---------------------------------------------------------------------------
# 12.6: Export payload — build_export_payload
# ---------------------------------------------------------------------------


class TestBuildExportPayload:
    """Validates: Requirements 12.6"""

    def test_returns_full_text(self):
        """Export payload should be the complete final text, unchanged."""
        text = "This is the humanized output text with full content."
        assert build_export_payload(text) == text

    def test_preserves_whitespace_and_newlines(self):
        """Export preserves all whitespace, newlines, and formatting."""
        text = "Line one.\n\nLine two.\n  Indented line.\n"
        assert build_export_payload(text) == text

    def test_empty_string(self):
        """Export of empty string returns empty string."""
        assert build_export_payload("") == ""

    def test_long_text(self):
        """Export handles large text without truncation."""
        text = "word " * 10000
        assert build_export_payload(text) == text


# ---------------------------------------------------------------------------
# 12.8: Export-disabled state — is_export_enabled
# ---------------------------------------------------------------------------


class TestIsExportEnabled:
    """Validates: Requirements 12.8"""

    def test_empty_string_disabled(self):
        """Empty output means export should be disabled."""
        assert is_export_enabled("") is False

    def test_non_empty_string_enabled(self):
        """Non-empty output means export should be enabled."""
        assert is_export_enabled("Some humanized text") is True

    def test_whitespace_only_enabled(self):
        """Whitespace-only is still truthy — export is enabled."""
        # A run that produces only whitespace still "completed"
        assert is_export_enabled("   ") is True

    def test_single_char_enabled(self):
        """Even a single character of output enables export."""
        assert is_export_enabled("x") is True


# ---------------------------------------------------------------------------
# 12.5: Disclaimer presence
# ---------------------------------------------------------------------------


class TestDisclaimer:
    """Validates: Requirements 12.5"""

    def test_disclaimer_constant_exists(self):
        """DISCLAIMER_TEXT should be a non-empty string."""
        assert isinstance(DISCLAIMER_TEXT, str)
        assert len(DISCLAIMER_TEXT) > 0

    def test_disclaimer_mentions_estimate(self):
        """Disclaimer should convey that scores are estimates, not guarantees."""
        lower = DISCLAIMER_TEXT.lower()
        assert "estimate" in lower
        # Should mention it's not a guarantee
        assert "not" in lower or "may not" in lower

    def test_disclaimer_content(self):
        """Disclaimer matches the expected text from app.py."""
        expected = (
            "Scores are estimates based on heuristic analysis and may not "
            "match specific AI detection tools."
        )
        assert DISCLAIMER_TEXT == expected
