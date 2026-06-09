"""Unit tests for humanizer.stage_iterative.IterativeParaphraser."""

from unittest.mock import patch

from humanizer.stage_iterative import IterativeParaphraser


class FakeHighSimilarity:
    def score(self, a, b):
        return 0.95


class FakeLowSimilarity:
    def score(self, a, b):
        return 0.70


def _mock_llm_identity(self, text, pass_index):
    """LLM mock that returns text unchanged (preserving placeholders)."""
    return text


def _mock_llm_append(self, text, pass_index):
    """LLM mock that appends a marker."""
    return text + " paraphrased"


def _mock_llm_fail(self, text, pass_index):
    """LLM mock that always fails."""
    raise RuntimeError("Connection error")


def _mock_llm_empty(self, text, pass_index):
    """LLM mock that returns empty."""
    return ""


class TestPassCountFormula:
    def test_aggression_0(self):
        assert IterativeParaphraser(aggression=0.0).pass_count == 1

    def test_aggression_025(self):
        assert IterativeParaphraser(aggression=0.25).pass_count == 2

    def test_aggression_05(self):
        assert IterativeParaphraser(aggression=0.5).pass_count == 3

    def test_aggression_075(self):
        assert IterativeParaphraser(aggression=0.75).pass_count == 4

    def test_aggression_1(self):
        assert IterativeParaphraser(aggression=1.0).pass_count == 5


class TestEmptyInput:
    def test_empty_string(self):
        p = IterativeParaphraser(aggression=0.5, seed=42)
        result = p.process_measured("")
        assert result.text == ""
        assert result.changed is False
        assert result.fell_back is False

    def test_whitespace_only(self):
        p = IterativeParaphraser(aggression=0.5, seed=42)
        result = p.process_measured("   ")
        assert result.text == "   "
        assert result.changed is False
        assert result.fell_back is False


class TestLLMFailures:
    @patch.object(IterativeParaphraser, "_llm_pass", _mock_llm_fail)
    def test_first_pass_failure_returns_original(self):
        """Req 1.8: first-pass failure → return original."""
        p = IterativeParaphraser(aggression=0.5, seed=42)
        result = p.process_measured("Hello world.")
        assert result.text == "Hello world."
        assert result.changed is False
        assert result.fell_back is True
        assert result.error is not None

    @patch.object(IterativeParaphraser, "_llm_pass", _mock_llm_empty)
    def test_first_pass_empty_returns_original(self):
        """Req 1.8: first-pass empty → return original."""
        p = IterativeParaphraser(aggression=0.5, seed=42)
        result = p.process_measured("Hello world.")
        assert result.text == "Hello world."
        assert result.changed is False
        assert result.fell_back is True

    def test_second_pass_failure_returns_last_good(self):
        """Req 1.6: LLM error with prior success → last good pass."""
        call_count = [0]

        def mock(self, text, pass_index):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Paraphrased once."
            raise RuntimeError("Timeout")

        with patch.object(IterativeParaphraser, "_llm_pass", mock):
            p = IterativeParaphraser(
                aggression=0.5, seed=42, similarity=FakeHighSimilarity()
            )
            result = p.process_measured("Original text.")
            assert result.text == "Paraphrased once."
            assert result.changed is True
            assert result.error is not None


class TestSimilarityFloor:
    @patch.object(IterativeParaphraser, "_llm_pass", _mock_llm_append)
    def test_below_floor_discards_pass(self):
        """Req 1.5: similarity < 0.80 → discard pass."""
        p = IterativeParaphraser(
            aggression=0.5, seed=42, similarity=FakeLowSimilarity()
        )
        result = p.process_measured("Hello world.")
        # First pass gets discarded, no prior good output → original
        assert result.text == "Hello world."
        assert result.changed is False
        assert result.fell_back is True


class TestProtectedSpans:
    @patch.object(IterativeParaphraser, "_llm_pass", _mock_llm_identity)
    def test_protected_terms_preserved(self):
        """Req 1.4: protected terms preserved."""
        p = IterativeParaphraser(
            aggression=0.0, seed=42, similarity=FakeHighSimilarity()
        )
        text = "The algorithm processes the coefficient matrix."
        result = p.process_measured(text)
        assert "algorithm" in result.text
        assert "coefficient" in result.text


class TestProcessDelegation:
    @patch.object(IterativeParaphraser, "_llm_pass", _mock_llm_identity)
    def test_process_returns_text(self):
        """process() delegates to process_measured() and returns text."""
        p = IterativeParaphraser(
            aggression=0.0, seed=42, similarity=FakeHighSimilarity()
        )
        text = "Test text."
        result = p.process(text)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Example tests: pass chaining and LLM failure recovery (Task 6.5)
# Requirements: 1.3, 1.6, 1.8, 1.9
# ---------------------------------------------------------------------------

from tests.conftest import FakeSimilarityEvaluator


class TestPassChainingAndLLMFailureRecovery:
    """Example tests using FakeSimilarityEvaluator and patch.object on _llm_pass."""

    def test_pass_output_chaining(self):
        """Req 1.3: Each pass feeds its output as input to the next pass.

        Mock _llm_pass to append a marker on each pass. With aggression=1.0
        (5 passes), the output should show markers from all 5 passes chained.
        """
        def mock_append_marker(self, text, pass_index):
            return text + f"[P{pass_index}]"

        evaluator = FakeSimilarityEvaluator(default=0.95)

        with patch.object(IterativeParaphraser, "_llm_pass", mock_append_marker):
            p = IterativeParaphraser(
                aggression=1.0, seed=42, similarity=evaluator
            )
            result = p.process("Start.")

        # aggression=1.0 → 5 passes; each appends [P0], [P1], ..., [P4]
        assert result == "Start.[P0][P1][P2][P3][P4]"

    def test_llm_error_with_prior_success_returns_last_good(self):
        """Req 1.6: LLM error with prior success → return last good pass output.

        Pass 1 succeeds, pass 2 raises RuntimeError. The output should be
        the result of pass 1.
        """
        call_count = [0]

        def mock_succeed_then_fail(self, text, pass_index):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Successfully paraphrased."
            raise RuntimeError("LLM connection error")

        evaluator = FakeSimilarityEvaluator(default=0.95)

        with patch.object(IterativeParaphraser, "_llm_pass", mock_succeed_then_fail):
            p = IterativeParaphraser(
                aggression=0.5, seed=42, similarity=evaluator
            )
            result = p.process_measured("Original text here.")

        assert result.text == "Successfully paraphrased."
        assert result.changed is True
        assert result.error is not None

    def test_first_pass_failure_no_prior_returns_original(self):
        """Req 1.8: First-pass failure with no prior success → return original.

        _llm_pass fails immediately on the first call. The output must equal
        the original input text unchanged.
        """
        def mock_always_fail(self, text, pass_index):
            raise RuntimeError("Service unavailable")

        evaluator = FakeSimilarityEvaluator(default=0.95)

        with patch.object(IterativeParaphraser, "_llm_pass", mock_always_fail):
            p = IterativeParaphraser(
                aggression=0.5, seed=42, similarity=evaluator
            )
            original = "This is the original academic text."
            result = p.process_measured(original)

        assert result.text == original
        assert result.changed is False
        assert result.fell_back is True
        assert result.error is not None

    def test_per_pass_timeout_treated_as_failure(self):
        """Req 1.9: Per-pass timeout treated as failure.

        A RuntimeError('timeout') raised by _llm_pass should be handled the
        same as any other LLM error — if no prior success exists, return
        original; if a prior success exists, return last good pass.
        """
        call_count = [0]

        def mock_succeed_then_timeout(self, text, pass_index):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Pass one done."
            raise RuntimeError("timeout")

        evaluator = FakeSimilarityEvaluator(default=0.95)

        with patch.object(IterativeParaphraser, "_llm_pass", mock_succeed_then_timeout):
            p = IterativeParaphraser(
                aggression=0.5, seed=42, similarity=evaluator
            )
            result = p.process_measured("Original input for timeout test.")

        # Timeout on pass 2 → last good pass (pass 1) is returned
        assert result.text == "Pass one done."
        assert result.changed is True
        assert result.error is not None
        assert "timeout" in result.error
