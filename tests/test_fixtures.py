"""Smoke tests for the shared test fixtures themselves."""

import pytest

from tests.conftest import (
    FakeClassifier,
    FakeLLMPassStream,
    FakeLLMResponse,
    FakeSimilarityEvaluator,
    fake_llm_pass_stream,
)


class TestFakeSimilarityEvaluator:
    """Verify FakeSimilarityEvaluator returns queued scores."""

    def test_returns_queued_scores_in_order(self):
        ev = FakeSimilarityEvaluator(scores=[0.95, 0.80, 0.70])
        assert ev.score("a", "b") == 0.95
        assert ev.score("c", "d") == 0.80
        assert ev.score("e", "f") == 0.70

    def test_returns_default_when_queue_empty(self):
        ev = FakeSimilarityEvaluator(scores=[0.9], default=0.5)
        assert ev.score("a", "b") == 0.9
        assert ev.score("a", "b") == 0.5
        assert ev.score("a", "b") == 0.5

    def test_is_available(self):
        ev = FakeSimilarityEvaluator()
        assert ev.is_available() is True

    def test_tracks_calls(self):
        ev = FakeSimilarityEvaluator(scores=[0.9])
        ev.score("hello", "world")
        assert ev.calls == [("hello", "world")]

    def test_reset(self):
        ev = FakeSimilarityEvaluator(scores=[0.9])
        ev.score("a", "b")
        ev.reset(scores=[0.5, 0.6])
        assert ev.score("c", "d") == 0.5
        assert len(ev.calls) == 1  # reset clears old calls


class TestFakeClassifier:
    """Verify FakeClassifier returns scores and fails on schedule."""

    def test_returns_queued_scores(self):
        clf = FakeClassifier(scores=[85.0, 60.0, 30.0])
        assert clf.score("text1") == 85.0
        assert clf.score("text2") == 60.0
        assert clf.score("text3") == 30.0

    def test_returns_default_when_queue_empty(self):
        clf = FakeClassifier(scores=[85.0], default=42.0)
        assert clf.score("text1") == 85.0
        assert clf.score("text2") == 42.0

    def test_fail_after(self):
        clf = FakeClassifier(scores=[85.0, 60.0, 30.0], fail_after=2)
        assert clf.score("text1") == 85.0
        assert clf.score("text2") == 60.0
        with pytest.raises(RuntimeError, match="simulated failure"):
            clf.score("text3")

    def test_rejects_empty_input(self):
        clf = FakeClassifier(scores=[50.0])
        with pytest.raises(FakeClassifier.InvalidInput):
            clf.score("")

    def test_rejects_whitespace_only(self):
        clf = FakeClassifier(scores=[50.0])
        with pytest.raises(FakeClassifier.InvalidInput):
            clf.score("   ")

    def test_rejects_over_10000_chars(self):
        clf = FakeClassifier(scores=[50.0])
        with pytest.raises(FakeClassifier.InvalidInput):
            clf.score("x" * 10_001)

    def test_is_available(self):
        clf = FakeClassifier()
        assert clf.is_available() is True

    def test_tracks_call_count(self):
        clf = FakeClassifier(scores=[50.0, 50.0])
        clf.score("a")
        clf.score("b")
        assert clf.call_count == 2

    def test_reset(self):
        clf = FakeClassifier(scores=[50.0], fail_after=1)
        clf.score("a")
        clf.reset(scores=[70.0], fail_after=None)
        assert clf.score("b") == 70.0
        assert clf.call_count == 1


class TestFakeLLMResponse:
    """Verify FakeLLMResponse factory methods."""

    def test_success_chunks(self):
        resp = FakeLLMResponse.success("Hello world!", chunk_size=5)
        assert resp.chunks == ["Hello", " worl", "d!"]
        assert resp.error is None

    def test_error_response(self):
        resp = FakeLLMResponse.error_response("timeout!")
        assert resp.error is not None
        assert resp.chunks is None

    def test_empty(self):
        resp = FakeLLMResponse.empty()
        assert resp.chunks == []
        assert resp.error is None


class TestFakeLLMPassStream:
    """Verify FakeLLMPassStream yields, raises, or returns empty."""

    def test_yields_chunks(self):
        responses = [FakeLLMResponse.success("rewritten text", chunk_size=8)]
        fake = FakeLLMPassStream(responses=responses)
        chunks = list(fake("input", "prompt", 0.7))
        assert "".join(chunks) == "rewritten text"

    def test_raises_on_error(self):
        responses = [FakeLLMResponse.error_response("API down")]
        fake = FakeLLMPassStream(responses=responses)
        with pytest.raises(RuntimeError, match="API down"):
            list(fake("input", "prompt", 0.7))

    def test_yields_empty(self):
        responses = [FakeLLMResponse.empty()]
        fake = FakeLLMPassStream(responses=responses)
        chunks = list(fake("input", "prompt", 0.7))
        assert chunks == []

    def test_consumes_queue_in_order(self):
        responses = [
            FakeLLMResponse.success("first"),
            FakeLLMResponse.success("second"),
        ]
        fake = FakeLLMPassStream(responses=responses)
        assert "".join(fake("t", "p", 0.5)) == "first"
        assert "".join(fake("t", "p", 0.5)) == "second"

    def test_falls_back_to_default(self):
        default = FakeLLMResponse.success("default output")
        fake = FakeLLMPassStream(responses=[], default=default)
        assert "".join(fake("t", "p", 0.5)) == "default output"

    def test_tracks_calls(self):
        fake = FakeLLMPassStream(
            responses=[FakeLLMResponse.success("x")]
        )
        list(fake("text", "sys_prompt", 0.8))
        assert fake.call_count == 1
        assert fake.calls == [("text", "sys_prompt", 0.8)]


class TestFakeLLMContextManager:
    """Verify the fake_llm_pass_stream context manager patches correctly."""

    def test_patch_applies(self):
        from humanizer.stage_llm_rewrite import LLMRewriter

        responses = [
            FakeLLMResponse.success("pass1 output"),
            FakeLLMResponse.success("pass2 output"),
        ]
        with fake_llm_pass_stream(responses) as patcher:
            rewriter = LLMRewriter(
                aggression=0.5,
                model="test-model",
                api_key="test-key",
                base_url="http://localhost",
            )
            result = rewriter.process("original text")
            # The fake patches _llm_pass_stream so no real HTTP call happens
            assert result is not None
