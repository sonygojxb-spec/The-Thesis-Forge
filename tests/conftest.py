"""
Shared test fixtures for deterministic offline testing.

Provides:
- FakeSimilarityEvaluator: scriptable score(a, b) returning queued values.
- FakeClassifier: scriptable score(text) with a fail_after switch.
- fake_llm_pass_stream: patches LLMRewriter._llm_pass_stream to yield scripted
  chunks, raise RuntimeError, or yield empty.

Requirements: 9.2, 9.4 (test infrastructure for deterministic offline runs)
"""

from __future__ import annotations

from collections import deque
from typing import Generator, List, Optional
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# FakeSimilarityEvaluator
# ---------------------------------------------------------------------------


class FakeSimilarityEvaluator:
    """A scriptable similarity evaluator for injection into stages.

    Scores are consumed in FIFO order from a queue. When the queue is empty,
    returns a configurable default score.

    Usage:
        evaluator = FakeSimilarityEvaluator(scores=[0.95, 0.80, 0.92])
        evaluator.score("a", "b")  # -> 0.95
        evaluator.score("a", "b")  # -> 0.80
        evaluator.score("a", "b")  # -> 0.92
        evaluator.score("a", "b")  # -> default (0.95)
    """

    def __init__(
        self,
        scores: Optional[List[float]] = None,
        default: float = 0.95,
    ) -> None:
        self._queue: deque[float] = deque(scores or [])
        self._default = default
        self._calls: List[tuple] = []

    def score(self, a: str, b: str) -> float:
        """Return the next queued score, or default if queue is exhausted."""
        self._calls.append((a, b))
        if self._queue:
            return self._queue.popleft()
        return self._default

    def is_available(self) -> bool:
        """Always available in tests."""
        return True

    @property
    def calls(self) -> List[tuple]:
        """List of (a, b) pairs passed to score()."""
        return self._calls

    def reset(self, scores: Optional[List[float]] = None) -> None:
        """Reset the queue and call log."""
        self._queue = deque(scores or [])
        self._calls.clear()


# ---------------------------------------------------------------------------
# FakeClassifier
# ---------------------------------------------------------------------------


class FakeClassifier:
    """A scriptable classifier for injection into stages.

    Scores are consumed in FIFO order. After `fail_after` successful calls
    (if set), subsequent calls raise RuntimeError to simulate classifier
    failure mid-loop.

    Usage:
        clf = FakeClassifier(scores=[85.0, 60.0, 30.0], fail_after=2)
        clf.score("text1")  # -> 85.0
        clf.score("text2")  # -> 60.0
        clf.score("text3")  # raises RuntimeError (fail_after=2 reached)
    """

    class InvalidInput(Exception):
        """Raised when input is empty or exceeds 10,000 characters."""

        def __init__(self, reason: str) -> None:
            self.reason = reason
            super().__init__(reason)

    def __init__(
        self,
        scores: Optional[List[float]] = None,
        default: float = 50.0,
        fail_after: Optional[int] = None,
    ) -> None:
        self._queue: deque[float] = deque(scores or [])
        self._default = default
        self._fail_after = fail_after
        self._call_count = 0
        self._calls: List[str] = []

    def score(self, text: str) -> float:
        """Return the next queued score or default.

        Raises RuntimeError after `fail_after` successful calls (if set).
        Raises InvalidInput for empty text or text > 10,000 characters.
        """
        # Validate input like the real classifier
        if not text or not text.strip():
            raise self.InvalidInput("Input text is empty")
        if len(text) > 10_000:
            raise self.InvalidInput(
                f"Input exceeds 10,000 characters (got {len(text)})"
            )

        self._call_count += 1
        self._calls.append(text)

        if self._fail_after is not None and self._call_count > self._fail_after:
            raise RuntimeError(
                f"FakeClassifier: simulated failure after {self._fail_after} calls"
            )

        if self._queue:
            return self._queue.popleft()
        return self._default

    def is_available(self) -> bool:
        """Always available in tests."""
        return True

    @property
    def calls(self) -> List[str]:
        """List of texts passed to score()."""
        return self._calls

    @property
    def call_count(self) -> int:
        """Total number of calls made to score()."""
        return self._call_count

    def reset(
        self,
        scores: Optional[List[float]] = None,
        fail_after: Optional[int] = None,
    ) -> None:
        """Reset the queue, call count, and fail_after."""
        self._queue = deque(scores or [])
        self._fail_after = fail_after
        self._call_count = 0
        self._calls.clear()


# ---------------------------------------------------------------------------
# Fake LLM helper
# ---------------------------------------------------------------------------


class FakeLLMResponse:
    """Represents a scripted LLM response for use with fake_llm_pass_stream.

    Each response can be:
    - A list of string chunks (simulates successful streaming)
    - A RuntimeError instance (simulates LLM API error)
    - An empty list (simulates empty response)
    """

    def __init__(
        self,
        chunks: Optional[List[str]] = None,
        error: Optional[RuntimeError] = None,
    ) -> None:
        self.chunks = chunks
        self.error = error

    @classmethod
    def success(cls, text: str, chunk_size: int = 10) -> "FakeLLMResponse":
        """Create a successful response that yields text in chunks."""
        chunks = [
            text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
        ]
        return cls(chunks=chunks)

    @classmethod
    def error_response(cls, message: str = "LLM API error: connection timeout") -> "FakeLLMResponse":
        """Create an error response that raises RuntimeError."""
        return cls(error=RuntimeError(message))

    @classmethod
    def empty(cls) -> "FakeLLMResponse":
        """Create a response that yields no content (empty)."""
        return cls(chunks=[])


class FakeLLMPassStream:
    """A configurable fake for LLMRewriter._llm_pass_stream.

    Queues scripted responses and replays them in order. When the queue is
    empty, falls back to a default response.

    Usage as a context manager to patch the real method:

        responses = [
            FakeLLMResponse.success("rewritten text"),
            FakeLLMResponse.error_response("timeout"),
            FakeLLMResponse.empty(),
        ]
        with fake_llm_pass_stream(responses) as fake:
            result = rewriter.process("input text")
    """

    def __init__(
        self,
        responses: Optional[List[FakeLLMResponse]] = None,
        default: Optional[FakeLLMResponse] = None,
    ) -> None:
        self._queue: deque[FakeLLMResponse] = deque(responses or [])
        self._default = default or FakeLLMResponse.empty()
        self._call_count = 0
        self._calls: List[tuple] = []

    def __call__(
        self, text: str, system_prompt: str, temperature: float
    ) -> Generator[str, None, None]:
        """Simulate _llm_pass_stream: yield chunks, raise, or yield nothing.

        When patched onto LLMRewriter as an instance attribute, Python calls
        this directly without passing `self` of the rewriter (because callable
        objects are not descriptors).
        """
        self._call_count += 1
        self._calls.append((text, system_prompt, temperature))

        response = self._queue.popleft() if self._queue else self._default

        if response.error is not None:
            raise response.error

        if response.chunks:
            for chunk in response.chunks:
                yield chunk

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def calls(self) -> List[tuple]:
        """List of (text, system_prompt, temperature) tuples."""
        return self._calls

    def reset(self, responses: Optional[List[FakeLLMResponse]] = None) -> None:
        """Reset the queue and call count."""
        self._queue = deque(responses or [])
        self._call_count = 0
        self._calls.clear()


def fake_llm_pass_stream(
    responses: Optional[List[FakeLLMResponse]] = None,
    default: Optional[FakeLLMResponse] = None,
):
    """Context manager that patches LLMRewriter._llm_pass_stream with a fake.

    Usage:
        responses = [FakeLLMResponse.success("rewritten")]
        with fake_llm_pass_stream(responses) as fake:
            rewriter = LLMRewriter(aggression=0.5, model="m", api_key="k",
                                   base_url="http://x")
            result = rewriter.process("original text")
            assert fake.call_count == 2  # pass1 + pass2

    Returns the FakeLLMPassStream instance so tests can inspect calls.
    """
    fake = FakeLLMPassStream(responses=responses, default=default)
    return patch(
        "humanizer.stage_llm_rewrite.LLMRewriter._llm_pass_stream",
        new=fake,
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_similarity() -> FakeSimilarityEvaluator:
    """Provide a fresh FakeSimilarityEvaluator instance."""
    return FakeSimilarityEvaluator()


@pytest.fixture
def fake_classifier() -> FakeClassifier:
    """Provide a fresh FakeClassifier instance."""
    return FakeClassifier()


@pytest.fixture
def fake_classifier_with_fail() -> FakeClassifier:
    """Provide a FakeClassifier that fails after 2 calls."""
    return FakeClassifier(fail_after=2)


@pytest.fixture
def llm_success_response():
    """Factory fixture for creating successful LLM responses."""
    def _make(text: str, chunk_size: int = 10) -> FakeLLMResponse:
        return FakeLLMResponse.success(text, chunk_size)
    return _make


@pytest.fixture
def llm_error_response():
    """Factory fixture for creating error LLM responses."""
    def _make(message: str = "LLM API error: connection timeout") -> FakeLLMResponse:
        return FakeLLMResponse.error_response(message)
    return _make


@pytest.fixture
def llm_empty_response():
    """Factory fixture for creating empty LLM responses."""
    return FakeLLMResponse.empty()
