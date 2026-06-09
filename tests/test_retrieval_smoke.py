"""
Smoke test: default Reference_Corpus loads into RetrievalService.

Validates: Requirements 7.1
"""

from humanizer.retrieval import RetrievalService, ReferenceEntry


class TestDefaultCorpusLoads:
    """Verify that the bundled default corpus loads correctly."""

    def test_corpus_has_at_least_one_entry(self):
        """Default corpus must load at least one ReferenceEntry."""
        service = RetrievalService()
        assert len(service.corpus) >= 1

    def test_all_entries_are_reference_entry_instances(self):
        """Every corpus item should be a ReferenceEntry."""
        service = RetrievalService()
        for entry in service.corpus:
            assert isinstance(entry, ReferenceEntry)

    def test_all_entries_have_non_empty_text(self):
        """Every entry must have a non-empty text field."""
        service = RetrievalService()
        for entry in service.corpus:
            assert entry.text.strip(), f"Entry {entry.id!r} has empty text"
