"""Steps that must carry on when part of their work fails (no LLM needed)."""

from karaage import extraction
from karaage.validation import apply_validation_filter, run_validation


def test_validation_after_failed_rephrasing_batch():
    # A failed rephrasing batch leaves its statements without a summary;
    # validation used to raise KeyError: 'summary' and fail the whole document.
    statements = [{"id": 0, "actor": "Ann", "statement": "S", "passage": "P", "rephrasing_error": "timeout"}]
    validated = apply_validation_filter(run_validation(statements))   # no LLM call is made
    assert [(s["valid"], s["invalid_reason"]) for s in validated] == [(False, "rephrasing error")]


def test_failed_passage_is_returned(monkeypatch):
    def failing_call(*args, **kwargs):
        raise RuntimeError("backend down")
    monkeypatch.setattr(extraction.llm, "complete_json", failing_call)
    passages = [{"text": "x" * 300, "segments": [0], "speaker_pcts": {"Ann": 100.0}, "speaker_chars": {"Ann": 300}}]
    statements, errors = extraction.extract_statements(passages, "main point")
    assert statements == [] and errors == [{"passage_index": 0, "error": "backend down"}]
