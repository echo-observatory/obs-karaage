"""Steps that must carry on when part of their work fails (no LLM needed)."""

from karaage import extraction
from karaage.validation import apply_validation_filter, run_validation


def test_validation_after_failed_rephrasing_batch():
    # A failed rephrasing batch leaves its statements without a summary;
    # validation used to raise KeyError: 'summary' and fail the whole document.
    statements = [{"id": 0, "actor": "Ann", "statement": "S", "excerpt": "P", "rephrasing_error": "timeout"}]
    validated = apply_validation_filter(run_validation(statements))   # no LLM call is made
    assert [(s["valid"], s["invalid_reason"]) for s in validated] == [(False, "rephrasing error")]


def test_failed_excerpt_is_returned(monkeypatch):
    def failing_call(*args, **kwargs):
        raise RuntimeError("backend down")
    monkeypatch.setattr(extraction.llm, "complete_json", failing_call)
    excerpts = [{"text": "x" * 300, "segments": [0], "actor_pcts": {"Ann": 100.0}, "actor_chars": {"Ann": 300}}]
    statements, errors = extraction.extract_statements(excerpts, "main point")
    assert statements == [] and errors == [{"excerpt_index": 0, "error": "RuntimeError: backend down"}]


def fake_segments_and_excerpts(n_excerpts, chars_per_excerpt):
    segments = [{"id": i, "text": "x" * chars_per_excerpt} for i in range(n_excerpts)]
    excerpts = [{"segments": [i]} for i in range(n_excerpts)]
    return segments, excerpts


def test_parts_fit_the_budget_and_are_balanced():
    segments, excerpts = fake_segments_and_excerpts(10, 400)   # about 100 tokens per excerpt
    parts = extraction.split_into_parts(segments, excerpts, budget=350)
    sizes = [extraction.estimate_tokens(p) for p in parts]
    assert len(parts) == 4 and max(sizes) <= 350 and max(sizes) - min(sizes) <= 110
    assert "\n".join(parts) == extraction.format_segments(segments)


def test_title_and_date_head_the_text():
    assert extraction.with_header("TEXT") == "TEXT"
    assert extraction.with_header("TEXT", title="T", published_at="2020-01-02") == "Title: T\nPublished: 2020-01-02\n\nTEXT"


def test_main_point_of_a_long_text_comes_from_part_summaries(monkeypatch):
    calls = []
    def fake_complete_model(system, user, model_cls, **kwargs):
        calls.append(model_cls.__name__)
        return model_cls(**{next(iter(model_cls.model_fields)): f"answer {len(calls)}"})
    monkeypatch.setattr(extraction.llm, "complete_model", fake_complete_model)
    monkeypatch.setattr(extraction.llm, "context_tokens", lambda: 8000)

    segments, excerpts = fake_segments_and_excerpts(5, 400)
    assert extraction.extract_main_point(segments, excerpts) == ("answer 1", "single", [])

    calls.clear()
    segments, excerpts = fake_segments_and_excerpts(100, 1200)   # about 30,000 tokens
    main_point, method, summaries = extraction.extract_main_point(segments, excerpts)
    assert method == "chunked" and len(summaries) == 5 and calls == ["PartSummary"] * 5 + ["MainPoint"]
