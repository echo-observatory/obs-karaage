"""Karaage on each test article, replaying recorded LLM responses (docs/design-1.1.0.md, section 7.1).

Each article's LLM responses are recorded in tests/recordings/<profile>/<id>.json,
and its output must match tests/recordings/<profile>/expected/<id>.json. To
re-record after changing prompts, schemas or the profile, with the profile's
model loaded in LM Studio:

    KARAAGE_LLM_MODE=record uv run pytest tests/test_articles.py

Recording an article rewrites its file from scratch. One article alone:
    KARAAGE_LLM_MODE=record uv run pytest "tests/test_articles.py::test_article[30]"

After a change to the output alone, which changes no LLM request, rewrite the
expected outputs from a replay instead:

    KARAAGE_UPDATE_EXPECTED=1 uv run pytest tests/test_articles.py
"""

import json
import os

import pytest

import karaage
from karaage import extraction
from conftest import ARTICLE_IDS, EXPECTED, INVENTED_TEXT, as_document, check_output, comparable


@pytest.mark.parametrize("article_id", ARTICLE_IDS)
def test_article(article_id, articles, run_article, resources):
    result = run_article(article_id)
    data, text = result.data, articles[article_id]["text"]
    check_output(data, text)
    excerpts = {p["text"] for p in result.excerpts}
    assert all(text[s["char_start"]:s["char_end"]] in excerpts for s in data["statements"])
    assert "<link" not in result.html_report   # self-contained
    expected = EXPECTED / f"{article_id}.json"
    if resources.client.mode == "record" or os.environ.get("KARAAGE_UPDATE_EXPECTED"):
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_text(json.dumps(comparable(data), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        assert comparable(data) == json.loads(expected.read_text(encoding="utf-8"))


def test_failed_step_gives_a_failed_output(resources, monkeypatch, tmp_path):
    def failing_attribution(*args, **kwargs):
        raise RuntimeError("backend down")
    monkeypatch.setattr("karaage.attribution.step_2_attribute_actors", failing_attribution)

    result = karaage.run({"id": 1001, "text": INVENTED_TEXT}, resources)
    data = result.data
    check_output(data, INVENTED_TEXT)
    assert data["document_id"] == "1001" and data["status"] == "failed"
    assert data["errors"] == [{"step": "attribution", "excerpt": None, "statement": None,
                               "message": "RuntimeError: backend down"}]
    assert list(data["provenance"]["step_seconds"]) == ["segmentation", "attribution"]
    assert "RuntimeError: backend down" in result.html_report

    karaage.save(result, data_path=tmp_path / "1001.json", html_path=tmp_path / "1001.html")
    assert json.loads((tmp_path / "1001.json").read_text(encoding="utf-8")) == data


def test_failed_excerpt_gives_a_partial_output(articles, resources, monkeypatch):
    if resources.client.mode != "replay":
        pytest.skip("replays article 19, whose later requests change")
    analyze_excerpt = extraction.analyze_excerpt
    def failing_first_excerpt(excerpts, excerpt_index, *args, **kwargs):
        if excerpt_index == 0:
            raise RuntimeError("backend down")
        return analyze_excerpt(excerpts, excerpt_index, *args, **kwargs)
    monkeypatch.setattr(extraction, "analyze_excerpt", failing_first_excerpt)

    resources.client.use_recording("19")
    data = karaage.run(as_document(articles[19]), resources).data
    check_output(data, articles[19]["text"])
    assert data["status"] == "partial"
    assert data["errors"][0] == {"step": "extraction", "excerpt": 1, "statement": None,
                                 "message": "RuntimeError: backend down"}


def test_document_needs_an_id_and_a_text(resources):
    for document in ["text alone", {"text": "no id"}, {"id": "", "text": "empty id"}, {"id": True, "text": "x"},
                     {"id": 1}, {"id": 1, "text": "x", "title": 3}, {"id": 1, "text": "x", "published_at": 2004}]:
        with pytest.raises((TypeError, ValueError)):
            karaage.run(document, resources)
