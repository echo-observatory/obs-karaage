"""`karaage run` on a JSONL file (docs/design-1.1.0.md, sections 5, 6.1 and 6.2)."""

import hashlib
import json

import pytest

import karaage
from karaage import cli
from karaage.batch import InputError, is_done, output_name, parse_fields, read_documents, run_documents
from conftest import EXPECTED, INVENTED_TEXT, as_document, check_output, comparable


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


def test_output_names():
    assert [output_name(i) for i in ("19", "doc-1_a.b~c", "a/b", "50%", "..", "é")] == [
        "19", "doc-1_a.b~c", "a%2Fb", "50%25", "%2E.", "%C3%A9"]
    ids = ["a/b", "a%2Fb", "a b", "a+b", "a%20b", ".x", "%2Ex", "x.", "é", "%C3%A9"]
    assert len({output_name(i) for i in ids}) == len(ids)


def test_field_mapping(tmp_path):
    path = write_jsonl(tmp_path / "docs.jsonl", [
        {"doc": 7, "body": "A text.", "headline": "A title", "date": "2020-01-02", "other": 1},
        {"doc": "b", "body": "Another text.", "date": None},
    ])
    fields = parse_fields(["id=doc", "text=body", "title=headline", "published_at=date"])
    assert read_documents(path, fields) == [
        {"id": 7, "text": "A text.", "title": "A title", "published_at": "2020-01-02"},
        {"id": "b", "text": "Another text."},
    ]
    for spec in ["published_at", "date=published_at", "title="]:
        with pytest.raises(InputError):
            parse_fields([spec])


def test_input_problems_are_listed_before_running(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join([
        '{"id": 1, "text": "ok"}',
        '{"id": 2, "text": "no closing brace"',
        '["not", "an", "object"]',
        '{"text": "no id"}',
        '{"id": 3}',
        '{"id": 1, "text": "same id as line 1"}',
        '{"id": "A", "text": "x"}',
        '{"id": "a", "text": "same file name as A on a file system that ignores case"}',
        "",
    ]), encoding="utf-8")
    with pytest.raises(InputError) as raised:
        read_documents(path)
    lines = [line.split(":")[0].strip() for line in str(raised.value).splitlines()[1:]]
    assert lines == ["line 2", "line 3", "line 4", "line 5", "line 6", "line 8"]


def test_done_means_ok_on_the_same_text():
    sha = hashlib.sha256("A text.".encode()).hexdigest()
    assert is_done({"status": "ok", "text_sha256": sha}, "A text.")
    assert not is_done({"status": "ok", "text_sha256": sha}, "A changed text.")
    assert not is_done({"status": "partial", "text_sha256": sha}, "A text.")
    assert not is_done(None, "A text.")


@pytest.fixture
def replay(resources, monkeypatch):
    if resources.client.mode != "replay":
        pytest.skip("replays recorded responses")
    monkeypatch.setattr(karaage, "setup", lambda **kwargs: resources)
    return resources


def test_run_command(tmp_path, articles, replay, monkeypatch):
    records = [{k: articles[i][k] for k in ("id", "text", "title", "date")} for i in (7, 19)]
    records.append({"id": "invented/1", "text": INVENTED_TEXT})   # no recorded responses: it fails
    out = tmp_path / "out"
    argv = ["run", "--input", str(write_jsonl(tmp_path / "docs.jsonl", records)), "--output", str(out),
            "--report", "--field", "published_at=date"]
    assert cli.main(argv) == 1   # a document failed
    assert sorted(p.name for p in out.iterdir()) == [
        "19.html", "19.json", "7.html", "7.json", "invented%2F1.html", "invented%2F1.json"]
    for article_id in (7, 19):
        data = json.loads((out / f"{article_id}.json").read_text(encoding="utf-8"))
        check_output(data, articles[article_id]["text"])
        assert comparable(data) == json.loads((EXPECTED / f"{article_id}.json").read_text(encoding="utf-8"))
    assert json.loads((out / "invented%2F1.json").read_text(encoding="utf-8"))["status"] == "failed"

    # Running again skips the documents that are ok, and retries the one that failed.
    calls, run = [], karaage.run
    monkeypatch.setattr(karaage, "run", lambda document, resources, **kw: calls.append(document["id"]) or run(document, resources, **kw))
    assert cli.main(argv) == 1 and calls == ["invented/1"]


def test_interrupted_run_resumes(tmp_path, articles, replay, monkeypatch):
    documents = [as_document(articles[i]) for i in (7, 19)]
    out = tmp_path / "out"
    calls, run = [], karaage.run

    def interrupted_at_19(document, resources, **kw):
        if document["id"] == 19:
            raise KeyboardInterrupt
        return run(document, resources, **kw)
    monkeypatch.setattr(karaage, "run", interrupted_at_19)
    with pytest.raises(KeyboardInterrupt):
        run_documents(documents, out, replay)
    assert [p.name for p in out.iterdir()] == ["7.json"]

    monkeypatch.setattr(karaage, "run", lambda document, resources, **kw: calls.append(document["id"]) or run(document, resources, **kw))
    statuses = run_documents(documents, out, replay)
    assert calls == [19] and statuses["skipped"] == 1 and sum(statuses.values()) == 2


def test_run_command_without_paraphrasing(tmp_path, articles, replay, capsys):
    from test_articles import WITHOUT_PARAPHRASING
    records = [{k: articles[i][k] for k in ("id", "text", "title", "date")} for i in WITHOUT_PARAPHRASING]
    out = tmp_path / "out"
    argv = ["run", "--input", str(write_jsonl(tmp_path / "docs.jsonl", records)), "--output", str(out),
            "--field", "published_at=date"]
    assert cli.main(argv + ["--no-paraphrase"]) == 0
    for article_id in WITHOUT_PARAPHRASING:
        data = json.loads((out / f"{article_id}.json").read_text(encoding="utf-8"))
        check_output(data, articles[article_id]["text"])
        expected = EXPECTED / f"{article_id}-no-paraphrase.json"
        assert comparable(data) == json.loads(expected.read_text(encoding="utf-8"))

    # Without the flag, these outputs are skipped as done, with a warning that they were made otherwise.
    capsys.readouterr()
    assert cli.main(argv) == 0
    assert "3 of them were made by another profile, Karaage version or paraphrasing setting" in capsys.readouterr().out
