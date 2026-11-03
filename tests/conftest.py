import json
import os
from pathlib import Path

import jsonschema
import pytest

import karaage
from karaage.output import output_schema

DATASET = Path(__file__).parent.parent / "data" / "test-datasets" / "wikinews-piracy-30" / "articles.jsonl"
ARTICLE_IDS = range(1, 31)   # the 30 articles of the test dataset
PROFILE = "qwen3.5-9b"
RECORDINGS = Path(__file__).parent / "recordings"
EXPECTED = RECORDINGS / PROFILE / "expected"

# What changes between runs: the version, the time, and the build LM Studio reports, which replay doesn't know.
VOLATILE = ("version", "commit", "inferred_at", "step_seconds", "model_build")

# An invented text, for tests that need no recorded response.
INVENTED_TEXT = (
    "The town council voted on Monday to close the old library. "
    "Mayor Jane Doe said the building was too expensive to repair, "
    "and that a new library would open next to the school within two years."
)


def as_document(article: dict) -> dict:
    """A test article as Karaage input: the dataset's `date` is the publication date."""
    return {"id": article["id"], "text": article["text"], "title": article["title"], "published_at": article["date"]}


@pytest.fixture(scope="session")
def articles():
    return {a["id"]: a for a in map(json.loads, DATASET.read_text(encoding="utf-8").splitlines())}


@pytest.fixture(scope="session")
def nlp():
    from karaage.segmentation import load_nlp_pipeline
    return load_nlp_pipeline()


@pytest.fixture(scope="session")
def resources():
    """The default profile on LM Studio, replaying recorded responses unless KARAAGE_LLM_MODE says otherwise."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KARAAGE_LLM_MODE", os.environ.get("KARAAGE_LLM_MODE") or "replay")
        mp.setenv("KARAAGE_RECORDINGS", str(RECORDINGS))
        yield karaage.setup(backend="lmstudio", profile=PROFILE)


@pytest.fixture(scope="session")
def run_article(articles, resources):
    """Run the pipeline on a test article, once per session, so that several tests can use its output."""
    results = {}

    def run(article_id):
        if article_id not in results:
            resources.client.use_recording(str(article_id))
            results[article_id] = karaage.run(as_document(articles[article_id]), resources)
        return results[article_id]
    return run


def comparable(data: dict) -> dict:
    """The output without what changes between runs."""
    return {**data, "provenance": {k: v for k, v in data["provenance"].items() if k not in VOLATILE}}


def check_output(data: dict, text: str) -> None:
    """Check an output against its JSON Schema, and its offsets against the text."""
    jsonschema.Draft202012Validator(output_schema()).validate(data)
    assert data["text_length"] == len(text)
    for s in data["statements"]:
        assert 0 <= s["char_start"] < s["char_end"] <= len(text)
    ids = [s["id"] for s in data["statements"]]
    assert len(ids) == len(set(ids))
