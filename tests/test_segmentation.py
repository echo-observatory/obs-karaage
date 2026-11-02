"""Segmentation and passages on the test dataset and on inputs that used to crash (no LLM needed)."""

import pytest

from conftest import ARTICLE_IDS
from karaage.attribution import AttributionResult, Source
from karaage.passages import step_3_passages
from karaage.segmentation import discourse_split, find_forced_boundaries, segment_text, step_1_segment


def assert_lossless(text, segments):
    assert "".join(s["text"] for s in segments) == text
    assert all(text[s["start"]:s["end"]] == s["text"] for s in segments)
    assert [s["id"] for s in segments] == list(range(len(segments)))


@pytest.mark.parametrize("article_id", ARTICLE_IDS)
def test_segmentation_and_passages(article_id, articles, nlp):
    text = articles[article_id]["text"]
    failures = []
    segments = segment_text(text, nlp, discourse_split(text, find_forced_boundaries(text)), failures=failures)
    assert_lossless(text, segments)

    # benepar used to fail on every sentence ending in whitespace (the last of each paragraph),
    # which then stayed unsplit.
    assert not [f for f in failures if f["text"] != f["text"].rstrip()]

    # Passages cover every segment exactly once, in order. The middle third is
    # attributed to one actor so that passages have two speakers to separate.
    third = len(segments) // 3
    sources = AttributionResult(sources=[Source(main="Actor", other=[], segments=[[third, 2 * third]])])
    passages, _ = step_3_passages(segments, sources)
    assert [i for p in passages for i in p["segments"]] == [s["id"] for s in segments]


@pytest.mark.parametrize("text", [
    # Over benepar's limit of 512 sub-word tokens: used to raise ValueError.
    "The minister said that " + "the new rules on file sharing will change how people use the internet and " * 45 + "nothing else.",
    # Nothing to parse: used to raise StopIteration.
    " \n\n \n",
])
def test_inputs_that_used_to_crash(text, nlp):
    segments, _ = step_1_segment(text, nlp)
    assert_lossless(text, segments)
