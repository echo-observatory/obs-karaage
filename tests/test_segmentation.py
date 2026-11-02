"""Segmentation and passages on the test dataset (no LLM needed)."""

import pytest

from conftest import ARTICLE_IDS
from karaage.attribution import AttributionResult, Source
from karaage.passages import step_3_passages
from karaage.segmentation import step_1_segment


@pytest.mark.parametrize("article_id", ARTICLE_IDS)
def test_segmentation_and_passages(article_id, articles, nlp):
    text = articles[article_id]["text"]
    segments, _ = step_1_segment(text, nlp)

    # Lossless, with offsets that point into the text.
    assert "".join(s["text"] for s in segments) == text
    assert all(text[s["start"]:s["end"]] == s["text"] for s in segments)
    assert [s["id"] for s in segments] == list(range(len(segments)))

    # Passages cover every segment exactly once, in order. The middle third is
    # attributed to one actor so that passages have two speakers to separate.
    third = len(segments) // 3
    sources = AttributionResult(sources=[Source(main="Actor", other=[], segments=[[third, 2 * third]])])
    passages, _ = step_3_passages(segments, sources)
    assert [i for p in passages for i in p["segments"]] == [s["id"] for s in segments]
