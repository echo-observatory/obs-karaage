"""The whole pipeline on the test dataset, replaying recorded LLM responses (docs/design-1.1.0.md, section 7.1).

Each article's LLM responses are recorded in tests/recordings/<profile>/<id>.json,
and its output must match tests/recordings/<profile>/expected/<id>.json. To
re-record after changing prompts, schemas or the profile, with the profile's
model loaded in LM Studio:

    KARAAGE_LLM_MODE=record uv run pytest tests/test_pipeline.py

Recording an article rewrites its file from scratch. One article alone:
    KARAAGE_LLM_MODE=record uv run pytest "tests/test_pipeline.py::test_pipeline[30]"

After a change to the output alone, which changes no LLM request, rewrite the
expected outputs from a replay instead:

    KARAAGE_UPDATE_EXPECTED=1 uv run pytest tests/test_pipeline.py
"""

import json
import os
from pathlib import Path

import pytest

import karaage
from conftest import ARTICLE_IDS

PROFILE = "qwen3.5-9b"
RECORDINGS = Path(__file__).parent / "recordings"
EXPECTED = RECORDINGS / PROFILE / "expected"


@pytest.fixture(scope="module")
def resources():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("KARAAGE_LLM_MODE", os.environ.get("KARAAGE_LLM_MODE") or "replay")
        mp.setenv("KARAAGE_RECORDINGS", str(RECORDINGS))
        yield karaage.setup(backend="lmstudio", profile=PROFILE)


def comparable(result) -> dict:
    """The output without what changes between runs: version and timestamp."""
    data = {k: v for k, v in result.data.items() if k not in ("karaage_version", "timestamp")}
    return {"step_errors": result.step_errors, "failures": result.failures, "data": data}


@pytest.mark.parametrize("article_id", ARTICLE_IDS)
def test_pipeline(article_id, articles, resources):
    resources.client.use_recording(str(article_id))
    output = comparable(karaage.run(articles[article_id]["text"], resources))
    expected = EXPECTED / f"{article_id}.json"
    if resources.client.mode == "record" or os.environ.get("KARAAGE_UPDATE_EXPECTED"):
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_text(json.dumps(output, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        assert output == json.loads(expected.read_text(encoding="utf-8"))
