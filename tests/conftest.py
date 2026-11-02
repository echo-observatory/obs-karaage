import json
from pathlib import Path

import pytest

DATASET = Path(__file__).parent.parent / "data" / "test-datasets" / "wikinews-piracy-30" / "articles.jsonl"
ARTICLE_IDS = range(1, 31)   # the 30 articles of the test dataset


@pytest.fixture(scope="session")
def articles():
    return {a["id"]: a for a in map(json.loads, DATASET.read_text(encoding="utf-8").splitlines())}


@pytest.fixture(scope="session")
def nlp():
    from karaage.segmentation import load_nlp_pipeline
    return load_nlp_pipeline()
