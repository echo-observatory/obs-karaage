# obs-karaage

Karaage extracts attributable statements from English news articles and rephrases them, each with a reference to the passage it comes from.
It is the "Extract positions" step of the pipeline; see the `pipeline` repo for how the components fit together.

**Status:** being restructured into a versioned Python package. The first release will be 1.1.0.
Until then, the code is the pre-1.1.0 version and its interface will change.
The decisions behind the restructuring are in [`docs/design-1.1.0.md`](docs/design-1.1.0.md).

## Inputs

One English text per document. Karaage does not translate: sources in other languages must be translated beforehand.

## Outputs

For each document, a JSON file with the main point of the text, the actors found in it, and the extracted statements, each with its actor, a rephrased statement, a one-sentence version and the range of segments it comes from.
Optionally, an HTML report for checking the results by hand.

For 1.1.0, the output will be aligned with the snapshot specification in `obs-corpus-snapshot`.

## Setup

Karaage needs Python 3.12 or later. With [uv](https://docs.astral.sh/uv/), from a clone of this repo:

```bash
uv sync                          # creates .venv with the exact versions in uv.lock
uv run karaage download-models   # benepar parser and embedding model
```

`uv sync` installs the versions pinned in `uv.lock`; library versions can change the output, so use it for any run whose results matter.
`pip install .` also works, but resolves versions within the bounds of `pyproject.toml` instead.
The spaCy model is installed as a dependency. `karaage download-models` puts the benepar parser in `~/nltk_data` and the embedding model in the Hugging Face cache.

The version comes from git tags: `1.1.0` on tag `v1.1.0`, and development versions such as `1.1.1.dev3+g1a2b3c4` in between. `karaage --version` prints it.

`Pipeline usage.ipynb` shows how to connect to an LLM served by LM Studio and run the pipeline on a batch of documents.
It predates the restructuring: skip its installation cell and run it in the environment above. It will be rewritten.

## Test dataset

`data/test-datasets/wikinews-piracy-30/` holds 30 English Wikinews articles about piracy, file sharing and DRM, under CC BY 2.5 or in the public domain.
See its README for sources, licences and attribution.
