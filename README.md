# obs-karaage

Karaage extracts attributable statements from English news articles and rephrases them, each with a reference to the passage it comes from.
It is the "Extract positions" step of the pipeline; see the `pipeline` repo for how the components fit together.

**Status:** being restructured into a versioned Python package. The first release will be 1.1.0.
Until then, the code is the pre-1.1.0 version and its interface will change.
The decisions behind the restructuring are in [`docs/design-1.1.0.md`](docs/design-1.1.0.md).

## Inputs

Documents, each with an id and an English text, and optionally a title and a publication date, given to the model as context when it states the text's main point.
Karaage does not translate: sources in other languages must be translated beforehand.

## Outputs

For each document, a JSON output described by the JSON Schema [`src/karaage/schemas/output.schema.json`](src/karaage/schemas/output.schema.json):
- a status (`ok`, `partial` or `failed`), with the errors, and warnings about input that looks wrong;
- the text's length and SHA-256;
- the main point of the text, and the actors found in it;
- every extracted statement, rejected ones included: its actor and the actor's aliases, a rephrased statement, a one-sentence version, the character offsets of the passage it comes from, and the validation result;
- provenance: Karaage version and commit, profile, backend and model.

Optionally, an HTML report for checking the results by hand. It is self-contained, so it opens offline.

The output maps onto the snapshot specification of `obs-corpus-snapshot` as described in [`docs/design-1.1.0.md`](docs/design-1.1.0.md), section 6.4.

## Setup

Karaage needs Python 3.12 or later. With [uv](https://docs.astral.sh/uv/), from a clone of this repo:

```bash
uv sync                          # creates .venv with the exact versions in uv.lock
uv run karaage download-models   # benepar parser, and the embedding model of the profile
```

`uv sync` installs the versions pinned in `uv.lock`; library versions can change the output, so use it for any run whose results matter.
`pip install .` also works, but resolves versions within the bounds of `pyproject.toml` instead.
The spaCy model is installed as a dependency. `karaage download-models` puts the benepar parser in `~/nltk_data` and the embedding model in the Hugging Face cache.

The version comes from git tags: `1.1.0` on tag `v1.1.0`, and development versions such as `1.1.1.dev3+g1a2b3c4` in between. `karaage --version` prints it.
Until 1.1.0 is released, development versions read `1.1.0.devN+g…`, counted from the tag `start-1.1.0.dev0`, which marks the start of the restructuring and is not a release.

## Configuration

Karaage needs two things: a **backend**, which says where the LLM runs, and a **profile**, which says which model to use and how to drive it.
Set them with `KARAAGE_*` environment variables, or in a `.env` file in the working directory (copy `.env.example`):

| Variable | Meaning | Default |
| --- | --- | --- |
| `KARAAGE_BACKEND` | `lmstudio`, `openrouter`, or `openai` for any other OpenAI-compatible server | `lmstudio` |
| `KARAAGE_BASE_URL` | API base URL | `http://127.0.0.1:1234/v1` for LM Studio, `https://openrouter.ai/api/v1` for OpenRouter |
| `KARAAGE_API_KEY` | API key, required for `openrouter` and `openai` | none needed for LM Studio |
| `KARAAGE_PROFILE` | a shipped profile name, or the path to a profile file | `qwen3.5-9b` |

The default profile `qwen3.5-9b` uses Qwen3.5 9B with reasoning off; the prompts were written for it.
- **LM Studio:** load `qwen/qwen3.5-9b` with a context length of at least 8,000 tokens. Karaage turns the model's reasoning off in each request, so no other setting is needed. It warns when the model is not loaded or its context is too short, and records the loaded build (format and quantisation).
- **OpenRouter:** set the API key in `.env`. Team rule: never send Retriever data to OpenRouter or any other external API.

A custom profile is a TOML file shaped like [`src/karaage/profiles/qwen3.5-9b.toml`](src/karaage/profiles/qwen3.5-9b.toml) with a name of its own; outputs label it `custom:<name>`.
The format is described in [`docs/design-1.1.0.md`](docs/design-1.1.0.md), section 4.2.

From Python:

```python
import karaage

resources = karaage.setup()   # or e.g. karaage.setup(backend="openrouter", profile="my-profile.toml")
result = karaage.run({"id": "doc-1", "text": text}, resources)
karaage.save(result, data_path="out/doc.json", html_path="out/doc.html")
```

`Pipeline usage.ipynb` shows how to run the pipeline on a batch of documents.
It predates the restructuring, and its installation and data-loading cells are out of date. It will be rewritten.

## Tests

```bash
uv run pytest
```

The tests need the models (`uv run karaage download-models`) but no LLM: they replay LLM responses recorded on the test dataset, stored in `tests/recordings/`, and compare each article's output with the recorded one. GitHub Actions runs them on every pull request.

After a change to prompts, schemas or the default profile, record again, with the profile's model loaded in LM Studio:

```bash
KARAAGE_LLM_MODE=record uv run pytest tests/test_pipeline.py
```

Each article's responses are in one file, `tests/recordings/qwen3.5-9b/<id>.json`, rewritten from scratch when the article is recorded again; `"tests/test_pipeline.py::test_pipeline[30]"` records article 30 alone.

Then check the changed outputs in `tests/recordings/qwen3.5-9b/expected/` before committing them.

A change to the output alone, which sends no new request to the LLM, needs no recording: `KARAAGE_UPDATE_EXPECTED=1 uv run pytest tests/test_pipeline.py` rewrites the expected outputs from a replay.

`tests/test_snapshot.py` converts the outputs into a snapshot bundle and checks it with the snapshot validator. The validator is private: clone `obs-corpus-snapshot` next to this repo, then run `uv run --with ../obs-corpus-snapshot pytest`. Without it, and in GitHub Actions, the test is skipped.

## Test dataset

`data/test-datasets/wikinews-piracy-30/` holds 30 English Wikinews articles about piracy, file sharing and DRM, under CC BY 2.5 or in the public domain.
See its README for sources, licences and attribution.
