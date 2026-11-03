# obs-karaage

Karaage infers attributable statements from English news articles. Each statement comes with the actor who made it, a rephrased version that stands alone, and the excerpt of the text it comes from.
It is the "Infer statements" step of the Observatory pipeline; see the `pipeline` repo for how the components fit together.

**Status:** 1.1.0, the first release of Karaage as a versioned package, is in preparation. Until its tag `v1.1.0` exists, install from a clone.

## Quick start

On the test dataset of 30 articles, with [uv](https://docs.astral.sh/uv/) and [LM Studio](https://lmstudio.ai/):

1. In LM Studio, download the model `qwen/qwen3.5-9b`, load it with a context length of at least 8,000 tokens, and start the local server.
2. From a clone of this repo:

   ```bash
   uv sync
   uv run karaage download-models
   uv run karaage run --input data/test-datasets/wikinews-piracy-30/articles.jsonl --output out/ --report --field published_at=date
   ```

3. Open `out/19.html` to see how one article was processed, and `out/19.json` for its output.

The run takes about 40 minutes on an Apple M5 Pro laptop. You can interrupt it at any time: the same command resumes where it stopped.

## How it works

Karaage runs seven steps on each text, one module each in `src/karaage/`:

1. **Segmentation:** splits the text into clause-level segments, with spaCy and benepar.
2. **Attribution:** an LLM assigns the segments to the actors quoted or cited in the text. Segments in the text's own voice go to `base_author`.
3. **Excerpts:** groups consecutive segments into excerpts of about 2,000 characters at most, each dominated by one actor where possible.
4. **Extraction:** an LLM states the main point of the text, then extracts up to four statements per actor and excerpt.
5. **Rephrasing:** an LLM rewrites each statement so that it stands alone, plus a one-sentence version.
6. **Paraphrasing:** statements too close to the source wording are paraphrased again, with an embedding check that the meaning is kept. It can be turned off (see [Paraphrasing](#paraphrasing)).
7. **Validation:** an LLM checks that each statement is in its excerpt and attributed to the right actor. Statements that fail are kept in the output, marked as rejected.

Karaage never modifies the text: all offsets point into the text as given.

## Installation

Karaage needs Python 3.12 or later.

**A release,** from its git tag:

```bash
pip install "obs-karaage @ git+https://github.com/echo-observatory/obs-karaage@v1.1.0"
karaage download-models
```

The package is named `obs-karaage` because `karaage` on PyPI is an unrelated project; the import name and the command are `karaage`. Karaage is not published on PyPI.

**From a clone,** with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/echo-observatory/obs-karaage.git
cd obs-karaage
git checkout v1.1.0              # or stay on main for the development version
uv sync                          # creates .venv with the exact versions in uv.lock
uv run karaage download-models
```

`uv sync` installs the library versions pinned in `uv.lock`. Library versions can change the output, so use it for any run whose results matter; `pip` resolves versions within the bounds of `pyproject.toml` instead.
In a clone, prefix commands with `uv run`, or activate `.venv`.

`karaage download-models` puts the benepar parser in `~/nltk_data` and the profile's embedding model in the Hugging Face cache. The spaCy model is installed as a dependency.

## Configuration

Karaage needs a **backend**, which says where the LLM runs, and a **profile**, which says which model to use and how to drive it.
Set them with `KARAAGE_*` environment variables, or in a `.env` file in the working directory or one of its parents. Copy [`.env.example`](.env.example) to start.

| Variable | Meaning | Default |
| --- | --- | --- |
| `KARAAGE_BACKEND` | `lmstudio`, `openrouter`, or `openai` for any other OpenAI-compatible server | `lmstudio` |
| `KARAAGE_BASE_URL` | API base URL | `http://127.0.0.1:1234/v1` for LM Studio, `https://openrouter.ai/api/v1` for OpenRouter |
| `KARAAGE_API_KEY` | API key, required for `openrouter` and `openai` | none needed for LM Studio |
| `KARAAGE_PROFILE` | a shipped profile name, or the path to a profile file | `qwen3.5-9b` |

> **Team rule:** Retriever data must never be sent to OpenRouter or any other external API, and is always processed with paraphrasing on. Karaage does not check this; process such data with a local backend such as LM Studio, without `--no-paraphrase`.

### Backends

- **LM Studio:** load the profile's model with a context length of at least the profile's `context_tokens` (8,000 for the default profile). Karaage turns the model's reasoning off in each request, so no other setting is needed. It warns when the model is not loaded or its context is too short, and records the loaded build (format and quantisation) in each output.
- **OpenRouter:** set `KARAAGE_API_KEY`. Requests ask for providers that support every parameter sent and don't keep the data. Each output records the providers that served it, since they may run the model at different precisions.

### Profiles

A profile fixes the model and how it is driven. A release ships named profiles, and outputs record the profile they were made with.

The default profile, `qwen3.5-9b`, runs Qwen3.5 9B with reasoning off; the prompts were written for it.
A custom profile is a TOML file shaped like [`src/karaage/profiles/qwen3.5-9b.toml`](src/karaage/profiles/qwen3.5-9b.toml), with a name of its own. Pass its path as the profile; outputs label it `custom:<name>`.

```toml
name = "my-profile"
description = "What this profile is for."
context_tokens = 8000            # LM Studio must load the model with at least this context length
temperature = 0
embedding_model = "intfloat/multilingual-e5-large-instruct"
# seed = 0                       # optional; sent with every request when set

[backends.lmstudio]              # one section per backend that can run the profile
model = "qwen/qwen3.5-9b"        # the model id on that backend
reasoning_off = "request-parameter"
```

`reasoning_off` says how to turn the model's reasoning off:
- `none`: the model does not reason;
- `no-think-tag`: `/no_think` before the system prompt;
- `prefill-empty-think`: an empty `<think></think>` block at the start of the answer;
- `request-parameter`: the backend's reasoning parameter.

Unknown keys are errors, so that a typo can't fall back to a default silently.

## Running

### Input

Documents, in English. Karaage does not translate: sources in other languages must be translated beforehand.

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Document id, unique in the run. Integers are accepted and written as strings. |
| `text` | yes | The text. |
| `title` | no | Title or headline, given to the model as context when it states the text's main point. |
| `published_at` | no | Publication date or time, used the same way. |

### Command line

`karaage run` reads a JSONL file, one document per line:

```bash
karaage run --input docs.jsonl --output out/ [--report] [--profile NAME] [--no-paraphrase] [--field FIELD=NAME]...
```

- **Output:** `out/<id>.json` for each document, and `out/<id>.html` with `--report`. Characters unsafe in file names are percent-encoded, e.g. id `a/b` gives `a%2Fb.json`.
- **Field names:** `--field FIELD=NAME` reads a field under another name, e.g. `--field published_at=date`.
- **Input check:** every line is checked before anything runs: valid JSON, an id and a text, and unique ids. Ids that differ only by case are refused, since some file systems ignore case.
- **Paraphrasing:** `--no-paraphrase` turns it off; see [Paraphrasing](#paraphrasing).
- **Resume:** running the same command again skips the documents whose output is `ok` and was made from the same text. Documents that failed or are partial are run again, so the same command also retries them.
- **Summary:** at the end, the number of documents per status, and the time per 1,000 characters.
- **Exit code:** 0 when no document failed, 1 when one did, 2 when the input can't be run.

`karaage run --help` lists the options.

### Python

```python
import karaage

resources = karaage.setup()   # or e.g. karaage.setup(backend="openrouter", profile="my-profile.toml")
result = karaage.run({"id": "doc-1", "text": text, "title": title}, resources)   # paraphrase=False turns it off
result.data                   # the output, described below
karaage.save(result, data_path="out/doc-1.json", html_path="out/doc-1.html")
```

`setup()` loads the models and connects to the backend once per session; its keyword arguments override the environment.
`run()` processes one document and does not raise when a step fails: the output then has the status `failed`, with the error.
`karaage.batch.read_documents()` and `karaage.batch.run_documents()` do what `karaage run` does.

[`examples/using-karaage-from-python.ipynb`](examples/using-karaage-from-python.ipynb) walks through these on the test dataset. To open it: `uv run --with jupyter jupyter lab examples/using-karaage-from-python.ipynb`.

### Paraphrasing

Statements must not reproduce their source's wording, for ethical and legal reasons. So step 6 paraphrases the statements whose wording is too close to their excerpt (ROUGE-2 precision above 0.3), and validation rejects those that stay too close ("too similar to source").

This isn't needed when the source's wording may be reproduced, typically when it is in the public domain or under a licence that allows it. `--no-paraphrase`, or `paraphrase=False` in Python, then turns off both the paraphrasing and the rejection. The similarity is still measured and reported in each statement's `diagnostics`, and the output records the setting in `provenance.paraphrase`.

Retriever data is always processed with paraphrasing on (team rule).

### Time

On an Apple M5 Pro laptop with LM Studio, the default profile takes about 12 seconds per 1,000 characters: about 40 minutes for the 30 test articles, which total 180,000 characters.

## Output

Each document's output is a JSON object, described by the JSON Schema [`src/karaage/schemas/output.schema.json`](src/karaage/schemas/output.schema.json). Every field is always present; empty lists and `null` stand for absent values.
[`tests/recordings/qwen3.5-9b/expected/19.json`](tests/recordings/qwen3.5-9b/expected/19.json) is an example, without the provenance fields that change from run to run.

| Field | Content |
| --- | --- |
| `document_id` | The input id, as a string. |
| `status` | `ok`, `partial` or `failed` (see below). |
| `errors` | What failed: `{"step", "excerpt", "statement", "message"}`, with `excerpt` (numbered from 1, as in the report) or `statement` saying which part, when only part of a step failed. |
| `warnings` | Input that looks wrong: `{"code", "message"}`, with the codes `text-not-nfc`, `text-has-carriage-return` and `text-not-english`. Karaage runs anyway. |
| `text_length`, `text_sha256` | Length of the text in Unicode code points, and SHA-256 of its UTF-8 bytes, to detect a later change to the text. |
| `main_point` | One sentence stating the central argument of the text. |
| `main_point_method` | `single`, or `chunked` when the text did not fit the model's context: its parts were summarised, and the summaries combined. |
| `actors` | The actors quoted or cited (*inferred actors*, in the snapshot specification's terms): `canonical_name`, `aliases`, and `kind` (`person`, `organisation`, or `null`). |
| `statements` | Every statement extracted, rejected ones included (below). |
| `provenance` | Karaage version and commit, profile and its settings, backend, model and its build, whether paraphrasing was on (`paraphrase`), `inferred_at` (UTC, `YYYY-MM-DDTHH:MM:SSZ`), and the duration of each step. |

Each statement has:
- `id`: `<document_id>-s<n>`, numbered from 1 in extraction order;
- `actor`: the actor's canonical name, or `base_author` for the text's own voice. `actor_aliases` lists the canonical name, then the aliases;
- `text`: the statement, rephrased to stand alone. `text_short`: a one-sentence version. Both are `null` if rephrasing failed;
- `char_start` and `char_end`: the offsets of the excerpt the statement comes from, in code points, end excluded;
- `valid` and `rejection_reason`: `validation error`, `rephrasing error`, `too similar to source`, `not found in source text` or `wrong actor attribution`;
- `diagnostics`: the ROUGE-2 precision against the excerpt, whether the statement was paraphrased, and the validator's answers.

**Status:**
- `ok`: every step ran without error. A document with no statements can be `ok`.
- `partial`: every step ran, but some excerpts or statements failed, and statements may be missing.
- `failed`: a step failed as a whole. There are no actors, statements or main point, but the output is still written, with the error.

**HTML report:** with `--report`, or `html_path` in `save()`, a page showing each step's result, for checking by hand. It is self-contained and opens offline.

### Snapshot bundles

Karaage outputs are meant to go into snapshot bundles (specification 1.0, in `obs-corpus-snapshot`). Karaage doesn't assemble bundles; a bundle builder does, with this mapping for `inferred_statements`, one row per valid statement:

| Snapshot field | Karaage field |
| --- | --- |
| `id` | `statements[].id` |
| `source_document_id` | `document_id` |
| `text`, `text_short` | `statements[].text`, `statements[].text_short` |
| `char_start`, `char_end` | `statements[].char_start`, `statements[].char_end` |
| `source_text_length` | `text_length` |
| `raw_actor_aliases` | `statements[].actor_aliases` |
| `inferred_at` | `provenance.inferred_at` |
| `inference_tool`, `inference_version` | `provenance.tool`, `provenance.version` |
| `inference_profile` (flex field) | `provenance.profile` |

`main_point` goes into `source_documents`, and the HTML report under `monitoring/`. The profile settings, and whether paraphrasing was on, go under `snapshot.statement_inference` in the descriptor.
Karaage's actors are not reconciled, so they don't fill `inferred_actors`. [`tests/test_snapshot.py`](tests/test_snapshot.py) builds a valid bundle from the test dataset this way.

## Versions

Versions are `MAJOR.MINOR.PATCH`, and every output records the version and the git commit that made it.
- **MAJOR:** the output format breaks.
- **MINOR:** anything that can change the statements: code, prompts, parameters, shipped profiles, models, dependencies.
- **PATCH:** no change to the output.

A released version is never changed: a fix is a new version.
The version comes from git tags: `1.1.0` on tag `v1.1.0`, and development versions such as `1.1.1.dev3+g1a2b3c4` in between. `karaage --version` prints it.
Until 1.1.0 is released, development versions read `1.1.0.devN+g…`, counted from the tag `start-1.1.0.dev0`, which marks the start of the restructuring and is not a release.

## Development

[`AGENTS.md`](AGENTS.md) holds the conventions for working on this repo, and [`docs/design-1.1.0.md`](docs/design-1.1.0.md) the decisions behind 1.1.0.

### Tests

```bash
uv run pytest
```

The tests need the models (`uv run karaage download-models`) but no LLM: they replay LLM responses recorded on the test dataset, stored in `tests/recordings/`, and compare each article's output with the recorded one. GitHub Actions runs them on every pull request.

After a change to prompts, schemas or the default profile, record again, with the profile's model loaded in LM Studio. It takes about 40 minutes:

```bash
KARAAGE_LLM_MODE=record uv run pytest tests/test_articles.py
```

Each article's responses are in one file, `tests/recordings/qwen3.5-9b/<id>.json`, rewritten from scratch when the article is recorded again; `"tests/test_articles.py::test_article[30]"` records article 30 alone.
Then check the changed outputs in `tests/recordings/qwen3.5-9b/expected/` before committing them. A change to the method is also checked by a human on the HTML reports of the test articles.

A change to the output alone, which sends no new request to the LLM, needs no recording: `KARAAGE_UPDATE_EXPECTED=1 uv run pytest tests/test_articles.py` rewrites the expected outputs from a replay.

`tests/test_snapshot.py` converts the outputs into a snapshot bundle and checks it with the snapshot validator. The validator is private: clone `obs-corpus-snapshot` next to this repo, then run `uv run --with ../obs-corpus-snapshot pytest`. Without it, and in GitHub Actions, the test is skipped.

## Test dataset

`data/test-datasets/wikinews-piracy-30/` holds 30 English Wikinews articles about piracy, file sharing and DRM, under CC BY 2.5 or in the public domain.
See its README for sources, licences and attribution. Its canonical home is `echo-observatory/obs-test-datasets`.

## Licence

Karaage is free software: you can redistribute it and modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License or (at your option) any later version. See [`LICENSE`](LICENSE). Copyright © 2026 the obs-karaage authors.

- `src/karaage/simple.min.css` is [Simple.css](https://simplecss.org), under the MIT licence; its notice is in the file.
- The test dataset has its own licences: see its README.
