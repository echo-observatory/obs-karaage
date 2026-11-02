# Karaage 1.1.0 design notes

These notes hold the decisions behind the 1.1.0 restructuring and the reasons for them.
Every 1.1.0 issue links here.
Read `AGENTS.md` first. When an issue and these notes disagree, or neither settles a question, ask before deciding (section 9).

## 1. Scope

Karaage takes one English text and returns the statements made in it, each with the actor who made it and the passage it comes from.
It runs seven steps, one module each:

1. **Segmentation** (`segmentation.py`): splits the text into clause-level segments with spaCy and benepar. Lossless: the segments joined give back the text.
2. **Attribution** (`attribution.py`): an LLM assigns ranges of segments to the actors quoted or cited in the text, then overlapping ranges are resolved.
3. **Passages** (`passages.py`): groups consecutive segments into short passages, each dominated by one actor where possible.
4. **Extraction** (`extraction.py`): an LLM states the main point of the text, then extracts up to four statements per actor and passage.
5. **Rephrasing** (`rephrasing.py`): an LLM rewrites each statement so that it stands alone, plus a one-sentence version.
6. **Paraphrasing** (`paraphrasing.py`): statements too close to the source wording (ROUGE-2 precision above 0.3) are paraphrased again, with an embedding check that the meaning is kept.
7. **Validation** (`validation.py`): an LLM checks that each statement is in the passage and attributed to the right actor.

In scope:
- The seven steps and the actor kinds (section 3).
- Writing one output file and one HTML report per document (section 6).
- Warning about input that looks wrong (not English, not normalised).

Out of scope:
- **Translation.** Karaage assumes English input, native or translated by a separate process.
- **Reading from specific databases** such as Retriever's. Karaage reads a generic input file (section 5).
- **Reconciling actors** across documents, including deciding who `base_author` is.
- **Assembling snapshot bundles.** Karaage output is input to a separate bundle builder (section 6.4).

Karaage never modifies the input text. All offsets point into the text exactly as given.

## 2. Terms

- **Segment:** the smallest unit, produced by step 1. A dict `{id, text, start, end}`; `start` and `end` are code-point offsets into the text, `end` excluded.
- **Actor:** a person, organisation or document that a segment is attributed to in step 2. The code calls it a *source*, with a `main` name and `other` names.
- **`base_author`:** the placeholder actor for every segment not attributed to anyone else, i.e. the text's own voice. It stays a placeholder in the output (section 3.3).
- **Document author:** whoever the text names as its author, such as a byline. Not part of 1.1.0 (section 3.4).
- **Passage:** consecutive segments grouped in step 3, at most 2,000 characters, short enough for a human to check.
- **Statement:** what one actor states in one passage, extracted in step 4 and then rephrased, possibly paraphrased, and validated.
- **Main point:** one sentence stating the central argument of the text, given as context to steps 4 and 5.
- **Profile** and **backend:** see section 4.

## 3. Decisions

### 3.1 English only

Karaage assumes English input. A cheap stopword check adds a warning to the output when the text does not look English; it never stops the run.
*Reason:* translation is a separate process with its own documentation; Karaage stays single-purpose.

### 3.2 Statement offsets cover the passage

A statement's `char_start` and `char_end` are those of its passage: the `start` of the passage's first segment and the `end` of its last.
*Reason:* the offsets let a human check the statement against its source. A passage gives them some context. Passage boundaries are also more reliable than attribution, so segment-level offsets would add precision that inherits attribution errors.

### 3.3 `base_author` stays a placeholder

Statements in the text's own voice keep the actor `base_author`. Karaage does not replace it with the document author's name.
*Reason:* method clarity. Linking `base_author` to the document's author is a later reconciliation step.

### 3.4 Actors

Each actor in the output has:
- `canonical_name`: the full name of a person, the official name of an organisation, or the title of a document;
- `aliases`: the other names the text uses for it, possibly empty;
- `kind`: `person`, `organisation` or `null` (documents, unknown). These are the snapshot specification's values, with its spelling. The attribution prompt lets the model choose among `person`, `organisation`, `document` and `unknown`, and the output maps the last two to `null`.

The **document author** is not part of 1.1.0. A separate LLM call, given the text (or its beginning and end) and asked who the text presents as its author, named quoted people and interviewees as authors in most of the 30 test articles, which have no bylines. Asking for a quote of the byline did not fix it. It will be revisited once there are evals (milestone "Quality").

There are no character offsets and no `source` field: everything Karaage reports is inferred from the text.

### 3.5 Main point of long documents

When the text does not fit the context:
1. Split it at passage boundaries into chunks that fit.
2. Summarise each chunk.
3. Combine the summaries into the main point.

A text that fits gets a single call. The output records which way was used.

### 3.6 Model choice through profiles

A release ships one or more named **profiles**; a profile fixes the model and how it is driven (section 4.2).
- **Shipped profiles** are part of the release. The prompts are tuned for them, and the reference run uses them.
- **Custom profiles:** a user can pass a profile file of their own. The output then names it `custom:<name>`.
- **Output:** every output records the profile.

*Reason:*
- Baking one model into a release would document it simply but prevents adapting to other machines and budgets.
- A free-form model setting would make outputs hard to compare.
- Profiles give a documented default, and flexibility that is visible in the output.

### 3.7 Versions

The rules are in `AGENTS.md`. Changing or adding a shipped profile is a MINOR change.

### 3.8 Rejected statements stay in the output

Statements that fail validation are kept, with `valid: false` and a reason. The bundle builder filters them.
*Reason:* monitoring. The rejection rate and reasons show how a profile performs.

### 3.9 Distribution

The distribution name is `obs-karaage`, because `karaage` on PyPI is an unrelated project. The import name stays `karaage`.
1.1.0 is not published on PyPI: install it from the git tag, e.g. `pip install "obs-karaage @ git+https://github.com/echo-observatory/obs-karaage@v1.1.0"`.

## 4. LLM configuration

### 4.1 Backend

The backend says where the model runs. It is local configuration and not part of the method.
Settings are read from environment variables, or from a `.env` file through python-dotenv, into a frozen dataclass (`karaage.config`).
`load_settings(**overrides)` returns it, with keyword arguments overriding the environment. The `.env` file is looked for in the working directory and its parents.

| Variable | Meaning | Default |
| --- | --- | --- |
| `KARAAGE_BACKEND` | `lmstudio`, `openrouter` or `openai` (any other OpenAI-compatible server) | `lmstudio` |
| `KARAAGE_BASE_URL` | API base URL | `http://127.0.0.1:1234/v1` for `lmstudio`, `https://openrouter.ai/api/v1` for `openrouter`, none for `openai` |
| `KARAAGE_API_KEY` | API key | `lm-studio` for `lmstudio`, required otherwise |
| `KARAAGE_PROFILE` | name of a shipped profile, or path to a profile file | the release's default profile |

For `openrouter`, requests set the provider options `require_parameters: true` and `data_collection: "deny"`. With `require_parameters`, only providers that support every parameter sent are used, which is one reason why profiles send no seed at temperature 0.

For `lmstudio`, `setup()` also reads LM Studio's own API (`/api/v1/models`) to record the loaded build of the model: publisher, format, quantisation and context length. It warns when the model is not loaded, because LM Studio would then load it with its own default settings, and when the loaded context is shorter than the profile's `context_tokens`.
Retriever data must never be sent to OpenRouter or any other external API; this is a team rule that the code does not enforce.

### 4.2 Profile

A profile is a TOML file. Shipped profiles live in `src/karaage/profiles/`, and their names are reserved.
The same model can have different ids, and need a different way of turning reasoning off, on each backend. So a profile has one section per backend it supports; a backend without a section cannot run the profile.

```toml
name = "qwen3.5-9b"
description = "Qwen3.5 9B with reasoning off. Default profile of Karaage 1.1.0; the prompts were written for this model."

context_tokens = 8000        # LM Studio must load the model with a context length of at least this
temperature = 0
embedding_model = "intfloat/multilingual-e5-large-instruct"
# seed = 0                   # optional; sent with every request when set

[backends.lmstudio]
model = "qwen/qwen3.5-9b"    # LM Studio's own build: MLX 4-bit on Apple silicon, GGUF on other systems
reasoning_off = "request-parameter"

[backends.openrouter]
model = "qwen/qwen3.5-9b"
reasoning_off = "request-parameter"
```

Unknown keys are errors, so that a typo cannot silently fall back to a default. `seed` is optional: at temperature 0 it changes nothing, and on OpenRouter it would exclude providers that do not support it.

`reasoning_off` is one of:
- `none`: the model does not reason;
- `no-think-tag`: prepend `/no_think` to the system prompt;
- `prefill-empty-think`: add an assistant turn `<think>\n\n</think>\n\n`;
- `request-parameter`: use the backend's reasoning parameter: top-level `reasoning_effort: "none"` on LM Studio and other OpenAI-compatible servers, `reasoning: {"effort": "none"}` on OpenRouter.

On LM Studio's `qwen/qwen3.5-9b` build (November 2026), only `request-parameter` and `prefill-empty-think` turn reasoning off. `/no_think` is ignored: a one-word question still produced 156 reasoning tokens. Measured with LM Studio's native `reasoning` and `chat_template_kwargs` options too, which are also ignored on the OpenAI-compatible endpoint.

The default profile keeps the pre-1.1.0 settings: model `qwen/qwen3.5-9b`, 8,000-token context, temperature 0, and the embedding model above. Qwen3.5 9B stays the default for 1.1.0 because the prompts were written for it; other models may get profiles in later releases.

### 4.3 LLM calls

All LLM calls go through one client (`karaage.llm`), activated by `setup()`. It:
- applies the profile: the model id for the backend, temperature, seed and reasoning switch-off;
- requests JSON output against a schema;
- retries with backoff through the OpenAI SDK's `max_retries`, replacing the pre-1.1.0 immediate retry;
- raises a clear error when the answer is cut off by the token limit, instead of failing later on invalid JSON;
- falls back to `reasoning_content` when `content` is empty, and strips `<think>` blocks, as LM Studio sometimes requires;
- records or replays responses for tests (section 7.1).

Two problems in the pre-1.1.0 code that this resolves:
- **Inconsistent reasoning switch-off.** Attribution prepends `/no_think`, which the current LM Studio build ignores (section 4.2); rephrasing, paraphrasing and validation pre-fill an empty `<think>` block; extraction (main point and passages) does neither. So on this build, attribution and extraction ran with reasoning on. Main-point output is capped at 250 tokens, so reasoning output would exhaust it. This may explain the truncated main points.
- **Token budgets that ignore the profile.** Budget constants assume an 8,000-token context; rephrasing even asks for up to 8,000 output tokens. Budgets that depend on the context size must derive from the profile's `context_tokens`.

## 5. Input

The Python API keeps its current shape:
- `karaage.setup(profile=None, backend=None, base_url=None, api_key=None)` loads the NLP models, the embedding model and the LLM client once, from the backend settings and a profile; the keyword arguments override the environment (section 4.1).
- `karaage.run(document, resources)` processes one document.
- `karaage.save(...)` writes the output files.

A document has:

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Document id, unique in the run. Integers are accepted and stored as strings. |
| `text` | yes | The text, in English. |
| `title` | no | Title or headline, given as context to the main point step. |
| `published_at` | no | Publication date or time, given as context in the same steps. |

The command line reads JSONL, one document per line: `karaage run --input docs.jsonl --output DIR [--profile NAME] [--report]`.
Field names can be mapped, e.g. the test dataset's `date` to `published_at`.

## 6. Output

### 6.1 Files

Each document gets `<id>.json`, plus `<id>.html` (the report) when asked for.
- **File names:** must be unique per document id. Characters unsafe in file names are encoded, never dropped or truncated in a way that could collide.
- **Report:** self-contained, with CSS inlined rather than loaded from a CDN, so it can be copied into a bundle and opened offline.

### 6.2 Document status

- `ok`: every step ran without error. A document with no statements can be `ok`.
- `partial`: every step ran, but some passages, batches or statements failed and are listed in `errors`, so statements may be missing.
- `failed`: a step failed as a whole. There are no statements, but the file is still written, with the error, so a rerun knows to retry it.

A rerun skips documents whose status is `ok` and retries the others.

### 6.3 Fields

The JSON Schema of the output lives in `src/karaage/schemas/` and is the reference. Every field is filled; empty lists and `null` stand for absent values.

```json
{
  "document_id": "19",
  "status": "ok",
  "errors": [],
  "warnings": [],
  "text_length": 5120,
  "text_sha256": "…",
  "main_point": "…",
  "main_point_method": "single",
  "actors": [
    {"canonical_name": "Jane Doe", "aliases": ["Doe", "the spokesperson"], "kind": "person"}
  ],
  "statements": [
    {
      "id": "19-s1",
      "actor": "Jane Doe",
      "actor_aliases": ["Jane Doe", "Doe", "the spokesperson"],
      "text": "…",
      "text_short": "…",
      "char_start": 812,
      "char_end": 2304,
      "valid": true,
      "rejection_reason": null,
      "diagnostics": {"rouge2_precision": 0.12, "paraphrased": false, "in_text": "yes", "actor_correct": "yes"}
    }
  ],
  "provenance": {
    "tool": "karaage",
    "version": "1.1.0",
    "commit": "…",
    "profile": "qwen3.5-9b",
    "profile_settings": {},
    "backend": "lmstudio",
    "model": "qwen/qwen3.5-9b",
    "model_reported": "qwen/qwen3.5-9b",
    "model_build": {"publisher": "qwen", "format": "mlx", "quantization": "4bit", "loaded": true, "context_length": 8192},
    "inferred_at": "2026-11-02T10:11:00Z",
    "step_seconds": {}
  }
}
```

- **Text fingerprint:** `text_length` counts code points and `text_sha256` is taken from the UTF-8 text, so a later change to the text can be detected.
- **Warnings:**
  - the text is not NFC-normalised;
  - it contains `\r`;
  - it does not look like English.
- **`main_point_method`:** `single` or `chunked`.
- **Statement ids:**
  - written `<document_id>-s<n>`, with `n` counted from 1 in extraction order (by passage, then actor);
  - rejected statements included, so ids don't depend on validation;
  - unique within a run.
- **Actor:** `actor` is the actor's `canonical_name`, or `base_author`. `actor_aliases` is the canonical name followed by the aliases; for `base_author` it is `["base_author"]`.
- **Rejection reasons:** `validation error`, `rephrasing error`, `too similar to source`, `not found in source text` or `wrong actor attribution`, as in `validation.py`.
- **Provenance:**
  - `commit` is the git commit Karaage was built from, recorded at build time, so that installed releases carry it too;
  - `profile_settings` is the resolved profile;
  - `model` is the id the profile asks for on this backend, and `model_reported` the id the server returns;
  - `model_build` is the loaded build as LM Studio reports it (section 4.1); on OpenRouter, the providers that served the document's requests, since they may run different precisions; `null` on other backends.
- **`inferred_at`:** UTC in the snapshot format `YYYY-MM-DDTHH:MM:SSZ`.

### 6.4 Mapping to the snapshot specification

The bundle builder (not Karaage) turns outputs into snapshot tables (spec 1.0, `obs-corpus-snapshot`). Karaage fields map as follows.

`inferred_statements`, one row per valid statement:

| Snapshot field | Karaage field |
| --- | --- |
| `id` | `statements[].id` (the builder may prefix it if one snapshot holds several runs of a document) |
| `source_document_id` | `document_id` |
| `claim_id` | none: set by the claim inference process |
| `text` | `statements[].text` |
| `text_short` | `statements[].text_short` |
| `char_start`, `char_end` | `statements[].char_start`, `char_end` |
| `source_text_length` | `text_length` |
| `raw_actor_aliases` | `statements[].actor_aliases` |
| `inferred_at` | `provenance.inferred_at` |
| `inference_tool`, `inference_version` | `provenance.tool`, `provenance.version` |
| `inference_profile` (flex field) | `provenance.profile` |

`source_documents`:
- `main_point` comes from `main_point`.
- `monitoring_report_path` points to the HTML report once the builder has copied it under `monitoring/`.
- `title`, `published_at` and the text come from the corpus, not from Karaage.

The profile settings go under `snapshot.statement_inference` in the bundle descriptor.
`inferred_actors` is not filled from Karaage output: its actors are reconciled ones, and Karaage's are raw.

## 7. Testing and quality

### 7.1 Recorded LLM responses

Tests run without an LLM server by replaying recorded responses.
- **Mode:** `KARAAGE_LLM_MODE` is `live` (default), `record` or `replay`.
- **Storage:** recordings are one JSON file per call in `tests/recordings/<profile>/` (`KARAAGE_RECORDINGS` sets the root), named by the SHA-256 of the request (backend, model id, messages, schema, generation parameters). One file per call keeps merges between branches simple.
- **Expected outputs:** recording also saves each article's output in `tests/recordings/<profile>/expected/`. In `replay` mode, the pipeline test compares the output with it, so any change in behaviour shows up as a failing test, and as a diff of the expected outputs when recorded again.
- **Same results on any machine:** the embedding model runs on the CPU, so that results do not depend on the machine's GPU.
- **Missing recording:** in `replay` mode, a missing recording fails the test with a clear message. It never falls back to a live call.
- **When to redo:** recordings come only from the test dataset, and are redone whenever prompts, schemas or profiles change. Doing so needs a live backend; the command is in `tests/test_pipeline.py` and the README.

### 7.2 Human check of method changes

There is no gold standard to measure extraction quality against.
A change to prompts, schemas or step logic is checked by a human on the HTML reports of the 30 test articles before merging. The PR says which articles were checked and what changed.

### 7.3 Continuous integration

GitHub Actions runs the tests in replay mode on every pull request. The benepar and embedding models are cached, since private repos on the free tier get 2,000 minutes a month; the spaCy model is a locked dependency.

## 8. Test dataset

`data/test-datasets/wikinews-piracy-30/` is a copy of the dataset in `echo-observatory/obs-test-datasets`, the canonical home. Fix it there, then copy it.
Useful cases:
- **Article 7:** an interview transcript with speaker labels in brackets.
- **Article 19:** names its reporter in the text, a test case for the document author (deferred, section 3.4).
- **Article 30:** about 39,500 characters, larger than the 8,000-token context, which tests long documents.

## 9. Working on an issue

- Read `AGENTS.md`, these notes and the issue.
- Have `pipeline`, `obs-corpus-snapshot` and `obs-test-datasets` cloned next to this repo. They are private, and the snapshot validator is installed from `../obs-corpus-snapshot`.
- **Stop and ask** before deciding anything that changes the output, the method or the configuration and is not settled here or in the issue. Internal code structure is yours to decide.
- **Without an LLM backend:** work in replay mode, and say in the PR what still needs a live check.
- **Outside the issue's scope:** don't change prompts or behaviour. Report what you notice instead.
- **Data:** never use data other than the test datasets, and never paste output from other corpora into issues, PRs, code or tests.
