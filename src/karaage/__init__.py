try:
    # Written from git tags by hatch-vcs when the package is built or installed.
    from karaage._version import __version__
except ImportError:  # source tree that was never installed
    __version__ = "0+unknown"
try:
    # Written by hatch_build.py: the git commit the package was built from.
    from karaage._build_info import commit as __commit_id__
except ImportError:
    __commit_id__ = None

import datetime
import html as html_lib
import time
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from karaage import llm
from karaage.config import load_settings
from karaage.profile import Profile, load_profile


@dataclass
class Resources:
    client: llm.LLMClient   # backend and profile
    nlp: object             # spacy.Language
    sim_model: SentenceTransformer

    @property
    def profile(self) -> Profile:
        return self.client.profile

    @property
    def model(self) -> str:
        return self.client.model

    @property
    def embedding_model(self) -> str:
        return self.client.profile.embedding_model

    @property
    def max_context_tokens(self) -> int:
        return self.client.profile.context_tokens


@dataclass
class RunResult:
    data: dict              # the output (docs/design-1.1.0.md, section 6.3; schemas/output.schema.json)
    html_report: str
    # intermediate state exposed for batch_debug
    all_statements: list
    excerpts: list


def load_embedding_model(model_name: str) -> SentenceTransformer:
    # On the CPU everywhere, so that results don't depend on the machine's GPU (Apple MPS, CUDA).
    # The model only embeds a few sentences per document.
    return SentenceTransformer(model_name, device="cpu")


def set_verbose(value: bool) -> None:
    from karaage import segmentation, attribution, excerpts, extraction, rephrasing, paraphrasing, validation, output
    for mod in [segmentation, attribution, excerpts, extraction, rephrasing, paraphrasing, validation, output]:
        mod.verbose = value


def setup(*, profile=None, backend=None, base_url=None, api_key=None) -> Resources:
    """Connect to the LLM backend and load the models, once per session.

    The backend settings and the profile come from KARAAGE_* environment
    variables or a .env file (docs/design-1.1.0.md, section 4); keyword
    arguments override them. `profile` is a shipped profile name or a path.
    The LLM is always called live.
    """
    return _setup(load_settings(profile=profile, backend=backend, base_url=base_url, api_key=api_key))


def _setup(settings, *, llm_mode: str = "live", recordings=None) -> Resources:
    """`setup()` from loaded settings, in an LLM mode that only the test suite sets (tests/conftest.py).

    `llm_mode` is `live`, `record` or `replay`, with the recordings under
    `recordings` (docs/design-1.1.0.md, section 7.1).
    """
    from karaage.segmentation import load_nlp_pipeline
    client = llm.LLMClient(settings, load_profile(settings.profile), mode=llm_mode, recordings=recordings)
    client.check()
    llm.activate(client)
    build = client.model_build
    build_str = f", {build['format']} {build['quantization']}" if build else ""
    replay = " (replaying recorded responses)" if client.mode == "replay" else ""
    print(f"Profile {client.profile.label}: {client.model}{build_str} on {client.backend} at {settings.base_url}{replay}")
    sim_model = load_embedding_model(client.profile.embedding_model)
    nlp = load_nlp_pipeline()
    return Resources(client=client, nlp=nlp, sim_model=sim_model)


def read_document(document) -> tuple[str, str, str | None, str | None]:
    """The id, text, title and publication date of a document (docs/design-1.1.0.md, section 5)."""
    if not isinstance(document, Mapping):
        raise TypeError("run() takes a document: a dict with an 'id' and a 'text'.")
    document_id = document.get("id")
    if document_id is None:
        raise ValueError("A document has no id.")
    if isinstance(document_id, bool) or not isinstance(document_id, (str, int)) or document_id == "":
        raise ValueError(f"A document id must be a non-empty string or an integer, not {document_id!r}.")
    text = document.get("text")
    if text is None:
        raise ValueError(f"Document {document_id} has no text.")
    if not isinstance(text, str):
        raise ValueError(f"The text of document {document_id} must be a string, not {type(text).__name__}.")
    title = document.get("title")
    if title is not None and not isinstance(title, str):
        raise ValueError(f"The title of document {document_id} must be a string, not {type(title).__name__}.")
    published_at = document.get("published_at")
    if isinstance(published_at, (datetime.date, datetime.datetime)):
        published_at = published_at.isoformat()
    if published_at is not None and not isinstance(published_at, str):
        raise ValueError(f"The publication date of document {document_id} must be a string or a date, "
                         f"not {type(published_at).__name__}.")
    return str(document_id), text, title or None, published_at or None


def run(document, resources: Resources, *, paraphrase: bool = True) -> RunResult:
    """Process one document: a dict with an `id` (string or integer) and a `text`,
    and optionally a `title` and a `published_at` date, given as context to the main point step.

    `paraphrase=False` turns off step 6, which rewords statements too close to their
    source, and validation's check for them. Only for sources whose wording may be
    reproduced, e.g. in the public domain; never for Retriever data (README).

    Never raises on a failing step: the output then has the status `failed`, with the error.
    """
    from karaage.segmentation import step_1_segment
    from karaage.attribution import step_2_attribute_actors
    from karaage.excerpts import step_3_excerpts
    from karaage.extraction import step_4_key_statements
    from karaage.rephrasing import step_5_rephrase
    from karaage.paraphrasing import step_6_paraphrase
    from karaage.validation import step_7_validate
    from karaage.output import STEPS, TOOL, build_output, build_html_report, build_actor_styles, error, timestamp

    document_id, text, title, published_at = read_document(document)
    llm.activate(resources.client)
    resources.client.start_document()
    step_seconds = {}
    failure = None          # the error of the step that failed as a whole
    model = resources.model

    def _step(name, fn, n_returns=2):
        nonlocal failure
        if failure:
            return (None,) * n_returns
        t0 = time.perf_counter()
        try:
            return fn()
        except Exception as exc:
            failure = error(name, f"{type(exc).__name__}: {exc}")
            print(f"  ❌ {STEPS[name]} failed: {exc}")
            return (None,) * n_returns
        finally:
            step_seconds[name] = round(time.perf_counter() - t0, 1)

    segments,       html_seg  = _step("segmentation", lambda: step_1_segment(text, resources.nlp))
    attribution,    html_attr = _step("attribution",  lambda: step_2_attribute_actors(segments, model, resources.max_context_tokens))
    excerpts_list,  html_exc  = _step("excerpts",     lambda: step_3_excerpts(segments, attribution))
    all_stmts, main_point, html_extr, excerpt_errors, main_point_method = _step("extraction", lambda: step_4_key_statements(segments, excerpts_list, title, published_at), n_returns=5)
    rephrased,      html_reph = _step("rephrasing",   lambda: step_5_rephrase(all_stmts, excerpts_list, main_point))
    paraphrased,    html_para = _step("paraphrasing", lambda: step_6_paraphrase(rephrased, resources.embedding_model, resources.sim_model, paraphrase))
    validated,      html_val  = _step("validation",   lambda: step_7_validate(paraphrased, check_similarity=paraphrase))

    provenance = {
        "tool": TOOL,
        "version": __version__,
        "commit": __commit_id__,
        **resources.client.provenance(),
        "paraphrase": paraphrase,
        "inferred_at": timestamp(datetime.datetime.now(datetime.timezone.utc)),
        "step_seconds": step_seconds,
    }
    data = build_output(
        document_id, text,
        failure=failure,
        excerpt_errors=excerpt_errors or [],
        segments=segments,
        attribution=attribution,
        statements=validated,
        main_point=main_point,
        main_point_method=main_point_method,
        provenance=provenance,
    )

    def _section_html(name, html_content):
        if failure and failure["step"] == name:
            exc_escaped = html_lib.escape(failure["message"])
            return (
                f'<p style="color:#721c24;background:#f8d7da;padding:8px;border-radius:4px">'
                f'<strong>❌ Step failed:</strong><br><code>{exc_escaped}</code></p>'
            )
        if name not in step_seconds:
            return '<p><em>Step skipped — a prior step failed.</em></p>'
        return html_content

    step_html = {
        "segmentation": html_seg, "attribution": html_attr, "excerpts": html_exc, "extraction": html_extr,
        "rephrasing": html_reph, "paraphrasing": html_para, "validation": html_val,
    }
    sections = [{"title": title, "html": _section_html(name, step_html[name])} for name, title in STEPS.items()]
    actor_styles = build_actor_styles(attribution) if attribution is not None else ""
    html_report = build_html_report(data, actor_styles, sections)

    print_time(step_seconds, chars=len(text), compact=True)

    return RunResult(
        data=data,
        html_report=html_report,
        all_statements=all_stmts or [],
        excerpts=excerpts_list or [],
    )

def print_time(step_seconds, chars, compact=False):
    total = round(sum(step_seconds.values()), 1)
    per_kchar = total / (max(chars, 1) / 1000)
    if compact:
        steps_str = " / ".join(f"{name}: {secs}s" for name, secs in step_seconds.items())
        print(f'⏳ {total}s --- {per_kchar:.1f} s/kchar --- {steps_str}')
    else:
        print('Step timing recap')
        print('-' * 36)
        for name, secs in step_seconds.items():
            print(f'  {name:<30} {secs:>6}s')
        print('-' * 36)
        print(f'  {"Total":<30} {total:>6}s')
        print(f'  {"-> Total per 1000 chars":<30} {per_kchar:.1f} s/kchar')


def save(
    result: RunResult,
    *,
    data_path=None,
    html_path=None,
) -> None:
    """Write the output as JSON to data_path, and the HTML report to html_path. Failed documents are written too."""
    from karaage.output import save_results, write_file
    if html_path is not None:   # before the data, whose ok status tells a rerun the document is done
        write_file(html_path, result.html_report)
    if data_path is not None:
        save_results(pathlib.Path(data_path), result.data)
