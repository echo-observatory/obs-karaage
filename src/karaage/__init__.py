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

import html as html_lib
import time
import pathlib
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from openai import OpenAI
from karaage._context import set_client, get_client


@dataclass
class Resources:
    model: str
    nlp: object          # spacy.Language
    sim_model: SentenceTransformer
    embedding_model: str
    enable_thinking: bool = False
    max_context_tokens: int = 8_000


@dataclass
class PipelineResult:
    data: dict
    step_times: dict
    step_errors: dict       # step name → error string, or "skipped"; empty on full success
    html_report: str
    # intermediate state exposed for batch_debug
    all_statements: list
    passages: list
    main_point: str


def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def create_llm_client(base_url: str, api_key: str = 'lm-studio') -> OpenAI:
    client = OpenAI(base_url=base_url, api_key=api_key)
    available = [m.id for m in client.models.list().data]
    print('Available models:', available)
    set_client(client)
    return client


def set_verbose(value: bool) -> None:
    from karaage import segmentation, attribution, passages, extraction, rephrasing, paraphrasing, validation, output
    for mod in [segmentation, attribution, passages, extraction, rephrasing, paraphrasing, validation, output]:
        mod.verbose = value


def setup(
    lm_studio_url: str,
    model: str,
    embedding_model: str,
    *,
    api_key: str = 'lm-studio',
    enable_thinking: bool = False,
    max_context_tokens: int = 8_000,
) -> Resources:
    from karaage.segmentation import load_nlp_pipeline
    create_llm_client(lm_studio_url, api_key=api_key)
    sim_model = load_embedding_model(embedding_model)
    nlp = load_nlp_pipeline()
    return Resources(model=model, nlp=nlp, sim_model=sim_model, embedding_model=embedding_model,
                     enable_thinking=enable_thinking, max_context_tokens=max_context_tokens)


def run(text: str, resources: Resources) -> PipelineResult:
    from karaage.segmentation import step_1_segment
    from karaage.attribution import step_2_attribute_authors
    from karaage.passages import step_3_passages
    from karaage.extraction import step_4_key_statements
    from karaage.rephrasing import step_5_rephrase
    from karaage.paraphrasing import step_6_paraphrase
    from karaage.validation import step_7_validate
    from karaage.output import build_results, build_html_report, build_speaker_styles

    step_times = {}
    step_errors = {}
    pipeline_stopped = False
    model = resources.model

    def _timed(name, fn, n_returns=2):
        nonlocal pipeline_stopped
        if pipeline_stopped:
            step_errors[name] = "skipped"
            return (None,) * n_returns
        t0 = time.perf_counter()
        try:
            out = fn()
            step_times[name] = int(time.perf_counter() - t0)
            return out
        except Exception as exc:
            step_times[name] = int(time.perf_counter() - t0)
            step_errors[name] = f"{type(exc).__name__}: {exc}"
            pipeline_stopped = True
            print(f"  ❌ {name} failed: {exc}")
            return (None,) * n_returns

    segments,       html_seg  = _timed("Segmentation",              lambda: step_1_segment(text, resources.nlp))
    sources,        html_attr = _timed("Attribution",         lambda: step_2_attribute_authors(segments, model, resources.enable_thinking, resources.max_context_tokens))
    passages_list,  html_pass = _timed("Passages",        lambda: step_3_passages(segments, sources))
    all_stmts, main_point, html_extr = _timed("Key statements extraction", lambda: step_4_key_statements(segments, passages_list, model), n_returns=3)
    rephrased,      html_reph = _timed("Rephrasing",                 lambda: step_5_rephrase(all_stmts, passages_list, main_point, model))
    paraphrased,    html_para = _timed("Paraphrasing",               lambda: step_6_paraphrase(rephrased, resources.embedding_model, model, resources.sim_model))
    filtered,       html_val  = _timed("Validation",                 lambda: step_7_validate(paraphrased, model))

    def _make_section_html(name, html_content):
        if name not in step_errors:
            return html_content
        error = step_errors[name]
        if error == "skipped":
            return '<p><em>Step skipped — a prior step failed.</em></p>'
        exc_escaped = html_lib.escape(error)
        return (
            f'<p style="color:#721c24;background:#f8d7da;padding:8px;border-radius:4px">'
            f'<strong>❌ Step failed:</strong><br><code>{exc_escaped}</code></p>'
        )

    if not step_errors:
        data = build_results(filtered, segments, sources, main_point)
        speaker_styles = build_speaker_styles(sources)
    else:
        data = {}
        speaker_styles = ""

    sections = [
        {"title": "Segmentation",              "html": _make_section_html("Segmentation",              html_seg)},
        {"title": "Attribution",               "html": _make_section_html("Attribution",               html_attr)},
        {"title": "Passages",                  "html": _make_section_html("Passages",                  html_pass)},
        {"title": "Key statements extraction", "html": _make_section_html("Key statements extraction", html_extr)},
        {"title": "Rephrasing",                "html": _make_section_html("Rephrasing",                html_reph)},
        {"title": "Paraphrasing",              "html": _make_section_html("Paraphrasing",              html_para)},
        {"title": "Validation",                "html": _make_section_html("Validation",                html_val)},
    ]
    html_report = build_html_report(speaker_styles, sections)

    print_time(step_times, chars=len(text), compact=True)

    return PipelineResult(
        data=data,
        step_times=step_times,
        step_errors=step_errors,
        html_report=html_report,
        all_statements=all_stmts or [],
        passages=passages_list or [],
        main_point=main_point or "",
    )

def print_time(step_times, chars, compact=False):
    if compact:
        total = sum(step_times.values())
        steps_str = " / ".join(
            f"{''.join(w[0].upper() for w in name.split())}: {secs}s"
            for name, secs in step_times.items()
        )
        print(f'⏳ {total}s --- {(total / (chars / 1000)):.1f} s/kchar --- {steps_str}')
    else:
        total = sum(step_times.values())
        print('Step timing recap')
        print('-' * 36)
        for name, secs in step_times.items():
            print(f'  {name:<32} {secs:>4}s')
        print('-' * 36)
        print(f'  {"Total":<32} {total:>4}s')
        print(f'  {"-> Total per 1000 chars":<32} {(total / (chars / 1000)):.1f} s/kchar')

def save(
    result: PipelineResult,
    *,
    data_path=None,
    html_path=None,
) -> None:
    from karaage.output import save_results
    if data_path is not None:
        save_results(pathlib.Path(data_path), result.data)
    if html_path is not None:
        dest = pathlib.Path(html_path)
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(result.html_report, encoding="utf-8")
