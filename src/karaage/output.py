"""The output of one document and its HTML report (docs/design-1.1.0.md, section 6).

The output's JSON Schema is schemas/output.schema.json.
"""

import datetime
import hashlib
import html as html_lib
import json
import os
import pathlib
import re
import unicodedata
from importlib.resources import files

from karaage.attribution import ACTOR_COLORS

verbose = False

TOOL = "karaage"

# The steps, as named in the output, with their titles in the HTML report.
STEPS = {
    "segmentation": "Segmentation",
    "attribution": "Attribution",
    "excerpts": "Excerpts",
    "extraction": "Key statements extraction",
    "rephrasing": "Rephrasing",
    "paraphrasing": "Paraphrasing",
    "validation": "Validation",
}

# A text looks English when enough of its words are frequent English function words.
# They make up 25% to 40% of the words of the test articles, and under 6% of the words
# of invented Danish, French, German and Spanish texts (tests/test_output.py).
ENGLISH_STOPWORDS = frozenset("""
    the of and to a in is that it for was on with as by be this are from his her he she
    have has had not but or an they which were their said will would been its who we you
""".split())
ENGLISH_MIN_SHARE = 0.15
ENGLISH_MIN_WORDS = 20      # shorter texts are not checked


def output_schema() -> dict:
    """The JSON Schema of the output."""
    return json.loads(files("karaage").joinpath("schemas", "output.schema.json").read_text(encoding="utf-8"))


def timestamp(value: datetime.datetime) -> str:
    """UTC in the snapshot format, YYYY-MM-DDTHH:MM:SSZ."""
    return value.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def text_warnings(text: str) -> list[dict]:
    """Warnings about input that looks wrong. Karaage runs anyway, on the text as given."""
    warnings = []
    if unicodedata.normalize("NFC", text) != text:
        warnings.append({"code": "text-not-nfc", "message": "The text is not in Unicode NFC."})
    if "\r" in text:
        warnings.append({"code": "text-has-carriage-return", "message": "The text contains carriage returns (\\r)."})
    words = re.findall(r"[^\W\d_]+", text.lower())
    if len(words) >= ENGLISH_MIN_WORDS:
        share = sum(word in ENGLISH_STOPWORDS for word in words) / len(words)
        if share < ENGLISH_MIN_SHARE:
            warnings.append({
                "code": "text-not-english",
                "message": f"The text does not look like English: only {share:.0%} of its words are common English words.",
            })
    return warnings


def snapshot_kind(kind):
    """The snapshot specification's actor kinds: person, organisation, or null for anything else."""
    return kind if kind in ("person", "organisation") else None


def error(step: str, message: str, *, excerpt: int | None = None, statement: str | None = None,
          answer: dict | None = None) -> dict:
    return {"step": step, "excerpt": excerpt, "statement": statement, "message": message, "answer": answer}


def statement_id(document_id: str, s: dict) -> str:
    """<document_id>-s<n>, n counted from 1 in extraction order (s["id"] counts from 0)."""
    return f"{document_id}-s{s['id'] + 1}"


def final_rouge2(s: dict) -> float | None:
    """The higher ROUGE-2 precision of the final long and short versions, as checked by validation."""
    scores = [
        s.get("paraphrased_rouge2_rephrased") if s.get("rephrased_paraphrased") else s.get("rouge2_rephrased"),
        s.get("paraphrased_rouge2_summary") if s.get("summary_paraphrased") else s.get("rouge2_summary"),
    ]
    scores = [score for score in scores if score is not None]
    return max(scores) if scores else None


def build_statement(s: dict, document_id: str, segments_by_id: dict, aliases_by_actor: dict) -> dict:
    text = s.get("paraphrased_statement") if s.get("rephrased_paraphrased") else s.get("rephrased_statement")
    text_short = s.get("paraphrased_summary") if s.get("summary_paraphrased") else s.get("summary")
    return {
        "id": statement_id(document_id, s),
        "actor": s["actor"],
        "actor_aliases": aliases_by_actor.get(s["actor"], [s["actor"]]),
        "text": text or None,
        "text_short": text_short or None,
        # The excerpt's offsets (docs/design-1.1.0.md, section 3.2).
        "char_start": segments_by_id[min(s["segments"])]["start"],
        "char_end": segments_by_id[max(s["segments"])]["end"],
        "valid": s["valid"],
        "rejection_reason": s["invalid_reason"],
        "diagnostics": {
            "rouge2_precision": final_rouge2(s),
            "paraphrased": bool(s.get("paraphrased")),
            "in_text": s.get("in_text"),
            "actor_correct": s.get("actor_correct"),
        },
    }


def statement_errors(statements: list, document_id: str) -> list[dict]:
    """Statements whose rephrasing batch or validation call failed."""
    errors = []
    for s in statements:
        if "rephrasing_error" in s:
            errors.append(error("rephrasing", s["rephrasing_error"], statement=statement_id(document_id, s),
                                answer=s.get("rephrasing_answer")))
    for s in statements:
        if s.get("validation_error"):
            errors.append(error("validation", s["validation_error"], statement=statement_id(document_id, s),
                                answer=s.get("validation_answer")))
    return errors


def build_output(
    document_id: str,
    text: str,
    *,
    failure: dict | None,
    excerpt_errors: list,
    segments: list | None,
    attribution,
    statements: list | None,
    main_point: str | None,
    main_point_method: str | None,
    provenance: dict,
) -> dict:
    """The output of one document (docs/design-1.1.0.md, section 6.3).

    failure: the error of a step that failed as a whole, or None. The document
    is then `failed`, and the results of the steps that ran are left out.
    excerpt_errors: the excerpts whose extraction failed ({"excerpt_index", "error"}).
    statements: the statements as validated by step 7.
    """
    if failure:
        status, errors = "failed", [failure]
        actors, statements_out, main_point, main_point_method = [], [], None, None
    else:
        errors = [error("extraction", e["error"], excerpt=e["excerpt_index"] + 1, answer=e.get("answer"))
                  for e in excerpt_errors]
        errors += statement_errors(statements, document_id)
        status = "partial" if errors else "ok"
        actor_list = list(getattr(attribution, "actors", attribution))
        actors = [
            {"canonical_name": actor.canonical_name, "aliases": list(actor.aliases), "kind": snapshot_kind(actor.kind)}
            for actor in actor_list
        ]
        aliases_by_actor = {actor.canonical_name: list(dict.fromkeys([actor.canonical_name, *actor.aliases])) for actor in actor_list}
        aliases_by_actor["base_author"] = ["base_author"]
        segments_by_id = {seg["id"]: seg for seg in segments}
        statements_out = [build_statement(s, document_id, segments_by_id, aliases_by_actor) for s in statements]
    return {
        "document_id": document_id,
        "status": status,
        "errors": errors,
        "warnings": text_warnings(text),
        "text_length": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "main_point": main_point,
        "main_point_method": main_point_method,
        "actors": actors,
        "statements": statements_out,
        "provenance": provenance,
    }


def build_actor_styles(attribution) -> str:
    ordered_actors = sorted(
        list(getattr(attribution, "actors", attribution)),
        key=lambda actor: min((start for start, _ in actor.segments), default=float("inf"))
    )
    style_rules = [
        f"  .actor{rank:02d} {{ background-color: {ACTOR_COLORS[(rank - 1) % len(ACTOR_COLORS)]}; }}"
        for rank in range(1, len(ordered_actors) + 1)
    ]
    return "\n".join(style_rules)


def _slugify(title: str) -> str:
    return re.sub(r'[^\w]+', '-', title.lower()).strip('-')


def write_file(dest, text: str) -> None:
    """Write a file whole or not at all, so that an interrupted run leaves no truncated output."""
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, dest)


def save_results(dest: pathlib.Path, result: dict) -> None:
    write_file(dest, json.dumps(result, indent=2, ensure_ascii=False))
    if verbose:
        print(f"Saved {len(result['statements'])} statements to {dest.resolve()}")


def _problems_html(data: dict) -> str:
    """The document's errors and warnings, for the top of the report."""
    def item(e):
        where = e["step"] + (f", excerpt {e['excerpt']}" if e["excerpt"] else "") + (f", {e['statement']}" if e["statement"] else "")
        return f"<li>{html_lib.escape(where)}: <code>{html_lib.escape(e['message'])}</code></li>"
    parts = []
    if data["errors"]:
        parts.append("<p><strong>Errors</strong></p>\n<ul>" + "".join(item(e) for e in data["errors"]) + "</ul>")
    if data["warnings"]:
        parts.append("<p><strong>Warnings</strong></p>\n<ul>"
                     + "".join(f"<li>{html_lib.escape(w['message'])}</li>" for w in data["warnings"]) + "</ul>")
    return "\n".join(parts)


def build_html_report(data: dict, actor_styles: str, sections: list) -> str:
    """The report of one document, self-contained so that it opens offline.

    data: the document's output. sections: list of {"title": str, "html": str}.
    """
    css = files("karaage").joinpath("simple.min.css").read_text(encoding="utf-8")
    provenance = data["provenance"]
    document_id = html_lib.escape(data["document_id"])
    nav = "\n".join(
        f'        <a href="#{_slugify(s["title"])}">{s["title"]}</a>'
        for s in sections
    )
    body = "\n\n".join(
        f'    <a name="{_slugify(s["title"])}"></a>\n'
        f'    <h2>{s["title"]}</h2>\n'
        f'    {s["html"]}'
        for s in sections
    )
    return f"""<!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <title>Monitoring Report: document {document_id}</title>
    <style>
{css}
    </style>
    <style>
        .seg0 {{ background-color: #eeeeee; }}
        .seg1 {{ background-color: #d4e0ef; }}
        mark {{ margin-right: 0.10em; font-style: normal; }}
        {actor_styles}
    </style>
    </head>
    <body>
    <header>
    <nav>
{nav}
    </nav>
    </header>
    <h1>Monitoring Report</h1>
    <p><small>Document {document_id} &mdash; status {data["status"]} &mdash; karaage v{html_lib.escape(provenance["version"])},
    profile {html_lib.escape(provenance["profile"])} &mdash; {provenance["inferred_at"]}</small></p>
    {_problems_html(data)}

{body}

    </body>
    </html>"""
