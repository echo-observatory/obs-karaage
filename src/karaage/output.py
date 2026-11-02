import json
import pathlib
import re
import datetime

from karaage.attribution import SOURCE_COLORS

verbose = False


def build_speaker_styles(sources) -> str:
    ordered_sources = sorted(
        list(getattr(sources, "sources", sources)),
        key=lambda src: min((start for start, _ in src.segments), default=float("inf"))
    )
    style_rules = [
        f"  .source{rank:02d} {{ background-color: {SOURCE_COLORS[(rank - 1) % len(SOURCE_COLORS)]}; }}"
        for rank in range(1, len(ordered_sources) + 1)
    ]
    return "\n".join(style_rules)


def _slugify(title: str) -> str:
    return re.sub(r'[^\w]+', '-', title.lower()).strip('-')


def snapshot_kind(kind):
    """The snapshot specification's actor kinds: person, organisation, or null for anything else."""
    return kind if kind in ("person", "organisation") else None


def build_results(
    validated_statements: list,
    segments: list,
    sources,
    main_point: str,
    main_point_method: str,
) -> dict:
    from karaage import __version__
    source_list = list(getattr(sources, "sources", sources))
    return {
        "karaage_version": __version__,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_text_main_point": main_point,
        "main_point_method": main_point_method,
        "source_text_length_chars": sum(len(seg["text"]) for seg in segments),
        "source_text_length_segments": len(segments),
        "actors": [
            {"name": src.main, "variations": src.other, "kind": snapshot_kind(src.kind)}
            for src in source_list
        ],
        "statements": [
            {
                "actor": s["actor"],
                "key_statement": s.get("paraphrased_summary") if s.get("summary_paraphrased") else s["summary"],
                "long_statement": s.get("paraphrased_statement") if s.get("rephrased_paraphrased") else s["rephrased_statement"],
                "segment_range": [min(s["segments"]), max(s["segments"])],
                "rouge2_precision": max(
                    s.get("paraphrased_rouge2_rephrased") if s.get("rephrased_paraphrased") else s.get("rouge2_rephrased", 0),
                    s.get("paraphrased_rouge2_summary") if s.get("summary_paraphrased") else s.get("rouge2_summary", 0),
                ),
            }
            for s in validated_statements
            if s["valid"]
        ],
    }


def save_results(dest: pathlib.Path, result: dict) -> None:
    dest = pathlib.Path(dest)
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"Saved {len(result['statements'])} statements to {dest.resolve()}")


def build_html_report(
    speaker_styles: str,
    sections: list,
) -> str:
    """sections: list of {"title": str, "html": str}"""
    from karaage import __version__
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
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
    <title>Monitoring Report</title>
    <link rel="stylesheet" href="https://cdn.simplecss.org/simple.min.css">
    <style>
        .seg0 {{ background-color: #eeeeee; }}
        .seg1 {{ background-color: #d4e0ef; }}
        mark {{ margin-right: 0.10em; font-style: normal; }}
        {speaker_styles}
    </style>
    </head>
    <body>
    <header>
    <nav>
{nav}
    </nav>
    </header>
    <h1>Monitoring Report</h1>
    <p><small>karaage v{__version__} &mdash; {timestamp}</small></p>

{body}

    </body>
    </html>"""


def save_html_report(
    dest: pathlib.Path,
    speaker_styles: str,
    sections: list,
) -> None:
    dest = pathlib.Path(dest)
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(build_html_report(speaker_styles, sections), encoding="utf-8")
    if verbose:
        print(f"Monitoring report written to {dest.resolve()}")
