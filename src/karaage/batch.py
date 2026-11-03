"""Run Karaage on a JSONL file of documents (docs/design-1.1.0.md, sections 5, 6.1 and 6.2)."""

import hashlib
import json
import time
import urllib.parse
from collections import Counter
from pathlib import Path

import karaage

FIELDS = ("id", "text", "title", "published_at")
MAX_NAME_LENGTH = 200   # characters of an encoded file name, under the usual limit of 255 bytes


class InputError(ValueError):
    """The input cannot be run as it is."""


def output_name(document_id: str) -> str:
    """The file name, without extension, of a document's output.

    ASCII letters and digits, `-`, `_`, `.` and `~` are kept. Everything else, `%`
    included, is percent-encoded, so that two ids never give the same name. A leading
    `.` is encoded too, so that names are neither hidden nor `.` or `..`.
    """
    name = urllib.parse.quote(document_id, safe="")
    return "%2E" + name[1:] if name.startswith(".") else name


def parse_fields(specs: list[str]) -> dict:
    """Field mapping from `FIELD=NAME` strings: Karaage field FIELD is read from input field NAME."""
    mapping = {}
    for spec in specs:
        field, sep, name = spec.partition("=")
        if not sep or not name or field not in FIELDS:
            raise InputError(f"--field {spec}: expected FIELD=NAME, with FIELD one of {', '.join(FIELDS)}.")
        mapping[field] = name
    return mapping


def read_documents(path, fields: dict | None = None) -> list[dict]:
    """Read a JSONL file, one document per line.

    `fields` maps Karaage fields to the input's field names, e.g. {"published_at": "date"}.
    Every line is checked before anything runs; InputError lists all the problems.
    """
    fields = {**{f: f for f in FIELDS}, **(fields or {})}
    documents, problems, lines_by_name = [], [], {}
    with open(path, encoding="utf-8-sig") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"line {n}: not valid JSON ({e.msg}).")
                continue
            if not isinstance(record, dict):
                problems.append(f"line {n}: not a JSON object.")
                continue
            document = {f: record[name] for f, name in fields.items() if record.get(name) is not None}
            try:
                document_id = karaage.read_document(document)[0]
            except (TypeError, ValueError) as e:
                problems.append(f"line {n}: {e}")
                continue
            name = output_name(document_id)
            key = name.lower()   # file systems may ignore case
            if len(name) > MAX_NAME_LENGTH:
                problems.append(f"line {n}: document id {document_id!r} is too long for a file name.")
            elif key in lines_by_name:
                problems.append(f"line {n}: document id {document_id!r} gives the same file name as the id "
                                f"on line {lines_by_name[key]}; ids must be unique, ignoring case.")
            else:
                lines_by_name[key] = n
            documents.append(document)
    if problems:
        raise InputError(f"{path}: {len(problems)} problem{'s' if len(problems) > 1 else ''}:\n  " + "\n  ".join(problems))
    return documents


def previous_output(path: Path) -> dict | None:
    """The output a previous run wrote, or None when there is none or it can't be read."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_done(previous: dict | None, text: str) -> bool:
    """A document is done when its output has the status ok and was made from the same text."""
    return (previous is not None and previous.get("status") == "ok"
            and previous.get("text_sha256") == hashlib.sha256(text.encode("utf-8")).hexdigest())


def run_documents(documents: list[dict], output_dir, resources, *, report: bool = False) -> Counter:
    """Run each document and write `<name>.json`, plus `<name>.html` with `report`.

    Documents whose output is already ok are skipped, so an interrupted run resumes where
    it stopped, and documents that failed or were partial are retried. Returns the number
    of documents per status, with `skipped` for those already done.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses, seconds, chars, other_runs = Counter(), 0.0, 0, 0
    current = (resources.client.profile.label, karaage.__version__)
    for i, document in enumerate(documents, 1):
        document_id = str(document["id"])
        name = output_name(document_id)
        previous = previous_output(output_dir / f"{name}.json")
        if is_done(previous, document["text"]):
            statuses["skipped"] += 1
            provenance = previous.get("provenance") or {}
            other_runs += (provenance.get("profile"), provenance.get("version")) != current
            continue
        print(f"[{i}/{len(documents)}] Document {document_id}")
        t0 = time.perf_counter()
        result = karaage.run(document, resources)
        seconds += time.perf_counter() - t0
        chars += len(document["text"])
        karaage.save(result, data_path=output_dir / f"{name}.json",
                     html_path=output_dir / f"{name}.html" if report else None)
        statuses[result.data["status"]] += 1
    print_summary(statuses, seconds, chars, other_runs)
    return statuses


def print_summary(statuses: Counter, seconds: float, chars: int, other_runs: int) -> None:
    counts = ", ".join(f"{statuses[s]} {s}" for s in ("ok", "partial", "failed", "skipped") if statuses[s])
    print(f"{sum(statuses.values())} documents: {counts or 'none'}.")
    if chars:
        print(f"{seconds:.0f} s in all, {seconds / (chars / 1000):.1f} s per 1,000 characters.")
    if statuses["skipped"]:
        print("Skipped documents already had an ok output, made from the same text.")
    if other_runs:
        print(f"⚠️  {other_runs} of them were made by another profile or Karaage version. "
              f"Use another output directory to run them again.")
    if statuses["failed"] or statuses["partial"]:
        print("Run the same command again to retry the failed and partial documents.")
