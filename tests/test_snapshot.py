"""The test dataset's outputs, converted into a snapshot bundle, pass the snapshot validator.

Karaage doesn't assemble bundles (docs/design-1.1.0.md, section 1). This test is a
minimal bundle builder following the mapping of section 6.4, to check that the
outputs fit snapshot specification 1.0. It needs the validator, installed from a
clone of obs-corpus-snapshot next to this repo:

    uv run --group snapshot pytest tests/test_snapshot.py
"""

import csv
import json

import pytest

import karaage
from conftest import ARTICLE_IDS

obs_corpus_snapshot = pytest.importorskip(
    "obs_corpus_snapshot", reason="needs the snapshot validator: uv run --group snapshot pytest")

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
UPDATED_AT = "2026-11-02T00:00:00Z"   # the test dataset records no ingestion time

# (name, type, required, description, constraints), as in Appendix A of the specification.
FIELDS = {
    "source_documents": [
        ("id", "string", True, "Document id, unique in the snapshot.", {}),
        ("text", "string", True, "Full text of the document; reference for statement offsets.", {}),
        ("origin_system", "string", True, "System the document was obtained from: wikinews.", {}),
        ("origin_id", "string", True, "Id of the Wikinews page.", {}),
        ("type", "string", True, "Kind of document, e.g. article.", {}),
        ("updated_at", "datetime", True, "Last time this document was created or modified in the corpus (UTC).", {}),
        ("published_at", "date", False, "Publication date (day precision).", {}),
        ("title", "string", False, "Headline or title.", {}),
        ("url", "string", False, "Public URL of the document.", {}),
        ("language", "string", False, "BCP 47 language tag of the document.", {}),
        ("main_point", "string", False, "The document's main point, as identified by Karaage.", {}),
        ("monitoring_report_path", "string", False,
         "Path of the latest processing monitoring report, relative to the bundle root and starting with monitoring/.", {}),
    ],
    "inferred_actors": [
        ("id", "string", True, "Actor id, unique in the snapshot.", {}),
        ("canonical_name", "string", True, "Canonical name of the actor.", {}),
        ("kind", "string", False, "Kind of actor: person or organisation; null when unknown.",
         {"enum": ["person", "organisation"]}),
    ],
    "inferred_statements": [
        ("id", "string", True, "Statement id, unique in the snapshot.", {}),
        ("source_document_id", "string", True, "Document the statement was inferred from.", {}),
        ("claim_id", "string", False, "Claim the statement belongs to; null when set aside.", {}),
        ("text", "string", True, "The statement, in English.", {}),
        ("inferred_at", "datetime", True, "When the statement was inferred (UTC).", {}),
        ("inference_tool", "string", True, "Name of the statement inference tool.", {}),
        ("inference_version", "string", True, "Version of the statement inference tool.", {}),
        ("source_text_length", "integer", True, "Length of the document text at inference time, in code points.",
         {"minimum": 1}),
        ("char_start", "integer", True, "Start of the source excerpt, 0-based, inclusive, in code points.", {"minimum": 0}),
        ("char_end", "integer", True, "End of the source excerpt, exclusive, in code points.", {"minimum": 1}),
        ("text_short", "string", False, "A shorter version of the statement, in English.", {}),
        ("raw_actor_aliases", "array", False,
         "Names given to the speaker by statement inference, before reconciliation (JSON array).", {}),
        ("inference_profile", "string", True, "Flex: the Karaage profile that inferred the statement.", {}),
    ],
    "inferred_claims": [
        ("id", "string", True, "Claim id, unique in the snapshot.", {}),
        ("actor_id", "string", True, "The actor making the claim.", {}),
        ("text", "string", True, "The claim, restated as one English text.", {}),
        ("inferred_at", "datetime", True, "When the claim was inferred (UTC).", {}),
        ("inference_tool", "string", True, "Name of the claim inference tool.", {}),
        ("inference_version", "string", True, "Version of the claim inference tool.", {}),
    ],
}
KEYS = {
    "inferred_statements": [
        {"fields": ["source_document_id"], "reference": {"resource": "source_documents", "fields": ["id"]}},
        {"fields": ["claim_id"], "reference": {"resource": "inferred_claims", "fields": ["id"]}},
    ],
    "inferred_claims": [
        {"fields": ["actor_id"], "reference": {"resource": "inferred_actors", "fields": ["id"]}},
    ],
}


def schema_field(name, type_, required, description, constraints):
    field = {"name": name, "type": type_}
    if type_ == "datetime":
        field["format"] = TIMESTAMP_FORMAT
    if name == "url":
        field["format"] = "uri"
    constraints = {"required": True, **constraints} if required else constraints
    if constraints:
        field["constraints"] = constraints
    return {**field, "description": description}


def descriptor(profile: str, profile_settings: dict) -> dict:
    resources = []
    for name, fields in FIELDS.items():
        schema = {"fields": [schema_field(*f) for f in fields], "primaryKey": ["id"]}
        if name in KEYS:
            schema["foreignKeys"] = KEYS[name]
        resources.append({"name": name, "type": "table", "path": f"data/{name}.csv", "format": "csv",
                          "mediatype": "text/csv", "encoding": "utf-8", "schema": schema})
    return {
        "$schema": "https://datapackage.org/profiles/2.0/datapackage.json",
        "name": "wikinews-piracy-30",
        "title": "Karaage outputs on the wikinews-piracy-30 test dataset",
        "created": UPDATED_AT,
        "snapshot": {
            "spec_version": "1.0.0",
            "producer": {"name": "obs-karaage tests/test_snapshot.py", "version": karaage.__version__},
            "statement_inference": {
                "description": f"Statements inferred by Karaage {karaage.__version__} with profile {profile}.",
                "profile": profile,
                "profile_settings": profile_settings,
            },
            "claim_inference": {"description": "None: this bundle has no claims, so every statement is set aside."},
        },
        "resources": resources,
    }


def write_table(path, name, rows):
    header = [f[0] for f in FIELDS[name]]
    with open(path / "data" / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows([[row[k] for k in header] for row in rows])


def test_outputs_make_a_valid_snapshot(tmp_path, articles, run_article, resources):
    if resources.client.mode != "replay":
        pytest.skip("converts replayed outputs")
    (tmp_path / "data").mkdir()
    (tmp_path / "monitoring").mkdir()
    documents, statements = [], []
    for article_id in ARTICLE_IDS:
        article, result = articles[article_id], run_article(article_id)
        data, provenance = result.data, result.data["provenance"]
        report_path = f"monitoring/{data['document_id']}.html"
        (tmp_path / report_path).write_text(result.html_report, encoding="utf-8")
        documents.append({
            "id": data["document_id"], "text": article["text"], "origin_system": "wikinews",
            "origin_id": str(article["page_id"]), "type": "article", "updated_at": UPDATED_AT,
            "published_at": article["date"], "title": article["title"], "url": article["url"], "language": "en",
            "main_point": data["main_point"], "monitoring_report_path": report_path,
        })
        statements += [{
            "id": s["id"], "source_document_id": data["document_id"], "claim_id": None,
            "text": s["text"], "inferred_at": provenance["inferred_at"],
            "inference_tool": provenance["tool"], "inference_version": provenance["version"],
            "source_text_length": data["text_length"], "char_start": s["char_start"], "char_end": s["char_end"],
            "text_short": s["text_short"], "raw_actor_aliases": json.dumps(s["actor_aliases"], ensure_ascii=False),
            "inference_profile": provenance["profile"],
        } for s in data["statements"] if s["valid"]]   # the bundle builder drops rejected statements

    write_table(tmp_path, "source_documents", documents)
    write_table(tmp_path, "inferred_actors", [])       # reconciled actors come from claim inference
    write_table(tmp_path, "inferred_statements", statements)
    write_table(tmp_path, "inferred_claims", [])
    (tmp_path / "datapackage.json").write_text(
        json.dumps(descriptor(provenance["profile"], provenance["profile_settings"]), indent=2), encoding="utf-8")

    report = obs_corpus_snapshot.validate(tmp_path, strict=True, allow=["statement-set-aside"])
    assert report.valid, report.format_text()
    assert report.stats["rows"]["inferred_statements"] == len(statements) > 0
