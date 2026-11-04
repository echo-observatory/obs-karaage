"""The output of one document (docs/design-1.1.0.md, section 6), built from invented step results."""

import jsonschema

from karaage.attribution import Actor
from karaage.output import build_output, output_schema, text_warnings

# Invented texts on the same subject, written for these tests.
ENGLISH = ("The council decided today to build a new school in the north of the town. The mayor said that the "
           "project should be finished in three years and that it would cost about twenty million euros. Many "
           "parents are pleased with the plan, but some neighbours are worried about the traffic on the road.")
NOT_ENGLISH = {
    "da": ("Kommunen har i dag besluttet at bygge en ny skole i den nordlige del af byen. Borgmesteren siger, at "
           "projektet skal være færdigt om tre år, og at det vil koste omkring to hundrede millioner kroner. Flere "
           "forældre er glade for planen, men nogle naboer er bekymrede for trafikken på vejen."),
    "de": ("Die Gemeinde hat heute beschlossen, im Norden der Stadt eine neue Schule zu bauen. Der Bürgermeister "
           "sagt, dass das Projekt in drei Jahren fertig sein soll und rund zwanzig Millionen Euro kosten wird. "
           "Viele Eltern freuen sich über den Plan, aber einige Nachbarn sorgen sich um den Verkehr auf der Straße."),
    "es": ("El ayuntamiento anunció ayer la construcción de una nueva escuela en el barrio norte de la ciudad. "
           "Según el alcalde, las obras durarán tres años y costarán unos veinte millones de euros. Muchos padres "
           "están contentos con el proyecto, pero algunos vecinos temen por el tráfico en la calle."),
    "fr": ("La mairie a annoncé hier la construction d'une nouvelle école dans le quartier nord de la ville. Selon "
           "le maire, les travaux devraient durer trois ans et coûter environ vingt millions d'euros. Plusieurs "
           "parents se réjouissent du projet, mais certains voisins s'inquiètent de la circulation dans la rue."),
}

PROVENANCE = {
    "tool": "karaage", "version": "1.1.0", "commit": None, "profile": "qwen3.5-9b",
    "profile_settings": {"name": "qwen3.5-9b", "description": "", "context_tokens": 8000, "temperature": 0.0,
                         "embedding_model": "e", "backends": {}, "seed": None, "custom": False},
    "backend": "lmstudio", "model": "qwen/qwen3.5-9b", "model_reported": "qwen/qwen3.5-9b", "model_build": None,
    "paraphrase": True,
    "inferred_at": "2026-11-03T10:11:00Z", "step_seconds": {"segmentation": 0.4},
}


def codes(text):
    return [w["code"] for w in text_warnings(text)]


def test_warnings():
    assert codes(ENGLISH) == []
    for language, text in NOT_ENGLISH.items():
        assert codes(text) == ["text-not-english"], language
    assert codes("Bonjour à tous.") == []   # too short to tell
    assert codes(ENGLISH.replace("school", "école")) == ["text-not-nfc"]
    assert codes(ENGLISH.replace(". ", ".\r\n")) == ["text-has-carriage-return"]


# An invented document: two segments, one excerpt, Jane Doe and the text's own voice.
TEXT = "The mayor spoke. Jane Doe said the school would open in May."
SEGMENTS = [{"id": 0, "text": "The mayor spoke. ", "start": 0, "end": 17},
            {"id": 1, "text": "Jane Doe said the school would open in May.", "start": 17, "end": 60}]
ACTORS = [Actor(canonical_name="Jane Doe", aliases=["Doe", "the mayor"], kind="person", segments=[[1, 1]])]


def statement(n, actor="Jane Doe", **fields):
    s = {"id": n, "actor": actor, "statement": "S", "excerpt": TEXT, "segments": [0, 1],
         "rephrased_statement": "The school opens in May.", "summary": "School opens in May.",
         "rouge2_rephrased": 0.1, "rouge2_summary": 0.2, "paraphrased": False,
         "in_text": "yes", "actor_correct": "yes", "valid": True, "invalid_reason": None}
    return {**s, **fields}


def output(**overrides):
    args = dict(failure=None, excerpt_errors=[], segments=SEGMENTS, attribution=ACTORS, statements=[],
                main_point="A school will open.", main_point_method="single", provenance=PROVENANCE)
    data = build_output("doc-7", TEXT, **{**args, **overrides})
    jsonschema.Draft202012Validator(output_schema()).validate(data)
    return data


def test_ok_output():
    data = output(statements=[
        statement(0),
        statement(1, actor="base_author", in_text="no", valid=False, invalid_reason="not found in source text"),
    ])
    assert data["status"] == "ok" and data["errors"] == [] and data["warnings"] == []
    assert data["text_length"] == 60 and len(data["text_sha256"]) == 64
    assert data["actors"] == [{"canonical_name": "Jane Doe", "aliases": ["Doe", "the mayor"], "kind": "person"}]
    valid, rejected = data["statements"]
    assert valid == {
        "id": "doc-7-s1", "actor": "Jane Doe", "actor_aliases": ["Jane Doe", "Doe", "the mayor"],
        "text": "The school opens in May.", "text_short": "School opens in May.",
        "char_start": 0, "char_end": 60, "valid": True, "rejection_reason": None,
        "diagnostics": {"rouge2_precision": 0.2, "paraphrased": False, "in_text": "yes", "actor_correct": "yes"},
    }
    assert (rejected["id"], rejected["actor_aliases"], rejected["valid"], rejected["rejection_reason"]) == (
        "doc-7-s2", ["base_author"], False, "not found in source text")


def test_partial_output_lists_what_failed():
    rephrasing_failed = {k: v for k, v in statement(3).items()
                         if k not in ("rephrased_statement", "summary", "rouge2_rephrased", "rouge2_summary")}
    data = output(
        excerpt_errors=[{"excerpt_index": 0, "error": "RuntimeError: backend down", "answer": None}],
        statements=[
            statement(2, validation_error="RuntimeError: timeout", in_text=None, actor_correct=None,
                      valid=False, invalid_reason="validation error"),
            {**rephrasing_failed, "rephrasing_error": "ValueError: cut off", "in_text": None, "actor_correct": None,
             "valid": False, "invalid_reason": "rephrasing error"},
        ],
    )
    assert data["status"] == "partial"
    assert data["errors"] == [
        {"step": "extraction", "excerpt": 1, "statement": None, "message": "RuntimeError: backend down", "answer": None},
        {"step": "rephrasing", "excerpt": None, "statement": "doc-7-s4", "message": "ValueError: cut off", "answer": None},
        {"step": "validation", "excerpt": None, "statement": "doc-7-s3", "message": "RuntimeError: timeout", "answer": None},
    ]
    s4 = data["statements"][1]
    assert (s4["text"], s4["text_short"], s4["diagnostics"]["rouge2_precision"]) == (None, None, None)


def test_errors_keep_the_answers_they_were_raised_on():
    answer = {"content": '{"a"', "finish_reason": "length", "usage": {"completion_tokens": 8000}, "provider": "InventedCloud"}
    rephrasing_failed = {k: v for k, v in statement(3).items()
                         if k not in ("rephrased_statement", "summary", "rouge2_rephrased", "rouge2_summary")}
    data = output(
        excerpt_errors=[{"excerpt_index": 0, "error": "ValueError: cut off", "answer": answer}],
        statements=[
            statement(2, validation_error="ValueError: cut off", validation_answer=answer, in_text=None, actor_correct=None,
                      valid=False, invalid_reason="validation error"),
            {**rephrasing_failed, "rephrasing_error": "ValueError: cut off", "rephrasing_answer": answer,
             "in_text": None, "actor_correct": None, "valid": False, "invalid_reason": "rephrasing error"},
        ],
    )
    jsonschema.Draft202012Validator(output_schema()).validate(data)
    assert [e["answer"] for e in data["errors"]] == [answer] * 3
    assert all("answer" not in s and "rephrasing_answer" not in s and "validation_answer" not in s
               for s in data["statements"])


def test_failed_output_has_no_results():
    failure = {"step": "rephrasing", "excerpt": None, "statement": None, "message": "RuntimeError: backend down",
               "answer": None}
    data = output(failure=failure, statements=None)
    assert data["status"] == "failed" and data["errors"] == [failure]
    assert (data["main_point"], data["main_point_method"], data["actors"], data["statements"]) == (None, None, [], [])


def test_schema_rejects_inconsistent_outputs():
    validator = jsonschema.Draft202012Validator(output_schema())
    data = output(statements=[statement(0)])
    assert not validator.is_valid({**data, "status": "partial"})            # partial without errors
    assert not validator.is_valid({**data, "status": "failed"})             # failed without errors
    assert not validator.is_valid({**data, "statements": [{**data["statements"][0], "rejection_reason": "too similar to source"}]})
    assert not validator.is_valid({**data, "extra": 1})
    paraphrase_off = {**data["provenance"], "paraphrase": False}
    rejected = {**data["statements"][0], "valid": False, "rejection_reason": "too similar to source"}
    assert not validator.is_valid({**data, "provenance": paraphrase_off, "statements": [rejected]})
    assert validator.is_valid({**data, "provenance": paraphrase_off})

    failed = output(failure={"step": "attribution", "excerpt": None, "statement": None, "message": "M", "answer": None},
                    statements=None)
    assert validator.is_valid(failed)
    def with_error(**fields):
        return {**failed, "errors": [{"step": "attribution", "excerpt": None, "statement": None, "message": "M", **fields}]}
    complete = {"content": "{", "finish_reason": "length", "usage": None, "provider": None}
    assert validator.is_valid(with_error(answer=complete))
    assert not validator.is_valid(with_error())                                  # an error always says
    assert not validator.is_valid(with_error(answer={"content": "{"}))           # incomplete
    assert not validator.is_valid(with_error(answer={**complete, "extra": 1}))
