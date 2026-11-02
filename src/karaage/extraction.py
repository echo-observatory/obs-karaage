import time
import json
import re
import html as html_lib
from html import unescape
from tqdm.auto import tqdm
from pydantic import BaseModel, Field
from karaage._context import llm_create

verbose = False

# Step-level configuration
main_point_length = 300
RESERVED_TOKENS_MAIN_POINT = main_point_length // 3 + 150

SYSTEM_PROMPT_MAIN_POINT = f'''You are a journalist-analyst analyzing a text to extract its central argument.

Your task:
Read the text and identify its central argument. The central argument expresses the core situation and what is at stake, from the journalist's perspective.

Rules:
- Make a single concise statement of under {main_point_length} characters.
- Use one single sentence if possible.
- Use well-formed, everyday language.
- Your answer must be derived from the article text, not from these instructions.
'''


class MainPoint(BaseModel):
    main_point: str = Field(description=f"The central argument of the article. Under {main_point_length} characters.",
                            #max_length=main_point_length # Note: we intentionally don't enforce the length here as it's not a priority.
                            )


# Step-level configuration
statement_target_length = 200
statement_max_length = 400

# Strip stray HTML tags and entities that may survive web scraping.
def clean_text(text):
    return unescape(re.sub(r'<[^>]+>', '', text))


SYSTEM_PROMPT_PASSAGE = f'''You are a journalist-analyst identifying "actor-statements" in a given passage of a text, as stated by a given actor.

A statement could be:
- A claim that the actor makes: a knowledge claim ("the reason for this is…"), an experience claim ("this feels like…"), or a more factual claim ("this has happened").
- A position that the actor takes: a political stance, a set of values, a role, a responsibility, a commitment, a personal experience that leads the actor to express a particular opinion or adopt a particular 'take' on a situation.
- A course of action that the actor proposes or opposes: a set of events that are happening at the behest of the actor, actions that the actor prompts others to take, or something that the actor is preventing others from doing.
- Or it could be any combination of the above.

Your task:
- Extract key statements per actor from the specified passage (max {statement_target_length} characters).

Rules:
- Focus only on the <focused_passage> and the specified actors.
- <context_before> and <context_after> are only provided as context — do not extract statements from them.
- If the actor doesn\'t make any clear statements in the passage, return an empty list for that actor.
- Account precisely for the statement(s) the actor makes, by including factual details if relevant to their argument.
- Assume the focused passage is from the specified actors.
- The actor "base_author" is a placeholder for the author of the text. Account for their voice, as they are a primary voice in the text. For base_author, extract framing choices and factual claims used to establish the article\'s narrative: the facts the author selects, how they characterise a situation, the causal links they assert. Scene-setting and background are valid statements when the author is making an editorial choice about what matters and why.
- Assume the reader is informed that your key statement is from the specified actor, and don\'t mention them in the statement itself unless it\'s relevant to the content of the statement.
- Use well-formed, everyday language.
'''


def extract_main_point(segments: list, model: str) -> str:
    article_text = "\n".join(f"[{seg['id']}] {seg['text']}" for seg in segments)
    t0 = time.perf_counter()
    response = llm_create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_MAIN_POINT},
            {"role": "user", "content": article_text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "MainPoint",
                "schema": MainPoint.model_json_schema(),
                "strict": True,
            },
        },
        max_tokens=RESERVED_TOKENS_MAIN_POINT,
        temperature=0,
    )
    elapsed = time.perf_counter() - t0
    msg = response.choices[0].message
    # LM Studio routes output to reasoning_content for some Qwen3 configurations;
    # fall back to it if content is empty.
    content = msg.content or getattr(msg, "reasoning_content", None) or ""
    if not content:
        raise ValueError(f"Model returned no content. Full message: {msg!r}")
    result = MainPoint.model_validate_json(content)
    if verbose:
        print(result.main_point)
        print(f"  ({len(result.main_point)} chars, {elapsed:.1f}s)")
    return result.main_point


def analyze_passage(passages, passage_index, main_point, model: str,
                    actor_contribution_threshold=40.0,
                    min_actor_chars=150,
                    offset_before=1, offset_after=1,
                    print_prompt=False):
    passage = passages[passage_index]

    # First gate: percentage share. Actors below this threshold are so marginal
    # in the passage (e.g. a single attribution tag) that asking about them
    # produces noise rather than signal.
    retained = [spk for spk, pct in passage['speaker_pcts'].items()
                if pct >= actor_contribution_threshold]

    # Second gate: absolute character count. An actor can clear the percentage
    # threshold yet still contribute very little text in absolute terms — this
    # happens in short passages where even 40% is only a sentence fragment or a
    # connective phrase. Below min_actor_chars there is simply not enough
    # material for a meaningful statement, so we skip those actors entirely
    # rather than sending thin content to the model.
    retained = [spk for spk in retained
                if passage['speaker_chars'][spk] >= min_actor_chars]

    # If no actors survive both gates, there is nothing to extract.
    if not retained:
        return {}, 0.0

    key_stmt_schema = {"type": "string", "maxLength": statement_max_length}

    # maxItems is a fixed cap rather than a character-derived formula.
    # The character-count heuristic (chars / 400) was false precision: it
    # conflated passage length with information density. In practice, even
    # lengthy turns rarely yield more than 3–4 distinct, non-redundant
    # statements. A fixed cap of 4 is simpler, more predictable, and lets the
    # model decide how many statements are actually warranted rather than
    # imposing an arbitrary ceiling based on word count.
    schema = {
        "type": "object",
        "properties": {
            spk: {
                "type": "array",
                "items": key_stmt_schema,
                "minItems": 0,
                "maxItems": 4,
            }
            for spk in retained
        },
        "required": retained,
        "additionalProperties": False,
    }

    actual_before = min(offset_before, passage_index)
    actual_after  = min(offset_after, len(passages) - 1 - passage_index)

    before_text = clean_text("".join(p['text'] for p in passages[passage_index - actual_before : passage_index]))
    focus_text  = clean_text(passage['text'])
    after_text  = clean_text("".join(p['text'] for p in passages[passage_index + 1 : passage_index + 1 + actual_after]))

    speakers_str = "".join(f"\n- {spk}" for spk in retained)

    prompt_parts = [
        f"Actors to extract statements for:{speakers_str}",
        f"<central_argument>\n{main_point}\n</central_argument>",
        "The excerpt below is an excerpt of a longer text. The central argument above captures its core meaning. Restrain your analysis to the focused passage, but use the surrounding passages for context.",
    ]
    if before_text:
        prompt_parts.append(f"<context_before>\n{before_text}\n</context_before>")
    prompt_parts.append(f"<focused_passage>\n{focus_text}\n</focused_passage>")
    if after_text:
        prompt_parts.append(f"<context_after>\n{after_text}\n</context_after>")

    prompt = "\n\n".join(prompt_parts)

    if print_prompt:
        print("# SYSTEM PROMPT #\n")
        print(SYSTEM_PROMPT_PASSAGE, "\n")
        print("# USER PROMPT #\n")
        print(prompt, "\n\n\n")

    t0 = time.perf_counter()
    response = llm_create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_PASSAGE},
            {"role": "user",   "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "PassageAnalysis", "schema": schema, "strict": True},
        },
        max_tokens=2000,
        temperature=0,
    )
    elapsed = time.perf_counter() - t0

    msg = response.choices[0].message
    # LM Studio routes output to reasoning_content for some Qwen3 configurations;
    # fall back to it if content is empty.
    content = msg.content or getattr(msg, "reasoning_content", None) or ""
    if not content:
        raise ValueError(f"Model returned no content. Full message: {msg!r}")

    return json.loads(content), elapsed


def extract_statements(passages: list, main_point: str, model: str) -> list[dict]:
    all_statements = []
    errors = []
    for i in tqdm(range(len(passages)), desc="Extracting statements"):
        try:
            result, elapsed = analyze_passage(passages, i, main_point, model)
            all_statements.append({"passage_index": i, "statements": result})
        except Exception as e:
            errors.append({"passage_index": i, "error": str(e)})
            print(f"  Error on passage {i}: {e}")
    total = sum(len(stmts) for entry in all_statements for stmts in entry["statements"].values())
    print(f"-> {total} statement{'s' if total != 1 else ''} extracted from {len(all_statements)} passage{'s' if len(all_statements) != 1 else ''}, {len(errors)} passage{'s' if len(errors) != 1 else ''} failed.")
    if errors:
        print("Failed passages:", [e["passage_index"] for e in errors])
    return all_statements


def key_statements_monitoring_html(all_statements: list, passages: list, main_point: str) -> str:
    """Return statements_content for the monitoring report."""
    method_description = (
        "Each passage is sent to an LLM together with one passage of context on each side "
        "and the article's central argument as an anchor. "
        "The model extracts key statements per actor "
        "(up to 4 statements per actor, each capped at 200 characters). "
        "<ul>"
        "<li><b>Actor filtering (two gates).</b> Only actors who clear both a "
        "<i>share gate</i> (≥40% of passage characters) and an "
        "<i>absolute gate</i> (≥150 characters of text) are queried. "
        "Actors below either threshold are too marginal to yield non-trivial statements.</li>"
        "<li><b>Statement types.</b> A statement can be a "
        "<i>claim</i> (knowledge, experience, or factual), "
        "a <i>position</i> (stance, values, commitment, or framing), "
        "or a <i>course of action</i> (something the actor proposes, opposes, or enacts). "
        "Any combination of the three is valid. This is enforced in the prompt.</li>"
        "<li><b>Base author treatment.</b> The author's own voice (<i>base_author</i>) is "
        "treated as a first-class actor. For this speaker the model targets framing choices "
        "and factual claims that establish the article's narrative: which facts are selected, "
        "how situations are characterised, and what causal links are asserted.</li>"
        "</ul>"
    )

    content_parts = []
    for entry in all_statements:
        idx = entry["passage_index"]
        passage = passages[idx]
        passage_html = f"<details><summary>Passage {idx + 1}</summary><p>{html_lib.escape(passage['text'])}</p></details>"
        statements_html = ""
        for actor, stmts in entry["statements"].items():
            for stmt in stmts:
                statements_html += html_lib.escape(stmt)
                statements_html += f"<br>— <em>{html_lib.escape(actor)}</em><br><br>"
        content_parts.append(f"<section>{passage_html}{statements_html}</section>")

    return (
        f"\n<article><h3>Method</h3>{method_description}</article>\n"
        f'<p><mark><strong>Central argument: </strong>{html_lib.escape(main_point)}</mark></p>\n'
        + "".join(content_parts)
        + "\n"
    )

def step_4_key_statements(segments, passages, MODEL):
    main_point = extract_main_point(segments, MODEL)
    all_statements = extract_statements(passages, main_point, MODEL)
    html_statements_content = key_statements_monitoring_html(all_statements, passages, main_point)
    return all_statements, main_point, html_statements_content