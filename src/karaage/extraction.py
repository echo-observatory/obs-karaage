import time
import json
import math
import re
import html as html_lib
from html import unescape
from tqdm.auto import tqdm
from pydantic import BaseModel, Field
from karaage import llm
from karaage.attribution import estimate_tokens

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


# A text longer than the context is summarised part by part, and the main point is
# taken from the summaries (docs/design-1.1.0.md, section 3.5).
part_summary_length = 1000
# Asked for a summary without more guidance, the model can enumerate points until the token
# limit (article 30); asked for one paragraph of prose, it stays near the length asked.
# The budget leaves room for twice that length.
RESERVED_TOKENS_PART_SUMMARY = 1_000
RESERVED_TOKENS_HEADER = 100   # title and publication date, when given

SYSTEM_PROMPT_PART_SUMMARY = f'''You are a journalist-analyst summarising one part of a longer text.

Your task:
Summarise this part of the text: the situation it describes, the claims and arguments made in it, who makes them, and what is at stake.

Rules:
- Write one paragraph of prose, without lists, of under {part_summary_length} characters.
- Keep the names of the people and organisations involved.
- Use well-formed, everyday language.
- Your answer must be derived from the text, not from these instructions.
'''


class PartSummary(BaseModel):
    summary: str = Field(description=f"Summary of this part of the text. Under {part_summary_length} characters.")


# Step-level configuration
statement_target_length = 200
statement_max_length = 400

# Strip stray HTML tags and entities that may survive web scraping.
def clean_text(text):
    return unescape(re.sub(r'<[^>]+>', '', text))


SYSTEM_PROMPT_EXCERPT = f'''You are a journalist-analyst identifying "actor-statements" in a given excerpt of a text, as stated by a given actor.

A statement could be:
- A claim that the actor makes: a knowledge claim ("the reason for this is…"), an experience claim ("this feels like…"), or a more factual claim ("this has happened").
- A position that the actor takes: a political stance, a set of values, a role, a responsibility, a commitment, a personal experience that leads the actor to express a particular opinion or adopt a particular 'take' on a situation.
- A course of action that the actor proposes or opposes: a set of events that are happening at the behest of the actor, actions that the actor prompts others to take, or something that the actor is preventing others from doing.
- Or it could be any combination of the above.

Your task:
- Extract key statements per actor from the specified excerpt (max {statement_target_length} characters).

Rules:
- Focus only on the <focused_excerpt> and the specified actors.
- <context_before> and <context_after> are only provided as context — do not extract statements from them.
- If the actor doesn\'t make any clear statements in the excerpt, return an empty list for that actor.
- Account precisely for the statement(s) the actor makes, by including factual details if relevant to their argument.
- Assume the focused excerpt is from the specified actors.
- The actor "base_author" is a placeholder for the author of the text. Account for their voice, as they are a primary voice in the text. For base_author, extract framing choices and factual claims used to establish the article\'s narrative: the facts the author selects, how they characterise a situation, the causal links they assert. Scene-setting and background are valid statements when the author is making an editorial choice about what matters and why.
- Assume the reader is informed that your key statement is from the specified actor, and don\'t mention them in the statement itself unless it\'s relevant to the content of the statement.
- Use well-formed, everyday language.
'''


def format_segments(segments: list) -> str:
    return "\n".join(f"[{seg['id']}] {seg['text']}" for seg in segments)


def with_header(text: str, title=None, published_at=None) -> str:
    """Put the title and publication date, when known, before the text."""
    header = "".join(f"{label}: {value}\n" for label, value in (("Title", title), ("Published", published_at)) if value)
    return f"{header}\n{text}" if header else text


def split_into_parts(segments: list, excerpts: list, budget: int) -> list[str]:
    """Group consecutive excerpts into parts of similar size, each within `budget` tokens.

    Parts are cut at the excerpt boundaries closest to evenly spaced points of the
    text; when a part would exceed the budget, the text is cut into one more part.
    """
    texts = [format_segments([segments[i] for i in p["segments"]]) for p in excerpts]
    sizes = [estimate_tokens(t) for t in texts]
    n_parts = math.ceil(sum(sizes) / budget)
    while True:
        target = sum(sizes) / n_parts
        parts, current, size, before = [], [], 0, 0
        for text, n in zip(texts, sizes):
            middle = before + n / 2
            if current and (size + n > budget or middle > (len(parts) + 1) * target):
                parts.append("\n".join(current))
                current, size = [], 0
            current.append(text)
            size += n
            before += n
        parts.append("\n".join(current))
        if len(parts) <= n_parts:
            return parts
        n_parts += 1


def extract_main_point(segments: list, excerpts: list, title=None, published_at=None) -> tuple[str, str, list[str]]:
    """Return the main point, how it was obtained (`single` or `chunked`), and the part summaries.

    A text that fits the context is read in one call. A longer one is split at excerpt
    boundaries into parts that fit; each part is summarised, and the main point is
    taken from the summaries.
    """
    t0 = time.perf_counter()
    article_text = with_header(format_segments(segments), title, published_at)
    if estimate_tokens(SYSTEM_PROMPT_MAIN_POINT + article_text) + RESERVED_TOKENS_MAIN_POINT <= llm.context_tokens():
        method, summaries, prompt = "single", [], article_text
    else:
        method = "chunked"
        budget = (llm.context_tokens() - estimate_tokens(SYSTEM_PROMPT_PART_SUMMARY)
                  - RESERVED_TOKENS_PART_SUMMARY - RESERVED_TOKENS_HEADER)
        summaries = [
            llm.complete_model(SYSTEM_PROMPT_PART_SUMMARY, with_header(part, title, published_at), PartSummary,
                               max_tokens=RESERVED_TOKENS_PART_SUMMARY).summary
            for part in split_into_parts(segments, excerpts, budget)
        ]
        prompt = with_header(
            "The text is too long to be read at once. Below are summaries of its consecutive parts, in order.\n\n"
            + "\n\n".join(f"[Part {i}] {s}" for i, s in enumerate(summaries, 1)),
            title, published_at,
        )
    result = llm.complete_model(SYSTEM_PROMPT_MAIN_POINT, prompt, MainPoint, max_tokens=RESERVED_TOKENS_MAIN_POINT)
    elapsed = time.perf_counter() - t0
    if verbose:
        print(result.main_point)
        print(f"  ({len(result.main_point)} chars, {method}, {elapsed:.1f}s)")
    return result.main_point, method, summaries


def analyze_excerpt(excerpts, excerpt_index, main_point,
                    actor_contribution_threshold=40.0,
                    min_actor_chars=150,
                    offset_before=1, offset_after=1,
                    print_prompt=False):
    excerpt = excerpts[excerpt_index]

    # First gate: percentage share. Actors below this threshold are so marginal
    # in the excerpt (e.g. a single attribution tag) that asking about them
    # produces noise rather than signal.
    retained = [actor for actor, pct in excerpt['actor_pcts'].items()
                if pct >= actor_contribution_threshold]

    # Second gate: absolute character count. An actor can clear the percentage
    # threshold yet still contribute very little text in absolute terms — this
    # happens in short excerpts where even 40% is only a sentence fragment or a
    # connective phrase. Below min_actor_chars there is simply not enough
    # material for a meaningful statement, so we skip those actors entirely
    # rather than sending thin content to the model.
    retained = [actor for actor in retained
                if excerpt['actor_chars'][actor] >= min_actor_chars]

    # If no actors survive both gates, there is nothing to extract.
    if not retained:
        return {}, 0.0

    key_stmt_schema = {"type": "string", "maxLength": statement_max_length}

    # maxItems is a fixed cap rather than a character-derived formula.
    # The character-count heuristic (chars / 400) was false precision: it
    # conflated excerpt length with information density. In practice, even
    # lengthy turns rarely yield more than 3–4 distinct, non-redundant
    # statements. A fixed cap of 4 is simpler, more predictable, and lets the
    # model decide how many statements are actually warranted rather than
    # imposing an arbitrary ceiling based on word count.
    schema = {
        "type": "object",
        "properties": {
            actor: {
                "type": "array",
                "items": key_stmt_schema,
                "minItems": 0,
                "maxItems": 4,
            }
            for actor in retained
        },
        "required": retained,
        "additionalProperties": False,
    }

    actual_before = min(offset_before, excerpt_index)
    actual_after  = min(offset_after, len(excerpts) - 1 - excerpt_index)

    before_text = clean_text("".join(p['text'] for p in excerpts[excerpt_index - actual_before : excerpt_index]))
    focus_text  = clean_text(excerpt['text'])
    after_text  = clean_text("".join(p['text'] for p in excerpts[excerpt_index + 1 : excerpt_index + 1 + actual_after]))

    actors_str = "".join(f"\n- {actor}" for actor in retained)

    prompt_parts = [
        f"Actors to extract statements for:{actors_str}",
        f"<central_argument>\n{main_point}\n</central_argument>",
        "The excerpt below is an excerpt of a longer text. The central argument above captures its core meaning. Restrain your analysis to the focused excerpt, but use the surrounding excerpts for context.",
    ]
    if before_text:
        prompt_parts.append(f"<context_before>\n{before_text}\n</context_before>")
    prompt_parts.append(f"<focused_excerpt>\n{focus_text}\n</focused_excerpt>")
    if after_text:
        prompt_parts.append(f"<context_after>\n{after_text}\n</context_after>")

    prompt = "\n\n".join(prompt_parts)

    if print_prompt:
        print("# SYSTEM PROMPT #\n")
        print(SYSTEM_PROMPT_EXCERPT, "\n")
        print("# USER PROMPT #\n")
        print(prompt, "\n\n\n")

    t0 = time.perf_counter()
    content = llm.complete_json(SYSTEM_PROMPT_EXCERPT, prompt, schema, name="ExcerptAnalysis", max_tokens=2000)
    elapsed = time.perf_counter() - t0

    return json.loads(content), elapsed


def extract_statements(excerpts: list, main_point: str) -> tuple[list[dict], list[dict]]:
    """Extract statements excerpt by excerpt. An excerpt that fails is skipped and
    returned in the list of errors, so that the document can be marked as partial."""
    all_statements = []
    errors = []
    for i in tqdm(range(len(excerpts)), desc="Extracting statements"):
        try:
            result, elapsed = analyze_excerpt(excerpts, i, main_point)
            all_statements.append({"excerpt_index": i, "statements": result})
        except Exception as e:
            errors.append({"excerpt_index": i, "error": f"{type(e).__name__}: {e}"})
            print(f"  Error on excerpt {i}: {e}")
    total = sum(len(stmts) for entry in all_statements for stmts in entry["statements"].values())
    print(f"-> {total} statement{'s' if total != 1 else ''} extracted from {len(all_statements)} excerpt{'s' if len(all_statements) != 1 else ''}, {len(errors)} excerpt{'s' if len(errors) != 1 else ''} failed.")
    if errors:
        print("Failed excerpts:", [e["excerpt_index"] for e in errors])
    return all_statements, errors


def key_statements_monitoring_html(all_statements: list, excerpts: list, main_point: str,
                                   main_point_method: str = "single", part_summaries=()) -> str:
    """Return statements_content for the monitoring report."""
    method_description = (
        "The article's central argument is extracted first, from the whole text; "
        "when the text is longer than the context, from summaries of its consecutive parts. "
        "Each excerpt is sent to an LLM together with one excerpt of context on each side "
        "and the article's central argument as an anchor. "
        "The model extracts key statements per actor "
        "(up to 4 statements per actor, each capped at 200 characters). "
        "<ul>"
        "<li><b>Actor filtering (two gates).</b> Only actors who clear both a "
        "<i>share gate</i> (≥40% of excerpt characters) and an "
        "<i>absolute gate</i> (≥150 characters of text) are queried. "
        "Actors below either threshold are too marginal to yield non-trivial statements.</li>"
        "<li><b>Statement types.</b> A statement can be a "
        "<i>claim</i> (knowledge, experience, or factual), "
        "a <i>position</i> (stance, values, commitment, or framing), "
        "or a <i>course of action</i> (something the actor proposes, opposes, or enacts). "
        "Any combination of the three is valid. This is enforced in the prompt.</li>"
        "<li><b>Base author treatment.</b> The author's own voice (<i>base_author</i>) is "
        "treated as a first-class actor. For this actor the model targets framing choices "
        "and factual claims that establish the article's narrative: which facts are selected, "
        "how situations are characterised, and what causal links are asserted.</li>"
        "</ul>"
    )

    content_parts = []
    for entry in all_statements:
        idx = entry["excerpt_index"]
        excerpt = excerpts[idx]
        excerpt_html = f"<details><summary>Excerpt {idx + 1}</summary><p>{html_lib.escape(excerpt['text'])}</p></details>"
        statements_html = ""
        for actor, stmts in entry["statements"].items():
            for stmt in stmts:
                statements_html += html_lib.escape(stmt)
                statements_html += f"<br>— <em>{html_lib.escape(actor)}</em><br><br>"
        content_parts.append(f"<section>{excerpt_html}{statements_html}</section>")

    parts_html = ""
    if main_point_method == "chunked":
        items = "".join(f"<li>{html_lib.escape(s)}</li>" for s in part_summaries)
        parts_html = (f"<details><summary>Taken from summaries of {len(part_summaries)} parts of the text, "
                      f"which is longer than the context</summary><ol>{items}</ol></details>\n")
    return (
        f"\n<article><h3>Method</h3>{method_description}</article>\n"
        f'<p><mark><strong>Central argument: </strong>{html_lib.escape(main_point)}</mark></p>\n'
        + parts_html
        + "".join(content_parts)
        + "\n"
    )

def step_4_key_statements(segments, excerpts, title=None, published_at=None):
    main_point, method, part_summaries = extract_main_point(segments, excerpts, title, published_at)
    all_statements, errors = extract_statements(excerpts, main_point)
    html_statements_content = key_statements_monitoring_html(all_statements, excerpts, main_point, method, part_summaries)
    return all_statements, main_point, html_statements_content, errors, method