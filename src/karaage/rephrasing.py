import json
import html as html_lib
from tqdm import tqdm
from pydantic import BaseModel, Field, create_model
from karaage import llm

verbose = False

# Step-level configuration (the token budget is the profile's context size: llm.context_tokens())
BATCH_SIZE = 4                # Statements per batch
PASSAGE_CONTEXT_OFFSET = 3    # Extra passages before/after the batch's passage range
ECHO_PREFIX_LENGTH = 80       # Characters of the original statement echoed back for cross-contamination detection
RESTATED_LENGTH_TARGET = 400  # Soft character target communicated in the prompt
SHORT_LENGTH_TARGET = 150     # Soft character target for the single-sentence headline version, communicated in the prompt
RESTATED_LENGTH_MAX = 800     # Hard character limit enforced via JSON schema
SHORT_LENGTH_MAX = 300        # Hard character limit for the single-sentence short version


SYSTEM_PROMPT_REPHRASING = f'''You are a journalist-analyst.

Your task:
Rephrase the statement made by the specified actor in the source text so that it is fully understandable even in the absence of the original context.
After writing the rephrased version, distill it into a self-contained, single-sentence summary capturing the essence of the statement.

You will be provided indications about where in the full text the statement is located, who the actor is, and what the article's central claim is.

Actor statements are made by an actor. The actor is either cited in the article, or the actor is the author of the article.

Rules:
- Rewrite the actor statement so that the source text's central claim is naturally present as a backdrop — as a shared reality that the actor's statement assumes, confirms, qualifies, challenges or draws conclusions from.
  The actor's stance toward the central claim should emerge naturally from the content and framing of the statement itself.
  Where it fits naturally, the source of the central claim may be named directly in the statement — for instance when an actor is explicitly reacting to or distancing themselves from it.
  Where it does not fit naturally, the central claim should simply function as implicit backdrop without being named.
  Where the actor is themselves the originator of the central claim — for instance the author of a report or the person making the announcement — they should reference it as their own work or decision rather than as an external source.

- If it is the case, make it explicit that:
    - The actor is arguing that there is a problem (then what is the problem, for whom, and what is causing it?)
    - The actor is proposing a particular solution (if so, what is it, and why do they believe it will work?)
    - The actor is contributing expert knowledge (if so, what kind, and how does it bear on the central claim?)
    - The actor is assigning blame, responsibility or the capacity to solve a problem to certain other actors, institutions, technologies or events (if so, to whom or what, and in what way?)
    - The actor is adopting a particular problem frame (if so, what is the underlying concern or question?)
    - The actor is expressing adherence to specific values or commitments (if so, which and with what consequences?)
  These dimensions, if relevant, should be naturally integrated into the flow of the statement — not listed or made explicit as separate points.

- Each statement must include sufficient context to be fully self-contained and understandable in isolation. This further includes:
    - Geographic and institutional context — specify relevant countries, institutions, or roles rather than assuming the reader knows the subject
    - Situational context — briefly explain the circumstances or motivation behind the claim.
  The output must never contain underspecified references like "they", "the report", "the measures", "the experts", "the government", "the organization", "the study", etc.
  Instead, these should be replaced with more specific descriptions that make the statement understandable on its own terms.

- Write the statement as if it were the actor's own position or claim, in an assertive style rendered in third person.
  Use modal verbs like "will", "should", or "must" to convey the actor's stance.
  The output must never contain first-person pronouns: "I", "we", "our", "us", "my", "me". Replace them with the relevant third-person subject (e.g. "Denmark", "the sector", "the farming industry", "the government").

- Never attribute the statement to anyone inside the text. Do not name the actor, their role, or their institution — not even in passing. Do not use attribution phrases of any kind: "X argues that", "X states that", "according to X", "she contends", "the council asserts", "he warns", "she emphasizes", and all similar constructions. Attribution is handled exclusively through metadata. The statement must read as a direct assertion of the position — not as a report of someone holding it.
  Wrong: "Anna Berg, spokesperson for the farmers' association, warns that the proposed limits on nitrogen use will push small farms out of business."
  Right: "The proposed limits on nitrogen use in agriculture will push small farms out of business, a cost that must be weighed against the expected gains for water quality."

- Aim for 2–3 sentences for the rephrasing. Prioritize self-containment over brevity — include enough geographic, institutional, and situational context for the statement to be fully understood in isolation. The hard upper limit is {RESTATED_LENGTH_MAX} characters; do not sacrifice context to stay short.

- For the summary, aim for one sentence capturing the essence of the rephrased statement, with the same rules: self-contained but not attributed to anyone. The hard upper limit is {SHORT_LENGTH_TARGET} characters.

- Use well-formed, everyday language.
'''


class StatementResult(BaseModel):
    # The model copies the opening characters of the original statement before
    # rephrasing. A short prefix is enough to anchor generation to the correct
    # item and to detect cross-contamination: if the echoed prefix does not match
    # the input, the model has drifted to a neighbouring statement.
    original: str = Field(description=f"First {ECHO_PREFIX_LENGTH} characters of the original statement, copied verbatim.", max_length=ECHO_PREFIX_LENGTH)
    rephrased: str = Field(description="Self-contained rephrasing of the statement.", max_length=RESTATED_LENGTH_MAX)
    summary: str = Field(description="Single-sentence summary of the rephrased statement.", max_length=SHORT_LENGTH_MAX)


def make_batch_schema(batch):
    fields = {str(s["id"]): (StatementResult, ...) for s in batch}
    return create_model("RephrasedStatementBatch", **fields)


def make_batch_prompt(batch, passages: list, main_point: str) -> str:
    n_passages = len(passages)
    min_p = min(s['passage_index'] for s in batch)
    max_p = max(s['passage_index'] for s in batch)
    start_p = max(0, min_p - PASSAGE_CONTEXT_OFFSET)
    end_p = min(n_passages - 1, max_p + PASSAGE_CONTEXT_OFFSET)
    context_passages = [{"id": i, "text": passages[i]['text']} for i in range(start_p, end_p + 1)]
    items = {str(s["id"]): {"passage_id": s["passage_index"], "actor": s["actor"], "statement": s["statement"]} for s in batch}
    return "\n\n".join([
        f"<task>\nFor each item: copy the first {ECHO_PREFIX_LENGTH} characters of its statement field verbatim into 'original', then write the rephrasing into 'rephrased'. Process each item independently, keyed by its id (as a string). Use the passage_id to locate the statement within the provided passages.\n</task>",
        f"<central_claim>\n{main_point}\n</central_claim>",
        f"<passages>{json.dumps(context_passages, ensure_ascii=False, indent=2)}</passages>",
        f"<statements>{json.dumps(items, ensure_ascii=False, indent=2)}</statements>",
    ])


def build_statements(all_statements: list, passages: list) -> list[dict]:
    statements = []
    stmt_id = 0
    for entry in all_statements:
        passage_index = entry['passage_index']
        passage_data = passages[passage_index]
        for actor, actor_statements in entry['statements'].items():
            for statement_text in actor_statements:
                statements.append({
                    'id': stmt_id,
                    'actor': actor,
                    'statement': statement_text,
                    'passage_index': passage_index,
                    'passage': passage_data['text'],
                    'segments': passage_data['segments'],
                })
                stmt_id += 1
    return statements


def rephrase_statements(statements: list, passages: list, main_point: str) -> list[dict]:
    for s in statements:
        s.pop('rephrased_statement', None)
        s.pop('rephrasing_error', None)
    batches = [statements[i:i+BATCH_SIZE] for i in range(0, len(statements), BATCH_SIZE)]
    for batch in tqdm(batches, desc="Rephrasing batches"):
        user_prompt = make_batch_prompt(batch, passages, main_point)
        try:
            schema = make_batch_schema(batch)
            result = llm.complete_model(SYSTEM_PROMPT_REPHRASING, user_prompt, schema,
                                        max_tokens=llm.context_tokens())
            rephrased = result.model_dump()
            for s in batch:
                s['rephrased_statement'] = rephrased[str(s["id"])]["rephrased"]
                s['summary'] = rephrased[str(s["id"])]["summary"]
        except Exception as e:
            for s in batch:
                s['rephrasing_error'] = f"{type(e).__name__}: {e}"
    n_ok  = sum(1 for s in statements if 'rephrased_statement' in s)
    n_err = sum(1 for s in statements if 'rephrasing_error'    in s)
    print(f"-> Rephrasing for {n_ok} statement{'s' if n_ok != 1 else ''} succeeded, {n_err} failed.")
    if verbose:
        print(statements)
    return statements

def step_5_rephrase(all_statements, passages, main_point):
    statements = build_statements(all_statements, passages)

    if verbose:
        print('Statements with context:')
        print(statements)
        # Make sanity check before hitting the model:
        # ensure no batch can exceed the token limit given the shared prompt + passage context + statement content + expected output length.
        rephrasing_statements_sanity_check(main_point, passages, statements)
    
    rephrased_statements = rephrase_statements(statements, passages, main_point)
    html_rephrasing_content = rephrasing_monitoring_html(rephrased_statements, main_point)
    return rephrased_statements, html_rephrasing_content

def rephrasing_statements_sanity_check(main_point, passages, statements):
    def estimate_tokens(text):
        return len(text) // 4

    all_passages_list = passages
    n_passages = len(all_passages_list)

    # Base shared context (system prompt + central claim — passage window varies per batch)
    shared_base_tokens = estimate_tokens(SYSTEM_PROMPT_REPHRASING + f"<central_claim>\n{main_point}\n</central_claim>")

    # Check every actual batch and find the worst case
    all_batches = [statements[i:i+BATCH_SIZE] for i in range(0, len(statements), BATCH_SIZE)]
    worst_total = 0
    worst_idx = 0

    for bi, batch in enumerate(all_batches):
        min_p = min(s['passage_index'] for s in batch)
        max_p = max(s['passage_index'] for s in batch)
        start_p = max(0, min_p - PASSAGE_CONTEXT_OFFSET)
        end_p = min(n_passages - 1, max_p + PASSAGE_CONTEXT_OFFSET)
        context_passages = [{"id": i, "text": all_passages_list[i]['text']} for i in range(start_p, end_p + 1)]

        passage_tokens   = estimate_tokens(json.dumps(context_passages))
        items            = {str(s["id"]): {"passage_id": s["passage_index"], "actor": s["actor"], "statement": s["statement"]} for s in batch}
        statement_tokens = estimate_tokens(json.dumps(items))
        # Output per statement: echoed prefix + rephrased + short
        output_tokens    = len(batch) * (ECHO_PREFIX_LENGTH // 4 + RESTATED_LENGTH_MAX // 4 + SHORT_LENGTH_MAX // 4)

        total = shared_base_tokens + passage_tokens + statement_tokens + output_tokens
        if total > worst_total:
            worst_total = total
            worst_idx   = bi

    print("Sanity check for token limits:")
    print(f"Worst batch:            #{worst_idx}  ({len(all_batches[worst_idx])} statements)")
    budget = llm.context_tokens()
    print(f"Total:                  ~{worst_total} / {budget}")
    assert worst_total <= budget, f"Exceeds the profile's context size ({worst_total} > {budget})"
    print("OK")

def batch_debug(all_statements, passages, main_point, batch_index):
    # Recompute batches (that's fine, it's not too expensive)
    statements = build_statements(all_statements, passages)
    batches = [statements[i:i+BATCH_SIZE] for i in range(0, len(statements), BATCH_SIZE)]

    debug_batch = batches[batch_index]
    debug_prompt = make_batch_prompt(debug_batch, passages, main_point)
    debug_schema = make_batch_schema(debug_batch)

    print("=" * 60)
    print("SYSTEM PROMPT")
    print("=" * 60)
    print(SYSTEM_PROMPT_REPHRASING)

    print("=" * 60)
    print(f"USER PROMPT  (batch {batch_index}, {len(debug_batch)} statements)")
    print("=" * 60)
    print(debug_prompt)

    print("=" * 60)
    print("RAW RESPONSE")
    print("=" * 60)
    debug_response = llm.active().request(
        SYSTEM_PROMPT_REPHRASING, debug_prompt, debug_schema.model_json_schema(),
        name=debug_schema.__name__, max_tokens=llm.context_tokens(),
    )
    raw = debug_response.choices[0].message.content
    print(raw)
    print("=" * 60)
    print(f"Finish reason: {debug_response.choices[0].finish_reason}")
    print(f"Usage: {debug_response.usage}")

def rephrasing_monitoring_html(statements: list, main_point: str) -> str:
    """Return rephrasing_content for the monitoring report."""
    method_description = (
        "The raw statements extracted in step 4 are rephrased into self-contained, assertive sentences "
        "that can be understood in isolation, without reference to the original article. "
        "They are then distilled into single-sentence summaries capturing their essence. "
        "<ul>"
        f"<li><b>Batch processing.</b> Statements are sent to the LLM in batches of {BATCH_SIZE}, "
        f"each batch accompanied by a window of passage context ({PASSAGE_CONTEXT_OFFSET} passages before and after "
        "the batch's passage range) and the article's central claim as anchor. "
        "We want small batches to mitigate cross-contamination between batch items, "
        "but we also need batches because loading the prompt is resource-intensive "
        "on a per-claim basis.</li>"
        f"<li><b>Echo-prefix guard.</b> The model is asked to copy the first {ECHO_PREFIX_LENGTH} characters of "
        "each original statement verbatim before rephrasing. "
        "We use it as simple anchoring: mitigate model confusion about which statement is being rephrased.</li>"
        "<li><b>Rephrasing rules.</b> The model rewrites each statement so that: "
        "(1) the central claim functions as an implicit backdrop (or explicit backdrop where the actor reacts to it directly); "
        "(2) relevant dimensions are surfaced — whether the actor identifies a problem, proposes a solution, "
        "contributes expertise, assigns blame, adopts a problem frame, or expresses values; "
        "(3) geographic, institutional, and situational context is added for self-containment; "
        "(4) the output uses third-person assertive style with no attribution phrases "
        "(<i>“X argues that…”</i>, <i>“according to X…”</i>) and no first-person pronouns.</li>"
        "</ul>"
        f"Each rephrased statement is capped at {RESTATED_LENGTH_MAX} characters. Attribution is handled "
        "exclusively through metadata (i.e. from earlier attribution step)."
    )
    content_parts = []
    for entry in statements:
        if 'rephrased_statement' in entry:
            rephrased = html_lib.escape(entry['rephrased_statement'])
            summary = html_lib.escape(entry['summary'])
        else:
            error = html_lib.escape(entry.get('rephrasing_error', 'unknown error'))
            rephrased = f'<em style="color:red">[ERROR: {error}]</em>'
            summary = f''
        content_parts.append(
            f"<p><code>{entry['id']}</code> "
            f"<strong>{html_lib.escape(entry['statement'])}</strong><br>"
            f"<code>Rephrased:</code> {rephrased}<br>"
            f"<code>Summary:</code> {summary}<br>"
            f"— {html_lib.escape(entry['actor'])}</p>"
        )
    return (
        f"\n<article><h3>Method</h3>{method_description}</article>\n"
        f'<p><mark><strong>Central argument: </strong>{html_lib.escape(main_point)}</mark></p>\n'
        + "".join(content_parts)
        + "\n"
    )
