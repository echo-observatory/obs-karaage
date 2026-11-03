import re
import html as html_lib
from tqdm import tqdm
from pydantic import BaseModel, Field
from sentence_transformers import util
from karaage import llm
from karaage.rephrasing import SHORT_LENGTH_MAX

verbose = False

# Step-level configuration
ROUGE2_THRESHOLD = 0.3
TARGET_COSINE = 0.75
MAX_ITERATIONS = 2

PARAPHRASE_SYSTEM = (
    "You are a precise paraphrasing assistant. Rewrite the given statement using "
    "completely different vocabulary and sentence structure while preserving every "
    "fact, number, proper noun, and the exact same meaning. Do not add or remove "
    "any information. Prefer simple, direct sentence structures over elaborate ones,"
    "even if this means breaking complex ideas into shorter sentences."
    "Where technical or domain-specific terms are used in the source"
    "(e.g. \"bottom trawling\", \"ecological integrity\", \"biomass\"),"
    "retain these keywords as they carry precise meaning — but restructure the"
    "sentences around them rather than preserving the original phrasing."
)


class ParaphraseResult(BaseModel):
    rephrased_statement: str = Field(
        description="Rewritten statement with different vocabulary and structure, identical meaning.",
        max_length=800,
    )


class ParaphraseSummaryResult(BaseModel):
    rephrased_statement: str = Field(
        description="Rewritten summary with different vocabulary and structure, identical meaning.",
        max_length=SHORT_LENGTH_MAX,
    )


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def bigrams(tokens):
    return list(zip(tokens, tokens[1:]))


def rouge2_precision(rephrased, excerpt):
    rep_bigrams = bigrams(tokenize(rephrased))
    if not rep_bigrams:
        return 0.0
    excerpt_bigram_set = set(bigrams(tokenize(excerpt)))
    matches = sum(1 for b in rep_bigrams if b in excerpt_bigram_set)
    return matches / len(rep_bigrams)


def compute_rouge2_scores(statements: list) -> list[dict]:
    for s in statements:
        if 'rephrased_statement' in s:
            s['rouge2_rephrased'] = round(rouge2_precision(s['rephrased_statement'], s['excerpt']), 4)
        if 'summary' in s:
            s['rouge2_summary'] = round(rouge2_precision(s['summary'], s['excerpt']), 4)
    return statements


def get_metrics(source, candidate, excerpt, sim_model):
    emb1 = sim_model.encode(source, convert_to_tensor=True)
    emb2 = sim_model.encode(candidate, convert_to_tensor=True)
    cosine = util.cos_sim(emb1, emb2).item()
    rouge2 = rouge2_precision(candidate, excerpt)
    return cosine, rouge2


def _paraphrase_text(text, initial_rouge2, excerpt, sim_model, schema):
    best_candidate = text
    best_rouge2 = initial_rouge2
    best_cosine = 1.0

    prompt = (
        "Rewrite this statement using completely different vocabulary and sentence "
        f"structure, while preserving every fact and the exact same meaning:\n\n{text}"
    )

    for i in range(MAX_ITERATIONS):
        result = llm.complete_model(PARAPHRASE_SYSTEM, prompt, schema, max_tokens=llm.context_tokens())
        candidate = result.rephrased_statement
        cosine, rouge2 = get_metrics(text, candidate, excerpt, sim_model)

        status = "OK" if cosine >= TARGET_COSINE and rouge2 < ROUGE2_THRESHOLD else "fail"
        if verbose:
            print(f"  iter {i+1}: cosine={cosine:.3f}  rouge2={rouge2:.4f}  [{status}]")

        if rouge2 < best_rouge2:
            best_candidate = candidate
            best_rouge2 = rouge2
            best_cosine = cosine

        if cosine >= TARGET_COSINE and rouge2 < ROUGE2_THRESHOLD:
            return candidate, cosine, rouge2, i + 1

        feedback = []
        if cosine < TARGET_COSINE:
            feedback.append("The meaning has drifted. Preserve all the original facts and framing exactly.")
        if rouge2 >= ROUGE2_THRESHOLD:
            feedback.append(
                f"Too many phrases are shared with the source text ({rouge2:.0%} bigram overlap). "
                "Use entirely different sentence structures and word choices."
            )

        prompt = (
            f"Your last attempt failed. Feedback: {' '.join(feedback)}\n\n"
            f"Original (meaning to preserve): {text}\n"
            f"Previous attempt: {candidate}\n\n"
            "Try again with a fresh approach."
        )

    return best_candidate, best_cosine, best_rouge2, MAX_ITERATIONS


def paraphrase_statements(statements: list, sim_model) -> list[dict]:
    flagged = [
        s for s in statements
        if s.get('rouge2_rephrased', 0) > ROUGE2_THRESHOLD or s.get('rouge2_summary', 0) > ROUGE2_THRESHOLD
    ]
    not_flagged = [
        s for s in statements
        if s.get('rouge2_rephrased', 0) <= ROUGE2_THRESHOLD and s.get('rouge2_summary', 0) <= ROUGE2_THRESHOLD
    ]
    if verbose:
        print(f"{len(flagged)} statements flagged (rouge2 > {ROUGE2_THRESHOLD}), {len(not_flagged)} below threshold")
    for s in not_flagged:
        s['paraphrased'] = False
        s['rephrased_paraphrased'] = False
        s['summary_paraphrased'] = False
    for s in tqdm(flagged, desc="Paraphrasing statements"):
        if verbose:
            print(f"\nStatement {s['id']} (rouge2_rephrased={s.get('rouge2_rephrased', 0):.4f}, rouge2_summary={s.get('rouge2_summary', 0):.4f})")
        s['rephrased_paraphrased'] = False
        s['summary_paraphrased'] = False
        if s.get('rouge2_rephrased', 0) > ROUGE2_THRESHOLD:
            candidate, cosine, rouge2, iters = _paraphrase_text(
                s['rephrased_statement'], s['rouge2_rephrased'], s['excerpt'], sim_model, ParaphraseResult,
            )
            s['paraphrased_statement'] = candidate
            s['paraphrased_rouge2_rephrased'] = round(rouge2, 4)
            s['paraphrased_cosine_rephrased'] = round(cosine, 4)
            s['paraphrased_iterations_rephrased'] = iters
            s['rephrased_paraphrased'] = True
            if verbose:
                print(f"  rephrased -> rouge2={rouge2:.4f}  cosine={cosine:.3f}  ({iters} iter(s))")
        if s.get('rouge2_summary', 0) > ROUGE2_THRESHOLD:
            candidate, cosine, rouge2, iters = _paraphrase_text(
                s['summary'], s['rouge2_summary'], s['excerpt'], sim_model, ParaphraseSummaryResult,
            )
            s['paraphrased_summary'] = candidate
            s['paraphrased_rouge2_summary'] = round(rouge2, 4)
            s['paraphrased_cosine_summary'] = round(cosine, 4)
            s['paraphrased_iterations_summary'] = iters
            s['summary_paraphrased'] = True
            if verbose:
                print(f"  summary   -> rouge2={rouge2:.4f}  cosine={cosine:.3f}  ({iters} iter(s))")
        s['paraphrased'] = s['rephrased_paraphrased'] or s['summary_paraphrased']
    return statements

def step_6_paraphrase(rephrased_statements, embedding_model, sim_model):
    scored_statements = compute_rouge2_scores(rephrased_statements)
    
    if verbose:
        print(f"{'ID':>3}  {'R2-Long':>8}  {'R2-Summ':>8}  Rephrased (truncated)")
        print('-' * 80)
        for s in scored_statements:
            print(f"{s['id']:>3}  {s.get('rouge2_rephrased', 0):>8.4f}  {s.get('rouge2_summary', 0):>8.4f}  {s.get('rephrased_statement', '')[:55]}...")

    paraphrased_statements = paraphrase_statements(scored_statements, sim_model)
    html_paraphrasing_content = paraphrasing_monitoring_html(paraphrased_statements, embedding_model)
    return paraphrased_statements, html_paraphrasing_content

def paraphrasing_monitoring_html(statements: list, embedding_model_name: str) -> str:
    """Return paraphrasing_content for the monitoring report."""
    method_description = (
        "Rephrased statements or summaries that borrow too much wording from the source excerpt are "
        "rewritten a second time through an adversarial paraphrasing loop designed to "
        "maximise lexical distance while preserving meaning. "
        "<ul>"
        "<li><b>ROUGE-2 precision screening.</b> For every statement the bigram overlap with its source "
        "excerpt is computed separately for the long rephrased version and for the single-sentence summary. "
        "A statement is flagged if either score exceeds the threshold (ROUGE-2 precision &gt; 0.30). "
        "Each version that exceeds the threshold is then paraphrased independently; versions below it are "
        "passed through unchanged. "
        "This metric is chosen because it works for comparing a short text (the statement) "
        "to a long one (the excerpt).</li>"
        f"<li><b>Adversarial paraphrasing loop (up to {MAX_ITERATIONS} iterations).</b> The LLM is asked to "
        "rewrite the flagged version using entirely different vocabulary and sentence structure "
        "while preserving every fact, number, proper noun, and the exact same meaning. "
        "Each candidate is evaluated against two criteria: "
        f"(1) <i>cosine similarity</i> ≥ 0.75 (semantic fidelity, measured with "
        f"<code>{embedding_model_name}</code>); "
        "(2) <i>ROUGE-2 precision</i> &lt; 0.30 (sufficient lexical distance from the excerpt). "
        "If a candidate fails, targeted feedback is returned to the model "
        "(“meaning has drifted” or “too many shared phrases”) and the loop retries. "
        "If no iteration meets both criteria, the candidate with the lowest ROUGE-2 is kept.</li>"
        "</ul>"
    )
    content_parts = []
    current_excerpt_id = -1
    for entry in statements:
        if entry['excerpt_index'] != current_excerpt_id:
            current_excerpt_id = entry['excerpt_index']
            content_parts.append(
                f"<details><summary>Excerpt {entry['excerpt_index'] + 1}</summary>"
                f"<p>{html_lib.escape(entry['excerpt'])}</p></details>"
            )
        section = f"<p><code>{entry['id']}</code> "
        if 'rephrased_statement' in entry:
            rephrased = html_lib.escape(entry['rephrased_statement'])
            r2_rep = html_lib.escape(str(entry.get('rouge2_rephrased', '')))
            summary = html_lib.escape(entry['summary'])
            r2_sum = html_lib.escape(str(entry.get('rouge2_summary', '')))
            section += (
                f"{rephrased}<br>"
                f"<small><code>ROUGE-2 (long): {r2_rep}</code></small>"
            )
            if entry.get('rephrased_paraphrased'):
                para = html_lib.escape(entry['paraphrased_statement'])
                para_r2 = html_lib.escape(str(entry['paraphrased_rouge2_rephrased']))
                para_cos = html_lib.escape(str(entry['paraphrased_cosine_rephrased']))
                section += (
                    f"</p><p style='margin-top: 0px; padding-left: 2em; color: steelblue'>"
                    f"<strong>&gt; Adversarial paraphrase (long):</strong><br>{para}<br>"
                    f"<small><code>ROUGE-2: {para_r2}, Cosine: {para_cos}</code></small>"
                )
            section += (
                f"</p><p style='margin-top: 0px'>"
                f"<em>Summary:</em> {summary}<br>"
                f"<small><code>ROUGE-2 (summary): {r2_sum}</code></small>"
            )
            if entry.get('summary_paraphrased'):
                para_s = html_lib.escape(entry['paraphrased_summary'])
                para_r2_s = html_lib.escape(str(entry['paraphrased_rouge2_summary']))
                para_cos_s = html_lib.escape(str(entry['paraphrased_cosine_summary']))
                section += (
                    f"</p><p style='margin-top: 0px; padding-left: 2em; color: steelblue'>"
                    f"<strong>&gt; Adversarial paraphrase (summary):</strong><br>{para_s}<br>"
                    f"<small><code>ROUGE-2: {para_r2_s}, Cosine: {para_cos_s}</code></small>"
                )
        else:
            error = html_lib.escape(entry.get('rephrasing_error', 'unknown error'))
            section += f'<em style="color:red">[ERROR: {error}]</em>'
        section += "</p>"
        content_parts.append(section)
    return (
        f"\n<article><h3>Method</h3>{method_description}</article>\n"
        + "".join(content_parts)
        + "\n"
    )
