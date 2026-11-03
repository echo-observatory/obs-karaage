import html as html_lib
from pydantic import BaseModel
from typing import Literal
import time
from karaage import llm
from karaage.paraphrasing import ROUGE2_THRESHOLD

verbose = False


class StatementValidation(BaseModel):
    in_text: Literal["yes", "uncertain", "no"]
    actor_correct: Literal["yes", "uncertain", "no"]


SYSTEM_PROMPT_VALIDATION = """You are validating statements extracted from a news article.
For each statement you receive the source passage, the statement text, and the actor it is attributed to.

Respond with:
- in_text: whether the statement's content is represented in the passage ("yes", "uncertain", or "no")
- actor_correct: whether the statement is correctly attributed to the stated actor ("yes", "uncertain", or "no")

"base_author" means the article's narrator/journalist, not a quoted source."""


def run_validation(statements: list) -> list[dict]:
    validated = []
    failed_ids = []
    for s in statements:
        if "rephrasing_error" in s:
            # The statement's rephrasing batch failed: there is nothing to validate,
            # and validate_statement rejects it as a rephrasing error.
            validated.append({**s, "in_text": None, "actor_correct": None})
            continue
        final_summary = s.get("paraphrased_summary") if s.get("summary_paraphrased") else s["summary"]
        user_prompt = f'Passage:\n{s["passage"]}\n\nStatement: "{final_summary}"\nAttributed to: {s["actor"]}'
        try:
            t0 = time.perf_counter()
            result = llm.complete_model(SYSTEM_PROMPT_VALIDATION, user_prompt, StatementValidation,
                                        max_tokens=llm.context_tokens())
            elapsed = time.perf_counter() - t0
            validated.append({**s, "in_text": result.in_text, "actor_correct": result.actor_correct})
            if verbose:
                print(f"[{s['id']:2d}] {s['actor']:<35} in_text={result.in_text:<9} actor={result.actor_correct:<9}  ({elapsed:.1f}s)")
        except Exception as e:
            validated.append({**s, "validation_error": f"{type(e).__name__}: {e}", "in_text": None, "actor_correct": None})
            failed_ids.append(s["id"])
            print(f"  ⚠️  Validation failed for statement [{s['id']}]: {e}")
    if failed_ids:
        print(f"  ⚠️  {len(failed_ids)} statement{'s' if len(failed_ids) != 1 else ''} could not be validated (IDs: {failed_ids}) — they will be excluded from results.")
    if verbose:
        keep = [s for s in validated if not s.get("validation_error") and s["in_text"] != "no" and s["actor_correct"] != "no"]
        drop = [s for s in validated if s.get("validation_error") or s["in_text"] == "no" or s["actor_correct"] == "no"]
        print(f"\nKeep: {len(keep)} / Drop: {len(drop)}")
        if drop:
            print("\nDropped statements:")
            for s in drop:
                print(f"  [{s['id']:2d}] {s['actor']}: {s['statement'][:80]}...")
                if s.get("validation_error"):
                    print(f"        validation_error={s['validation_error']}")
                else:
                    print(f"        in_text={s['in_text']}, actor_correct={s['actor_correct']}")
    return validated


# Validation criteria (applied in priority order — first match wins)
#   0. Validation error: the LLM call itself failed
#   1. Rephrasing error: rephrased_statement is an empty string
#   2. Too similar to source: rephrased but ROUGE-2 precision above threshold
#   3. Not in text: LLM judged the statement not grounded in the passage
#   4. Wrong actor: LLM judged the attribution incorrect

def validate_statement(s):
    if s.get("validation_error"):
        return False, "validation error"
    if not s.get("rephrased_statement", "").strip():
        return False, "rephrasing error"
    rouge2_rephrased = s.get("paraphrased_rouge2_rephrased") if s.get("rephrased_paraphrased") else s.get("rouge2_rephrased", 0)
    rouge2_summary = s.get("paraphrased_rouge2_summary") if s.get("summary_paraphrased") else s.get("rouge2_summary", 0)
    if max(rouge2_rephrased or 0, rouge2_summary or 0) > ROUGE2_THRESHOLD:
        return False, "too similar to source"
    if s.get("in_text") != "yes":
        return False, "not found in source text"
    if s.get("actor_correct") == "no":
        return False, "wrong actor attribution"
    return True, None


def apply_validation_filter(validated_statements: list) -> list[dict]:
    for s in validated_statements:
        s["valid"], s["invalid_reason"] = validate_statement(s)
    valid_count = sum(1 for s in validated_statements if s["valid"])
    invalid_count = len(validated_statements) - valid_count
    print(f"📦 {valid_count} statement{'s' if valid_count != 1 else ''} retained ({invalid_count} candidate{'s' if invalid_count != 1 else ''} rejected)")
    if verbose:
        for s in validated_statements:
            status = "✓" if s["valid"] else f"✗  {s['invalid_reason']}"
            print(f"  [{s['id']:2d}] {status:<42}  {s['actor']}: {s['statement'][:60]}...")
    return validated_statements

def step_7_validate(paraphrased_statements):
    validated_statements = run_validation(paraphrased_statements)
    filtered_statements = apply_validation_filter(validated_statements)
    html_validation_content = validation_monitoring_html(filtered_statements)
    return filtered_statements, html_validation_content

def validation_monitoring_html(validated_statements: list) -> str:
    """Return validation_content for the monitoring report."""
    method_description = (
        "Each statement is checked against two independent signals, then filtered by four criteria applied in priority order. "
        "<ul>"
        "<li><b>LLM grounding check.</b> The model receives the source passage, the statement’s summary "
        "(paraphrased version if available, otherwise the rephrasing-step summary), and its attributed actor. "
        "It returns two three-way verdicts: <i>in_text</i> (is the summary’s content represented in the passage?) "
        "and <i>actor_correct</i> (is the attribution right?) — each “yes”, “uncertain”, or “no”. "
        "Validating the summary rather than the original extraction serves as a consistency check: "
        "a summary that has drifted from the passage will be caught here.</li>"
        "<li><b>Filtering.</b> A statement is dropped if: "
        "(1) its rephrased form is empty (rephrasing error); "
        "(2) the ROUGE-2 precision of either the long or the summary version remains above 0.30 after paraphrasing "
        "(still too close to the source); "
        "(3) <i>in_text</i> is not “yes” (content not found or uncertain); "
        "(4) <i>actor_correct</i> is “no” (attribution judged wrong; “uncertain” passes).</li>"
        "</ul>"
    )
    content_parts = []
    for entry in validated_statements:
        if entry.get("valid"):
            badge = '<span style="background:#d4edda;color:#155724;padding:2px 6px;border-radius:3px;font-size:0.85em">✔</span> '
            opc_att = ""
        else:
            reason = html_lib.escape(entry.get("invalid_reason") or "unknown")
            badge = f'<span style="background:#f8d7da;color:#721c24;padding:2px 6px;border-radius:3px;font-size:0.85em">Invalid: {reason}</span><br>'
            opc_att = "style='opacity: 0.3;'"
        final_summary = entry.get("paraphrased_summary") if entry.get("summary_paraphrased") else entry.get("summary", "")
        final_rephrased = entry.get("paraphrased_statement") if entry.get("rephrased_paraphrased") else entry.get("rephrased_statement", "")
        section = f"<p><code>{entry['id']}</code> {badge}<span {opc_att}><em>{html_lib.escape(entry['actor'])}:</em><br><strong>{html_lib.escape(final_summary)}</strong><br>"
        if entry.get('validation_error'):
            err = html_lib.escape(entry['validation_error'])
            section += f'<em style="color:red">[VALIDATION ERROR: {err}]</em><br>'
        elif final_rephrased.strip():
            section += f"{html_lib.escape(final_rephrased)}<br>"
        else:
            error = html_lib.escape(entry.get('rephrasing_error', 'empty rephrased statement'))
            section += f'<em style="color:red">[ERROR: {error}]</em><br>'
        section += "</span></p>"
        content_parts.append(section)
    return (
        f"\n<article><h3>Method</h3>{method_description}</article>\n"
        + "".join(content_parts)
        + "\n"
    )
