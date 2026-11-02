import re
import html as html_lib
import spacy
import benepar
from transformers import T5Tokenizer, T5TokenizerFast

verbose = False


def load_nlp_pipeline() -> spacy.Language:
    # transformers 5.x removed build_inputs_with_special_tokens from T5 tokenizers,
    # but benepar still calls it to locate special-token positions. For T5 it's a no-op.
    def _t5_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        return token_ids_0
    for _cls in (T5Tokenizer, T5TokenizerFast):
        if not hasattr(_cls, 'build_inputs_with_special_tokens'):
            _cls.build_inputs_with_special_tokens = _t5_build_inputs_with_special_tokens
    nlp = spacy.load('en_core_web_md')
    if not nlp.has_pipe('benepar'):
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
    return nlp


# ═══════════════════════════════════════════════════════════════════════════════
# FOUR-LAYER SEGMENTATION PIPELINE
#
# Goal: split text into segments fine enough that each one can be labelled
# with a single "source" (narrator, named speaker, etc.), while preserving
# every character so that concatenating all segments reconstructs the original
# text exactly.
#
# Layer 0 — Forced-boundary detection
#   Finds formatting artefacts (section headings merged into body prose)
#   that should always produce a segment boundary.
#
# Layer 1 — Discourse split
#   Splits at colon-introduced direct speech.  This is a discourse-level
#   boundary that the constituency parser cannot reliably detect.
#
# Layer 2 — Clause extraction  (benepar constituency parse)
#   Within each Layer-1 segment, first splits into sentences using spaCy's
#   dependency parser (without benepar), then runs benepar on each sentence
#   individually to find syntactic clause boundaries — the points where the
#   attributed source can change mid-sentence.
#
# Layer 3 — Micro-segment merging
#   Absorbs very short fragments (≤ 2 chars) produced by Layer 2 into their
#   neighbours, keeping the segment list clean and attribution-friendly.
# ═══════════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────────
# LAYER 0  —  Forced-boundary detection
# ───────────────────────────────────────────────────────────────────────────────
#
# Problem
# -------
# When articles are scraped from HTML, section headings are sometimes merged
# directly into the body text with no separating punctuation.  Example:
#
#   "...a ruling is expected before the summer. Higher fees ahead Under
#    the new agreement, shipping companies..."
#
# "Higher fees ahead" is a section heading.  Without intervention,
# benepar receives an enormous run-on sentence and parses it poorly.
#
# Detection heuristic
# -------------------
# After a sentence-ending period, look for a short phrase where:
#   - the FIRST word starts with a capital letter,
#   - ALL REMAINING words start with a lower-case letter.
# This is "sentence-case" heading style (first word only capitalised).
#
# Two guards prevent common false positives:
#   (a) Exclusion list: words that routinely start a new sentence but are
#       not headings (conjunctions, pronouns, prepositions, articles).
#   (b) Additional-word constraint: every word after the first must begin
#       with a lower-case letter, which blocks proper-noun sequences such as
#       "European Commission" (the second word starts with a capital).
#
# Known limitation
# ----------------
# Headings written as full sentences ("The climate may pay the price") are
# NOT detected, because their additional words look identical to ordinary
# prose.  Reliable detection of those cases requires POS tagging (checking
# for verbs) and is left as a future improvement.

_NOT_HEADING_STARTS = (
    "Because|But|And|Or|While|Since|Although|However|Furthermore|Moreover|"
    "Additionally|In|On|At|By|For|From|As|This|These|That|There|"
    "It|He|She|They|We|You|I|The|A|An"
)

# Breakdown of the pattern:
#   \.\s+               sentence-ending period + whitespace
#   (?!...\b)           guard (a): first word must NOT be in exclusion list
#   (?P<header>         named capture group for the heading phrase
#     [A-Z][a-z]+       first word: one capital + one-or-more lower-case
#     (?:\s+[a-z]\w+)   additional words: each starts with lower-case
#     {1,4}             between 1 and 4 additional words
#   )
#   (?=\s+[A-Z][a-z])   lookahead: what follows is another capitalised word
#                        (the true start of the next sentence)
_FORCED_BOUNDARY_RE = re.compile(
    r'\.\s+'
    r'(?!(?:' + _NOT_HEADING_STARTS + r')\b)'
    r'(?P<header>[A-Z][a-z]+(?:\s+[a-z]\w+){1,4})'
    r'(?=\s+[A-Z][a-z])'
)


def find_forced_boundaries(text):
    """
    Return a list of character positions where a segment boundary must be
    placed regardless of what the parser finds.

    Each position marks the FIRST CHARACTER of the text that follows the
    detected heading — i.e., the start of the next body sentence.
    The heading itself stays attached to the preceding segment.
    """
    boundaries = []
    for m in _FORCED_BOUNDARY_RE.finditer(text):
        # m.end() sits right after the last character of the heading phrase.
        # From there, the text continues with whitespace and then [A-Z].
        # We skip the whitespace to land exactly on the capital letter.
        after_header = text[m.end():]
        leading_space = len(after_header) - len(after_header.lstrip())
        boundaries.append(m.end() + leading_space)
    return boundaries


# ───────────────────────────────────────────────────────────────────────────────
# LAYER 1  —  Discourse split
# ───────────────────────────────────────────────────────────────────────────────
#
# Problem
# -------
# Constituency parsers model SYNTACTIC structure (which phrase contains which),
# but the attribution boundary in colon-introduced direct speech is a
# DISCOURSE-level signal, not a syntactic one:
#
#   X said in a written reply: "It is clear that..."
#
# The colon marks where the reporting clause ends and the quoted content begins.
# The parser sees one sentence and may or may not generate a nested S node
# for the quote — its behaviour is inconsistent for this pattern.
#
# Why not split on every quotation mark?
# ----------------------------------------
# Scare quotes and embedded technical terms appear throughout:
#   the 'protected' areas,  "clear objectives",  "restore"
# These are noun phrases, not attributed speech, and must NOT create splits.
# The colon immediately before an opening quote is the reliable discriminator.
#
# Guard against false positives
# -----------------------------
# Colons in clock times ("10:30") and ratios ("2:1") are excluded by
# checking that the character immediately before the colon is not a digit.
#
# Quote characters handled
# ------------------------
#   "  straight double quote (U+0022)
#   "  left curly double quote (U+201C)
#   '  left curly single quote (U+2018)  — used for some direct speech

_COLON_QUOTE_RE = re.compile(
    r'(:\s*)'                    # the colon and any trailing whitespace
    '(?=["“‘])'        # lookahead: straight ", curly ", or curly '
                                 # Non-raw string so \u escapes are resolved.
                                 # (lookahead does not consume the quote character,
                                 #  so it remains at the start of the next segment)
)


def discourse_split(text, forced_boundaries):
    """
    Split `text` into coarse segments at discourse-level boundaries.

    Two kinds of boundaries are applied:
      1. Positions from `forced_boundaries` (Layer 0, section headings).
      2. Positions after a colon that introduces a direct quotation.

    The split is placed AFTER the colon and any following whitespace, so:
      - the reporting clause keeps its colon  →  "X said:"
      - the quoted content starts with the opening quote  →  '"Y..."'

    All characters are preserved; segments reconstruct `text` exactly.

    Returns a list of dicts: {'text': str, 'start': int, 'end': int}
    where start/end are character offsets into the original `text`.
    """
    # Begin with position 0 (start of text is always a boundary) and add
    # any forced positions from Layer 0.
    split_points = {0} | set(forced_boundaries)

    for m in _COLON_QUOTE_RE.finditer(text):
        colon_pos = m.start()

        # Guard: skip colons that are part of a time or ratio.
        if colon_pos > 0 and text[colon_pos - 1].isdigit():
            continue

        # m.end() is right after the colon and whitespace — the opening
        # quote character is the very next character in the text.
        split_points.add(m.end())

    # Sort the split points, then turn adjacent pairs into (start, end) spans.
    split_points = sorted(split_points)

    segments = []
    for i, start in enumerate(split_points):
        end = split_points[i + 1] if i + 1 < len(split_points) else len(text)
        if start < end:                 # defensive: skip zero-length spans
            segments.append({"text": text[start:end], "start": start, "end": end})

    return segments


# ───────────────────────────────────────────────────────────────────────────────
# LAYER 2  —  Clause extraction  (benepar constituency parse)
# ───────────────────────────────────────────────────────────────────────────────
#
# What is a constituency parse?
# ------------------------------
# A constituency parser turns a sentence into a tree where every node is a
# labelled phrase.  Leaf nodes are individual words; internal nodes are
# phrases (NP, VP, PP, …) or clauses (S, SBAR, …).
#
# Clause-level labels we track:
#   S    — declarative clause
#   SBAR — subordinate clause, usually introduced by a complementiser
#           ("that", "which", "because", "whether", …)
#   SINV — inverted clause ("Said the minister:")
#   SQ   — interrogative clause
#
# Two-pass strategy
# -----------------
# benepar is a sentence-level parser: it has no inter-sentence context and
# processes each sentence independently regardless of whether it receives one
# sentence or many.  We exploit this by making the sentence boundary explicit:
#
#   Pass 1 — spaCy alone (benepar disabled) detects sentence boundaries
#             using its dependency parser.
#   Pass 2 — benepar is run on each sentence individually to find the
#             clause-level S/SBAR nodes that mark attribution boundaries.
#
# Running benepar per-sentence also avoids a Python 3.14 retokenizer bug
# that fires when benepar processes multi-sentence input (the retokenizer
# exhausts its offset iterator before the last word and raises AssertionError).
# Sentences where benepar still fails are returned unsplit; any speaker
# change within them will be missed downstream.
#
# Trailing-punctuation absorption
# --------------------------------
# A constituency span ends at the last WORD of the clause; the comma (and
# any closing quotation mark) that follows it in the text is NOT included in
# the span.  Without intervention this punctuation ends up at the start of
# the NEXT segment, producing awkward comma-leading fragments such as:
#
#   ',' says Anders Andersen and continues:'
#   ', the council's recommendations are clear.'
#
# To prevent this, after every extracted clause we absorb the immediately
# following "punctuation bridge" — a comma, an optional closing quote, and
# an optional space — into the clause's end.  The next segment then starts
# cleanly on the first word of real content.
#
# Offset translation
# ------------------
# Pass 1 runs on each Layer-1 segment's text, so sentence start_char/end_char
# are relative to that segment.  Pass 2 runs on each sentence's text in
# isolation, so benepar's span.start_char/end_char are relative to that
# sentence.  To produce offsets into the ORIGINAL full text we add
# sent_base = segment['start'] + sent.start_char to every benepar offset.

CLAUSE_LABELS = {"S", "SBAR", "SINV", "SQ"}

# Characters that count as closing quotation marks for the trailing-bridge
# absorption rule.
_CLOSING_QUOTES_BRIDGE = {'"', "'", "”", "’"}


def _advance_past_trailing_punct(text, pos, end):
    """
    From position `pos` in `text`, step over the punctuation "bridge" that
    typically closes a clause in the source text.

    The bridge has up to three parts (each optional except the comma):
      1. A comma — the standard separator after a subordinate clause.
      2. A closing quotation mark — closes an embedded direct quote.
      3. A single space — separates the bridge from the next clause.

    Only a comma at `pos` triggers absorption.  If `text[pos]` is not a
    comma, the function returns `pos` unchanged.

    Examples
    --------
      "...the 1.5-degree target," says X"   →  pos lands on 's'
      "...the EU objective, the council..."  →  pos lands on 't'
      "...protection schemes."              →  pos unchanged (no comma)
    """
    if pos >= end or text[pos] != ',':
        return pos          # no comma → nothing to absorb
    pos += 1                # step over the comma
    if pos < end and text[pos] in _CLOSING_QUOTES_BRIDGE:
        pos += 1            # step over an immediately following closing quote
    if pos < end and text[pos] == ' ':
        pos += 1            # step over the space that separates the bridge
    return pos


def _direct_embedded_clauses(sent):
    """
    Return clause-labelled spans that are direct children of the sentence root.

    benepar stores every parse-tree node in `sent._.constituents` as a flat
    list of (possibly overlapping) Span objects.  We filter in two steps:

    Step 1 — keep clause-labelled spans strictly smaller than the full sentence
             (the root S node, which spans the whole sentence, is excluded).
    Step 2 — from those, keep only spans NOT enclosed by any other clause span
             in the list.  These are the outermost embedded clauses — the ones
             directly beneath the root — which are the attribution boundaries.
    """
    all_spans = list(sent._.constituents)

    # Step 1
    clausal = [
        span for span in all_spans
        if any(label in CLAUSE_LABELS for label in span._.labels)
        and not (span.start == sent.start and span.end == sent.end)
    ]

    # Step 2  —  "other strictly contains span" means:
    #   other starts at or before span AND ends at or after span,
    #   but is not the identical span.
    direct = [
        span for span in clausal
        if not any(
            other.start <= span.start
            and other.end >= span.end
            and not (other.start == span.start and other.end == span.end)
            for other in clausal
            if other is not span
        )
    ]

    return sorted(direct, key=lambda s: s.start_char)


def extract_clauses(segment, nlp, failures=None):
    """
    Apply the constituency parser to one coarse segment and return fine-grained
    clause-level segments, with offsets back into the original full text.

    Two passes are made:
      1. spaCy alone (benepar disabled) detects sentence boundaries.
      2. benepar is run on each sentence individually to find clause splits.
         Sentences where benepar fails are returned unsplit; any speaker
         change within them will be missed downstream.

    `segment` — dict {'text', 'start', 'end'} from Layer 1.
    `nlp`     — spaCy pipeline with benepar added.
    """
    base = segment["start"]

    # Pass 1 — sentence detection without benepar.
    with nlp.select_pipes(disable=["benepar"]):
        spacy_doc = nlp(segment["text"])

    result = []
    cursor = 0

    def extend_last_segment(end):
        """Extend the last segment to `end` (offset in the coarse segment), to cover
        the end of a sentence that benepar did not see, such as trailing whitespace."""
        last = result[-1]
        if last["end"] < base + end:
            last["text"] += segment["text"][last["end"] - base:end]
            last["end"] = base + end

    for sent in spacy_doc.sents:

        # Whitespace between consecutive sentences within this segment.
        if sent.start_char > cursor:
            result.append({
                "text":  segment["text"][cursor:sent.start_char],
                "start": base + cursor,
                "end":   base + sent.start_char,
            })

        sent_text = segment["text"][sent.start_char:sent.end_char]
        sent_base = base + sent.start_char
        # benepar fails on every sentence that ends in whitespace, such as the last
        # sentence of a paragraph, so it parses the sentence without it. The
        # whitespace is added back to the sentence's last segment below.
        parse_text = sent_text.rstrip()

        if not parse_text:
            # A whitespace-only sentence: nothing to parse.
            result.append({"text": sent_text, "start": sent_base, "end": sent_base + len(sent_text)})
            cursor = sent.end_char
            continue

        # Pass 2 — benepar constituency parse for this sentence.
        try:
            benepar_doc = nlp(parse_text)
            # benepar returns exactly one sentence when given one sentence.
            benepar_sent = next(iter(benepar_doc.sents))
        except (AssertionError, ValueError, StopIteration):
            # benepar fails on some sentences: AssertionError (tokenizer offset
            # mismatch), ValueError (more than 512 sub-word tokens), StopIteration
            # (no sentence returned). Return this sentence unsplit; any speaker
            # change inside it will be missed downstream.
            if failures is not None:
                failures.append({"text": sent_text, "start": sent_base, "end": sent_base + len(sent_text)})
            result.append({
                "text":  sent_text,
                "start": sent_base,
                "end":   sent_base + len(sent_text),
            })
            cursor = sent.end_char
            continue

        embedded = _direct_embedded_clauses(benepar_sent)

        if not embedded:
            # No embedded clause found: the whole sentence is one segment.
            result.append({
                "text":  sent_text[benepar_sent.start_char:benepar_sent.end_char],
                "start": sent_base + benepar_sent.start_char,
                "end":   sent_base + benepar_sent.end_char,
            })
            extend_last_segment(sent.end_char)
            cursor = sent.end_char
            continue

        # Walk through the direct embedded clauses in order, emitting:
        #   • the text before the first clause (reporting verb, subject, …)
        #   • each clause, extended to absorb its trailing punctuation bridge
        #   • gaps between consecutive clauses (e.g. "and", "but")
        #   • trailing text after the last clause (period, closing quote, …)
        inner = benepar_sent.start_char

        for clause in embedded:
            if clause.start_char > inner:           # text before this clause
                result.append({
                    "text":  sent_text[inner:clause.start_char],
                    "start": sent_base + inner,
                    "end":   sent_base + clause.start_char,
                })

            clause_end = _advance_past_trailing_punct(
                sent_text, clause.end_char, benepar_sent.end_char
            )

            result.append({                         # the clause itself
                "text":  sent_text[clause.start_char:clause_end],
                "start": sent_base + clause.start_char,
                "end":   sent_base + clause_end,
            })
            inner = clause_end

        if inner < benepar_sent.end_char:           # trailing punctuation, etc.
            result.append({
                "text":  sent_text[inner:benepar_sent.end_char],
                "start": sent_base + inner,
                "end":   sent_base + benepar_sent.end_char,
            })

        extend_last_segment(sent.end_char)
        cursor = sent.end_char

    # Content after the last sentence within this segment (edge case).
    if cursor < len(segment["text"]):
        result.append({
            "text":  segment["text"][cursor:],
            "start": base + cursor,
            "end":   base + len(segment["text"]),
        })

    return result


# ───────────────────────────────────────────────────────────────────────────────
# LAYER 3  —  Micro-segment merging
# ───────────────────────────────────────────────────────────────────────────────
#
# Problem
# -------
# Layer 2 sometimes isolates very short fragments — a lone space, a period,
# a closing quotation mark, a ", " pair — that carry no independent attribution
# meaning.  These are better absorbed into a neighbouring segment.
#
# Definition
# ----------
# A "micro-segment" is any segment whose text is 2 characters or shorter.
#
# Merge direction
# ---------------
# The default is to absorb a micro into the PREVIOUS segment, because most
# micro-fragments are tail punctuation (a period or closing quote that ends
# the preceding content).
#
# The one exception is an OPENING quotation mark: it starts a new quoted
# voice and semantically belongs to the quoted content that follows it, so
# it should be prepended to the NEXT segment instead.
#
# Distinguishing opening from closing quotation marks
# ----------------------------------------------------
# Unicode "curly" quotes are directional — unambiguous:
#   Opening:  " (U+201C)  ' (U+2018)
#   Closing:  " (U+201D)  ' (U+2019)
#
# ASCII straight quotes (" and ') are ambiguous.  Heuristic: look at the
# character immediately before the micro in the original text.  If that
# character is a letter or sentence-ending punctuation (.!?), the quote is
# almost certainly closing (it ends the content that just finished).
# Otherwise it is treated as opening.
#
# Algorithm — two passes on the pre-marked micro list
# ----------------------------------------------------
# All micro-segments are identified BEFORE any merging begins.  This matters
# when two micros sit next to each other: each gets its direction decided
# independently, rather than the second one seeing an already-grown first
# neighbour and incorrectly staying put.
#
#   Pass 1  (left → right):  opening-quote micros are prepended to the next
#           segment (accumulated as a "pending prefix" until a non-opening-
#           quote segment is reached).
#
#   Pass 2  (left → right):  all remaining micros are appended to the
#           previous segment.
#
# Edge cases
# ----------
# • Opening-quote micro at the END of the list — no next segment exists.
#   Fall back: attach to the previous segment instead.
# • Any micro at the START of the list after Pass 2 — no previous exists.
#   Fall back: prepend to the next segment.

_OPEN_QUOTES  = {'“', '‘'}   # " and ' (Unicode opening curly quotes)
_CLOSE_QUOTES = {'”', '’'}   # " and ' (Unicode closing curly quotes)


def _is_opening_quote_micro(seg, original_text):
    """
    Return True if this micro-segment is an OPENING quotation mark that
    should be glued to the next segment rather than the previous one.
    """
    text = seg["text"]
    has_open  = any(ch in _OPEN_QUOTES  for ch in text)
    has_close = any(ch in _CLOSE_QUOTES for ch in text)

    # Unicode directional quotes: unambiguous.
    if has_open and not has_close:
        return True
    if has_close:
        return False   # closing quote (possibly mixed with a period, e.g. '."')

    # ASCII straight quotes only: use context from the original text.
    if any(ch in ('"', "'") for ch in text):
        pos = seg["start"]
        # A closing quote follows a word character or sentence-ending punctuation.
        if pos > 0 and (original_text[pos - 1].isalpha() or original_text[pos - 1] in ".!?"):
            return False   # likely closing → glue to previous
        return True        # likely opening → glue to next

    return False           # not a quote at all → glue to previous (Pass 2)


def merge_microsegments(segments, original_text):
    """
    Absorb micro-segments (text length ≤ 2) into their neighbours.

    Pass 1: opening-quote micros are prepended to the following segment.
    Pass 2: all remaining micros are appended to the preceding segment.

    The lossless guarantee is maintained: each merged segment's text is a
    simple concatenation, and its start/end span the full merged range.
    """
    # ── Pass 1: opening-quote micros → glue to next ───────────────────────────────────────
    # We walk left to right, accumulating opening-quote micros in `pending`.
    # As soon as we reach a segment that is NOT an opening-quote micro, we
    # flush the pending list by prepending it to that segment.
    pass1   = []
    pending = []   # opening-quote micros not yet attached to a host segment

    for seg in segments:
        if len(seg["text"]) <= 2 and _is_opening_quote_micro(seg, original_text):
            pending.append(seg)
        else:
            if pending:
                all_segs = pending + [seg]
                pass1.append({
                    "text":  "".join(s["text"] for s in all_segs),
                    "start": all_segs[0]["start"],
                    "end":   all_segs[-1]["end"],
                })
                pending = []
            else:
                pass1.append(seg)

    # Fallback: opening-quote micro(s) at the very end of the list (no next
    # segment exists).  Attach them to the last result segment instead.
    if pending:
        extra = "".join(s["text"] for s in pending)
        if pass1:
            prev = pass1[-1]
            pass1[-1] = {
                "text":  prev["text"] + extra,
                "start": prev["start"],
                "end":   pending[-1]["end"],
            }
        else:
            # Degenerate: the entire input consisted of opening-quote micros.
            pass1.append({
                "text":  extra,
                "start": pending[0]["start"],
                "end":   pending[-1]["end"],
            })

    # ── Pass 2: remaining micros → glue to previous ──────────────────────────────────────
    # Walk left to right.  When a micro is encountered and there is already a
    # segment in the result, merge it into that last segment.
    pass2 = []
    for seg in pass1:
        if len(seg["text"]) <= 2 and pass2:
            prev = pass2[-1]
            pass2[-1] = {
                "text":  prev["text"] + seg["text"],
                "start": prev["start"],
                "end":   seg["end"],
            }
        else:
            pass2.append(seg)

    # Fallback: the very first segment is still micro after Pass 2 (there was
    # no previous to absorb it).  Prepend it to the next segment.
    if len(pass2) >= 2 and len(pass2[0]["text"]) <= 2:
        merged_text = pass2[0]["text"] + pass2[1]["text"]
        pass2 = [{
            "text":  merged_text,
            "start": pass2[0]["start"],
            "end":   pass2[1]["end"],
        }] + pass2[2:]

    return pass2


# ───────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ───────────────────────────────────────────────────────────────────────────────

def segment_text(text, nlp, coarse, failures=None):
    """
    Run layers 2–3 on pre-computed coarse segments and return fine-grained
    segment dicts.

    coarse — output of discourse_split() (layers 0+1, computed by the caller)

    Each returned dict has: {'id': int, 'text': str, 'start': int, 'end': int}

    Guarantee: ''.join(seg['text'] for seg in result) == text
    """
    fine = []
    # Sentences benepar could not parse, which stay unsplit.
    benepar_failures = failures if failures is not None else []
    for seg in coarse:
        fine.extend(extract_clauses(seg, nlp, failures=benepar_failures))  # Layer 2
    if verbose and benepar_failures:
        print(f"benepar failed on {len(benepar_failures)} sentence(s) — those were left as single-sentence segments")
    raw = merge_microsegments(fine, text)          # Layer 3
    return [
        {"id": i, "text": seg["text"], "start": seg["start"], "end": seg["end"]}
        for i, seg in enumerate(raw)
    ]


def step_1_segment(text, nlp):
    """
    Run the full segmentation pipeline and return (segments, html_content).

    Layers 0+1 (forced-boundary detection and discourse split) are computed
    here and passed to segment_text, which handles layers 2+3.

    segments     — list of dicts: {'id': int, 'text': str, 'start': int, 'end': int}
    html_content — monitoring HTML string for the report
    """
    forced = find_forced_boundaries(text)          # Layer 0
    coarse = discourse_split(text, forced)         # Layer 1

    if verbose:
        # Give a sense of what is going on at this stage
        print(f"Forced boundaries from Layer 0: {forced}\n")
        print(f"Coarse segments from Layer 1: {len(coarse)}\n")
        for i, seg in enumerate(coarse):
            preview = seg["text"][:80].replace("\n", "↵")
            ellipsis = "…" if len(seg["text"]) > 80 else ""
            print(f"[{i:3d}] ({seg['start']:5d}–{seg['end']:5d})  {repr(preview + ellipsis)}")

    # Compute layers 2+3
    segments = segment_text(text, nlp, coarse)

    html_content = segmentation_monitoring_html(segments)
    return segments, html_content


def segmentation_monitoring_html(segments: list) -> str:
    method_description = (
        "The text is split into fine-grained segments using a four-layer pipeline. "
        "<ul><li><b>Layer 0</b> detects section headings that were merged into body prose during HTML scraping "
        "and forces a boundary before the following sentence. </li>"
        "<li><b>Layer 1</b> splits at discourse-level boundaries: colons that introduce direct quotations "
        "(e.g. <i>X said: “…”</i>), while ignoring scare quotes and embedded terms. </li>"
        "<li><b>Layer 2</b> applies a two-pass strategy per coarse segment: first spaCy’s dependency "
        "parser (without benepar) detects sentence boundaries; then benepar runs on each sentence "
        "individually to split further at syntactic clause boundaries—the S, SBAR, SINV, and SQ nodes "
        "directly beneath the sentence root—which mark where the attributed speaker can change "
        "mid-sentence. This design aims to mitigate issues when Benepar fails occasionally.</li>"
        "<li><b>Layer 3</b> absorbs any stray micro-fragments (≤2 chars, e.g. lone commas or quotes) "
        "into their neighbours. </li>"
        "</ul>"
        "The result is a lossless partition of the original text: concatenating all segments "
        "reconstructs the input exactly."
    )
    content = "".join(
        f'<mark class="seg{seg["id"] % 2}">{html_lib.escape(seg["text"])}</mark>'
        for seg in segments
    )
    return f"<article><h3>Method</h3>{method_description}</article><p>{content}</p>"
