from __future__ import annotations
import time
import html as html_lib
from pydantic import BaseModel
from karaage import llm

verbose = False


class Source(BaseModel):
    main: str
    other: list[str]
    segments: list[list[int]]  # list of [start, end] inclusive ranges


class AttributionResult(BaseModel):
    sources: list[Source]


# Step-level configuration
RESERVED_TOKENS_ATTRIBUTION = 2_500   # prompt framing + expected output

SYSTEM_PROMPT_ATTRIBUTION = """\
You analyze a text to identify who or what is the declared source of information in each segment. A source is a person, an organization, or a document.

The text is divided into numbered segments. For each source, you list the segment IDs containing what they said.

Important: when text contains an attribution like "X said", "according to Y", or "Z noted", attribute the surrounding quoted or reported claim to that source — not the segment that contains the words "X said" themselves.

Example: in `[5] The deal will close by March, [6] she confirmed.` — segment 5 is attributed to her, not segment 6.

For each source, provide:
- "main": canonical name
- "other": list of alternative names or references found in the text (may be empty)
- "segments": list of [start, end] inclusive segment ID ranges containing what the source said

Rules:
- Do not include the base author/narrator.
- Prefer a human as the source when one is named. When no human is named but a document, study, or report is identified, use that as the "main" name.
- All quoted or reported text must be attributed to a source. Use context to infer the source when not explicitly named.
- Ranges are inclusive: [3, 5] covers segments 3, 4, and 5. A single segment is [n, n].
- If a source's statements appear in non-contiguous parts, list each contiguous block as a separate range.
"""


ATTRIBUTION_SCHEMA = {
    "type": "object",
    "title": "AttributionResult",
    "properties": {
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "main": {"type": "string"},
                    "other": {"type": "array", "items": {"type": "string"}},
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                },
                "required": ["main", "other", "segments"],
            },
        }
    },
    "required": ["sources"],
}


def estimate_tokens(text: str) -> int:
    # ~4 characters per token for English (conservative)
    return len(text) // 4


def format_segments_range(segments: list, start: int, end: int) -> str:
    return "\n".join(
        f"[{seg['id']}] {seg['text']}"
        for seg in segments
        if start <= seg["id"] <= end
    )


def build_prompt(formatted_segments: str) -> str:
    return (
        "Read the text below and identify all speakers with their segment ranges. "
        "Each segment is labelled [ID].\n\n"
        f"{formatted_segments}"
    )


def find_split(segments: list, start: int, end: int) -> int:
    """Return a segment ID near the middle of [start, end] that ends a sentence."""
    ids = [seg["id"] for seg in segments if start <= seg["id"] <= end]
    mid_pos = len(ids) // 2
    sentence_enders = ('.', '!', '?', '."', '!"', '?"')
    search_radius = max(1, len(ids) // 4)

    for offset in range(0, search_radius + 1):
        for delta in ([0] if offset == 0 else [offset, -offset]):
            idx = mid_pos + delta
            if 0 <= idx < len(ids):
                candidate = ids[idx]
                text = segments[candidate]["text"].strip()
                if any(text.endswith(e) for e in sentence_enders):
                    return candidate
    return ids[mid_pos]  # fallback: exact middle regardless of punctuation


def merge_results(left: AttributionResult, right: AttributionResult) -> AttributionResult:
    """Merge two AttributionResult objects by combining sources with the same main name."""
    by_name: dict[str, Source] = {}
    for src in left.sources + right.sources:
        if src.main in by_name:
            existing = by_name[src.main]
            by_name[src.main] = Source(
                main=src.main,
                other=list(set(existing.other + src.other)),
                segments=existing.segments + src.segments,
            )
        else:
            by_name[src.main] = src
    return AttributionResult(sources=list(by_name.values()))


def run_attribution(segments: list, start: int, end: int, max_context_tokens: int) -> AttributionResult:
    formatted = format_segments_range(segments, start, end)
    prompt = build_prompt(formatted)
    estimated = estimate_tokens(SYSTEM_PROMPT_ATTRIBUTION + prompt) + RESERVED_TOKENS_ATTRIBUTION

    if verbose:
        print(f"  Segments [{start}–{end}]: ~{estimated:,} tokens estimated", end="")

    if estimated > max_context_tokens and end > start + 1:
        mid = find_split(segments, start, end)
        if verbose:
            print(f"  → splitting at segment {mid}")
        left = run_attribution(segments, start, mid, max_context_tokens)
        right = run_attribution(segments, mid + 1, end, max_context_tokens)
        return merge_results(left, right)

    if verbose:
        print("  → calling LLM", end="")

    # Cap the answer at what fits in the context: with no limit, a model caught in a
    # repetition loop keeps generating until the request times out.
    max_tokens = max(RESERVED_TOKENS_ATTRIBUTION,
                     max_context_tokens - estimate_tokens(SYSTEM_PROMPT_ATTRIBUTION + prompt))
    content = llm.complete_json(SYSTEM_PROMPT_ATTRIBUTION, prompt, ATTRIBUTION_SCHEMA,
                                name="AttributionResult", strict=False, max_tokens=max_tokens)
    return AttributionResult.model_validate_json(content)


def display_result(result: AttributionResult, segments: list) -> None:
    for src in result.sources:
        aliases = f"  (also: {', '.join(src.other)})" if src.other else ""
        ranges = ", ".join(f"[{s},{e}]" for s, e in src.segments)
        print(f"Source: {src.main}{aliases}")
        print(f"  Segments: {ranges}")
        if src.segments:
            first_start = src.segments[0][0]
            preview = segments[first_start]["text"][:70].strip() if first_start < len(segments) else ""
            print(f"  Preview:  \"{preview}…\"")
        print()


# ============================================================
# resolve_source_ranges
# ============================================================
# Cleans up the raw LLM output by resolving overlapping and
# near-adjacent ranges in four stages:
#
#   1. Same-source cleanup  — merge overlapping ranges and merge
#      ranges whose gap (in characters) is below merge_gap_threshold.
#   2. Partial-overlap fix   — for any two ranges from *different*
#      sources that partially overlap (neither contains the other),
#      choose the edit that drops the fewest characters, leaving the
#      pair either disjoint or properly nested.
#   3. Nesting resolution    — when a range from one source fully
#      contains a range from another, drop the outer (larger) range.
#      Tie-break by source index (lower index wins).
#   4. Validation            — assert that no cross-source overlaps
#      survive stages 1-3.

def resolve_source_ranges(
    result: AttributionResult,
    segments: list,
    merge_gap_threshold: int = 100,
) -> AttributionResult:
    """
    Parameters
    ----------
    result               : AttributionResult from the LLM pipeline.
    segments             : List of segment dicts with keys 'id', 'text', 'start', 'end'.
    merge_gap_threshold  : Same-source ranges whose intervening gap is
                           strictly less than this many characters are merged.

    Returns
    -------
    A new AttributionResult with cleaned-up, non-overlapping ranges.
    """

    # ------------------------------------------------------------------
    # Precompute per-segment character counts for fast range summation.
    # ------------------------------------------------------------------
    seg_lengths: dict[int, int] = {seg["id"]: len(seg["text"]) for seg in segments}

    def char_len(start: int, end: int) -> int:
        """Total characters in all segments [start, end] inclusive."""
        return sum(seg_lengths.get(i, 0) for i in range(start, end + 1))

    # ------------------------------------------------------------------
    # Working copy: sid -> list of mutable [start, end] pairs.
    # We never mutate the original AttributionResult.
    # ------------------------------------------------------------------
    source_ranges: dict[str, list[list[int]]] = {
        str(i): [list(r) for r in src.segments]
        for i, src in enumerate(result.sources)
    }

    # ==================================================================
    # STAGE 1 — Same-source merge
    # ==================================================================
    # Two ranges of the same source are merged when they overlap,
    # are adjacent (gap of 0 segments), or the character count of the
    # gap between them is strictly less than merge_gap_threshold.

    def merge_same_source(ranges: list[list[int]]) -> list[list[int]]:
        if not ranges:
            return []
        # Work left-to-right on a sorted copy.
        ordered = sorted(ranges, key=lambda r: r[0])
        merged: list[list[int]] = [ordered[0][:]]

        for cur in ordered[1:]:
            prev = merged[-1]
            # cur[0] <= prev[1]+1 covers both overlapping and directly adjacent.
            if cur[0] <= prev[1] + 1:
                prev[1] = max(prev[1], cur[1])
                continue
            # There is a real gap; check its character length.
            # The gap spans segments (prev[1]+1) through (cur[0]-1).
            gap_chars = char_len(prev[1] + 1, cur[0] - 1)
            if gap_chars < merge_gap_threshold:
                prev[1] = max(prev[1], cur[1])
            else:
                merged.append(cur[:])

        return merged

    for sid in source_ranges:
        source_ranges[sid] = merge_same_source(source_ranges[sid])

    # ==================================================================
    # STAGE 2 — Cross-source partial-overlap resolution
    # ==================================================================
    # A *partial* overlap between R1=[a,b] and R2=[c,d] (different
    # sources) is the case where one range starts first and ends inside
    # the other:
    #   a < c ≤ b < d   (R1 starts first)
    # or symmetrically c < a ≤ d < b.
    #
    # Label the three disjoint sub-ranges:
    #   L = [a, c-1]   — left part, in R1 only
    #   O = [c, b]     — overlap,   in both
    #   R = [b+1, d]   — right part, in R2 only
    #
    # Four one-shot edits resolve the partial overlap by omitting
    # exactly one contiguous piece:
    #   opt1 — drop L from R1 → R1 becomes [c,b]    → R1 nested inside R2
    #   opt2 — drop O from R1 → R1 becomes [a,c-1]  → disjoint
    #   opt3 — drop O from R2 → R2 becomes [b+1,d]  → disjoint
    #   opt4 — drop R from R2 → R2 becomes [c,b]    → R2 nested inside R1
    #
    # Selection criterion (sort key, ascending):
    #   1. Fewest characters dropped.
    #   2. Prefer disjoint over nested (is_nested = 0 < 1).
    #   3. Prefer trimming R2 over R1 (prefers_trim_r1 = 0 < 1).

    def resolve_one_partial_overlap(
        r1: list[int], r2: list[int]
    ) -> tuple[list[int], list[int]]:
        """
        Precondition: r1[0] < r2[0] ≤ r1[1] < r2[1]  (r1 starts first).
        Returns (new_r1, new_r2) after applying the best edit.
        """
        a, b = r1
        c, d = r2

        len_L = char_len(a, c - 1)   # chars removed by opt1
        len_O = char_len(c, b)        # chars removed by opt2 or opt3
        len_R = char_len(b + 1, d)   # chars removed by opt4

        # Each tuple: (dropped_chars, is_nested, prefers_trim_r1, new_r1, new_r2)
        options = [
            (len_L, 1, 1, [c, b],     list(r2)      ),  # opt1: drop L
            (len_O, 0, 1, [a, c - 1], list(r2)      ),  # opt2: drop O from R1
            (len_O, 0, 0, list(r1),   [b + 1, d]    ),  # opt3: drop O from R2
            (len_R, 1, 0, list(r1),   [c, b]        ),  # opt4: drop R
        ]
        options.sort(key=lambda o: (o[0], o[1], o[2]))
        _, _, _, new_r1, new_r2 = options[0]
        return new_r1, new_r2

    MAX_ITERS = 10_000
    for _it in range(MAX_ITERS):
        found_partial = False
        sids = list(source_ranges.keys())
        for i in range(len(sids)):
            if found_partial:
                break
            for j in range(i + 1, len(sids)):
                if found_partial:
                    break
                ranges1 = source_ranges[sids[i]]
                ranges2 = source_ranges[sids[j]]
                for ri, r1 in enumerate(ranges1):
                    if found_partial:
                        break
                    for rj, r2 in enumerate(ranges2):
                        a, b = r1
                        c, d = r2
                        if a < c <= b < d:
                            # R1 starts first.
                            nr1, nr2 = resolve_one_partial_overlap(r1, r2)
                            ranges1[ri] = nr1
                            ranges2[rj] = nr2
                            found_partial = True
                            break
                        elif c < a <= d < b:
                            # R2 starts first — call with args swapped.
                            nr2, nr1 = resolve_one_partial_overlap(r2, r1)
                            ranges1[ri] = nr1
                            ranges2[rj] = nr2
                            found_partial = True
                            break
        if not found_partial:
            break
    else:
        raise RuntimeError(
            f"Partial-overlap resolution did not converge after {MAX_ITERS} iterations."
        )

    # Ranges trimmed to start > end (degenerate) are discarded.
    for sid in source_ranges:
        source_ranges[sid] = [r for r in source_ranges[sid] if r[0] <= r[1]]

    # ==================================================================
    # STAGE 3 — Nesting resolution
    # ==================================================================
    # After stage 2, every cross-source pair of ranges is either
    # disjoint or nested (one fully inside the other).  When nested,
    # we keep the smaller range (inner, more specific) and discard the
    # outer (larger).
    #
    # Tie-break: if both ranges span the same number of characters
    # (e.g. they are identical), the range belonging to the source
    # with the lower numeric index is kept.
    #
    # We iterate because discarding an outer range can reveal a new
    # nesting pair that was previously hidden beneath it.

    for _it in range(MAX_ITERS):
        # Build a flat snapshot of all live ranges with precomputed char counts.
        live: list[tuple[str, int, int, int, int]] = []
        for sid, ranges in source_ranges.items():
            for ri, r in enumerate(ranges):
                live.append((sid, ri, r[0], r[1], char_len(r[0], r[1])))

        found_nesting = False
        for ii in range(len(live)):
            if found_nesting:
                break
            sid1, ri1, a1, b1, cl1 = live[ii]
            for jj in range(len(live)):
                if ii == jj:
                    continue
                sid2, ri2, a2, b2, cl2 = live[jj]
                if sid1 == sid2:
                    continue
                # Check: does range ii (sid1) fully contain range jj (sid2)?
                if a1 <= a2 and b2 <= b1:
                    # ii is the outer.  Keep the smaller; if equal keep lower index.
                    # outer_wins=True  → discard the inner (jj)
                    # outer_wins=False → discard the outer (ii)
                    outer_wins = cl1 < cl2 or (cl1 == cl2 and int(sid1) < int(sid2))
                    if outer_wins:
                        source_ranges[sid2][ri2] = [-1, -1]   # tombstone inner
                    else:
                        source_ranges[sid1][ri1] = [-1, -1]   # tombstone outer
                    found_nesting = True
                    break

        # Sweep out tombstones before the next iteration.
        for sid in source_ranges:
            source_ranges[sid] = [r for r in source_ranges[sid] if r[0] >= 0]

        if not found_nesting:
            break
    else:
        raise RuntimeError(
            f"Nesting resolution did not converge after {MAX_ITERS} iterations."
        )

    # ==================================================================
    # STAGE 4 — Validation
    # ==================================================================
    # Confirm that no two ranges from different sources share even a
    # single segment.  Raises AssertionError if any overlap remains.

    all_final: list[tuple[str, int, int]] = [
        (sid, r[0], r[1])
        for sid, ranges in source_ranges.items()
        for r in ranges
    ]
    for ii, (sid1, a1, b1) in enumerate(all_final):
        for sid2, a2, b2 in all_final[ii + 1:]:
            if sid1 == sid2:
                continue
            if max(a1, a2) <= min(b1, b2):
                raise AssertionError(
                    f"Overlap remains after resolution: "
                    f"source {sid1} [{a1},{b1}] overlaps source {sid2} [{a2},{b2}]"
                )

    # ==================================================================
    # Reconstruct AttributionResult
    # ==================================================================
    # Sources whose every range was discarded are omitted entirely.

    new_sources: list[Source] = []
    for i, src in enumerate(result.sources):
        remaining = [r for r in source_ranges.get(str(i), []) if r[0] <= r[1]]
        if remaining:
            new_sources.append(Source(
                main=src.main,
                other=src.other,
                segments=remaining,
            ))
    return AttributionResult(sources=new_sources)


def extract_author_attribution(segments: list, max_context_tokens: int) -> AttributionResult:
    seg_ids = [seg["id"] for seg in segments]
    result = run_attribution(segments, seg_ids[0], seg_ids[-1], max_context_tokens)
    return resolve_source_ranges(result, segments)


def step_2_attribute_authors(
    segments: list,
    model: str,
    max_context_tokens: int,
) -> tuple[AttributionResult, str]:
    if verbose:
        print("\nAttributionResult schema:\n")
        print(AttributionResult.model_json_schema())

    sources = extract_author_attribution(segments, max_context_tokens)

    if verbose:
        print(f"Results ({len(sources.sources)} sources):\n")
        display_result(sources, segments)
    else:
        print(f"Attribution -> {len(sources.sources)} author{'s' if len(sources.sources) != 1 else ''}/source{'s' if len(sources.sources) != 1 else ''} identified.")

    html_content = attribution_monitoring_html(sources, segments, model)
    return sources, html_content


def range_resolution_test():
    # ============================================================
    # Tests for resolve_speaker_ranges
    # ============================================================
    # All test segments are exactly 10 characters long, so:
    #   char_len(a, b) == (b - a + 1) * 10
    # This makes expected character counts trivial to verify by hand.

    _S = "XXXXXXXXXX"  # 10-char placeholder
    test_segments = [{"id": i, "text": _S, "start": i * 10, "end": (i + 1) * 10 - 1} for i in range(150)]

    # ------------------------------------------------------------------
    # Craft a set of speakers that exercises every branch of the algorithm.
    # ------------------------------------------------------------------
    _test_speakers = {

        # ---- STAGE 1a: same-speaker gap merge (gap < threshold) --------
        # Gap between [0,3] and [5,9] is segment 4 alone = 10 chars < 100.
        # Expected: merged to [[0,9]]
        "0": Source(main="Alice",  other=[], segments=[[0, 3], [5, 9]]),

        # ---- STAGE 1b: same-speaker overlap merge ----------------------
        # [20,27] and [23,30] overlap at segments 23-27.
        # Expected: merged to [[20,30]]
        "1": Source(main="Bob",    other=[], segments=[[20, 27], [23, 30]]),

        # ---- STAGE 2: cross-speaker partial overlap --------------------
        # Carol=[40,48], Dave=[45,55]
        #   L = [40,44] = 5 segs = 50 chars
        #   O = [45,48] = 4 segs = 40 chars   ← minimum drop
        #   R = [49,55] = 7 segs = 70 chars
        # Both opt2 and opt3 drop O (40 chars, tie).  opt3 wins (prefer
        # trimming R2, key bit = prefers_trim_r1=0).
        # opt3: Carol stays [40,48], Dave → [49,55]
        "2": Source(main="Carol",  other=[], segments=[[40, 48]]),
        "3": Source(main="Dave",   other=[], segments=[[45, 55]]),

        # ---- STAGE 3: cross-speaker nesting, keep smaller --------------
        # Eve=[60,79] = 200 chars (outer), Frank=[65,70] = 60 chars (inner).
        # Frank is smaller → discard Eve.
        # Expected: Eve removed, Frank keeps [[65,70]]
        "4": Source(main="Eve",    other=[], segments=[[60, 79]]),
        "5": Source(main="Frank",  other=[], segments=[[65, 70]]),

        # ---- STAGE 3: multi-level nesting (three speakers) -------------
        # Grace=[80,99]=200ch > Henry=[83,96]=140ch > Iris=[86,92]=70ch
        # Iteration 1: Grace contains Henry → discard Grace (larger).
        # Iteration 2: Henry contains Iris  → discard Henry (larger).
        # Expected: Grace and Henry removed, Iris keeps [[86,92]]
        "6": Source(main="Grace",  other=[], segments=[[80, 99]]),
        "7": Source(main="Henry",  other=[], segments=[[83, 96]]),
        "8": Source(main="Iris",   other=[], segments=[[86, 92]]),

        # ---- STAGE 3: tie-break (identical ranges, lower index wins) ---
        # Kim and Leo both have [110,115] = 60 chars each.
        # Tie broken by list index: 9 < 10 → Kim wins.
        # Expected: Leo removed, Kim keeps [[110,115]]
        "9":  Source(main="Kim",   other=[], segments=[[110, 115]]),
        "10": Source(main="Leo",   other=[], segments=[[110, 115]]),

        # ---- STAGE 1c: same-speaker gap NOT merged (gap == threshold) --
        # Gap between [120,124] and [135,139] = segs 125-134 = 100 chars.
        # Condition is strictly < 100, so 100 chars == threshold → NOT merged.
        # Expected: [[120,124],[135,139]] unchanged
        "11": Source(main="Mike",  other=[], segments=[[120, 124], [135, 139]]),
    }

    test_input  = AttributionResult(sources=list(_test_speakers.values()))
    test_output = resolve_source_ranges(test_input, test_segments, merge_gap_threshold=100)

    # ------------------------------------------------------------------
    # Helper: return the ranges for a source by name, or None if removed.
    # ------------------------------------------------------------------
    def _ranges(res: AttributionResult, name: str):
        for src in res.sources:
            if src.main == name:
                return src.segments
        return None

    # ------------------------------------------------------------------
    # Assertions — one per scenario.
    # ------------------------------------------------------------------
    assert _ranges(test_output, "Alice")  == [[0, 9]],               f"Alice: {_ranges(test_output, 'Alice')}"
    assert _ranges(test_output, "Bob")    == [[20, 30]],              f"Bob: {_ranges(test_output, 'Bob')}"
    assert _ranges(test_output, "Carol")  == [[40, 48]],              f"Carol: {_ranges(test_output, 'Carol')}"
    assert _ranges(test_output, "Dave")   == [[49, 55]],              f"Dave: {_ranges(test_output, 'Dave')}"
    assert _ranges(test_output, "Eve")    is None,                    "Eve should have been removed"
    assert _ranges(test_output, "Frank")  == [[65, 70]],              f"Frank: {_ranges(test_output, 'Frank')}"
    assert _ranges(test_output, "Grace")  is None,                    "Grace should have been removed"
    assert _ranges(test_output, "Henry")  is None,                    "Henry should have been removed"
    assert _ranges(test_output, "Iris")   == [[86, 92]],              f"Iris: {_ranges(test_output, 'Iris')}"
    assert _ranges(test_output, "Kim")    == [[110, 115]],            f"Kim: {_ranges(test_output, 'Kim')}"
    assert _ranges(test_output, "Leo")    is None,                    "Leo should have been removed"
    assert _ranges(test_output, "Mike")   == [[120, 124], [135, 139]], f"Mike: {_ranges(test_output, 'Mike')}"

    print("All tests passed.\n")


SOURCE_COLORS = [
    "#ffd6a5",  # source01 — soft orange
    "#caffbf",  # source02 — soft green
    "#9bf6ff",  # source03 — soft cyan
    "#a0c4ff",  # source04 — soft blue
    "#bdb2ff",  # source05 — soft purple
    "#ffc6ff",  # source06 — soft pink
    "#fdffb6",  # source07 — soft yellow
    "#ffadad",  # source08 — soft red
]


def attribution_monitoring_html(
    sources: AttributionResult,
    segments: list,
    model: str,
) -> str:
    """Return attribution_content for the monitoring report."""
    method_description = (
        f"The fine-grained segments are sent to an LLM (<strong>{model}</strong>) "
        "that identifies every speaker or source "
        "quoted or referenced in the text and maps each one to the inclusive segment-ID ranges "
        "it authored or spoke. "
        "<ul>"
        "<li><b>Context-window splitting.</b> When the total prompt would exceed the token budget, "
        "the segment list is recursively bisected at a sentence boundary; "
        "the two halves are attributed independently and their results merged by matching source names.</li>"
        "<li><b>Range resolution (4 stages).</b> "
        "The goal is to ensure each segment is attributed to exactly 1 source "
        "(i.e. actor, by default the text author) as an acceptable simplification. "
        "(1) Same-source ranges that overlap or whose character gap falls below a threshold are merged. "
        "(2) Partial cross-source overlaps are resolved by dropping the fewest characters from one side. "
        "(3) When one source range fully contains another, the outer (larger) range is discarded. "
        "(4) The result is validated to confirm no cross-source overlap survives.</li>"
        "</ul>"
    )

    ordered_sources = sorted(
        list(getattr(sources, "sources", sources)),
        key=lambda src: min((start for start, _ in src.segments), default=float("inf"))
    )

    seg_to_class: dict[int, str] = {}
    seg_to_name: dict[int, str] = {}
    for rank, src in enumerate(ordered_sources, start=1):
        css_class = f"source{rank:02d}"
        for start, end in src.segments:
            for seg_id in range(start, end + 1):
                seg_to_class[seg_id] = css_class
                seg_to_name[seg_id] = src.main

    parts = []
    for seg in segments:
        seg_id = seg["id"]
        text = html_lib.escape(seg["text"])
        css_class = seg_to_class.get(seg_id)
        if css_class:
            title = html_lib.escape(seg_to_name[seg_id])
            parts.append(f'<mark class="{css_class}" title="{title}">{text}</mark>')
        else:
            parts.append(f'<span>{text}</span>')
    marked_text = "".join(parts)

    legend_items = []
    for rank, src in enumerate(ordered_sources, start=1):
        css_class = f"source{rank:02d}"
        also_names = " aka ".join(f'"{html_lib.escape(n)}"' for n in src.other)
        also = f' <span>aka {also_names}</span>' if src.other else ""
        legend_items.append(
            f'  <li>'
            f'<mark class="{css_class}"><strong>{html_lib.escape(src.main)}</strong></mark>{also}'
            f'</li>'
        )
    legend_html = "\n".join(legend_items)

    attribution_content = f"""
<article><h3>Method</h3>{method_description}</article>

<h3>Speakers</h3>
<ul>
{legend_html}
</ul>

<p>{marked_text}</p>"""

    return attribution_content
