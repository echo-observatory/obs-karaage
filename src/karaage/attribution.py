from __future__ import annotations
import time
import html as html_lib
from pydantic import BaseModel, ConfigDict, Field
from karaage import llm

verbose = False


# What the model sees calls actors "sources": the system prompt, and the answer format with
# its "main" and "other" names (the user prompt also says "speakers"). This wording is kept
# on purpose: "source" covers organisations and documents, where "speaker" suggests a person.
# The aliases read the answer into the names used everywhere else, those of the output
# (docs/design-1.1.0.md, section 2).

class Actor(BaseModel):
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)
    canonical_name: str = Field(alias="main")
    aliases: list[str] = Field(alias="other")
    kind: str | None = None    # one of ACTOR_KINDS
    segments: list[list[int]]  # list of [start, end] inclusive ranges


class AttributionResult(BaseModel):
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)
    actors: list[Actor] = Field(alias="sources")


# Step-level configuration
RESERVED_TOKENS_ATTRIBUTION = 2_500   # prompt framing + expected output
# Every list in the answer has a maximum, so that a model caught in a repetition loop
# has to close it. Loops seen on the test dataset: an alias list ("we", "I", "we", ...)
# and the list of actors itself. The largest legitimate lists seen are 28 actors,
# 26 ranges for one actor and 10 aliases.
MAX_ACTORS = 40
MAX_RANGES = 40
MAX_ALIASES = 10
# The outputs keep "person" and "organisation" (the snapshot specification's kinds) and
# give null for the others (docs/design-1.1.0.md, section 3.4).
ACTOR_KINDS = ["person", "organisation", "document", "unknown"]

SYSTEM_PROMPT_ATTRIBUTION = """\
You analyze a text to identify who or what is the declared source of information in each segment. A source is a person, an organization, or a document.

The text is divided into numbered segments. For each source, you list the segment IDs containing what they said.

Important: when text contains an attribution like "X said", "according to Y", or "Z noted", attribute the surrounding quoted or reported claim to that source — not the segment that contains the words "X said" themselves.

Example: in `[5] The deal will close by March, [6] she confirmed.` — segment 5 is attributed to her, not segment 6.

For each source, provide:
- "main": canonical name: the full name of a person, the official name of an organization, or the title of a document, as far as the text gives it
- "other": list of the other names and descriptions the text uses for the source (may be empty); do not include pronouns
- "kind": "person", "organisation", "document", or "unknown"
- "segments": list of [start, end] inclusive segment ID ranges containing what the source said

Rules:
- Do not include the base author/narrator.
- "other" lists names and descriptions only, never pronouns such as "I", "we", "she", "they" or "our".
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
            "maxItems": MAX_ACTORS,
            "items": {
                "type": "object",
                "properties": {
                    "main": {"type": "string"},
                    "other": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_ALIASES},
                    "kind": {"type": "string", "enum": ACTOR_KINDS},
                    "segments": {
                        "type": "array",
                        "maxItems": MAX_RANGES,
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                },
                "required": ["main", "other", "kind", "segments"],
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
    """Merge two AttributionResult objects by combining actors with the same canonical name."""
    by_name: dict[str, Actor] = {}
    for actor in left.actors + right.actors:
        if actor.canonical_name in by_name:
            existing = by_name[actor.canonical_name]
            by_name[actor.canonical_name] = Actor(
                canonical_name=actor.canonical_name,
                aliases=list(dict.fromkeys(existing.aliases + actor.aliases)),   # keeps the order, unlike a set
                kind=existing.kind or actor.kind,
                segments=existing.segments + actor.segments,
            )
        else:
            by_name[actor.canonical_name] = actor
    return AttributionResult(actors=list(by_name.values()))


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
    return normalize_answer(AttributionResult.model_validate_json(content))


def normalize_answer(result: AttributionResult) -> AttributionResult:
    """Repair what a looping model, or a backend that ignores the schema, may return.

    Every range becomes a [start, end] pair: [n] becomes [n, n], [] is dropped, and a
    longer list is read as a list of single segments. An actor listed several times is
    merged into one, and repeated aliases are kept once.
    """
    actors = []
    for actor in result.actors:
        ranges = []
        for r in actor.segments:
            if len(r) == 2:
                ranges.append(list(r))
            elif len(r) == 1:
                ranges.append([r[0], r[0]])
            elif len(r) > 2:
                ranges.extend([i, i] for i in r)
        actors.append(Actor(canonical_name=actor.canonical_name, aliases=list(dict.fromkeys(actor.aliases)), kind=actor.kind, segments=ranges))
    return merge_results(AttributionResult(actors=actors), AttributionResult(actors=[]))


def display_result(result: AttributionResult, segments: list) -> None:
    for actor in result.actors:
        aliases = f"  (also: {', '.join(actor.aliases)})" if actor.aliases else ""
        ranges = ", ".join(f"[{s},{e}]" for s, e in actor.segments)
        print(f"Actor: {actor.canonical_name}{aliases}")
        print(f"  Segments: {ranges}")
        if actor.segments:
            first_start = actor.segments[0][0]
            preview = segments[first_start]["text"][:70].strip() if first_start < len(segments) else ""
            print(f"  Preview:  \"{preview}…\"")
        print()


# ============================================================
# resolve_actor_ranges
# ============================================================
# Cleans up the raw LLM output by resolving overlapping and
# near-adjacent ranges in four stages:
#
#   1. Same-actor cleanup   — merge overlapping ranges and merge
#      ranges whose gap (in characters) is below merge_gap_threshold.
#   2. Partial-overlap fix   — for any two ranges from *different*
#      actors that partially overlap (neither contains the other),
#      choose the edit that drops the fewest characters, leaving the
#      pair either disjoint or properly nested.
#   3. Nesting resolution    — when a range from one actor fully
#      contains a range from another, drop the outer (larger) range.
#      Tie-break by actor index (lower index wins).
#   4. Validation            — assert that no cross-actor overlaps
#      survive stages 1-3.

def resolve_actor_ranges(
    result: AttributionResult,
    segments: list,
    merge_gap_threshold: int = 100,
) -> AttributionResult:
    """
    Parameters
    ----------
    result               : AttributionResult read from the model's answer.
    segments             : List of segment dicts with keys 'id', 'text', 'start', 'end'.
    merge_gap_threshold  : Same-actor ranges whose intervening gap is
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
    actor_ranges: dict[str, list[list[int]]] = {
        str(i): [list(r) for r in actor.segments]
        for i, actor in enumerate(result.actors)
    }

    # ==================================================================
    # STAGE 1 — Same-actor merge
    # ==================================================================
    # Two ranges of the same actor are merged when they overlap,
    # are adjacent (gap of 0 segments), or the character count of the
    # gap between them is strictly less than merge_gap_threshold.

    def merge_same_actor(ranges: list[list[int]]) -> list[list[int]]:
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

    for sid in actor_ranges:
        actor_ranges[sid] = merge_same_actor(actor_ranges[sid])

    # ==================================================================
    # STAGE 2 — Cross-actor partial-overlap resolution
    # ==================================================================
    # A *partial* overlap between R1=[a,b] and R2=[c,d] (different
    # actors) is the case where one range starts first and ends inside
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
        sids = list(actor_ranges.keys())
        for i in range(len(sids)):
            if found_partial:
                break
            for j in range(i + 1, len(sids)):
                if found_partial:
                    break
                ranges1 = actor_ranges[sids[i]]
                ranges2 = actor_ranges[sids[j]]
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
    for sid in actor_ranges:
        actor_ranges[sid] = [r for r in actor_ranges[sid] if r[0] <= r[1]]

    # ==================================================================
    # STAGE 3 — Nesting resolution
    # ==================================================================
    # After stage 2, every cross-actor pair of ranges is either
    # disjoint or nested (one fully inside the other).  When nested,
    # we keep the smaller range (inner, more specific) and discard the
    # outer (larger).
    #
    # Tie-break: if both ranges span the same number of characters
    # (e.g. they are identical), the range belonging to the actor
    # with the lower numeric index is kept.
    #
    # We iterate because discarding an outer range can reveal a new
    # nesting pair that was previously hidden beneath it.

    for _it in range(MAX_ITERS):
        # Build a flat snapshot of all live ranges with precomputed char counts.
        live: list[tuple[str, int, int, int, int]] = []
        for sid, ranges in actor_ranges.items():
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
                        actor_ranges[sid2][ri2] = [-1, -1]   # tombstone inner
                    else:
                        actor_ranges[sid1][ri1] = [-1, -1]   # tombstone outer
                    found_nesting = True
                    break

        # Sweep out tombstones before the next iteration.
        for sid in actor_ranges:
            actor_ranges[sid] = [r for r in actor_ranges[sid] if r[0] >= 0]

        if not found_nesting:
            break
    else:
        raise RuntimeError(
            f"Nesting resolution did not converge after {MAX_ITERS} iterations."
        )

    # ==================================================================
    # STAGE 4 — Validation
    # ==================================================================
    # Confirm that no two ranges from different actors share even a
    # single segment.  Raises AssertionError if any overlap remains.

    all_final: list[tuple[str, int, int]] = [
        (sid, r[0], r[1])
        for sid, ranges in actor_ranges.items()
        for r in ranges
    ]
    for ii, (sid1, a1, b1) in enumerate(all_final):
        for sid2, a2, b2 in all_final[ii + 1:]:
            if sid1 == sid2:
                continue
            if max(a1, a2) <= min(b1, b2):
                raise AssertionError(
                    f"Overlap remains after resolution: "
                    f"actor {sid1} [{a1},{b1}] overlaps actor {sid2} [{a2},{b2}]"
                )

    # ==================================================================
    # Reconstruct AttributionResult
    # ==================================================================
    # Actors whose every range was discarded are omitted entirely.

    new_actors: list[Actor] = []
    for i, actor in enumerate(result.actors):
        remaining = [r for r in actor_ranges.get(str(i), []) if r[0] <= r[1]]
        if remaining:
            new_actors.append(Actor(
                canonical_name=actor.canonical_name,
                aliases=actor.aliases,
                kind=actor.kind,
                segments=remaining,
            ))
    return AttributionResult(actors=new_actors)


def extract_actor_attribution(segments: list, max_context_tokens: int) -> AttributionResult:
    seg_ids = [seg["id"] for seg in segments]
    result = run_attribution(segments, seg_ids[0], seg_ids[-1], max_context_tokens)
    return resolve_actor_ranges(result, segments)


def step_2_attribute_actors(
    segments: list,
    model: str,
    max_context_tokens: int,
) -> tuple[AttributionResult, str]:
    if verbose:
        print("\nAttributionResult schema:\n")
        print(AttributionResult.model_json_schema())

    attribution = extract_actor_attribution(segments, max_context_tokens)

    n = len(attribution.actors)
    if verbose:
        print(f"Results ({n} actors):\n")
        display_result(attribution, segments)
    else:
        print(f"Attribution -> {n} actor{'s' if n != 1 else ''} identified.")

    html_content = attribution_monitoring_html(attribution, segments, model)
    return attribution, html_content


ACTOR_COLORS = [
    "#ffd6a5",  # actor01 — soft orange
    "#caffbf",  # actor02 — soft green
    "#9bf6ff",  # actor03 — soft cyan
    "#a0c4ff",  # actor04 — soft blue
    "#bdb2ff",  # actor05 — soft purple
    "#ffc6ff",  # actor06 — soft pink
    "#fdffb6",  # actor07 — soft yellow
    "#ffadad",  # actor08 — soft red
]


def attribution_monitoring_html(
    attribution: AttributionResult,
    segments: list,
    model: str,
) -> str:
    """Return attribution_content for the monitoring report."""
    method_description = (
        f"The fine-grained segments are sent to an LLM (<strong>{model}</strong>) "
        "that identifies every actor (person, organisation or document) "
        "quoted or referenced in the text and maps each one to the inclusive segment-ID ranges "
        "of what it said or wrote. "
        "<ul>"
        "<li><b>Context-window splitting.</b> When the total prompt would exceed the token budget, "
        "the segment list is recursively bisected at a sentence boundary; "
        "the two halves are attributed independently and their results merged by matching actor names.</li>"
        "<li><b>Range resolution (4 stages).</b> "
        "The goal is to ensure each segment is attributed to exactly 1 actor "
        "(by default base_author, the text's own voice) as an acceptable simplification. "
        "(1) Same-actor ranges that overlap or whose character gap falls below a threshold are merged. "
        "(2) Partial cross-actor overlaps are resolved by dropping the fewest characters from one side. "
        "(3) When one actor's range fully contains another's, the outer (larger) range is discarded. "
        "(4) The result is validated to confirm no cross-actor overlap survives.</li>"
        "</ul>"
    )

    ordered_actors = sorted(
        list(getattr(attribution, "actors", attribution)),
        key=lambda actor: min((start for start, _ in actor.segments), default=float("inf"))
    )

    seg_to_class: dict[int, str] = {}
    seg_to_name: dict[int, str] = {}
    for rank, actor in enumerate(ordered_actors, start=1):
        css_class = f"actor{rank:02d}"
        for start, end in actor.segments:
            for seg_id in range(start, end + 1):
                seg_to_class[seg_id] = css_class
                seg_to_name[seg_id] = actor.canonical_name

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
    for rank, actor in enumerate(ordered_actors, start=1):
        css_class = f"actor{rank:02d}"
        also_names = " aka ".join(f'"{html_lib.escape(n)}"' for n in actor.aliases)
        also = f' <span>aka {also_names}</span>' if actor.aliases else ""
        legend_items.append(
            f'  <li>'
            f'<mark class="{css_class}"><strong>{html_lib.escape(actor.canonical_name)}</strong></mark> ({html_lib.escape(str(actor.kind))}){also}'
            f'</li>'
        )
    legend_html = "\n".join(legend_items)

    attribution_content = f"""
<article><h3>Method</h3>{method_description}</article>

<h3>Actors</h3>
<ul>
{legend_html}
</ul>

<p>{marked_text}</p>"""

    return attribution_content
