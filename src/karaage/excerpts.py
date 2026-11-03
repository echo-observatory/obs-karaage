import html as html_lib
from karaage.attribution import ACTOR_COLORS

verbose = False

# Step-level configuration
EXCERPT_MAX_CHARS = 2000


def build_actor_map(segments, attribution):
    """
    Return a dict mapping every segment index (int) to its actor's canonical name (str).

    Attribution ranges are inclusive [start, end] pairs. Segments not covered by
    any range are assigned 'base_author' — the text's own voice, which is
    treated as a first-class actor throughout the analysis rather than as
    neutral background text.
    """
    actor_map = {}
    for actor in getattr(attribution, "actors", attribution):
        for start, end in actor.segments:
            for i in range(start, end + 1):
                actor_map[i] = actor.canonical_name
    for seg in segments:
        if seg["id"] not in actor_map:
            actor_map[seg["id"]] = 'base_author'
    return actor_map


def split_into_excerpts(segments, actor_map, min_chars=200, max_chars=3000, hhi_threshold=65):
    """
    Partition segments into excerpts using a window-based HHI scan.

    Parameters
    ----------
    segments : list[dict]
        Ordered list of segment dicts with keys 'id', 'text', 'start', 'end'.
    actor_map : dict[int, str]
        Segment index → actor name (from build_actor_map).
    min_chars : int
        Minimum excerpt length before a cut is considered.
    max_chars : int
        Hard ceiling: an excerpt is sealed immediately upon reaching this length.
    hhi_threshold : float
        Minimum HHI (0–100) for the primary cut rule.
        65 ≈ dominant actor holds ~89 % of chars in a two-actor excerpt.

    Returns
    -------
    list[dict]
        Each excerpt dict contains: 'text', 'segments', 'actor_chars',
        'actor_pcts', 'dominant_actor', 'dominant_ratio', 'hhi'.

    Algorithm
    ---------
    For each new excerpt start, accumulate segments one by one and record the
    HHI at every cut point that has reached min_chars (up to max_chars).
    Then choose the cut by the following priority:
      1. Primary: the *largest* cut with HHI ≥ hhi_threshold.
         Rationale: once purity is good enough, prefer more context.
      2. Fallback: the cut with the best HHI overall (first occurrence on tie).
         Rationale: HHI tends to decrease as segments accumulate, so the first
         occurrence of the maximum is the smallest excerpt — accepting a compact
         but pure excerpt when no clean cut exists anywhere in the window.
    """
    def hhi(actor_chars):
        """HHI = 100 × Σ (cᵢ/C)²; ranges from 0 (fragmented) to 100 (single voice)."""
        total = sum(actor_chars.values())
        if total == 0:
            return 0
        return 100 * sum((c / total) ** 2 for c in actor_chars.values())

    def make_excerpt(segs, actor_chars):
        """Build the excerpt dict from an accumulated segment list and char counts."""
        text = ''.join(t for _, t, _ in segs)
        total = sum(actor_chars.values())
        dominant = max(actor_chars, key=actor_chars.get)
        # Sort actors by share descending so dominant appears first everywhere.
        actor_pcts = {
            actor: round(chars / total * 100, 1)
            for actor, chars in sorted(actor_chars.items(), key=lambda x: -x[1])
        }
        return {
            'text': text,
            'segments': [i for i, _, _ in segs],
            'actor_chars': dict(actor_chars),
            'actor_pcts': actor_pcts,
            'dominant_actor': dominant,
            'dominant_ratio': actor_chars[dominant] / total if total else 0,
            'hhi': hhi(actor_chars),
        }

    excerpts = []
    start_idx = 0  # index into segments list

    while start_idx < len(segments):
        segs = []
        total_chars = 0
        actor_chars = {}
        # Each entry: (hhi_score, ordered_index, segs_snapshot, sp_chars_snapshot).
        # Snapshots are taken only once min_chars is reached so we never cut too early.
        candidates = []

        for i in range(start_idx, len(segments)):
            seg = segments[i]
            idx = seg["id"]
            text = seg["text"]
            actor = actor_map[idx]

            segs.append((idx, text, actor))
            total_chars += len(text)
            actor_chars[actor] = actor_chars.get(actor, 0) + len(text)

            if total_chars >= min_chars:
                candidates.append((hhi(actor_chars), i, list(segs), dict(actor_chars)))

            if total_chars >= max_chars:
                break  # hard ceiling reached; seal with whatever is best among candidates

        if not candidates:
            # Tail of the article is shorter than min_chars; flush as a single excerpt.
            if segs:
                excerpts.append(make_excerpt(segs, actor_chars))
            break

        good = [(score, i, s, c) for score, i, s, c in candidates if score >= hhi_threshold]
        if good:
            # Primary rule: take the rightmost (largest) excerpt that meets the threshold.
            _, end_i, best_segs, best_sp = good[-1]
        else:
            # Fallback: take the excerpt with the highest HHI regardless of threshold.
            # max() returns the first element that achieves the maximum, so ties resolve
            # to the smallest excerpt — consistent with the fallback rationale above.
            _, end_i, best_segs, best_sp = max(candidates, key=lambda x: x[0])

        excerpts.append(make_excerpt(best_segs, best_sp))
        start_idx = end_i + 1

    return excerpts


def optimize_excerpts(excerpts, segments, actor_map, min_chars=200, max_chars=3000):
    """
    Refine excerpt boundaries by hill-climbing on the length-weighted HHI sum.

    Objective
    ---------
    Maximise Σᵢ |Pᵢ| · HHI(Pᵢ), where |Pᵢ| is the character length of excerpt i.
    Because total article length is constant, this is equivalent to maximising the
    length-weighted *average* HHI — rewarding purity where there is more text.

    Moves
    -----
    At each boundary between adjacent excerpts (i, i+1) two moves are considered:
      • shift right: move the last segment of excerpt i to the front of excerpt i+1.
      • shift left:  move the first segment of excerpt i+1 to the end of excerpt i.
    Only boundary moves are considered; this preserves text order and the total
    number of excerpts. A move is accepted only when both resulting excerpts remain
    within [min_chars, max_chars].

    Termination
    -----------
    The algorithm repeats until no single boundary move improves the objective.
    Convergence is guaranteed: the objective strictly increases with each accepted
    move and is bounded above by 100 × (total chars).

    Limitation
    ----------
    Hill-climbing finds a *local* optimum. Improvements requiring two simultaneous
    boundary shifts are not detected. This is an acceptable trade-off for the
    typical excerpt counts and segment lengths seen here.
    """
    def compute_hhi(actor_chars):
        total = sum(actor_chars.values())
        if total == 0:
            return 0
        return 100 * sum((c / total) ** 2 for c in actor_chars.values())

    def make_excerpt(seg_indices):
        """Rebuild a full excerpt dict from a list of segment indices."""
        text = ''.join(segments[i]["text"] for i in seg_indices)
        actor_chars = {}
        for i in seg_indices:
            actor = actor_map[i]
            actor_chars[actor] = actor_chars.get(actor, 0) + len(segments[i]["text"])
        total = sum(actor_chars.values())
        dominant = max(actor_chars, key=actor_chars.get)
        actor_pcts = {
            actor: round(chars / total * 100, 1)
            for actor, chars in sorted(actor_chars.items(), key=lambda x: -x[1])
        }
        return {
            'text': text,
            'segments': list(seg_indices),
            'actor_chars': actor_chars,
            'actor_pcts': actor_pcts,
            'dominant_actor': dominant,
            'dominant_ratio': actor_chars[dominant] / total if total else 0,
            'hhi': compute_hhi(actor_chars),
        }

    def weighted_score(p):
        """Contribution of one excerpt to the global objective."""
        return len(p['text']) * p['hhi']

    # Work on mutable lists of segment indices; excerpt dicts are rebuilt at the end.
    seg_lists = [list(p['segments']) for p in excerpts]
    moves_count = 0
    improved = True

    while improved:
        improved = False
        best_delta = 0
        best_move = None  # (boundary_index, new_left_seg_list, new_right_seg_list)

        for b in range(len(seg_lists) - 1):
            left, right = seg_lists[b], seg_lists[b + 1]
            left_p, right_p = make_excerpt(left), make_excerpt(right)
            current = weighted_score(left_p) + weighted_score(right_p)

            # Candidate A: shift last segment of left → beginning of right.
            if len(left) > 1:
                nl, nr = make_excerpt(left[:-1]), make_excerpt([left[-1]] + right)
                if min_chars <= len(nl['text']) <= max_chars and min_chars <= len(nr['text']) <= max_chars:
                    delta = weighted_score(nl) + weighted_score(nr) - current
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (b, left[:-1], [left[-1]] + right)

            # Candidate B: shift first segment of right → end of left.
            if len(right) > 1:
                nl, nr = make_excerpt(left + [right[0]]), make_excerpt(right[1:])
                if min_chars <= len(nl['text']) <= max_chars and min_chars <= len(nr['text']) <= max_chars:
                    delta = weighted_score(nl) + weighted_score(nr) - current
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (b, left + [right[0]], right[1:])

        if best_move:
            b, new_left, new_right = best_move
            seg_lists[b], seg_lists[b + 1] = new_left, new_right
            improved = True
            moves_count += 1
    if verbose:
        print(f"Optimization: {moves_count} boundary move(s) applied.")
    return [make_excerpt(sl) for sl in seg_lists]


def weighted_hhi(excerpts):
    """Length-weighted average HHI across all excerpts (the optimization objective)."""
    total_chars = sum(len(p['text']) for p in excerpts)
    return sum(len(p['text']) * p['hhi'] for p in excerpts) / total_chars


def step_3_excerpts(segments: list, attribution) -> list[dict]:
    actor_map = build_actor_map(segments, attribution)
    excerpts = split_into_excerpts(segments, actor_map, max_chars=EXCERPT_MAX_CHARS)
    if verbose:
        print(f"Initial weighted HHI: {weighted_hhi(excerpts):.1f}")
    excerpts = optimize_excerpts(excerpts, segments, actor_map, max_chars=EXCERPT_MAX_CHARS)
    if verbose:
        print(f"Final   weighted HHI: {weighted_hhi(excerpts):.1f}")

    if verbose:
        print()
        for i, p in enumerate(excerpts):
            actors_str = ', '.join(f'{actor}: {pct}%' for actor, pct in p['actor_pcts'].items())
            seg_range = f"segs {p['segments'][0]}–{p['segments'][-1]}"
            print(f"--- Excerpt {i+1} [{seg_range}] | {len(p['text'])} chars | HHI: {p['hhi']:.1f} | {actors_str} ---")
            print(p['text'][:120], '...' if len(p['text']) > 120 else '')
            print()
    
    html_excerpts_content = excerpts_monitoring_html(excerpts, segments, attribution, EXCERPT_MAX_CHARS)

    return excerpts, html_excerpts_content


def excerpts_monitoring_html(
    excerpts: list,
    segments: list,
    attribution,
    excerpt_max_chars: int,
) -> str:
    """Return excerpts_content for the monitoring report."""
    method_description = (
        "The attributed segments are grouped into excerpts large enough to provide meaningful "
        "context for downstream LLM analysis, while keeping each excerpt dominated by a single actor. "
        "Actor purity is measured with the <b>Herfindahl–Hirschman Index (HHI)</b>, "
        "ranging from 0 (fully fragmented) to 100 (single voice). "
        "Segments not covered by any attribution are assigned to <i>base_author</i>, "
        "the text author's own voice, which is treated as a first-class actor. "
        "Splitting proceeds in two phases. "
        "<ul>"
        "<li><b>Phase 1 — Greedy initial split.</b> Segments are accumulated until the excerpt "
        "reaches <i>min_chars</i> (200). At every eligible cut point the HHI is recorded up to "
        f"<i>max_chars</i> ({excerpt_max_chars:,}). The cut is chosen by: "
        "(1) <i>primary rule</i> — the largest cut whose HHI ≥ 65 (corresponding to a dominant "
        "actor holding ~89 % of characters in a two-actor excerpt); "
        "(2) <i>fallback</i> — the cut with the highest HHI when no cut meets the threshold, "
        "which tends to be the smallest excerpt since HHI decreases as more actors accumulate.</li>"
        "<li><b>Phase 2 — Hill-climbing boundary refinement.</b> Every boundary between adjacent "
        "excerpts is examined; the single-segment shift to the other side if it improves the global objective. "
        "The process repeats until no improving move exists. "
        "Convergence is guaranteed because the objective strictly increases with each accepted move.</li>"
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

    seg_to_excerpt_idx: dict[int, int] = {}
    for i, p in enumerate(excerpts):
        for seg_id in p["segments"]:
            seg_to_excerpt_idx[seg_id] = i

    body_parts = []
    current_excerpt_idx = None
    for seg in segments:
        seg_id = seg["id"]
        excerpt_idx = seg_to_excerpt_idx.get(seg_id)
        if excerpt_idx != current_excerpt_idx:
            if current_excerpt_idx is not None:
                body_parts.append("</p>\n")
            p = excerpts[excerpt_idx]
            seg_range = f"{p['segments'][0]}–{p['segments'][-1]}"
            actors_str = " · ".join(
                f"{html_lib.escape(actor)}: {pct}%"
                for actor, pct in p["actor_pcts"].items()
            )
            info = f'HHI: {p["hhi"]:.1f}% — {actors_str}'
            body_parts.append(f"<h3>Excerpt {excerpt_idx + 1} [{seg_range}]</h3>\n")
            body_parts.append(f"<i>{info}</i>\n")
            body_parts.append("<p>")
            current_excerpt_idx = excerpt_idx
        text = html_lib.escape(seg["text"])
        css_class = seg_to_class.get(seg_id)
        if css_class:
            title = html_lib.escape(seg_to_name[seg_id])
            body_parts.append(f'<mark class="{css_class}" title="{title}">{text}</mark>')
        else:
            body_parts.append(f"<span>{text}</span>")
    if current_excerpt_idx is not None:
        body_parts.append("</p>\n")
    excerpts_html = "".join(body_parts)

    legend_items = []
    for rank, actor in enumerate(ordered_actors, start=1):
        css_class = f"actor{rank:02d}"
        also_names = " aka ".join(f'"{html_lib.escape(n)}"' for n in actor.aliases)
        also = f' <span>aka {also_names}</span>' if actor.aliases else ""
        legend_items.append(
            f'  <li>'
            f'<mark class="{css_class}"><strong>{html_lib.escape(actor.canonical_name)}</strong></mark>{also}'
            f'</li>'
        )
    legend_html = "\n".join(legend_items)

    excerpts_content = f"""
<article><h3>Method</h3>{method_description}</article>

<h3>Actors</h3>
<ul>
{legend_html}
</ul>

{excerpts_html}"""

    return excerpts_content
