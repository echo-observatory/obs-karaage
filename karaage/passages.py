import html as html_lib
from karaage.attribution import SOURCE_COLORS

verbose = False

# Step-level configuration
PASSAGE_MAX_CHARS = 2000


def build_speaker_map(segments, sources):
    """
    Return a dict mapping every segment index (int) to its speaker name (str).

    Attribution ranges in `sources` are inclusive [start, end] pairs. Segments
    not covered by any range are assigned 'base_author' — the journalist's own
    voice, which is treated as a first-class speaker throughout the analysis
    rather than as neutral background text.
    """
    speaker_map = {}
    for source_data in getattr(sources, "sources", sources):
        speaker = source_data.main
        for start, end in source_data.segments:
            for i in range(start, end + 1):
                speaker_map[i] = speaker
    for seg in segments:
        if seg["id"] not in speaker_map:
            speaker_map[seg["id"]] = 'base_author'
    return speaker_map


def split_into_passages(segments, speaker_map, min_chars=200, max_chars=3000, hhi_threshold=65):
    """
    Partition segments into passages using a window-based HHI scan.

    Parameters
    ----------
    segments : list[dict]
        Ordered list of segment dicts with keys 'id', 'text', 'start', 'end'.
    speaker_map : dict[int, str]
        Segment index → speaker name (from build_speaker_map).
    min_chars : int
        Minimum passage length before a cut is considered.
    max_chars : int
        Hard ceiling: a passage is sealed immediately upon reaching this length.
    hhi_threshold : float
        Minimum HHI (0–100) for the primary cut rule.
        65 ≈ dominant speaker holds ~89 % of chars in a two-speaker passage.

    Returns
    -------
    list[dict]
        Each passage dict contains: 'text', 'segments', 'speaker_chars',
        'speaker_pcts', 'dominant_speaker', 'dominant_ratio', 'hhi'.

    Algorithm
    ---------
    For each new passage start, accumulate segments one by one and record the
    HHI at every cut point that has reached min_chars (up to max_chars).
    Then choose the cut by the following priority:
      1. Primary: the *largest* cut with HHI ≥ hhi_threshold.
         Rationale: once purity is good enough, prefer more context.
      2. Fallback: the cut with the best HHI overall (first occurrence on tie).
         Rationale: HHI tends to decrease as segments accumulate, so the first
         occurrence of the maximum is the smallest passage — accepting a compact
         but pure passage when no clean cut exists anywhere in the window.
    """
    def hhi(speaker_chars):
        """HHI = 100 × Σ (cᵢ/C)²; ranges from 0 (fragmented) to 100 (single voice)."""
        total = sum(speaker_chars.values())
        if total == 0:
            return 0
        return 100 * sum((c / total) ** 2 for c in speaker_chars.values())

    def make_passage(segs, speaker_chars):
        """Build the passage dict from an accumulated segment list and char counts."""
        text = ''.join(t for _, t, _ in segs)
        total = sum(speaker_chars.values())
        dominant = max(speaker_chars, key=speaker_chars.get)
        # Sort speakers by share descending so dominant appears first everywhere.
        speaker_pcts = {
            spk: round(chars / total * 100, 1)
            for spk, chars in sorted(speaker_chars.items(), key=lambda x: -x[1])
        }
        return {
            'text': text,
            'segments': [i for i, _, _ in segs],
            'speaker_chars': dict(speaker_chars),
            'speaker_pcts': speaker_pcts,
            'dominant_speaker': dominant,
            'dominant_ratio': speaker_chars[dominant] / total if total else 0,
            'hhi': hhi(speaker_chars),
        }

    passages = []
    start_idx = 0  # index into segments list

    while start_idx < len(segments):
        segs = []
        total_chars = 0
        sp_chars = {}
        # Each entry: (hhi_score, ordered_index, segs_snapshot, sp_chars_snapshot).
        # Snapshots are taken only once min_chars is reached so we never cut too early.
        candidates = []

        for i in range(start_idx, len(segments)):
            seg = segments[i]
            idx = seg["id"]
            text = seg["text"]
            speaker = speaker_map[idx]

            segs.append((idx, text, speaker))
            total_chars += len(text)
            sp_chars[speaker] = sp_chars.get(speaker, 0) + len(text)

            if total_chars >= min_chars:
                candidates.append((hhi(sp_chars), i, list(segs), dict(sp_chars)))

            if total_chars >= max_chars:
                break  # hard ceiling reached; seal with whatever is best among candidates

        if not candidates:
            # Tail of the article is shorter than min_chars; flush as a single passage.
            if segs:
                passages.append(make_passage(segs, sp_chars))
            break

        good = [(score, i, s, c) for score, i, s, c in candidates if score >= hhi_threshold]
        if good:
            # Primary rule: take the rightmost (largest) passage that meets the threshold.
            _, end_i, best_segs, best_sp = good[-1]
        else:
            # Fallback: take the passage with the highest HHI regardless of threshold.
            # max() returns the first element that achieves the maximum, so ties resolve
            # to the smallest passage — consistent with the fallback rationale above.
            _, end_i, best_segs, best_sp = max(candidates, key=lambda x: x[0])

        passages.append(make_passage(best_segs, best_sp))
        start_idx = end_i + 1

    return passages


def optimize_passages(passages, segments, speaker_map, min_chars=200, max_chars=3000):
    """
    Refine passage boundaries by hill-climbing on the length-weighted HHI sum.

    Objective
    ---------
    Maximise Σᵢ |Pᵢ| · HHI(Pᵢ), where |Pᵢ| is the character length of passage i.
    Because total article length is constant, this is equivalent to maximising the
    length-weighted *average* HHI — rewarding purity where there is more text.

    Moves
    -----
    At each boundary between adjacent passages (i, i+1) two moves are considered:
      • shift right: move the last segment of passage i to the front of passage i+1.
      • shift left:  move the first segment of passage i+1 to the end of passage i.
    Only boundary moves are considered; this preserves text order and the total
    number of passages. A move is accepted only when both resulting passages remain
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
    typical passage counts and segment lengths seen here.
    """
    def compute_hhi(speaker_chars):
        total = sum(speaker_chars.values())
        if total == 0:
            return 0
        return 100 * sum((c / total) ** 2 for c in speaker_chars.values())

    def make_passage(seg_indices):
        """Rebuild a full passage dict from a list of segment indices."""
        text = ''.join(segments[i]["text"] for i in seg_indices)
        sp_chars = {}
        for i in seg_indices:
            spk = speaker_map[i]
            sp_chars[spk] = sp_chars.get(spk, 0) + len(segments[i]["text"])
        total = sum(sp_chars.values())
        dominant = max(sp_chars, key=sp_chars.get)
        speaker_pcts = {
            spk: round(chars / total * 100, 1)
            for spk, chars in sorted(sp_chars.items(), key=lambda x: -x[1])
        }
        return {
            'text': text,
            'segments': list(seg_indices),
            'speaker_chars': sp_chars,
            'speaker_pcts': speaker_pcts,
            'dominant_speaker': dominant,
            'dominant_ratio': sp_chars[dominant] / total if total else 0,
            'hhi': compute_hhi(sp_chars),
        }

    def weighted_score(p):
        """Contribution of one passage to the global objective."""
        return len(p['text']) * p['hhi']

    # Work on mutable lists of segment indices; passage dicts are rebuilt at the end.
    seg_lists = [list(p['segments']) for p in passages]
    moves_count = 0
    improved = True

    while improved:
        improved = False
        best_delta = 0
        best_move = None  # (boundary_index, new_left_seg_list, new_right_seg_list)

        for b in range(len(seg_lists) - 1):
            left, right = seg_lists[b], seg_lists[b + 1]
            left_p, right_p = make_passage(left), make_passage(right)
            current = weighted_score(left_p) + weighted_score(right_p)

            # Candidate A: shift last segment of left → beginning of right.
            if len(left) > 1:
                nl, nr = make_passage(left[:-1]), make_passage([left[-1]] + right)
                if min_chars <= len(nl['text']) <= max_chars and min_chars <= len(nr['text']) <= max_chars:
                    delta = weighted_score(nl) + weighted_score(nr) - current
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (b, left[:-1], [left[-1]] + right)

            # Candidate B: shift first segment of right → end of left.
            if len(right) > 1:
                nl, nr = make_passage(left + [right[0]]), make_passage(right[1:])
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
    return [make_passage(sl) for sl in seg_lists]


def weighted_hhi(passages):
    """Length-weighted average HHI across all passages (the optimization objective)."""
    total_chars = sum(len(p['text']) for p in passages)
    return sum(len(p['text']) * p['hhi'] for p in passages) / total_chars


def step_3_passages(segments: list, sources) -> list[dict]:
    speaker_map = build_speaker_map(segments, sources)
    passages = split_into_passages(segments, speaker_map, max_chars=PASSAGE_MAX_CHARS)
    if verbose:
        print(f"Initial weighted HHI: {weighted_hhi(passages):.1f}")
    passages = optimize_passages(passages, segments, speaker_map, max_chars=PASSAGE_MAX_CHARS)
    if verbose:
        print(f"Final   weighted HHI: {weighted_hhi(passages):.1f}")

    if verbose:
        print()
        for i, p in enumerate(passages):
            speakers_str = ', '.join(f'{spk}: {pct}%' for spk, pct in p['speaker_pcts'].items())
            seg_range = f"segs {p['segments'][0]}–{p['segments'][-1]}"
            print(f"--- Passage {i+1} [{seg_range}] | {len(p['text'])} chars | HHI: {p['hhi']:.1f} | {speakers_str} ---")
            print(p['text'][:120], '...' if len(p['text']) > 120 else '')
            print()
    
    html_passages_content = passages_monitoring_html(passages, segments, sources, PASSAGE_MAX_CHARS)

    return passages, html_passages_content


def passages_monitoring_html(
    passages: list,
    segments: list,
    sources,
    passage_max_chars: int,
) -> str:
    """Return passages_content for the monitoring report."""
    method_description = (
        "The attributed segments are grouped into passages large enough to provide meaningful "
        "context for downstream LLM analysis, while keeping each passage dominated by a single speaker. "
        "Speaker purity is measured with the <b>Herfindahl–Hirschman Index (HHI)</b>, "
        "ranging from 0 (fully fragmented) to 100 (single voice). "
        "Segments not covered by any attribution are assigned to <i>base_author</i>, "
        "the text author's own voice, which is treated as a first-class speaker. "
        "Splitting proceeds in two phases. "
        "<ul>"
        "<li><b>Phase 1 — Greedy initial split.</b> Segments are accumulated until the passage "
        "reaches <i>min_chars</i> (200). At every eligible cut point the HHI is recorded up to "
        f"<i>max_chars</i> ({passage_max_chars:,}). The cut is chosen by: "
        "(1) <i>primary rule</i> — the largest cut whose HHI ≥ 65 (corresponding to a dominant "
        "speaker holding ~89 % of characters in a two-speaker passage); "
        "(2) <i>fallback</i> — the cut with the highest HHI when no cut meets the threshold, "
        "which tends to be the smallest passage since HHI decreases as more speakers accumulate.</li>"
        "<li><b>Phase 2 — Hill-climbing boundary refinement.</b> Every boundary between adjacent "
        "passages is examined; the single-segment shift to the other side if it improves the global objective. "
        "The process repeats until no improving move exists. "
        "Convergence is guaranteed because the objective strictly increases with each accepted move.</li>"
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

    seg_to_passage_idx: dict[int, int] = {}
    for i, p in enumerate(passages):
        for seg_id in p["segments"]:
            seg_to_passage_idx[seg_id] = i

    body_parts = []
    current_passage_idx = None
    for seg in segments:
        seg_id = seg["id"]
        passage_idx = seg_to_passage_idx.get(seg_id)
        if passage_idx != current_passage_idx:
            if current_passage_idx is not None:
                body_parts.append("</p>\n")
            p = passages[passage_idx]
            seg_range = f"{p['segments'][0]}–{p['segments'][-1]}"
            speakers_str = " · ".join(
                f"{html_lib.escape(spk)}: {pct}%"
                for spk, pct in p["speaker_pcts"].items()
            )
            info = f'HHI: {p["hhi"]:.1f}% — {speakers_str}'
            body_parts.append(f"<h3>Passage {passage_idx + 1} [{seg_range}]</h3>\n")
            body_parts.append(f"<i>{info}</i>\n")
            body_parts.append("<p>")
            current_passage_idx = passage_idx
        text = html_lib.escape(seg["text"])
        css_class = seg_to_class.get(seg_id)
        if css_class:
            title = html_lib.escape(seg_to_name[seg_id])
            body_parts.append(f'<mark class="{css_class}" title="{title}">{text}</mark>')
        else:
            body_parts.append(f"<span>{text}</span>")
    if current_passage_idx is not None:
        body_parts.append("</p>\n")
    passages_html = "".join(body_parts)

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

    passages_content = f"""
<article><h3>Method</h3>{method_description}</article>

<h3>Speakers</h3>
<ul>
{legend_html}
</ul>

{passages_html}"""

    return passages_content
