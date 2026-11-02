"""Range resolution: overlapping and nested attribution ranges (moved from attribution.py)."""

from karaage.attribution import (ATTRIBUTION_SCHEMA, MAX_ALIASES, MAX_RANGES, MAX_SOURCES, AttributionResult,
                                 Source, normalize_answer, resolve_source_ranges)


def test_range_resolution():
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


def test_schema_bounds_every_list():
    # Unbounded lists let a looping model generate until the token limit (articles 11, 17 and 24).
    sources = ATTRIBUTION_SCHEMA["properties"]["sources"]
    source = sources["items"]["properties"]
    assert sources["maxItems"] == MAX_SOURCES and source["segments"]["maxItems"] == MAX_RANGES
    assert source["segments"]["items"]["minItems"] == source["segments"]["items"]["maxItems"] == 2
    assert source["other"]["maxItems"] == MAX_ALIASES


def test_normalize_answer():
    # Ranges as a backend that ignores the schema may return them (pre-1.1.0 runs crashed on these).
    raw = AttributionResult(sources=[
        Source(main="Ann", other=[], segments=[[4], [6, 8]]),
        Source(main="Bob", other=[], segments=[[]]),
        Source(main="Cy", other=[], segments=[[10, 12, 14]]),
    ])
    normalized = normalize_answer(raw)
    assert [s.segments for s in normalized.sources] == [[[4, 4], [6, 8]], [], [[10, 10], [12, 12], [14, 14]]]

    segments = [{"id": i, "text": "XXXXXXXXXX", "start": i * 10, "end": (i + 1) * 10} for i in range(20)]
    resolved = resolve_source_ranges(normalized, segments)
    assert {s.main: s.segments for s in resolved.sources} == {"Ann": [[4, 8]], "Cy": [[10, 14]]}


def test_repeated_actors_and_aliases_are_merged():
    # A looping model may list the same actor again and again (article 24).
    raw = AttributionResult(sources=[
        Source(main="Ann", other=["A.", "we", "A."], segments=[[1, 2]]),
        Source(main="Bob", other=[], segments=[[5, 5]]),
        Source(main="Ann", other=["Dr Ann"], segments=[[1, 2], [8, 9]]),
    ])
    merged = normalize_answer(raw)
    assert [(s.main, s.other, s.segments) for s in merged.sources] == [
        ("Ann", ["A.", "we", "Dr Ann"], [[1, 2], [1, 2], [8, 9]]),
        ("Bob", [], [[5, 5]]),
    ]
