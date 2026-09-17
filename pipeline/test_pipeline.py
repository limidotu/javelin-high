"""Select, windows, and drop-file complete checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from review_clips import _parse_review, assert_review_complete, review_ladder_rows
from clip_window import (
    REVIEW_DUR,
    clamp_trim,
    clip_window,
    default_trim,
    final_source_span,
    max_clip_dur,
    parse_clip_time,
    review_window,
    trim_from_row,
)
from pick import (
    REVIEW_BATCH,
    SCAN_CAP,
    batches,
    cap_by_score,
    child_review_groups,
    fallback_reel,
    fill_partial_reel,
    gens_dir,
    inbox_mp4s,
    next_gen_n,
    pick_reel,
    pick_source_mp4,
    wait_until_complete,
)
from scan_io import (
    child_scan_spans,
    load_all_keeps,
    merge_windows,
    iter_window_segments,
    run_with_retries,
    scan_ladder_rows,
    should_retry_window,
    should_split_span,
    uncovered_windows,
)
from scan_local import scan_prompt, span_timeout_ms


def _row(start: float, score: int, event: str = "kill") -> dict:
    return {
        "start_s": start,
        "end_s": start + 8.0,
        "score": score,
        "event": event,
        "keep": True,
    }


class CapByScoreTests(unittest.TestCase):
    def test_never_pads_past_available(self) -> None:
        rows = [_row(10.0, 9), _row(20.0, 8)]
        got = cap_by_score(rows, limit=75)
        self.assertEqual(len(got), 2)

    def test_keeps_highest_scores(self) -> None:
        rows = [_row(float(i), 8 if i < 70 else 10) for i in range(80)]
        got = cap_by_score(rows, limit=SCAN_CAP)
        self.assertEqual(len(got), 75)
        tens = [r for r in got if r["score"] == 10]
        self.assertEqual(len(tens), 10)

    def test_empty_limit_returns_empty(self) -> None:
        self.assertEqual(cap_by_score([_row(1.0, 9)], limit=0), [])
        self.assertEqual(cap_by_score([], limit=8), [])


class BatchTests(unittest.TestCase):
    def test_batches_of_ten_leave_a_remainder(self) -> None:
        items = list(range(23))
        got = batches(items, n=REVIEW_BATCH)
        self.assertEqual(len(got), 3)
        self.assertEqual(got[0], list(range(10)))
        self.assertEqual(got[-1], [20, 21, 22])
        self.assertTrue(all(len(b) <= 10 for b in got))


class PickReelTests(unittest.TestCase):
    def test_takes_eight_then_sorts_chrono(self) -> None:
        reviews = [
            {**_row(90.0, 8), "keep": True},
            {**_row(10.0, 10), "keep": True},
            {**_row(50.0, 9), "keep": True},
            {**_row(20.0, 9), "keep": True},
            {**_row(30.0, 9), "keep": True},
            {**_row(40.0, 9), "keep": True},
            {**_row(60.0, 9), "keep": True},
            {**_row(70.0, 9), "keep": True},
            {**_row(80.0, 9), "keep": True},
            {**_row(5.0, 10), "keep": False},
        ]
        got = pick_reel(reviews, n=8, min_n=8)
        self.assertEqual(len(got), 8)
        self.assertEqual([r["start_s"] for r in got], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])

    def test_fills_to_eight_from_high_scores(self) -> None:
        reviews = [{**_row(10.0, 9), "keep": True}]
        reviews.extend({**_row(20.0 + i, 7), "keep": False} for i in range(9))
        got = pick_reel(reviews, n=10, min_n=8)
        self.assertEqual(len(got), 8)
        self.assertEqual(got[0]["start_s"], 10.0)

    def test_caps_at_ten_keeps(self) -> None:
        reviews = [{**_row(float(i), 9), "keep": True} for i in range(12)]
        got = pick_reel(reviews, n=10, min_n=8)
        self.assertEqual(len(got), 10)

    def test_fewer_rows_than_min_uses_all(self) -> None:
        reviews = [
            {**_row(10.0, 9), "keep": True},
            {**_row(20.0, 4), "keep": False},
        ]
        got = pick_reel(reviews, n=10, min_n=8)
        self.assertEqual(len(got), 2)

    def test_fallback_uses_two_point_five_ranks(self) -> None:
        rows = [_row(80.0, 8), _row(10.0, 10), _row(40.0, 9)]
        got = fallback_reel(rows, n=2)
        self.assertEqual([r["start_s"] for r in got], [10.0, 40.0])

    def test_partial_fill_adds_unreviewed_scan_clips(self) -> None:
        reviewed = [{**_row(10.0, 9), "keep": True, "file": "c01.mp4"}]
        candidates = [
            {**_row(10.0, 9), "file": "c01.mp4"},
            {**_row(20.0, 10), "file": "c02.mp4"},
            {**_row(30.0, 8), "file": "c03.mp4"},
        ]
        got = fill_partial_reel(reviewed, candidates, partial=True, n=3)
        self.assertEqual([r["file"] for r in got], ["c01.mp4", "c02.mp4", "c03.mp4"])

    def test_complete_review_does_not_pad(self) -> None:
        reviewed = [{**_row(10.0, 9), "keep": True, "file": "c01.mp4"}]
        candidates = reviewed + [{**_row(20.0, 10), "file": "c02.mp4"}]
        got = fill_partial_reel(reviewed, candidates, partial=False, n=8)
        self.assertEqual(len(got), 1)


class UncoveredWindowTests(unittest.TestCase):
    def test_finished_prefix_leaves_the_tail(self) -> None:
        gaps = uncovered_windows([(0, 1800)], video_end=3600, window_s=600)
        self.assertEqual(gaps, [(1800, 2400), (2400, 3000), (3000, 3600)])

    def test_no_gaps_when_full(self) -> None:
        self.assertEqual(uncovered_windows([(0, 600), (600, 1200)], video_end=1200, window_s=600), [])


class WindowSegmentTests(unittest.TestCase):
    def test_ten_minute_window_is_ten_clips(self) -> None:
        got = iter_window_segments(1200, 1800)
        self.assertEqual(len(got), 10)
        self.assertEqual(got[0], (1200, 1260))
        self.assertEqual(got[-1], (1740, 1800))
        self.assertTrue(all((b - a) <= 60 for a, b in got))

    def test_short_tail_is_one_clip(self) -> None:
        self.assertEqual(iter_window_segments(6600, 6632), [(6600, 6632)])

    def test_never_exceeds_ten_videos(self) -> None:
        got = iter_window_segments(0, 1800, segment_s=60, max_videos=10)
        self.assertLessEqual(len(got), 10)
        self.assertEqual(got[0][0], 0)
        self.assertEqual(got[-1][1], 1800)


class ScanLadderTests(unittest.TestCase):
    def test_ten_min_splits_to_two_fives(self) -> None:
        self.assertEqual(child_scan_spans(0, 600), [(0, 300), (300, 600)])

    def test_five_min_splits_to_sixty(self) -> None:
        got = child_scan_spans(0, 300)
        self.assertEqual(len(got), 5)
        self.assertEqual(got[0], (0, 60))
        self.assertEqual(got[-1], (240, 300))

    def test_minute_has_no_children(self) -> None:
        self.assertEqual(child_scan_spans(0, 60), [])
        self.assertEqual(child_scan_spans(6600, 6632), [])

    def test_ten_min_ok_is_one_call(self) -> None:
        calls: list[tuple[int, int]] = []

        def attempt(a: int, b: int) -> list[dict]:
            calls.append((a, b))
            return [{"start_s": a}]

        got = scan_ladder_rows(0, 600, attempt)
        self.assertEqual(calls, [(0, 600)])
        self.assertEqual(len(got), 1)

    def test_ten_min_fail_tries_two_fives(self) -> None:
        calls: list[tuple[int, int]] = []

        def attempt(a: int, b: int) -> list[dict]:
            calls.append((a, b))
            if (a, b) == (0, 600):
                raise RuntimeError("400 INVALID_ARGUMENT")
            return [{"start_s": a}]

        got = scan_ladder_rows(0, 600, attempt)
        self.assertEqual(calls, [(0, 600), (0, 300), (300, 600)])
        self.assertEqual([r["start_s"] for r in got], [0, 300])

    def test_one_five_fail_breaks_only_that_half(self) -> None:
        calls: list[tuple[int, int]] = []

        def attempt(a: int, b: int) -> list[dict]:
            calls.append((a, b))
            if (a, b) in {(0, 600), (0, 300)}:
                raise RuntimeError("400 INVALID_ARGUMENT")
            return [{"start_s": a}]

        got = scan_ladder_rows(0, 600, attempt)
        sixties = [(i, i + 60) for i in range(0, 300, 60)]
        self.assertEqual(calls, [(0, 600), (0, 300), *sixties, (300, 600)])
        self.assertEqual(len(got), 6)

    def test_both_fives_fail_uses_ten_sixties(self) -> None:
        calls: list[tuple[int, int]] = []

        def attempt(a: int, b: int) -> list[dict]:
            calls.append((a, b))
            if b - a > 60:
                raise RuntimeError("400 INVALID_ARGUMENT")
            return [{"start_s": a}]

        got = scan_ladder_rows(0, 600, attempt)
        self.assertEqual(calls[0], (0, 600))
        self.assertEqual(calls[1], (0, 300))
        self.assertEqual(len([c for c in calls if c[1] - c[0] == 60]), 10)
        self.assertEqual(len(got), 10)

    def test_quota_does_not_split(self) -> None:
        self.assertFalse(should_split_span(RuntimeError("429 RESOURCE_EXHAUSTED")))
        self.assertTrue(should_split_span(RuntimeError("400 INVALID_ARGUMENT")))

        def attempt(_a: int, _b: int) -> list[dict]:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

        with self.assertRaises(RuntimeError):
            scan_ladder_rows(0, 600, attempt)

    def test_prompt_names_clip_length(self) -> None:
        self.assertIn("600 seconds", scan_prompt(600))
        self.assertIn("Prefer 0 to 8 items", scan_prompt(600))
        self.assertIn("Prefer 0 or 1 item", scan_prompt(60))
        self.assertGreater(span_timeout_ms(600), span_timeout_ms(60))


class ReviewLadderTests(unittest.TestCase):
    def test_ten_clips_split_to_two_fives(self) -> None:
        group = [{"i": i} for i in range(10)]
        self.assertEqual([len(g) for g in child_review_groups(group)], [5, 5])

    def test_five_clips_split_to_singles(self) -> None:
        group = [{"i": i} for i in range(5)]
        self.assertEqual(child_review_groups(group), [[{"i": i}] for i in range(5)])

    def test_failed_ten_retries_as_fives(self) -> None:
        group = [{"i": i} for i in range(10)]
        sizes: list[int] = []

        def attempt(g: list[dict]) -> list[dict]:
            sizes.append(len(g))
            if len(g) == 10:
                raise RuntimeError("400 INVALID_ARGUMENT")
            return g

        got = review_ladder_rows(
            group, attempt, quota_left=0, sleeper=lambda _s: None, sleep_s=0
        )
        self.assertEqual(sizes, [10, 5, 5])
        self.assertEqual(len(got), 10)

    def test_429_retries_same_batch(self) -> None:
        hits = {"n": 0}
        group = [{"i": 0}]

        def attempt(g: list[dict]) -> list[dict]:
            hits["n"] += 1
            if hits["n"] < 3:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return g

        got = review_ladder_rows(
            group, attempt, quota_left=3, sleeper=lambda _s: None, sleep_s=0
        )
        self.assertEqual(hits["n"], 3)
        self.assertEqual(got, group)

    def test_split_resets_429_quota(self) -> None:
        group = [{"i": 0}, {"i": 1}]
        hits = {"n1": 0}

        def attempt(g: list[dict]) -> list[dict]:
            if len(g) == 2:
                raise RuntimeError("400 INVALID_ARGUMENT")
            hits["n1"] += 1
            if hits["n1"] == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return g

        got = review_ladder_rows(
            group, attempt, quota_left=0, sleeper=lambda _s: None, sleep_s=0
        )
        self.assertEqual(len(got), 2)


class RetryWindowTests(unittest.TestCase):
    def test_invalid_argument_retries(self) -> None:
        exc = RuntimeError("400 INVALID_ARGUMENT. Request contains an invalid argument.")
        self.assertTrue(should_retry_window(exc))

    def test_quota_does_not_retry(self) -> None:
        self.assertFalse(should_retry_window(RuntimeError("429 RESOURCE_EXHAUSTED")))

    def test_retries_then_succeeds(self) -> None:
        hits = {"n": 0}

        def flaky() -> str:
            hits["n"] += 1
            if hits["n"] < 3:
                raise RuntimeError("400 INVALID_ARGUMENT. {'error': {'code': 400}}")
            return "ok"

        got = run_with_retries(flaky, tries=3, sleep_s=0, sleeper=lambda _s: None)
        self.assertEqual(got, "ok")
        self.assertEqual(hits["n"], 3)

    def test_raises_after_all_tries(self) -> None:
        def always_400() -> str:
            raise RuntimeError("400 INVALID_ARGUMENT. boom")

        with self.assertRaises(RuntimeError):
            run_with_retries(always_400, tries=2, sleep_s=0, sleeper=lambda _s: None)


class DropFolderTests(unittest.TestCase):
    def test_wait_returns_a_stable_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "game.mp4"
            path.write_bytes(b"not-empty")
            got = wait_until_complete(path, stable_s=0.0, poll_s=0.01, timeout_s=2.0)
            self.assertEqual(got, path)

    def test_inbox_ignores_non_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            inbox = Path(raw)
            (inbox / "note.txt").write_text("x", encoding="utf-8")
            (inbox / "a.mp4").write_bytes(b"mp4")
            got = inbox_mp4s(inbox)
            self.assertEqual([p.name for p in got], ["a.mp4"])

    def test_next_gen_increments(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            folder = gens_dir(root)
            (folder / "gen_6").mkdir(parents=True)
            (folder / "gen_2").mkdir()
            self.assertEqual(next_gen_n(root), 7)

    def test_next_gen_is_one_when_folder_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            self.assertEqual(next_gen_n(Path(raw)), 1)

    def test_done_folder_is_used_when_inbox_empty(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            inbox = Path(raw) / "inbox"
            done = inbox / "done"
            inbox.mkdir()
            done.mkdir()
            (done / "game.mp4").write_bytes(b"mp4")
            path, from_inbox = pick_source_mp4(inbox, done, Path(raw) / "source.mp4")
            self.assertEqual(path.name, "game.mp4")
            self.assertFalse(from_inbox)

    def test_inbox_beats_done(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            inbox = Path(raw) / "inbox"
            done = inbox / "done"
            inbox.mkdir()
            done.mkdir()
            (inbox / "new.mp4").write_bytes(b"new")
            (done / "old.mp4").write_bytes(b"old")
            path, from_inbox = pick_source_mp4(inbox, done, Path(raw) / "source.mp4")
            self.assertEqual(path.name, "new.mp4")
            self.assertTrue(from_inbox)


class ReviewParseTests(unittest.TestCase):
    def test_fills_missing_index(self) -> None:
        text = json.dumps({"clips": [{"index": 1, "keep": True, "score": 9, "event": "kill"}]})
        got = _parse_review(text, 3)
        self.assertEqual(len(got), 3)
        self.assertFalse(got[0]["keep"])
        self.assertTrue(got[1]["keep"])
        self.assertEqual(got[1]["score"], 9)
        self.assertFalse(got[2]["keep"])

    def test_incomplete_review_raises(self) -> None:
        text = json.dumps({"clips": [{"index": 0, "keep": True, "score": 9, "event": "kill"}]})
        rows = _parse_review(text, 3)
        with self.assertRaises(RuntimeError) as ctx:
            assert_review_complete(rows)
        self.assertIn("incomplete review", str(ctx.exception))

    def test_model_keep_at_seven_stays_keep(self) -> None:
        text = json.dumps({"clips": [{"index": 0, "keep": True, "score": 7, "event": "kill"}]})
        got = _parse_review(text, 1)
        self.assertTrue(got[0]["keep"])
        self.assertEqual(got[0]["score"], 7)


    def test_parses_clip_relative_start_end(self) -> None:
        text = json.dumps(
            {
                "clips": [
                    {
                        "index": 0,
                        "keep": True,
                        "score": 9,
                        "event": "kill",
                        "start": 1.5,
                        "end": 19.0,
                    }
                ]
            }
        )
        got = _parse_review(text, 1)
        self.assertEqual(got[0]["trim_start_s"], 1.5)
        self.assertEqual(got[0]["trim_end_s"], 19.0)


class TrimWindowTests(unittest.TestCase):
    def test_review_window_is_twenty_and_ends_after_the_event(self) -> None:
        start, dur = review_window(100.0, 108.0)
        self.assertEqual(dur, REVIEW_DUR)
        self.assertAlmostEqual(start + dur, 110.0, places=1)

    def test_cool_window_is_forty(self) -> None:
        from clip_window import COOL_WATCH_DUR, cool_review_window

        start, dur = cool_review_window(100.0, 108.0)
        self.assertEqual(dur, COOL_WATCH_DUR)
        self.assertAlmostEqual(start + dur, 110.0, places=1)

    def test_short_trim_expands_to_twelve(self) -> None:
        a, b = clamp_trim(8.0, 12.0, window_s=20.0)
        self.assertAlmostEqual(b - a, 12.0)
        self.assertGreaterEqual(a, 0.0)
        self.assertLessEqual(b, 20.0)

    def test_normal_trim_caps_at_eighteen(self) -> None:
        from clip_window import MAX_CLIP_DUR

        a, b = clamp_trim(0.0, 35.0, window_s=40.0)
        self.assertAlmostEqual(b - a, MAX_CLIP_DUR)
        self.assertAlmostEqual(b, 35.0)

    def test_cool_trim_can_be_forty(self) -> None:
        a, b = clamp_trim(0.0, 38.0, window_s=40.0, max_s=40.0)
        self.assertAlmostEqual(b - a, 38.0)
        self.assertEqual(max_clip_dur(10), 40.0)
        self.assertEqual(max_clip_dur(8), 18.0)

    def test_omitted_trim_is_fifteen_late(self) -> None:
        a, b = default_trim(20.0)
        self.assertAlmostEqual(b - a, 15.0)
        self.assertAlmostEqual(b, 20.0)

    def test_clock_string_and_seconds(self) -> None:
        self.assertEqual(parse_clip_time("00:03"), 3.0)
        self.assertEqual(parse_clip_time(4.5), 4.5)

    def test_match_clock_is_ignored(self) -> None:
        a, b = trim_from_row({"start": 76.0, "end": 91.0}, window_s=20.0)
        self.assertAlmostEqual(b - a, 15.0)

    def test_final_span_maps_to_source(self) -> None:
        start, dur = final_source_span(100.0, 40.0, 2.0, 18.0, max_s=40.0)
        self.assertAlmostEqual(start, 102.0)
        self.assertAlmostEqual(dur, 16.0)

    def test_only_one_clip_gets_forty(self) -> None:
        from clip_window import one_cool_caps

        rows = [
            {"score": 9, "trim_start_s": 0.0, "trim_end_s": 12.0},
            {"score": 10, "trim_start_s": 0.0, "trim_end_s": 36.0},
            {"score": 10, "trim_start_s": 0.0, "trim_end_s": 28.0},
        ]
        self.assertEqual(one_cool_caps(rows), [18.0, 40.0, 18.0])


class ScanIoTests(unittest.TestCase):
    def test_load_keeps_skips_low_scores(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "vertex_best_0_600.json"
            path.write_text(
                json.dumps(
                    {
                        "highlights": [
                            {
                                "source_start": "01:00",
                                "source_end": "01:08",
                                "event": "kill",
                                "keep": True,
                                "score": 9,
                            },
                            {
                                "source_start": "02:00",
                                "source_end": "02:08",
                                "event": "miss",
                                "keep": True,
                                "score": 6,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = merge_windows(load_all_keeps(Path(raw)))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event"], "kill")
            self.assertLessEqual(len(cap_by_score(rows)), SCAN_CAP)


class ClipWindowTests(unittest.TestCase):
    def test_event_pads_to_fifteen(self) -> None:
        start, dur = clip_window(100.0, 103.0)
        self.assertEqual(dur, 15.0)
        self.assertLessEqual(start, 100.0)
        self.assertGreaterEqual(start + dur, 103.0)


if __name__ == "__main__":
    unittest.main()
