"""Select, windows, and drop-file complete checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from review_clips import _parse_review
from scan_io import load_all_keeps, merge_windows, uncovered_windows
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
    fallback_reel,
    inbox_mp4s,
    next_gen_n,
    pick_reel,
    wait_until_complete,
)


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
        got = pick_reel(reviews, n=8)
        self.assertEqual(len(got), 8)
        self.assertEqual([r["start_s"] for r in got], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])

    def test_fewer_keeps_makes_a_shorter_reel(self) -> None:
        reviews = [
            {**_row(10.0, 9), "keep": True},
            {**_row(20.0, 4), "keep": False},
        ]
        got = pick_reel(reviews, n=8)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["start_s"], 10.0)

    def test_fallback_uses_two_point_five_ranks(self) -> None:
        rows = [_row(80.0, 8), _row(10.0, 10), _row(40.0, 9)]
        got = fallback_reel(rows, n=2)
        self.assertEqual([r["start_s"] for r in got], [10.0, 40.0])


class UncoveredWindowTests(unittest.TestCase):
    def test_finished_prefix_leaves_the_tail(self) -> None:
        gaps = uncovered_windows([(0, 1800)], video_end=3600, window_s=600)
        self.assertEqual(gaps, [(1800, 2400), (2400, 3000), (3000, 3600)])

    def test_no_gaps_when_full(self) -> None:
        self.assertEqual(uncovered_windows([(0, 600), (600, 1200)], video_end=1200, window_s=600), [])


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
            (root / "gen_6").mkdir()
            (root / "gen_2").mkdir()
            self.assertEqual(next_gen_n(root), 7)


class ReviewParseTests(unittest.TestCase):
    def test_fills_missing_index(self) -> None:
        text = json.dumps({"clips": [{"index": 1, "keep": True, "score": 9, "event": "kill"}]})
        got = _parse_review(text, 3)
        self.assertEqual(len(got), 3)
        self.assertFalse(got[0]["keep"])
        self.assertTrue(got[1]["keep"])
        self.assertEqual(got[1]["score"], 9)
        self.assertFalse(got[2]["keep"])

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

    def test_short_trim_expands_to_ten(self) -> None:
        a, b = clamp_trim(8.0, 12.0, window_s=20.0)
        self.assertAlmostEqual(b - a, 10.0)
        self.assertGreaterEqual(a, 0.0)
        self.assertLessEqual(b, 20.0)

    def test_normal_trim_caps_at_twenty(self) -> None:
        a, b = clamp_trim(0.0, 35.0, window_s=40.0, max_s=20.0)
        self.assertAlmostEqual(b - a, 20.0)
        self.assertAlmostEqual(b, 35.0)

    def test_cool_trim_can_be_forty(self) -> None:
        a, b = clamp_trim(0.0, 38.0, window_s=40.0, max_s=40.0)
        self.assertAlmostEqual(b - a, 38.0)
        self.assertEqual(max_clip_dur(10), 40.0)
        self.assertEqual(max_clip_dur(8), 20.0)

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
        self.assertEqual(one_cool_caps(rows), [20.0, 40.0, 20.0])


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
