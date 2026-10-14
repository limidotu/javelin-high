"""Argon pack and 9:16 render. No Vertex call."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from argon import fit_pace, fit_to_target, pack_timeline, probe_video, render_stage, render_tiktok, restore_openings, screen_of, stream_seconds
from gemini_defaults import uses_thinking_level
from media import ffmpeg_bin


def _clip(start: str, end: str, score: float, center: float, hook: str) -> dict:
    return {
        "start_time": start,
        "end_time": end,
        "excitement_score": score,
        "action_center_x": center,
        "on_screen_hook": hook,
        "is_hook_candidate": False,
    }


class PackTests(unittest.TestCase):
    def test_hook_opens_and_overlap_is_dropped(self) -> None:
        edl = {
            "clips": [
                _clip("00:10.0", "00:16.0", 9, 0.2, "BLOCK"),
                _clip("00:12.0", "00:14.0", 8, 0.5, "DIG"),
                _clip("00:30.0", "00:36.0", 7, 0.8, "ACE"),
            ]
        }
        timeline = pack_timeline(edl, 60.0, 12.0, hook=True)
        self.assertTrue(timeline[0]["is_opening_teaser"])
        self.assertLessEqual(sum(row["duration_sec"] for row in timeline), 12.05)
        self.assertGreater(timeline[0]["duration_sec"], 0)

    def test_long_clip_caps_at_nine(self) -> None:
        edl = {"clips": [_clip("00:00.0", "00:20.0", 9, 1.4, "KILL")]}
        timeline = pack_timeline(edl, 30.0, 60.0, hook=False)
        self.assertEqual(len(timeline), 1)
        self.assertAlmostEqual(timeline[0]["duration_sec"], 9.0)
        self.assertLessEqual(timeline[0]["action_center_x"], 0.85)

    def test_short_clip_is_dropped(self) -> None:
        edl = {"clips": [_clip("00:01.0", "00:01.5", 10, 0.5, "NO")]}
        self.assertEqual(pack_timeline(edl, 10.0, 60.0, hook=False), [])

    def test_thinking_level_skips_25(self) -> None:
        self.assertFalse(uses_thinking_level("gemini-2.5-flash"))
        self.assertTrue(uses_thinking_level("gemini-3.8-flash"))
        self.assertTrue(uses_thinking_level("gemini-4-argon"))


class RenderTests(unittest.TestCase):
    def test_smart_crop_writes_a_vertical_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "in.mp4"
            dest = root / "tiktok.mp4"
            subprocess.run(
                [
                    ffmpeg_bin(), "-y",
                    "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(source),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            render_tiktok(
                source,
                [{
                    "start_sec": 0.0,
                    "duration_sec": 1.0,
                    "action_center_x": 0.3,
                    "on_screen_hook": "WAIT FOR THE BLOCK",
                }],
                dest,
                layout="smart-crop",
                captions=True,
            )
            self.assertTrue(dest.is_file())
            self.assertGreater(dest.stat().st_size, 1000)
            meta = probe_video(dest)
            self.assertEqual(meta["width"], 1080)
            self.assertEqual(meta["height"], 1920)

    def test_blur_sides_writes_a_vertical_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "in.mp4"
            dest = root / "tiktok.mp4"
            subprocess.run(
                [
                    ffmpeg_bin(), "-y",
                    "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(source),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            render_tiktok(
                source,
                [{
                    "start_sec": 0.0,
                    "duration_sec": 0.6,
                    "action_center_x": 0.5,
                    "on_screen_hook": "ACE",
                }],
                dest,
                layout="blurred-bg",
                captions=True,
            )
            meta = probe_video(dest)
            self.assertEqual(meta["width"], 1080)
            self.assertEqual(meta["height"], 1920)


class FitTests(unittest.TestCase):
    def test_nine_plays_sum_to_sixty(self) -> None:
        spans = []
        cursor = 0.0
        for dur in (15, 13, 12, 12, 16, 13.5, 12, 15, 15):
            spans.append((cursor, cursor + dur))
            cursor += dur
        scores = [4, 4, 4, 4, 4, 4, 4, 9, 7]
        notes = {}
        for index, (start, end) in enumerate(spans):
            notes[index] = {
                "impact_s": start + (end - start) * 0.72,
                "score": scores[index],
                "label": "kill",
                "center_x": 0.4,
            }
        shots = fit_to_target(spans, notes, 60)
        self.assertEqual(len(shots), 9)
        total = sum(screen_of(shot["in_sec"], shot["out_sec"], shot["impact_s"]) for shot in shots)
        self.assertAlmostEqual(total, 60.0, delta=0.03)
        for shot in shots:
            self.assertGreaterEqual(shot["impact_s"], shot["in_sec"])
            self.assertLessEqual(shot["impact_s"], shot["out_sec"])
            self.assertGreaterEqual(shot["in_sec"], shot["span_a"] - 0.001)
            self.assertLessEqual(shot["out_sec"], shot["span_b"] + 0.001)
        self.assertGreater(shots[7]["screen_s"], shots[0]["screen_s"])

    def test_pre_share_keeps_the_approach(self) -> None:
        notes = {0: {"impact_s": 12.0, "in_s": 2.0, "out_s": 14.0, "score": 9, "label": "diving dig"}}
        shots = fit_to_target([(0.0, 15.0)], notes, 12)
        pre = shots[0]["impact_s"] - shots[0]["in_sec"]
        post = shots[0]["out_sec"] - shots[0]["impact_s"]
        self.assertGreater(pre, post)
        self.assertAlmostEqual(screen_of(shots[0]["in_sec"], shots[0]["out_sec"], shots[0]["impact_s"]), 12.0, delta=0.03)

    def test_high_score_keeps_the_marked_open(self) -> None:
        shots = [
            {"in_sec": 10.97, "out_sec": 15.0, "impact_s": 14.0, "span_a": 0.0, "span_b": 15.0, "score": 6, "prefer_in": None},
            {"in_sec": 23.74, "out_sec": 27.77, "impact_s": 26.8, "span_a": 15.0, "span_b": 28.0, "score": 4, "prefer_in": None},
            {"in_sec": 114.24, "out_sec": 123.64, "impact_s": 122.4, "span_a": 108.5, "span_b": 123.64, "score": 9, "prefer_in": 113.5},
        ]
        before = sum(screen_of(shot["in_sec"], shot["out_sec"], shot["impact_s"]) for shot in shots)
        restore_openings(shots)
        self.assertAlmostEqual(shots[2]["in_sec"], 113.5, delta=0.05)
        self.assertLess(shots[1]["in_sec"], 24.3)
        self.assertAlmostEqual(sum(screen_of(shot["in_sec"], shot["out_sec"], shot["impact_s"]) for shot in shots), before, delta=0.03)

    def test_pace_lands_on_sixty_without_speed(self) -> None:
        notes = {
            0: {"keep": True, "in_s": 0.0, "out_s": 25.0, "energy": 4, "center_x": 0.5},
            1: {"keep": True, "in_s": 25.0, "out_s": 50.0, "energy": 4, "center_x": 0.5},
            2: {"keep": True, "in_s": 50.0, "out_s": 68.0, "energy": 9, "center_x": 0.4},
        }
        shots = fit_pace([(0.0, 25.0), (25.0, 50.0), (50.0, 80.0)], notes, 60)
        total = sum(shot["out_sec"] - shot["in_sec"] for shot in shots)
        self.assertAlmostEqual(total, 60.0, delta=0.03)
        self.assertEqual(shots[2]["in_sec"], 50.0)
        self.assertGreaterEqual(shots[2]["out_sec"], 68.0)
        self.assertLess(shots[0]["screen_s"], 25.0)


class StageTests(unittest.TestCase):
    def test_stage_is_vertical_and_hits_target(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "in.mp4"
            dest = root / "tiktok.mp4"
            subprocess.run(
                [
                    ffmpeg_bin(), "-y",
                    "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(source),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            target = screen_of(0.0, 1.2, 0.6)
            render_stage(
                source,
                [{
                    "in_sec": 0.0,
                    "out_sec": 1.2,
                    "impact_s": 0.6,
                    "center_x": 0.4,
                    "label": "Kill",
                    "span_a": 0.0,
                    "span_b": 1.2,
                }],
                dest,
                layout="blurred-bg",
                captions=True,
                target_s=target,
            )
            meta = probe_video(dest)
            picture_s = stream_seconds(dest, "0:v:0")
            sound_s = stream_seconds(dest, "0:a:0")
            self.assertEqual(meta["width"], 1080)
            self.assertEqual(meta["height"], 1920)
            self.assertAlmostEqual(picture_s, target, delta=0.05)
            self.assertAlmostEqual(sound_s, picture_s, delta=0.05)


if __name__ == "__main__":
    unittest.main()
