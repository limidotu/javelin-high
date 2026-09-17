"""Behavior tests for clip windows and 9:16 crop boxes."""

from __future__ import annotations

import unittest

from follow_cam import (
    SmoothCam,
    aim,
    best_play_path,
    choose_detection,
    clip_window,
    crop_rect,
    damp,
    hold_after_rally,
    lock_play_court,
    smooth_track,
)


class ClipWindowTests(unittest.TestCase):
    def test_short_event_pads_to_fifteen_seconds(self) -> None:
        start, dur = clip_window(100.0, 103.0)
        self.assertEqual(dur, 15.0)
        self.assertLessEqual(start, 100.0)
        self.assertGreaterEqual(start + dur, 103.0)

    def test_long_event_stays_at_prefer_length(self) -> None:
        start, dur = clip_window(100.0, 120.0)
        self.assertEqual(dur, 15.0)
        self.assertGreaterEqual(start + dur, 120.0)

    def test_window_does_not_start_before_zero(self) -> None:
        start, dur = clip_window(0.5, 2.0)
        self.assertGreaterEqual(start, 0.0)
        self.assertEqual(dur, 15.0)


class CropRectTests(unittest.TestCase):
    def test_crop_stays_inside_1280x720(self) -> None:
        x, y, w, h = crop_rect(1280, 720, cx=0.5, cy=0.4, zoom=1.4)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, 1280)
        self.assertLessEqual(y + h, 720)
        self.assertAlmostEqual(w / h, 9 / 16, places=2)

    def test_higher_zoom_makes_a_smaller_box(self) -> None:
        _, _, w1, h1 = crop_rect(1280, 720, cx=0.5, cy=0.5, zoom=1.1)
        _, _, w2, h2 = crop_rect(1280, 720, cx=0.5, cy=0.5, zoom=1.8)
        self.assertLess(w2, w1)
        self.assertLess(h2, h1)


class AimTests(unittest.TestCase):
    def test_slow_ball_beside_person_aims_at_the_ball(self) -> None:
        persons = [(0.40, 0.62, 0.04, 0.40)]
        target, in_play = aim(
            ball=(0.42, 0.66),
            ball_speed=0.04,
            persons=persons,
            event="ace",
            in_play=False,
        )
        self.assertTrue(in_play)
        self.assertAlmostEqual(target.cx, 0.42, places=1)
        self.assertAlmostEqual(target.cy, 0.66, places=1)

    def test_fast_ball_aims_at_ball(self) -> None:
        persons = [(0.40, 0.62, 0.04, 0.40)]
        target, in_play = aim(
            ball=(0.70, 0.30),
            ball_speed=0.80,
            persons=persons,
            event="kill",
            in_play=False,
        )
        self.assertTrue(in_play)
        self.assertAlmostEqual(target.cx, 0.70, places=1)
        self.assertAlmostEqual(target.cy, 0.30, places=1)

    def test_rally_stays_on_ball_when_already_in_play(self) -> None:
        persons = [(0.50, 0.50, 0.04, 0.40)]
        target, in_play = aim(
            ball=(0.55, 0.44),
            ball_speed=0.10,
            persons=persons,
            event="block",
            in_play=True,
        )
        self.assertTrue(in_play)
        self.assertAlmostEqual(target.cx, 0.55, places=1)


class SmoothCamTests(unittest.TestCase):
    def test_deadzone_holds_still(self) -> None:
        cam = SmoothCam(1280, 720, zoom=1.4, cx=0.5, cy=0.4, alpha=0.2, dead=0.03)
        first = cam.box()
        cam.follow(0.51, 0.41)
        self.assertEqual(cam.box(), first)

    def test_crop_size_stays_fixed_while_panning(self) -> None:
        cam = SmoothCam(1280, 720, zoom=1.4, cx=0.3, cy=0.4, alpha=0.8, dead=0.0)
        w0, h0 = cam.box()[2], cam.box()[3]
        for x in (0.2, 0.8, 0.55):
            cam.follow(x, 0.4)
            box = cam.box()
            self.assertEqual(box[2], w0)
            self.assertEqual(box[3], h0)

    def test_follow_moves_toward_the_ball(self) -> None:
        cam = SmoothCam(1280, 720, zoom=1.4, cx=0.2, cy=0.4, alpha=0.5, dead=0.0)
        cam.follow(0.8, 0.4)
        self.assertGreater(cam.cx, 0.2)


class ChooseDetectionTests(unittest.TestCase):
    def test_keeps_the_near_blob_and_drops_the_far_one(self) -> None:
        cands = [(9.0, 0.90, 0.60), (8.0, 0.42, 0.31)]
        picked = choose_detection(cands, last=(0.40, 0.30), dt=1 / 60, speed=0.2)
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertAlmostEqual(picked[0], 0.42, places=2)
        self.assertAlmostEqual(picked[1], 0.31, places=2)

    def test_needs_a_strong_blob_to_start_a_track(self) -> None:
        weak = choose_detection([(3.0, 0.5, 0.4)], last=None, dt=1 / 60, speed=0.0)
        strong = choose_detection([(9.0, 0.5, 0.4)], last=None, dt=1 / 60, speed=0.0)
        light = choose_detection([(12.0, 0.5, 0.10)], last=None, dt=1 / 60, speed=0.0)
        person = choose_detection([(12.0, 0.5, 0.66)], last=None, dt=1 / 60, speed=0.0)
        held = choose_detection([(9.0, 0.48, 0.54)], last=None, dt=1 / 60, speed=0.0)
        self.assertIsNone(weak)
        self.assertEqual(strong, (0.5, 0.4))
        self.assertIsNone(light)
        self.assertIsNone(person)
        self.assertIsNone(held)

    def test_rejects_a_teleport_on_the_next_frame(self) -> None:
        picked = choose_detection(
            [(12.0, 0.70, 0.55)],
            last=(0.28, 0.40),
            dt=1 / 60,
            speed=2.4,
        )
        self.assertIsNone(picked)


class BestPlayPathTests(unittest.TestCase):
    def test_keeps_the_moving_air_ball_over_a_held_spare(self) -> None:
        frames: list[list[tuple[float, float, float]]] = []
        for i in range(24):
            spare = (9.0, 0.42, 0.55)
            play = (8.5, 0.58 + i * 0.004, 0.30)
            frames.append([spare, play])
        xs, ys = best_play_path(frames, hint_cx=0.60, court_tol=0.36)
        hit = [(x, y) for x, y in zip(xs, ys) if x is not None]
        self.assertGreater(len(hit), 10)
        self.assertGreater(float(hit[-1][0]), 0.55)
        self.assertLess(float(hit[-1][1]), 0.40)

    def test_ignores_a_ball_on_the_other_court(self) -> None:
        frames: list[list[tuple[float, float, float]]] = []
        for i in range(20):
            home = (8.5, 0.32 + i * 0.003, 0.28)
            other = (12.0, 0.88, 0.30)
            frames.append([home, other])
        xs, _ys = best_play_path(frames, hint_cx=0.35, court_tol=0.36)
        hit = [x for x in xs if x is not None]
        self.assertGreater(len(hit), 8)
        self.assertLess(max(hit), 0.55)

    def test_drops_a_carried_spare_even_when_it_is_denser(self) -> None:
        frames: list[list[tuple[float, float, float]]] = []
        for i in range(30):
            spare = (11.0, 0.38 + i * 0.002, 0.52)
            play = (8.0, 0.68 + i * 0.003, 0.26 + 0.06 * ((i % 6) / 6.0))
            frames.append([spare, play])
        xs, ys = best_play_path(frames, hint_cx=0.70, court_tol=0.40)
        hit = [(x, y) for x, y in zip(xs, ys) if x is not None]
        self.assertGreater(len(hit), 5)
        xs_hit = [h[0] for h in hit]
        self.assertGreater(sum(xs_hit) / len(xs_hit), 0.60)

    def test_drops_a_sideline_other_court_track(self) -> None:
        frames: list[list[tuple[float, float, float]]] = []
        for i in range(20):
            home = (8.5, 0.48 + i * 0.002, 0.30)
            other = (12.0, 0.90, 0.28)
            frames.append([home, other])
        xs, _ys = best_play_path(frames, hint_cx=0.50, court_tol=0.50)
        hit = [x for x in xs if x is not None]
        self.assertGreater(len(hit), 6)
        self.assertLess(max(hit), 0.70)

    def test_lock_play_court_stays_on_the_hint_court(self) -> None:
        frames: list[list[tuple[float, float, float]]] = []
        for i in range(24):
            home = (8.5, 0.30 + i * 0.004, 0.28)
            other = (12.0, 0.86, 0.30)
            frames.append([home, other])
        cx, tol = lock_play_court(frames, hint_cx=0.32)
        self.assertLess(cx, 0.50)
        self.assertLessEqual(tol, 0.28)


class HoldAfterRallyTests(unittest.TestCase):
    def test_stops_following_walkers_after_the_rally(self) -> None:
        xs: list[float | None] = [0.40, 0.42, 0.45, 0.48, 0.70, 0.72]
        ys: list[float | None] = [0.30, 0.28, 0.32, 0.29, 0.55, 0.56]
        ox, oy = hold_after_rally(xs, ys)
        self.assertIsNotNone(ox[4])
        self.assertLess(float(oy[4]), 0.35)
        self.assertAlmostEqual(float(oy[5]), float(oy[4]))
    def test_camera_pans_up_with_a_high_set(self) -> None:
        from follow_cam import plan_camera

        xs: list[float | None] = [0.50] * 12
        ys: list[float | None] = [0.40, 0.36, 0.32, 0.28, 0.24, 0.20, 0.16, 0.14, 0.14, 0.18, 0.22, 0.26]
        _ox, oy = plan_camera(xs, ys, hint_cx=0.50)
        self.assertLess(float(oy[7]), float(oy[0]))
        self.assertLess(float(oy[7]), 0.40)


class TrackQualityTests(unittest.TestCase):
    def test_rising_set_keeps_moving_up(self) -> None:
        from follow_cam import extend_rising_set

        xs: list[float | None] = [0.50, 0.51, 0.52, 0.53] + [None] * 6
        ys: list[float | None] = [0.40, 0.34, 0.28, 0.22] + [None] * 6
        ox, oy = extend_rising_set(xs, ys, extra=4)
        self.assertIsNotNone(ox[5])
        self.assertLess(float(oy[5]), 0.22)

    def test_walker_track_scores_near_zero(self) -> None:
        from follow_cam import track_quality

        xs: list[float | None] = [0.40 + i * 0.002 for i in range(20)]
        ys: list[float | None] = [0.52] * 20
        self.assertEqual(track_quality(xs, ys), 0.0)


class SmoothTrackTests(unittest.TestCase):
    def test_fills_a_short_gap_and_holds_a_long_gap(self) -> None:
        xs: list[float | None] = [0.20, 0.21, None, 0.23, 0.24] + [None] * 25 + [0.79, 0.80]
        ys: list[float | None] = [0.40, 0.40, None, 0.40, 0.40] + [None] * 25 + [0.40, 0.40]
        ox, oy = smooth_track(xs, ys, sigma=1.0, max_gap=5)
        self.assertAlmostEqual(float(ox[2]), 0.22, places=1)
        self.assertLess(float(ox[10]), 0.35)
        self.assertGreater(float(ox[-1]), 0.55)

    def test_drops_a_one_frame_teleport(self) -> None:
        xs: list[float | None] = [0.20, 0.21, 0.90, 0.22, 0.23]
        ys: list[float | None] = [0.40, 0.40, 0.70, 0.40, 0.40]
        ox, _ = smooth_track(xs, ys, sigma=1.0, max_gap=4)
        self.assertLess(float(ox[2]), 0.40)


class SoftmaxSampleTests(unittest.TestCase):
    def test_softmax_prefers_the_higher_score(self) -> None:
        from cut_vertex_reel import softmax_sample

        rows = [
            {"start_s": 10.0, "score": 8, "id": "low"},
            {"start_s": 20.0, "score": 10, "id": "high"},
            {"start_s": 30.0, "score": 8, "id": "low2"},
        ]
        n_high = 0
        for seed in range(40):
            picked = softmax_sample(rows, k=1, temperature=0.35, seed=seed)
            if picked[0]["id"] == "high":
                n_high += 1
        self.assertGreater(n_high, 20)


class DampTests(unittest.TestCase):
    def test_damp_moves_toward_the_target(self) -> None:
        pos, vel = 0.0, 0.0
        for _ in range(30):
            pos, vel = damp(pos, vel, target=1.0, dt=1 / 30)
        self.assertGreater(pos, 0.5)
        self.assertLess(abs(pos - 1.0), 0.15)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
