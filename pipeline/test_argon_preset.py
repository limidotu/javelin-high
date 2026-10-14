"""Check the accepted Argon preset through the request and publish paths."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import argon
import argon_video
import run_job
import web

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "argon_players_best_light.json").read_text())


class PresetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spans = [tuple(row) for row in FIXTURE["spans"]]
        self.edl = copy.deepcopy(FIXTURE["decisions"])

    def test_accepted_winner_keeps_every_interval_and_crop(self) -> None:
        argon.validate_pace_plan(self.edl, self.spans, 60)
        shots = argon.fit_pace(self.spans, argon.notes_by_index(self.edl, len(self.spans)), 60)
        expected = [row for row in self.edl["shots"] if row["keep"]]
        self.assertEqual(len(shots), len(expected))
        for shot, row in zip(shots, expected):
            self.assertAlmostEqual(shot["in_sec"], row["in_s"], places=4)
            self.assertAlmostEqual(shot["out_sec"], row["out_s"], places=4)
            self.assertEqual(shot["center_x"], row["center_x"])

    def test_reference_reel_generates_the_exact_tested_prompt(self) -> None:
        prompt = argon.build_pace_prompt("volleyball", FIXTURE["video_s"], self.spans, FIXTURE["target_s"])
        self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), FIXTURE["prompt_sha256"])

    def test_leftmost_legal_crop_does_not_become_a_center_crop(self) -> None:
        self.edl["shots"][0]["center_x"] = 0.0
        argon.validate_pace_plan(self.edl, self.spans, 60)
        shots = argon.fit_pace(self.spans, argon.notes_by_index(self.edl, len(self.spans)), 60)
        self.assertEqual(shots[0]["center_x"], 0.0)

    def test_missing_duplicate_and_out_of_bounds_shots_are_rejected(self) -> None:
        missing = copy.deepcopy(self.edl)
        missing["shots"].pop()
        duplicate = copy.deepcopy(self.edl)
        duplicate["shots"][-1]["index"] = 0
        outside = copy.deepcopy(self.edl)
        outside["shots"][0]["out_s"] = 15.5
        for plan in (missing, duplicate, outside):
            with self.subTest(plan=plan), self.assertRaises(argon.InvalidArgonPlan):
                argon.validate_pace_plan(plan, self.spans, 60)

    def test_short_plan_and_nonfinite_crop_are_rejected(self) -> None:
        short = copy.deepcopy(self.edl)
        short["shots"][0]["in_s"] += 2
        invalid_crop = copy.deepcopy(self.edl)
        invalid_crop["shots"][0]["center_x"] = float("nan")
        for plan in (short, invalid_crop):
            with self.subTest(plan=plan), self.assertRaises(argon.InvalidArgonPlan):
                argon.validate_pace_plan(plan, self.spans, 60)

    def test_new_reel_gets_its_own_bounds_sport_and_target(self) -> None:
        prompt = argon.build_pace_prompt("basketball", 200.25, [(0, 40), (40, 200.25)], 75)
        self.assertIn("200.25-second landscape reel", prompt)
        self.assertIn("75-second basketball short", prompt)
        self.assertIn("1: 40.000 to 200.250", prompt)
        self.assertIn("exactly 75.000 seconds", prompt)
        self.assertIn("steals", prompt)
        self.assertNotIn("123.64", prompt)
        self.assertNotIn("108.500", prompt)

    def test_request_uses_original_uri_and_tested_video_settings(self) -> None:
        client = MagicMock()
        client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(self.edl))
        with patch.object(argon.genai, "Client", return_value=client):
            got = argon.ask_argon("gs://private-reels/source.mp4", "brief", argon.PACE_SCHEMA, "gemini-4-argon", "HIGH")
        self.assertEqual(got, self.edl)
        request = client.models.generate_content.call_args.kwargs
        video = request["contents"][0]
        self.assertEqual(video.file_data.file_uri, "gs://private-reels/source.mp4")
        self.assertIsNone(video.inline_data)
        self.assertEqual(video.video_metadata.fps, 8)
        self.assertEqual(request["config"].temperature, 0.7)
        self.assertEqual(request["config"].thinking_config.thinking_level.value, "HIGH")
        self.assertEqual(request["config"].media_resolution.value, "MEDIA_RESOLUTION_HIGH")
        self.assertEqual(request["config"].max_output_tokens, 32768)

    def test_web_and_cli_default_to_the_accepted_preset(self) -> None:
        options = web.job_options({})
        self.assertTrue(options["argon"])
        self.assertEqual(options["thinking"], "HIGH")
        self.assertEqual(options["argon_model"], "gemini-4-argon")
        with patch("sys.argv", ["run_job.py"]), patch.object(run_job, "run_job", return_value=0) as job:
            self.assertEqual(run_job.main(), 0)
        self.assertTrue(job.call_args.kwargs["argon"])
        self.assertEqual(job.call_args.kwargs["thinking"], "HIGH")
        with patch("sys.argv", ["run_job.py", "--no-argon"]), patch.object(run_job, "run_job", return_value=0) as job:
            run_job.main()
        self.assertFalse(job.call_args.kwargs["argon"])


class PublishTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "reel.mp4"
        self.source.write_bytes(b"original reel")
        self.dest = self.root / "tiktok.mp4"
        self.edl = copy.deepcopy(FIXTURE["decisions"])
        self.spans = [tuple(row) for row in FIXTURE["spans"]]
        self.meta = {"duration": FIXTURE["video_s"], "width": 1280, "height": 720}
        self.upload = {"video_uri": "gs://private-reels/original.mp4", "sha256": "fixture", "bytes": 13}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_bad_timing_is_sent_back_to_argon_before_publication(self) -> None:
        short = copy.deepcopy(self.edl)
        short["shots"][0]["in_s"] += 2

        def render(source, shots, dest, **kwargs):
            self.assertFalse(self.dest.exists())
            self.assertEqual(len(shots), 7)
            dest.write_bytes(b"complete video")

        with patch.object(argon, "probe_video", return_value=self.meta), patch.object(argon, "upload_reel", return_value=self.upload), patch.object(argon, "ask_argon", side_effect=[short, self.edl]) as ask, patch.object(argon, "render_stage", side_effect=render):
            result = argon.polish_reel(self.source, self.dest, spans=self.spans)
        self.assertEqual(ask.call_count, 2)
        self.assertIn("58.000", ask.call_args_list[1].args[1])
        self.assertEqual(ask.call_args_list[0].args[0], self.upload["video_uri"])
        self.assertEqual(self.dest.read_bytes(), b"complete video")
        self.assertEqual(result["prompt_version"], "players-best-light-v1")
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(result["video_fps"], 8)
        self.assertEqual(result["source_input"], self.upload)

    def test_invalid_plan_stops_before_rendering(self) -> None:
        short = copy.deepcopy(self.edl)
        short["shots"][0]["in_s"] += 2
        with patch.object(argon, "probe_video", return_value=self.meta), patch.object(argon, "upload_reel", return_value=self.upload), patch.object(argon, "ask_argon", return_value=short) as ask, patch.object(argon, "render_stage") as render:
            with self.assertRaises(argon.InvalidArgonPlan):
                argon.polish_reel(self.source, self.dest, spans=self.spans)
        self.assertEqual(ask.call_count, 3)
        render.assert_not_called()
        self.assertFalse(self.dest.exists())

    def test_failed_render_does_not_replace_a_completed_video(self) -> None:
        self.dest.write_bytes(b"previous complete video")

        def render(source, shots, dest, **kwargs):
            dest.write_bytes(b"partial video")
            raise RuntimeError("render failed")

        with patch.object(argon, "probe_video", return_value=self.meta), patch.object(argon, "upload_reel", return_value=self.upload), patch.object(argon, "ask_argon", return_value=self.edl), patch.object(argon, "render_stage", side_effect=render):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                argon.polish_reel(self.source, self.dest, spans=self.spans)
        self.assertEqual(self.dest.read_bytes(), b"previous complete video")

    def test_short_source_stops_before_upload_or_model_call(self) -> None:
        with patch.object(argon, "probe_video", return_value={**self.meta, "duration": 30}), patch.object(argon, "upload_reel") as upload, patch.object(argon, "ask_argon") as ask:
            with self.assertRaises(ValueError):
                argon.polish_reel(self.source, self.dest)
        upload.assert_not_called()
        ask.assert_not_called()


class UploadTests(unittest.TestCase):
    def test_original_bytes_are_uploaded_to_a_private_bucket(self) -> None:
        client = MagicMock()
        client.lookup_bucket.return_value = None
        bucket = client.bucket.return_value
        client.create_bucket.return_value = bucket
        blob = bucket.blob.return_value
        blob.exists.return_value = False
        blob.name = "reels/cached.mp4"
        client.__enter__.return_value = client
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "reel.mp4"
            source.write_bytes(b"original full-resolution bytes")
            with patch.object(argon_video.storage, "Client", return_value=client), patch.dict("os.environ", {"ARGON_VIDEO_BUCKET": "private-reels"}):
                result = argon_video.upload_reel(source)
            blob.upload_from_filename.assert_called_once_with(str(source), content_type="video/mp4", if_generation_match=0, timeout=300)
            self.assertEqual(result["bytes"], source.stat().st_size)
        self.assertTrue(bucket.iam_configuration.uniform_bucket_level_access_enabled)
        self.assertEqual(bucket.iam_configuration.public_access_prevention, "enforced")
        self.assertEqual(result["video_uri"], "gs://private-reels/reels/cached.mp4")


if __name__ == "__main__":
    unittest.main()
