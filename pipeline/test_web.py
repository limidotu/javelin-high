"""One-page upload: size cap, MP4 check, and job status."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

import web

FTPY = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 20


class LooksLikeMp4Tests(unittest.TestCase):
    def test_ftyp_is_mp4(self) -> None:
        self.assertTrue(web.looks_like_mp4(FTPY))

    def test_text_is_not_mp4(self) -> None:
        self.assertFalse(web.looks_like_mp4(b"hello world"))

    def test_short_is_not_mp4(self) -> None:
        self.assertFalse(web.looks_like_mp4(b"ft"))

    def test_filename(self) -> None:
        self.assertTrue(web.filename_is_mp4("Game.MP4"))
        self.assertFalse(web.filename_is_mp4("game.mp4.exe"))
        self.assertFalse(web.filename_is_mp4("game.mov"))


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._root = web.ROOT
        self._max = web.MAX_BYTES
        self._thread = web.RUN_IN_THREAD
        web.ROOT = self.root
        web.MAX_BYTES = 64
        web.RUN_IN_THREAD = False
        web.reset_state()
        self.client = TestClient(web.app)

    def tearDown(self) -> None:
        web.reset_state()
        web.ROOT = self._root
        web.MAX_BYTES = self._max
        web.RUN_IN_THREAD = self._thread
        self.tmp.cleanup()

    def test_page_has_one_file_input_and_one_out(self) -> None:
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.text.count('type="file"'), 1)
        self.assertEqual(res.text.count('id="out"'), 1)
        self.assertEqual(res.text.count('id="pick"'), 1)
        self.assertNotIn("<nav", res.text)
        self.assertNotIn('role="tab"', res.text)

    def test_reject_non_mp4_name(self) -> None:
        res = self.client.post(
            "/upload",
            content=FTPY,
            headers={"x-filename": "a.txt"},
        )
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["detail"], web.ERR_MP4)

    def test_reject_non_mp4_bytes(self) -> None:
        res = self.client.post(
            "/upload",
            content=b"hello world this is not a movie file at all!!",
            headers={"x-filename": "a.mp4"},
        )
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["detail"], web.ERR_MP4)

    def test_reject_oversize_header(self) -> None:
        res = self.client.post(
            "/upload",
            content=b"x",
            headers={"content-length": "100", "x-filename": "a.mp4"},
        )
        self.assertEqual(res.status_code, 413)
        self.assertEqual(res.json()["detail"], web.ERR_SIZE)

    def test_reject_oversize_body(self) -> None:
        res = self.client.post(
            "/upload",
            content=b"\x00\x00\x00\x18ftypisom" + b"\x00" * 80,
            headers={"x-filename": "a.mp4"},
        )
        self.assertEqual(res.status_code, 413)
        self.assertEqual(res.json()["detail"], web.ERR_SIZE)

    def test_upload_runs_job_and_serves_reel(self) -> None:
        seen = {}

        def fake(source: Path, gen_n: int, options: dict) -> int:
            dest = self.root / "generations" / f"gen_{gen_n}"
            dest.mkdir(parents=True)
            (dest / "reel.mp4").write_bytes(b"reel")
            (dest / "tiktok.mp4").write_bytes(b"tiktok")
            self.assertTrue(source.is_file())
            seen.update(options)
            return 0

        with patch.object(web, "run_pipeline", fake):
            res = self.client.post(
                "/upload",
                content=FTPY,
                headers={
                    "x-filename": "game.mp4",
                    "x-scan-model": "gemini-2.5-pro",
                    "x-argon": "0",
                },
            )
        self.assertEqual(res.status_code, 200)
        body = self.client.get("/status").json()
        self.assertEqual(body["status"], "done")
        self.assertTrue(body["reel"])
        self.assertTrue(body["tiktok"])
        self.assertEqual(seen["scan_model"], "gemini-2.5-pro")
        self.assertFalse(seen["argon"])
        reel = self.client.get("/reel")
        self.assertEqual(reel.status_code, 200)
        self.assertEqual(reel.content, b"reel")
        tiktok = self.client.get("/tiktok")
        self.assertEqual(tiktok.status_code, 200)
        self.assertEqual(tiktok.content, b"tiktok")

    def test_busy_rejects_second_upload(self) -> None:
        web.set_state(status="working", detail="Working", reel=None)
        res = self.client.post(
            "/upload",
            content=FTPY,
            headers={"x-filename": "game.mp4"},
        )
        self.assertEqual(res.status_code, 409)
        self.assertEqual(res.json()["detail"], web.ERR_BUSY)

    def test_reel_missing_when_idle(self) -> None:
        res = self.client.get("/reel")
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json()["detail"], web.ERR_REEL)

    def test_job_failure_sets_error(self) -> None:
        with patch.object(web, "run_pipeline", lambda source, gen_n, options: 1):
            res = self.client.post(
                "/upload",
                content=FTPY,
                headers={"x-filename": "game.mp4"},
            )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.client.get("/status").json()["status"], "error")
        self.assertEqual(self.client.get("/status").json()["detail"], web.ERR_JOB)

    def test_bad_model_is_rejected(self) -> None:
        res = self.client.post(
            "/upload",
            content=FTPY,
            headers={"x-filename": "a.mp4", "x-scan-model": "not a model"},
        )
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["detail"], "Model id is not valid.")

    def test_page_has_model_controls(self) -> None:
        res = self.client.get("/")
        self.assertIn('id="scanModel"', res.text)
        self.assertIn('id="reviewModel"', res.text)
        self.assertIn('id="argonOn"', res.text)
        self.assertNotIn("tabManage", res.text)
        self.assertNotIn("Good to see you", res.text)

    def test_argon_error_still_serves_reel(self) -> None:
        def fake(source: Path, gen_n: int, options: dict) -> int:
            dest = self.root / "generations" / f"gen_{gen_n}"
            dest.mkdir(parents=True)
            (dest / "reel.mp4").write_bytes(b"reel")
            raise RuntimeError("Argon returned no clips")

        with patch.object(web, "run_pipeline", fake):
            res = self.client.post(
                "/upload",
                content=FTPY,
                headers={"x-filename": "game.mp4"},
            )
        self.assertEqual(res.status_code, 200)
        body = self.client.get("/status").json()
        self.assertEqual(body["status"], "done")
        self.assertIn("Argon returned no clips", body["detail"])
        self.assertTrue(body["reel"])
        self.assertFalse(body["tiktok"])


if __name__ == "__main__":
    unittest.main()
