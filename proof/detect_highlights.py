"""PROTOTYPE: motion peaks -> 9:16 clips. Not production."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "sample.mp4"
OUT_DIR = ROOT / "out"
PEAK_DIR = OUT_DIR / "peaks"
CLIP_DIR = OUT_DIR / "clips"


def ffmpeg_bin() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def motion_series(path: Path, sample_fps: float = 8.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / sample_fps)))
    prev = None
    times = []
    energy = []
    cx = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        small = cv2.resize(gray, (160, 90))
        if prev is not None:
            diff = cv2.absdiff(small, prev)
            energy.append(float(diff.mean()))
            times.append(idx / src_fps)
            col = diff.mean(axis=0)
            xs = np.arange(col.size)
            w = col + 1e-6
            cx.append(float((xs * w).sum() / w.sum()) / col.size)
        prev = small
        idx += 1
    cap.release()
    return np.array(times), np.array(energy), np.array(cx)


def smooth(y: np.ndarray, win: int = 7) -> np.ndarray:
    if y.size < win:
        return y
    k = np.ones(win) / win
    return np.convolve(y, k, mode="same")


def peaks(times: np.ndarray, y: np.ndarray, min_gap: float = 8.0, top_n: int = 4) -> list[int]:
    dy = np.gradient(y)
    cand = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and dy[i - 1] > 0:
            cand.append(i)
    cand.sort(key=lambda i: y[i], reverse=True)
    picked: list[int] = []
    for i in cand:
        t = times[i]
        if all(abs(t - times[j]) >= min_gap for j in picked):
            picked.append(i)
        if len(picked) >= top_n:
            break
    picked.sort(key=lambda i: times[i])
    return picked


def grab_frame(path: Path, t: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_bin(),
            "-y",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            str(dest),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def cut_vertical(path: Path, start: float, dur: float, crop_x_frac: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    crop_w = int(h * 9 / 16)
    crop_w = min(crop_w, w)
    cx = int(crop_x_frac * w)
    x = max(0, min(w - crop_w, cx - crop_w // 2))
    vf = f"crop={crop_w}:{h}:{x}:0,scale=1080:1920"
    subprocess.run(
        [
            ffmpeg_bin(),
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-i",
            str(path),
            "-t",
            f"{dur:.3f}",
            "-vf",
            vf,
            "-af",
            "loudnorm",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(dest),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def concat(clips: list[Path], dest: Path) -> None:
    lst = dest.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in clips), encoding="utf-8")
    subprocess.run(
        [
            ffmpeg_bin(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(lst),
            "-c",
            "copy",
            str(dest),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    if not SAMPLE.exists():
        print(f"missing {SAMPLE}", file=sys.stderr)
        return 1
    times, energy, cx = motion_series(SAMPLE)
    sm = smooth(energy)
    idx = peaks(times, sm)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    clip_paths = []
    for n, i in enumerate(idx, start=1):
        t = float(times[i])
        score = float(sm[i])
        crop_x = float(cx[i]) if i < len(cx) else 0.45
        start = max(0.0, t - 2.0)
        frame_path = PEAK_DIR / f"peak_{n}_{t:06.2f}.jpg"
        clip_path = CLIP_DIR / f"clip_{n}.mp4"
        grab_frame(SAMPLE, t, frame_path)
        cut_vertical(SAMPLE, start, 7.0, crop_x, clip_path)
        clip_paths.append(clip_path)
        records.append(
            {
                "rank": n,
                "t_in_sample_s": round(t, 3),
                "t_in_source_s": round(480.0 + t, 3),
                "motion": round(score, 4),
                "crop_x_frac": round(crop_x, 3),
                "frame": str(frame_path),
                "clip": str(clip_path),
            }
        )
    reel = OUT_DIR / "vertical_reel.mp4"
    concat(clip_paths, reel)
    report = {
        "sample": str(SAMPLE),
        "sample_offset_s": 480,
        "n_samples": int(times.size),
        "mean_motion": round(float(energy.mean()), 4) if energy.size else 0,
        "max_motion": round(float(energy.max()), 4) if energy.size else 0,
        "peaks": records,
        "reel": str(reel),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
