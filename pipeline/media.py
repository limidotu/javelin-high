"""FFmpeg 16:9 cuts and small Gemini proxies. No crop. No follow."""

from __future__ import annotations

import subprocess
from pathlib import Path


def ffmpeg_bin() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


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

SCAN_PROXY_VF = "fps=1,scale=640:360"
REVIEW_PROXY_VF = "scale=854:480,fps=10"
INLINE_MAX_BYTES = 18_000_000


def probe_secs(path: Path) -> float:
    ff = ffmpeg_bin()
    proc = subprocess.run(
        [ff, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    )
    err = proc.stderr or ""
    for line in err.splitlines():
        if "Duration:" in line:
            clock = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = clock.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def cut_wide(source: Path, start: float, dur: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_bin(),
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-i",
            str(source),
            "-t",
            f"{dur:.3f}",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _encode_proxy(
    source: Path,
    dest: Path,
    vf: str,
    extra_in: list[str] | None = None,
    crf: str = "32",
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_bin(), "-y"]
    if extra_in:
        cmd.extend(extra_in)
    cmd.extend(
        [
            "-i",
            str(source),
            "-vf",
            vf,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ]
    )
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dest


def make_scan_proxy(source: Path, start_s: float, end_s: float, dest: Path) -> Path:
    dur = max(0.1, end_s - start_s)
    extra = ["-ss", f"{max(0.0, start_s):.3f}", "-t", f"{dur:.3f}"]
    return _encode_proxy(source, dest, SCAN_PROXY_VF, extra_in=extra)


def make_review_proxy(clip: Path, dest: Path) -> Path:
    return _encode_proxy(clip, dest, REVIEW_PROXY_VF, crf="28")


def assert_inline_ok(path: Path) -> None:
    size = path.stat().st_size
    if size > INLINE_MAX_BYTES:
        raise RuntimeError(f"proxy too large for Vertex inline: {path.name} {size} bytes")


def concat_wide(clips: list[Path], dest: Path) -> None:
    concat(clips, dest)
