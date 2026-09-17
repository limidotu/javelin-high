"""Render one 15 s x 3-clip short for a camera iteration. No Vertex call."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cut_vertex_reel import CLIP_DIR, SOURCE, load_all_keeps, merge_windows, slug, softmax_sample
from detect_highlights import concat, ffmpeg_bin
from follow_cam import clip_window, MIN_TRACK_Q, render_follow

OUT = ROOT / "out"


def dump_stills(reel: Path, n: int) -> None:
    still_dir = OUT / "diag" / f"iter_{n}"
    still_dir.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_bin()
    for t in (3, 8, 10, 13, 18, 23, 25, 28, 33, 38, 40, 43):
        dest = still_dir / f"t{t:02d}.jpg"
        subprocess.run(
            [ff, "-y", "-ss", str(t), "-i", str(reel), "-frames:v", "1", str(dest)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if not SOURCE.exists():
        print(f"missing {SOURCE}", file=sys.stderr)
        return 1
    rows = softmax_sample(load_all_keeps(), k=12, temperature=0.30, seed=100 + n)
    windows = merge_windows(rows)
    windows.sort(key=lambda w: (-int(w["score"]), w["start_s"]))
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    for w in windows:
        if len(clips) >= 3:
            break
        start, dur = clip_window(w["start_s"], w["end_s"])
        dest = CLIP_DIR / f"iter{n}_{len(clips)+1:02d}_{slug(w['event'])}.mp4"
        print(f"iter{n} {dest.name} {start:.1f}s +{dur:.1f}s score={w['score']}", flush=True)
        quality = render_follow(SOURCE, start, dur, dest, hint_cx=w["crop"], event=w["event"])
        if quality < MIN_TRACK_Q or not dest.exists():
            dest.unlink(missing_ok=True)
            print("  skip weak rally", flush=True)
            continue
        clips.append(dest)
    if len(clips) < 2:
        print("not enough strong rallies", file=sys.stderr)
        return 1
    reel = OUT / f"iter_{n}.mp4"
    concat(clips, reel)
    dump_stills(reel, n)
    print(str(reel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
