"""Polish one locked reel into a 60 second vertical cut. Edit the paths, then run this file."""

import json
import shutil
from pathlib import Path

from argon import polish_reel, probe_video, render_stage

SOURCE = Path(r"C:\Users\dlash\Downloads\reel.mp4")
DEST = Path(r"D:\school\cursor\javelin\high\pipeline\generations\gen_14\tiktok_v5.mp4")
COPY_TO = Path(r"C:\Users\dlash\Downloads\tiktok_v5.mp4")
KEEPS = Path(r"D:\school\cursor\javelin\high\pipeline\generations\gen_14\keeps.json")
NOTES = Path(r"D:\school\cursor\javelin\high\pipeline\generations\gen_14\tiktok_v5_argon.json")
PROXY = Path(r"D:\school\cursor\javelin\high\pipeline\generations\gen_14\argon_v2")
THINKING = "HIGH"
TARGET_S = 60.0
LAYOUT = "blurred-bg"
SPORT = "volleyball"
RENDER_ONLY = True


def main() -> None:
    print(f"SOURCE {SOURCE}", flush=True)
    print(f"DEST {DEST}", flush=True)
    if RENDER_ONLY:
        shots = json.loads(NOTES.read_text(encoding="utf-8"))["shots_locked"]
        print(f"RENDER shots={len(shots)} from {NOTES.name}", flush=True)
        render_stage(SOURCE, shots, DEST, layout=LAYOUT, captions=False, target_s=TARGET_S, pace=True)
        meta = probe_video(DEST)
        print(f"DONE {DEST.name} {meta['duration']:.2f}s {meta['width']}x{meta['height']}", flush=True)
        if abs(meta["duration"] - TARGET_S) > 0.08:
            raise RuntimeError(f"TikTok is {meta['duration']:.2f}s. Target is {TARGET_S:.2f}s.")
        shutil.copy2(DEST, COPY_TO)
        print(f"COPY {COPY_TO}", flush=True)
        return
    polish_reel(
        SOURCE,
        DEST,
        sport=SPORT,
        thinking=THINKING,
        layout=LAYOUT,
        captions=False,
        hook=False,
        target_s=TARGET_S,
        keeps=KEEPS,
        proxy_dir=PROXY,
    )
    meta = probe_video(DEST)
    print(f"DONE {DEST.name} {meta['duration']:.2f}s {meta['width']}x{meta['height']}", flush=True)
    if abs(meta["duration"] - TARGET_S) > 0.08:
        raise RuntimeError(f"TikTok is {meta['duration']:.2f}s. Target is {TARGET_S:.2f}s.")
    shutil.copy2(DEST, COPY_TO)
    print(f"COPY {COPY_TO}", flush=True)


if __name__ == "__main__":
    main()
