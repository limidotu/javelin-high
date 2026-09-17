"""Cut a 9:16 reel from Vertex keep timestamps. Needs proof/source.mp4."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detect_highlights import concat
from follow_cam import clip_window, MIN_TRACK_Q, render_follow

SOURCE = ROOT / "source.mp4"
SCAN_DIR = ROOT / "scan"
HIT_GLOB = "vertex_best_*.json"
HIT_FALLBACK = "vertex_hl_*.json"
CROP = {"left": 0.35, "right": 0.65, "near": 0.5, "far": 0.5}
MIN_SCORE = 8
MAX_CLIPS = 8
SOFTMAX_T = 0.30
RNG_SEED = 42
TARGET_SECS = 120.0


def parse_clock(text: str) -> float:
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"bad clock {text!r}")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "play"


def merge_windows(rows: list[dict]) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["start_s"])
    out: list[dict] = []
    for row in rows:
        if out and row["start_s"] <= out[-1]["end_s"] + 1.5:
            out[-1]["end_s"] = max(out[-1]["end_s"], row["end_s"])
            out[-1]["event"] = f"{out[-1]['event']}+{row['event']}"
            out[-1]["score"] = max(int(out[-1]["score"]), int(row["score"]))
            continue
        out.append(dict(row))
    return out


def load_all_keeps(scan_dir: Path | None = None) -> list[dict]:
    base = scan_dir if scan_dir is not None else SCAN_DIR
    files = sorted(base.glob(HIT_GLOB))
    if not files:
        files = sorted(base.glob(HIT_FALLBACK))
    if not files:
        files = sorted(SCAN_DIR.glob(HIT_GLOB))
    if not files:
        files = sorted((ROOT / "out").glob(HIT_GLOB))
    keeps: list[dict] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for h in payload.get("highlights", []):
            try:
                score = int(h.get("score") or 0)
            except (TypeError, ValueError):
                score = 0
            if not h.get("keep"):
                continue
            if score and score < MIN_SCORE:
                continue
            side = str(h.get("team_side") or "").lower()
            keeps.append(
                {
                    "start_s": parse_clock(h["source_start"]),
                    "end_s": parse_clock(h["source_end"]),
                    "event": str(h.get("event") or "play"),
                    "team_side": side,
                    "crop": CROP.get(side, 0.5),
                    "score": score,
                }
            )
    return keeps


def load_keeps() -> list[dict]:
    return softmax_sample(load_all_keeps(), k=MAX_CLIPS, temperature=SOFTMAX_T, seed=RNG_SEED)


def _weighted_sample(
    rows: list[dict],
    k: int,
    temperature: float,
    seed: int,
) -> list[dict]:
    if not rows or k <= 0:
        return []
    k = min(k, len(rows))
    scores = np.array([float(r["score"]) for r in rows], dtype=np.float64)
    logits = (scores - scores.max()) / max(temperature, 1e-6)
    logits = np.clip(logits, -40.0, 40.0)
    p = np.exp(logits)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=k, replace=False, p=p)
    return [rows[int(i)] for i in idx]


def softmax_sample(
    rows: list[dict],
    k: int,
    temperature: float = 0.45,
    seed: int = 42,
) -> list[dict]:
    """Sample k clips. Take every score-10 play first, then weight the rest."""
    if not rows:
        return []
    k = min(k, len(rows))
    tens = [r for r in rows if int(r["score"]) >= 10]
    rest = [r for r in rows if int(r["score"]) < 10]
    picked = _weighted_sample(tens, min(k, len(tens)), temperature, seed)
    if len(picked) < k:
        picked.extend(_weighted_sample(rest, k - len(picked), temperature, seed + 1))
    picked.sort(key=lambda r: (-int(r["score"]), r["start_s"]))
    return picked


def main() -> int:
    from gen_reel import main as gen_main

    sys.argv = [sys.argv[0], "1"] if len(sys.argv) < 2 else sys.argv
    return gen_main()


if __name__ == "__main__":
    raise SystemExit(main())
