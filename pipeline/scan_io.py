"""Read and write 2.5 scan JSON. Map window clocks to source time."""

from __future__ import annotations

import json
import re
from pathlib import Path

HIT_GLOB = "vertex_best_*.json"
MIN_SCORE = 8
WINDOW_S = 600


def clock(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def parse_clock(text: str) -> float:
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"bad clock {text!r}")


def parse_rel(text: str) -> float:
    parts = [int(p) for p in str(text).strip().replace(".", ":").split(":")]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"bad clock {text!r}")


def to_source(t: float, window_start: int, window_end: int) -> float | None:
    """Map a model clock to match seconds. Models mix window-relative and match time."""
    span = window_end - window_start
    if window_start <= t <= window_end:
        return float(t)
    if 0 <= t <= span + 20:
        return float(window_start + t)
    return None


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted(spans)
    out: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for a, b in ordered[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def uncovered_windows(
    spans: list[tuple[int, int]],
    video_end: int,
    window_s: int = WINDOW_S,
) -> list[tuple[int, int]]:
    merged = merge_spans(spans)
    gaps: list[tuple[int, int]] = []
    t = 0
    for a, b in merged:
        if t < a:
            gaps.append((t, a))
        t = max(t, b)
    if t < video_end:
        gaps.append((t, video_end))
    out: list[tuple[int, int]] = []
    for a, b in gaps:
        cur = a
        while cur < b:
            nxt = min(cur + window_s, b)
            out.append((cur, nxt))
            cur = nxt
    return out


def normalize_rows(
    raw: object,
    window_start: int,
    window_end: int,
    video_end: int,
) -> list[dict]:
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = raw.get("highlights") or raw.get("clips") or []
    else:
        rows = []
    out: list[dict] = []
    for h in rows:
        if not isinstance(h, dict):
            continue
        try:
            rel_a = parse_rel(str(h.get("start", "0")))
            rel_b = parse_rel(str(h.get("end", "0")))
        except ValueError:
            continue
        if rel_b < rel_a:
            rel_a, rel_b = rel_b, rel_a
        src_a = to_source(rel_a, window_start, window_end)
        src_b = to_source(rel_b, window_start, window_end)
        if src_a is None or src_b is None:
            continue
        if src_b < src_a:
            src_a, src_b = src_b, src_a
        if src_a >= float(video_end):
            continue
        src_b = min(src_b, float(video_end))
        try:
            score_n = int(h.get("score"))
        except (TypeError, ValueError):
            score_n = 0
        side = str(h.get("team_side") or "").lower()
        if side in {"near", "far"}:
            side = "left" if side == "near" else "right"
        out.append(
            {
                "start": clock(rel_a),
                "end": clock(rel_b),
                "source_start": clock(src_a),
                "source_end": clock(src_b),
                "score": score_n,
                "event": str(h.get("event") or "play"),
                "team_side": side or "left",
                "keep": score_n >= MIN_SCORE,
                "reason": str(h.get("reason") or ""),
            }
        )
    return out


def dest_path(start_s: int, end_s: int, out_dir: Path) -> Path:
    return out_dir / f"vertex_best_{start_s}_{end_s}.json"


def existing_spans(out_dir: Path) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for path in out_dir.glob(HIT_GLOB):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        try:
            spans.append((int(parts[-2]), int(parts[-1])))
        except ValueError:
            continue
    return spans


def is_deadline(exc: BaseException) -> bool:
    text = str(exc)
    return (
        "DEADLINE_EXCEEDED" in text
        or "504" in text
        or "local timeout" in text
        or isinstance(exc, TimeoutError)
    )


def is_rate_limit(exc: BaseException) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def write_payload(payload: dict, out_dir: Path) -> int:
    req = payload["requested_window"]
    start_s = int(str(req["start"]).rstrip("s"))
    end_s = int(str(req["end"]).rstrip("s"))
    path = dest_path(start_s, end_s, out_dir)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    n = sum(1 for h in payload["highlights"] if h.get("keep"))
    print(f"wrote {path.name} items={len(payload['highlights'])} keeps={n}", flush=True)
    for h in payload["highlights"]:
        flag = "KEEP" if h["keep"] else "drop"
        print(
            f"  {flag} {h['source_start']}-{h['source_end']} {h['score']} {h['event']} {h['reason']}",
            flush=True,
        )
    return n


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


def load_all_keeps(scan_dir: Path) -> list[dict]:
    files = sorted(scan_dir.glob(HIT_GLOB))
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
            keeps.append(
                {
                    "start_s": parse_clock(h["source_start"]),
                    "end_s": parse_clock(h["source_end"]),
                    "event": str(h.get("event") or "play"),
                    "team_side": str(h.get("team_side") or "").lower(),
                    "score": score,
                }
            )
    return keeps
