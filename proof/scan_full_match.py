"""Scan the full YouTube match in 10-minute windows. Elite plays only."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

from google import genai
from google.genai import types

from gemini_defaults import SCAN_MODEL, VERTEX_LOCATION, VERTEX_PROJECT

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "scan"
PROJECT = VERTEX_PROJECT
YOUTUBE = "https://www.youtube.com/watch?v=sbGPehFw7n8"
MODEL = SCAN_MODEL
VIDEO_END_S = 6632
WINDOW_S = 600
MIN_SCORE = 8
HTTP_TIMEOUT_MS = 60_000
MAX_429_STREAK = 3

PROMPT = """
Amateur indoor volleyball. Pick THE BEST highlight moments only.

KEEP only elite plays: stuff blocks, powerful kills that land, diving digs
that save a point, then a kill, clean aces nobody touches. Both teams.

REJECT routine rallies, average spikes, walking, huddles, serve setup,
funny misses, net luck, dead time, and anything you would skip on a recap.

Return JSON only: a list of objects with keys
start,end,score,event,team_side,reason.
Times are MM:SS relative to THIS window, not the full match.
score is 0-10. Use 8, 9, or 10 only for a keep. Prefer 2 to 5 items.
team_side is left or right. Pad each rally about 6 to 10 seconds.
If this window has no elite play, return [].
""".strip()


def clock(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


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
    video_end: int | None = None,
    window_s: int | None = None,
) -> list[tuple[int, int]]:
    end = VIDEO_END_S if video_end is None else video_end
    win = WINDOW_S if window_s is None else window_s
    merged = merge_spans(spans)
    gaps: list[tuple[int, int]] = []
    t = 0
    for a, b in merged:
        if t < a:
            gaps.append((t, a))
        t = max(t, b)
    if t < end:
        gaps.append((t, end))
    out: list[tuple[int, int]] = []
    for a, b in gaps:
        cur = a
        while cur < b:
            nxt = min(cur + win, b)
            out.append((cur, nxt))
            cur = nxt
    return out


def normalize_rows(
    raw: object,
    window_start: int,
    window_end: int,
    video_end: int | None = None,
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
        limit = float(VIDEO_END_S if video_end is None else video_end)
        if src_a >= limit:
            continue
        src_b = min(src_b, limit)
        score = h.get("score")
        try:
            score_n = int(score)
        except (TypeError, ValueError):
            score_n = 0
        event = str(h.get("event") or "play")
        side = str(h.get("team_side") or "").lower()
        if side in {"near", "far"}:
            side = "left" if side == "near" else "right"
        keep = score_n >= MIN_SCORE
        out.append(
            {
                "start": clock(rel_a),
                "end": clock(rel_b),
                "source_start": clock(src_a),
                "source_end": clock(src_b),
                "score": score_n,
                "event": event,
                "team_side": side or "left",
                "keep": keep,
                "reason": str(h.get("reason") or ""),
            }
        )
    return out


def scan_window(client: genai.Client, start_s: int, end_s: int) -> dict:
    video = types.Part(
        file_data=types.FileData(file_uri=YOUTUBE, mime_type="video/mp4"),
        video_metadata=types.VideoMetadata(
            start_offset=f"{start_s}s",
            end_offset=f"{end_s}s",
            fps=1,
        ),
    )
    t0 = time.time()
    print(f"  call {start_s}-{end_s} ...", flush=True)

    def _run():
        return client.models.generate_content(
            model=MODEL,
            contents=[video, PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS),
            ),
        )

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_run)
    try:
        response = future.result(timeout=HTTP_TIMEOUT_MS / 1000.0)
    except FuturesTimeout as exc:
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"local timeout {start_s}-{end_s}") from exc
    else:
        pool.shutdown(wait=True, cancel_futures=False)
    elapsed = round(time.time() - t0, 1)
    print(f"  ok {start_s}-{end_s} {elapsed}s", flush=True)
    text = response.text or "[]"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = []
    rows = normalize_rows(parsed, start_s, end_s)
    usage = getattr(response, "usage_metadata", None)
    payload = {
        "model": MODEL,
        "backend": "vertex-ai",
        "project": PROJECT,
        "location": VERTEX_LOCATION,
        "youtube": YOUTUBE,
        "requested_window": {
            "start": f"{start_s}s",
            "end": f"{end_s}s",
            "label": f"{clock(start_s)}-{clock(end_s)}",
        },
        "elapsed_s": elapsed,
        "prompt_token_count": getattr(usage, "prompt_token_count", None),
        "candidates_token_count": getattr(usage, "candidates_token_count", None),
        "thoughts_token_count": getattr(usage, "thoughts_token_count", None),
        "total_token_count": getattr(usage, "total_token_count", None),
        "timestamp_note": "Times in start/end are window-relative. source_* is match time.",
        "highlights": rows,
    }
    return payload


def dest_path(start_s: int, end_s: int, out_dir: Path | None = None) -> Path:
    return (out_dir or OUT) / f"vertex_best_{start_s}_{end_s}.json"


def existing_spans(out_dir: Path | None = None) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for path in (out_dir or OUT).glob("vertex_best_*.json"):
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


def scan_range(client: genai.Client, start_s: int, end_s: int) -> list[dict]:
    """One call. Do not retry. Do not split. Skip a dead window."""
    try:
        return [scan_window(client, start_s, end_s)]
    except Exception as exc:
        if is_rate_limit(exc):
            raise
        if is_deadline(exc):
            print(f"  skip {start_s}-{end_s} deadline", flush=True)
            return []
        raise


def write_payload(payload: dict, out_dir: Path | None = None) -> int:
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


def main() -> int:
    global OUT
    if len(sys.argv) > 1:
        OUT = Path(sys.argv[1])
        if not OUT.is_absolute():
            OUT = ROOT / OUT
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"MODEL={MODEL} location={VERTEX_LOCATION} out={OUT}", flush=True)
    client = genai.Client(vertexai=True, project=PROJECT, location=VERTEX_LOCATION)
    total_keep = 0
    streak_429 = 0
    spans = existing_spans()
    todo = uncovered_windows(spans)
    print(f"gaps={todo[:8]}... n={len(todo)}", flush=True)
    for start_s, end_s in todo:
        print(f"SCAN {MODEL} {start_s}s-{end_s}s", flush=True)
        try:
            payloads = scan_range(client, start_s, end_s)
        except Exception as exc:
            if is_rate_limit(exc):
                streak_429 += 1
                print(f"  skip {start_s}-{end_s} 429 streak={streak_429}", flush=True)
                if streak_429 >= MAX_429_STREAK:
                    print("FAIL too many 429s", file=sys.stderr, flush=True)
                    return 1
                continue
            print(f"FAIL {start_s}s-{end_s}s {exc}", file=sys.stderr, flush=True)
            return 1
        streak_429 = 0
        for payload in payloads:
            total_keep += write_payload(payload)
        spans = existing_spans()
    print(f"TOTAL_KEEP={total_keep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
