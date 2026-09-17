"""Score clips with gemini-3.8-flash. At most 10 videos per request."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

from google import genai
from google.genai import types

from clip_window import REVIEW_DUR, default_trim, trim_from_row
from gemini_defaults import REVIEW_MODEL, REVIEW_THINKING_LEVEL, VERTEX_LOCATION, VERTEX_PROJECT
from media import assert_inline_ok, make_review_proxy
from scan_io import is_rate_limit, run_with_retries, should_split_span
from pick import REVIEW_BATCH, batches, child_review_groups, pick_reel

ROOT = Path(__file__).resolve().parent
MODEL = REVIEW_MODEL
HTTP_TIMEOUT_MS = 300_000
MAX_429_STREAK = 3
BATCH_TRIES = 3
RETRY_SLEEP_S = 8.0
REVIEW_429_SLEEP_S = 45.0

PROMPT = """
Amateur indoor volleyball. You rank highlight clips.

Videos are in order. The first video is index 0. The next is index 1.
Each clip is 20 seconds. Extra time is before the finish. Score the best
play in the clip, not the pad.

Set start and end in THIS clip, in seconds from 0 to 20.
Ideal span is 15 seconds. 12 to 18 is OK. Do not stretch to 20 seconds
to include walking or dead time. End on a dead ball (point over).
Do not end on a diving dig that keeps the ball alive. Cut walking.

KEEP a clip if it shows an elite play: stuff blocks, powerful kills that land,
diving digs that save a point, then a kill, or a clean ace nobody touches.
Both teams. Use 8, 9, or 10 only for a keep.

REJECT routine rallies, average spikes, walking only, huddles, serve setup
with no hit, funny misses, net luck, and dead time with no elite play.

Return JSON only:
{"clips": [{"index": 0, "keep": true, "score": 9, "event": "kill", "start": 2.0, "end": 17.0, "reason": "..."}]}

Give one object per video. score is 0-10. start and end are seconds in this clip.
""".strip()

LONG_PROMPT = """
Amateur indoor volleyball. This is ONE 40-second window.
The long elite rally is near the END of the clip. Ignore a short error
at the start if a long rally comes after.

Set start and end in THIS clip, in seconds from 0 to 40.
Use up to 40 seconds. End on a dead ball (point over).
Do not end on a diving dig that keeps the ball alive.

Return JSON only:
{"clips": [{"index": 0, "keep": true, "score": 9, "event": "rally", "start": 5.0, "end": 39.0, "reason": "..."}]}
""".strip()


def _parse_review(text: str, batch_len: int, window_s: float = REVIEW_DUR) -> list[dict]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        rows = parsed.get("clips") or parsed.get("highlights") or []
    elif isinstance(parsed, list):
        rows = parsed
    else:
        rows = []
    by_index: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= batch_len:
            continue
        try:
            score = int(row.get("score") or 0)
        except (TypeError, ValueError):
            score = 0
        keep = bool(row.get("keep"))
        clip_window_s = float(row.get("window_dur_s") or window_s)
        trim_a, trim_b = trim_from_row(row, clip_window_s)
        by_index[idx] = {
            "index": idx,
            "keep": keep,
            "score": score,
            "event": str(row.get("event") or "play"),
            "reason": str(row.get("reason") or ""),
            "trim_start_s": round(trim_a, 3),
            "trim_end_s": round(trim_b, 3),
        }
    out: list[dict] = []
    default_a, default_b = default_trim(window_s)
    for i in range(batch_len):
        if i in by_index:
            out.append(by_index[i])
        else:
            out.append(
                {
                    "index": i,
                    "keep": False,
                    "score": 0,
                    "event": "play",
                    "reason": "missing review",
                    "trim_start_s": round(default_a, 3),
                    "trim_end_s": round(default_b, 3),
                }
            )
    return out


def assert_review_complete(rows: list[dict]) -> list[dict]:
    missing = sum(1 for r in rows if r.get("reason") == "missing review")
    if missing:
        raise RuntimeError(
            f"400 INVALID_ARGUMENT incomplete review missing={missing}"
        )
    return rows


def review_batch(
    client: genai.Client,
    clips: list[dict],
    proxy_dir: Path,
    prompt: str = PROMPT,
) -> list[dict]:
    parts: list[types.Part] = []
    for i, rec in enumerate(clips):
        src = Path(rec["path"])
        proxy = proxy_dir / f"{src.stem}_rev.mp4"
        print(f"  proxy review {i} {src.name}", flush=True)
        make_review_proxy(src, proxy)
        assert_inline_ok(proxy)
        parts.append(types.Part.from_bytes(data=proxy.read_bytes(), mime_type="video/mp4"))
    t0 = time.time()
    print(f"  call review n={len(parts)} ...", flush=True)

    def _run():
        return client.models.generate_content(
            model=MODEL,
            contents=[*parts, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                thinking_config=types.ThinkingConfig(
                    thinking_level=REVIEW_THINKING_LEVEL
                ),
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS),
            ),
        )

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_run)
    try:
        response = future.result(timeout=HTTP_TIMEOUT_MS / 1000.0)
    except FuturesTimeout as exc:
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError("local timeout review batch") from exc
    else:
        pool.shutdown(wait=True, cancel_futures=False)
    elapsed = round(time.time() - t0, 1)
    print(f"  ok review {elapsed}s", flush=True)
    window_s = max(float(c.get("window_dur_s") or REVIEW_DUR) for c in clips)
    rows = assert_review_complete(
        _parse_review(response.text or "{}", len(clips), window_s=window_s)
    )
    usage = getattr(response, "usage_metadata", None)
    out: list[dict] = []
    for rec, row in zip(clips, rows):
        merged = dict(rec)
        merged.update(row)
        merged["review_model"] = MODEL
        merged["review_elapsed_s"] = elapsed
        merged["prompt_token_count"] = getattr(usage, "prompt_token_count", None)
        merged["candidates_token_count"] = getattr(usage, "candidates_token_count", None)
        out.append(merged)
        flag = "KEEP" if merged["keep"] else "drop"
        print(
            f"  {flag} {merged.get('file')} {merged['score']} {merged['event']} "
            f"trim={merged.get('trim_start_s')}-{merged.get('trim_end_s')} {merged['reason']}",
            flush=True,
        )
    return out


def review_ladder_rows(
    group: list[dict],
    attempt,
    *,
    quota_left: int = MAX_429_STREAK,
    sleeper=time.sleep,
    sleep_s: float = REVIEW_429_SLEEP_S,
) -> list[dict]:
    """Try one batch. On 400 or timeout, split. On 429, wait and retry this batch."""
    try:
        return list(attempt(group) or [])
    except Exception as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        if is_rate_limit(exc):
            if quota_left <= 0:
                print(f"  skip batch n={len(group)} 429", flush=True)
                return []
            print(f"  wait 429 n={len(group)} left={quota_left}", flush=True)
            sleeper(sleep_s)
            return review_ladder_rows(
                group,
                attempt,
                quota_left=quota_left - 1,
                sleeper=sleeper,
                sleep_s=sleep_s,
            )
        kids = child_review_groups(group)
        if not kids or not should_split_span(exc):
            print(f"  skip batch n={len(group)} {exc}", flush=True)
            return []
        print(f"  split review n={len(group)} -> {[len(k) for k in kids]}", flush=True)
        out: list[dict] = []
        for kid in kids:
            out.extend(
                review_ladder_rows(
                    kid,
                    attempt,
                    quota_left=MAX_429_STREAK,
                    sleeper=sleeper,
                    sleep_s=sleep_s,
                )
            )
        return out


def review_clips(
    clips: list[dict],
    dest: Path,
    *,
    proxy_dir: Path | None = None,
    prompt: str = PROMPT,
) -> dict:
    """Review in batches of 10. Keep finished batches if later calls fail."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    proxies = proxy_dir if proxy_dir is not None else dest.parent / "review"
    proxies.mkdir(parents=True, exist_ok=True)
    client = genai.Client(
        vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION
    )
    reviewed: list[dict] = []
    groups = batches(clips, n=REVIEW_BATCH)
    for i, group in enumerate(groups, start=1):
        print(f"REVIEW {MODEL} batch {i}/{len(groups)} n={len(group)}", flush=True)

        def _attempt(g: list[dict]) -> list[dict]:
            tries = BATCH_TRIES if len(g) <= 1 else 1
            return run_with_retries(
                lambda g=g: review_batch(client, g, proxies, prompt=prompt),
                tries=tries,
                sleep_s=RETRY_SLEEP_S,
            )

        reviewed.extend(review_ladder_rows(group, _attempt))
    partial = len(reviewed) < len(clips)
    payload = {
        "model": MODEL,
        "partial": partial,
        "n_input": len(clips),
        "n_reviewed": len(reviewed),
        "clips": reviewed,
        "picked": pick_reel(reviewed),
    }
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"wrote {dest.name} reviewed={len(reviewed)} picked={len(payload['picked'])} PARTIAL={partial}",
        flush=True,
    )
    return payload


def main() -> int:
    manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "candidates.json"
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "review.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    clips = rows["clips"] if isinstance(rows, dict) else rows
    review_clips(clips, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
