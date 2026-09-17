"""Scan a local MP4 with gemini-2.5-flash in 10-minute windows."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

from google import genai
from google.genai import types

from gemini_defaults import SCAN_MODEL, VERTEX_LOCATION, VERTEX_PROJECT
from media import assert_inline_ok, make_scan_proxy, probe_secs
from scan_io import (
    SEGMENT_S,
    WINDOW_S,
    clock,
    existing_spans,
    scan_ladder_rows,
    is_rate_limit,
    normalize_rows,
    run_with_retries,
    uncovered_windows,
    write_payload,
)

ROOT = Path(__file__).resolve().parent
MODEL = SCAN_MODEL
HTTP_TIMEOUT_MS = 180_000
MAX_429_STREAK = 3
WINDOW_TRIES = 3
RETRY_SLEEP_S = 8.0
PROMPT_BODY = """
Amateur indoor volleyball. This clip is about {span_s} seconds.
Pick THE BEST highlight moments only.

KEEP only elite plays: stuff blocks, powerful kills that land, diving digs
that save a point, then a kill, clean aces nobody touches. Both teams.

REJECT routine rallies, average spikes, walking, huddles, serve setup,
funny misses, net luck, dead time, and anything you would skip on a recap.

Return JSON only: a list of objects with keys
start,end,score,event,team_side,reason.
Times are MM:SS relative to THIS clip, not the full match.
score is 0-10. Use 8, 9, or 10 only for a keep. {prefer}
If this clip has no elite play, return [].
""".strip()


def scan_prompt(span_s: int) -> str:
    if span_s <= 90:
        prefer = "Prefer 0 or 1 item."
    elif span_s <= 360:
        prefer = "Prefer 0 to 4 items."
    else:
        prefer = "Prefer 0 to 8 items."
    return PROMPT_BODY.format(span_s=max(1, int(span_s)), prefer=prefer)


def span_timeout_ms(span_s: int) -> int:
    if span_s <= SEGMENT_S:
        return HTTP_TIMEOUT_MS
    if span_s <= 360:
        return 240_000
    return 300_000


def _parse_model_json(text: str) -> object:
    raw = (text or "").strip()
    try:
        return json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []


def scan_span(
    client: genai.Client,
    source: Path,
    start_s: int,
    end_s: int,
    proxy_dir: Path,
    video_end: int,
    *,
    window_start: int,
    window_end: int,
) -> list[dict]:
    dest = proxy_dir / f"w_{window_start}_{window_end}_{start_s}_{end_s}.mp4"
    print(f"  proxy {start_s}-{end_s} ...", flush=True)
    make_scan_proxy(source, start_s, end_s, dest)
    assert_inline_ok(dest)
    video = types.Part.from_bytes(data=dest.read_bytes(), mime_type="video/mp4")
    t0 = time.time()
    print(f"  call {start_s}-{end_s} ...", flush=True)
    timeout_ms = span_timeout_ms(end_s - start_s)
    prompt = scan_prompt(end_s - start_s)

    def _run():
        return client.models.generate_content(
            model=MODEL,
            contents=[video, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                http_options=types.HttpOptions(timeout=timeout_ms),
            ),
        )

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_run)
    try:
        response = future.result(timeout=timeout_ms / 1000.0)
    except FuturesTimeout as exc:
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"local timeout {start_s}-{end_s}") from exc
    else:
        pool.shutdown(wait=True, cancel_futures=False)
    elapsed = round(time.time() - t0, 1)
    print(f"  ok {start_s}-{end_s} {elapsed}s", flush=True)
    parsed = _parse_model_json(response.text or "[]")
    return normalize_rows(parsed, start_s, end_s, video_end=video_end)


def scan_window_file(
    client: genai.Client,
    source: Path,
    start_s: int,
    end_s: int,
    proxy_dir: Path,
    video_end: int,
) -> dict:
    print(f"  window {start_s}-{end_s} try {end_s - start_s}s", flush=True)
    t0 = time.time()

    def attempt(a: int, b: int) -> list[dict]:
        tries = WINDOW_TRIES if (b - a) <= SEGMENT_S else 1
        if (a, b) != (start_s, end_s):
            print(f"  split to {a}-{b} ({b - a}s)", flush=True)
        return run_with_retries(
            lambda: scan_span(
                client,
                source,
                a,
                b,
                proxy_dir,
                video_end,
                window_start=start_s,
                window_end=end_s,
            ),
            tries=tries,
            sleep_s=RETRY_SLEEP_S,
        )

    rows = scan_ladder_rows(start_s, end_s, attempt)
    elapsed = round(time.time() - t0, 1)
    print(
        f"  ok window {start_s}-{end_s} {elapsed}s keeps={sum(1 for h in rows if h.get('keep'))}",
        flush=True,
    )
    return {
        "model": MODEL,
        "backend": "vertex-ai",
        "project": VERTEX_PROJECT,
        "location": VERTEX_LOCATION,
        "source": str(source),
        "requested_window": {
            "start": f"{start_s}s",
            "end": f"{end_s}s",
            "label": f"{clock(start_s)}-{clock(end_s)}",
        },
        "elapsed_s": elapsed,
        "prompt_token_count": None,
        "candidates_token_count": None,
        "thoughts_token_count": None,
        "total_token_count": None,
        "timestamp_note": "Times in start/end are clip-relative. source_* is match time.",
        "highlights": rows,
        "partial_ok": True,
    }


def scan_range_file(
    client: genai.Client,
    source: Path,
    start_s: int,
    end_s: int,
    proxy_dir: Path,
    video_end: int,
) -> list[dict]:
    try:
        return [
            scan_window_file(
                client, source, start_s, end_s, proxy_dir, video_end
            )
        ]
    except Exception as exc:
        if is_rate_limit(exc):
            raise
        if isinstance(exc, FileNotFoundError):
            raise
        print(f"  skip {start_s}-{end_s} {exc}", flush=True)
        return []


def scan_local(
    source: Path,
    out_dir: Path,
    *,
    video_end: int | None = None,
    proxy_dir: Path | None = None,
) -> dict:
    """Scan what we can. Stop on a 429 streak. Keep finished windows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = proxy_dir if proxy_dir is not None else out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    end = video_end if video_end is not None else int(round(probe_secs(source)))
    print(
        f"MODEL={MODEL} location={VERTEX_LOCATION} out={out_dir} source={source} end={end}",
        flush=True,
    )
    client = genai.Client(
        vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION
    )
    total_keep = 0
    streak_429 = 0
    partial = False
    spans = existing_spans(out_dir)
    todo = uncovered_windows(spans, video_end=end, window_s=WINDOW_S)
    print(f"gaps={todo[:8]} n={len(todo)}", flush=True)
    for start_s, end_s in todo:
        print(f"SCAN {MODEL} {start_s}s-{end_s}s", flush=True)
        try:
            payloads = scan_range_file(
                client, source, start_s, end_s, windows_dir, end
            )
        except Exception as exc:
            if is_rate_limit(exc):
                streak_429 += 1
                partial = True
                print(f"  skip {start_s}-{end_s} 429 streak={streak_429}", flush=True)
                if streak_429 >= MAX_429_STREAK:
                    print("PARTIAL too many 429s; keep finished windows", flush=True)
                    break
                continue
            print(f"  skip {start_s}-{end_s} {exc}", flush=True)
            partial = True
            continue
        if not payloads:
            partial = True
            continue
        streak_429 = 0
        for payload in payloads:
            total_keep += write_payload(payload, out_dir)
        spans = existing_spans(out_dir)
    leftover = uncovered_windows(spans, video_end=end, window_s=WINDOW_S)
    if leftover:
        partial = True
    print(f"TOTAL_KEEP={total_keep} PARTIAL={partial}", flush=True)
    return {
        "keeps": total_keep,
        "partial": partial,
        "scan_dir": str(out_dir),
        "video_end": end,
    }


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "source.mp4"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "scan_local"
    if not source.is_absolute():
        source = (ROOT / source).resolve()
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if not source.exists():
        print(f"missing {source}", file=sys.stderr)
        return 1
    result = scan_local(source, out_dir)
    print(json.dumps({k: result[k] for k in ("keeps", "partial", "video_end")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
