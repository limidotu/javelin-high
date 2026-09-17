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
from scan_full_match import (
    HTTP_TIMEOUT_MS,
    MAX_429_STREAK,
    PROMPT,
    clock,
    existing_spans,
    is_deadline,
    is_rate_limit,
    normalize_rows,
    uncovered_windows,
    write_payload,
)
from v1_media import assert_inline_ok, make_scan_proxy, probe_secs

ROOT = Path(__file__).resolve().parent
MODEL = SCAN_MODEL
WINDOW_S = 600


def scan_window_file(
    client: genai.Client,
    source: Path,
    start_s: int,
    end_s: int,
    proxy_dir: Path,
    video_end: int,
) -> dict:
    proxy = proxy_dir / f"w_{start_s}_{end_s}.mp4"
    print(f"  proxy {start_s}-{end_s} ...", flush=True)
    make_scan_proxy(source, start_s, end_s, proxy)
    assert_inline_ok(proxy)
    video = types.Part.from_bytes(data=proxy.read_bytes(), mime_type="video/mp4")
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
    rows = normalize_rows(parsed, start_s, end_s, video_end=video_end)
    usage = getattr(response, "usage_metadata", None)
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
        "prompt_token_count": getattr(usage, "prompt_token_count", None),
        "candidates_token_count": getattr(usage, "candidates_token_count", None),
        "thoughts_token_count": getattr(usage, "thoughts_token_count", None),
        "total_token_count": getattr(usage, "total_token_count", None),
        "timestamp_note": "Times in start/end are window-relative. source_* is match time.",
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
        return [scan_window_file(client, source, start_s, end_s, proxy_dir, video_end)]
    except Exception as exc:
        if is_rate_limit(exc):
            raise
        if is_deadline(exc):
            print(f"  skip {start_s}-{end_s} deadline", flush=True)
            return []
        raise


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
            if is_deadline(exc):
                partial = True
                print(f"  skip {start_s}-{end_s} deadline", flush=True)
                continue
            print(f"FAIL {start_s}s-{end_s}s {exc}", file=sys.stderr, flush=True)
            partial = True
            break
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
