"""Scan a YouTube window with Vertex Gemini. Print JSON only. No secrets."""

from __future__ import annotations

import json
import sys

from google import genai
from google.genai import types

from gemini_defaults import DEFAULT_MODEL, VERTEX_LOCATION, VERTEX_PROJECT

PROJECT = VERTEX_PROJECT
YOUTUBE = "https://www.youtube.com/watch?v=sbGPehFw7n8"
PROMPT = """
Amateur indoor volleyball gym drop-in. Find highlight moments for a 9:16 rec-sports short.

KEEP: kills/spikes that land or are strong attacks, blocks, athletic jumps,
clean digs that save a play, aces. Show BOTH teams (left and right).

REJECT: walking, huddles, standing, serve setup, funny misses, shanks,
comedy errors, balls into the net as jokes, dead time.

Return JSON only: a list of objects with keys
start,end,score,event,team_side,reason.
Times are MM:SS relative to this window. score is 0-10. Prefer 6 to 12 items.
team_side is left or right. Pad each rally about 6 to 10 seconds.
""".strip()


def main() -> int:
    start = sys.argv[1] if len(sys.argv) > 1 else "1200s"
    end = sys.argv[2] if len(sys.argv) > 2 else "1800s"
    model = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL
    client = genai.Client(vertexai=True, project=PROJECT, location=VERTEX_LOCATION)
    video = types.Part(
        file_data=types.FileData(file_uri=YOUTUBE, mime_type="video/mp4"),
        video_metadata=types.VideoMetadata(start_offset=start, end_offset=end, fps=1),
    )
    print(f"SCAN {model} {start}-{end}", flush=True)
    response = client.models.generate_content(
        model=model,
        contents=[video, PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            http_options=types.HttpOptions(timeout=300_000),
        ),
    )
    text = response.text or "[]"
    path = f"/tmp/hl_{start}_{end}.json"
    open(path, "w", encoding="utf-8").write(text)
    parsed = json.loads(text)
    if isinstance(parsed, list):
        hs = parsed
    elif isinstance(parsed, dict):
        hs = parsed.get("highlights") or parsed.get("clips") or []
    else:
        hs = []
    print(f"N={len(hs)} FILE={path}", flush=True)
    for h in hs:
        print(
            f"{h.get('start')} {h.get('score')} {h.get('event')} {h.get('team_side')} {h.get('reason')}",
            flush=True,
        )
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
