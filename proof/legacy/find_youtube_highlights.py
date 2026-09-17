"""Ask Gemini for highlight timestamps on a public YouTube URL."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from gemini_defaults import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".env"
OUT = ROOT / "out" / "gemini_youtube.json"
YOUTUBE = "https://www.youtube.com/watch?v=sbGPehFw7n8"

PROMPT = """
This is amateur indoor volleyball from a gym drop-in.
Find highlight moments for a 9:16 rec-sports short that shows BOTH teams at their best.

KEEP: kills/spikes that land or are strong attacks, blocks, athletic jumps,
clean digs that save a play, aces. Mix near-side and far-side.

REJECT: walking, huddles, standing, serve setup, funny misses, shanks,
comedy errors, balls into the net as jokes, dead time.

Return JSON only:
{
  "sport": "volleyball|basketball|mixed|other",
  "highlights": [
    {
      "start": "H:MM:SS",
      "end": "H:MM:SS",
      "score": 0 to 10,
      "event": "kill|block|dig|ace|spike|set|other",
      "team_side": "near|far|both|unknown",
      "reason": "one short sentence"
    }
  ]
}

Give 6 to 12 candidates in this time window. Prefer score >= 7.
Pad each rally about 6 to 10 seconds.
""".strip()


def load_env() -> None:
    if not ENV.exists():
        return
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    load_env()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("missing GEMINI_API_KEY", file=sys.stderr)
        return 2
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    start = sys.argv[1] if len(sys.argv) > 1 else "1200s"
    end = sys.argv[2] if len(sys.argv) > 2 else "1500s"
    video = types.Part(
        file_data=types.FileData(file_uri=YOUTUBE),
        video_metadata=types.VideoMetadata(start_offset=start, end_offset=end, fps=2),
    )
    print(f"calling gemini on youtube {start}-{end}...", flush=True)
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=[video, PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            http_options=types.HttpOptions(timeout=180_000),
        ),
    )
    parsed = json.loads(response.text)
    parsed["_usage"] = {
        "prompt": getattr(response.usage_metadata, "prompt_token_count", None),
        "output": getattr(response.usage_metadata, "candidates_token_count", None),
        "total": getattr(response.usage_metadata, "total_token_count", None),
    }
    parsed["_window"] = {"start": start, "end": end}
    OUT.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(parsed.get("highlights", [])), "usage": parsed["_usage"]}))
    for h in parsed.get("highlights", []):
        print(f"{h.get('start')} {h.get('score')} {h.get('event')} {h.get('team_side')} {h.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
