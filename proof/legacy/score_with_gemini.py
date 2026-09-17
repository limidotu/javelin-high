"""Score highlight clips with Gemini. Loads proof/.env. Never print the key."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from gemini_defaults import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".env"
REPORT = ROOT / "out" / "report.json"
OUT = ROOT / "out" / "gemini_scores.json"

PROMPT = """
You score amateur indoor volleyball clips for a 9:16 rec-sports highlight short.

Keep a clip only if it shows a strong play: kill/spike that lands or is a real attack,
a block, a clean dig that saves a point, a jump serve ace, or a set that clearly
leads into an attack. Show both teams at their best. Near-side and far-side plays
both count.

Reject: walking, huddles, standing around, serve toss with no hit, funny misses,
shanks, balls into the net as comedy, and any clip whose main joke is a mistake.

Return JSON only with this shape:
{
  "keep": true or false,
  "score": 0 to 10,
  "event": "kill|block|dig|ace|spike|set|other|dead",
  "team_side": "near|far|both|unknown",
  "reason": "one short sentence"
}
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

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    client = genai.Client(api_key=key)
    results = []
    for peak in report["peaks"]:
        clip = Path(peak["clip"])
        uploaded = client.files.upload(file=str(clip), config={"mime_type": "video/mp4"})
        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state.name == "FAILED":
            results.append({"error": "upload_failed", "clip": str(clip)})
            continue
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=[
                types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
                PROMPT,
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        parsed = json.loads(response.text)
        parsed["clip"] = str(clip)
        parsed["t_in_source_s"] = peak["t_in_source_s"]
        parsed["rank"] = peak["rank"]
        parsed["usage"] = {
            "prompt": getattr(response.usage_metadata, "prompt_token_count", None),
            "output": getattr(response.usage_metadata, "candidates_token_count", None),
        }
        results.append(parsed)
        print(json.dumps({k: parsed[k] for k in parsed if k != "clip"}))
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    kept = [r for r in results if r.get("keep")]
    print(json.dumps({"n": len(results), "kept": len(kept)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
