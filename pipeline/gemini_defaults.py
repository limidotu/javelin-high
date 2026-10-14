"""Default Gemini models for the reel job."""

import re

SCAN_MODEL = "gemini-2.5-flash"
REVIEW_MODEL = "gemini-3.8-flash"
ARGON_MODEL = "gemini-4-argon"
DEFAULT_MODEL = REVIEW_MODEL
REVIEW_THINKING_LEVEL = "HIGH"
ARGON_THINKING = "HIGH"
ARGON_VIDEO_FPS = 8.0
ARGON_TEMPERATURE = 0.7
ARGON_MAX_OUTPUT_TOKENS = 32768
ARGON_PROMPT_VERSION = "players-best-light-v1"
VERTEX_PROJECT = "gen-lang-client-0442353446"
VERTEX_LOCATION = "global"
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}")


def uses_thinking_level(model: str) -> bool:
    name = (model or "").lower()
    return name.startswith("gemini-3") or name.startswith("gemini-4")


def clean_model(name: str | None, fallback: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return fallback
    if _MODEL_RE.fullmatch(raw) is None:
        raise ValueError("Model id is not valid.")
    return raw
