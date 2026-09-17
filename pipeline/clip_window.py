"""Source windows for 16:9 clips. No crop."""

from __future__ import annotations

MIN_DUR = 15.0
MAX_DUR = 15.0
PREFER_DUR = 15.0
CONTACT_AT = 0.58
REVIEW_DUR = 20.0
COOL_WATCH_DUR = 40.0
MIN_CLIP_DUR = 12.0
MAX_CLIP_DUR = 18.0
MAX_COOL_DUR = 40.0
COOL_SCAN_SCORE = 9
DEFAULT_CLIP_DUR = 15.0


def clip_window(
    event_start: float,
    event_end: float,
    *,
    min_s: float = MIN_DUR,
    max_s: float = MAX_DUR,
    prefer_s: float = PREFER_DUR,
    video_end: float | None = None,
) -> tuple[float, float]:
    """Return (start, duration) in seconds. Peak sits late in the clip."""
    if event_end < event_start:
        event_start, event_end = event_end, event_start
    span = event_end - event_start
    dur = min(max_s, max(min_s, min(prefer_s, span + 1.5)))
    start = event_end - CONTACT_AT * dur
    if start < 0:
        start = 0.0
    if video_end is not None and start + dur > video_end:
        start = max(0.0, video_end - dur)
        dur = min(dur, max(0.0, video_end - start))
    return start, dur


def review_window(
    event_start: float,
    event_end: float,
    *,
    video_end: float | None = None,
    watch_s: float = REVIEW_DUR,
) -> tuple[float, float]:
    """Watch window. Extra time is before the finish."""
    if event_end < event_start:
        event_start, event_end = event_end, event_start
    end = event_end + 2.0
    start = end - watch_s
    dur = watch_s
    if start < 0:
        start = 0.0
        dur = min(watch_s, end if video_end is None else min(end, video_end))
    if video_end is not None and start + dur > video_end:
        start = max(0.0, video_end - dur)
        dur = min(dur, max(0.0, video_end - start))
    return start, dur


def cool_review_window(
    event_start: float,
    event_end: float,
    *,
    video_end: float | None = None,
) -> tuple[float, float]:
    return review_window(
        event_start, event_end, video_end=video_end, watch_s=COOL_WATCH_DUR
    )


def max_clip_dur(scan_score: int) -> float:
    if int(scan_score) >= COOL_SCAN_SCORE:
        return MAX_COOL_DUR
    return MAX_CLIP_DUR


def one_cool_caps(rows: list[dict]) -> list[float]:
    """At most one clip may exceed 18 s. Pick the longest high-score trim."""
    n = len(rows)
    caps = [MAX_CLIP_DUR] * n
    winner: int | None = None
    winner_key = (-1, -1.0)
    for i, rec in enumerate(rows):
        score = int(rec.get("score") or 0)
        if score < COOL_SCAN_SCORE:
            continue
        a = rec.get("trim_start_s")
        b = rec.get("trim_end_s")
        if a is None or b is None:
            continue
        dur = float(b) - float(a)
        if dur <= MAX_CLIP_DUR:
            continue
        key = (score, dur)
        if key > winner_key:
            winner = i
            winner_key = key
    if winner is not None:
        caps[winner] = MAX_COOL_DUR
    return caps


def parse_clip_time(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.lower().endswith("s") and ":" not in text:
        text = text[:-1].strip()
    if ":" in text:
        parts = text.split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        if len(nums) == 2:
            return nums[0] * 60.0 + nums[1]
        if len(nums) == 3:
            return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]
        return None
    try:
        return float(text)
    except ValueError:
        return None


def default_trim(window_s: float = REVIEW_DUR) -> tuple[float, float]:
    """15 s late in the 20 s window when 3.8 omits start/end."""
    dur = min(DEFAULT_CLIP_DUR, max(0.0, window_s))
    start = max(0.0, window_s - dur)
    return start, start + dur


def clamp_trim(
    rel_a: float,
    rel_b: float,
    *,
    window_s: float = REVIEW_DUR,
    min_s: float = MIN_CLIP_DUR,
    max_s: float = MAX_CLIP_DUR,
) -> tuple[float, float]:
    """Keep a 12-18 s span, or up to 40 s when max_s allows."""
    window_s = max(0.0, window_s)
    if rel_b < rel_a:
        rel_a, rel_b = rel_b, rel_a
    rel_a = max(0.0, min(rel_a, window_s))
    rel_b = max(0.0, min(rel_b, window_s))
    dur = rel_b - rel_a
    if window_s <= min_s:
        return 0.0, window_s
    if dur < min_s:
        mid = (rel_a + rel_b) / 2.0 if dur > 0 else rel_a
        rel_a = mid - min_s / 2.0
        rel_b = mid + min_s / 2.0
        if rel_a < 0:
            rel_b -= rel_a
            rel_a = 0.0
        if rel_b > window_s:
            rel_a -= rel_b - window_s
            rel_b = window_s
        rel_a = max(0.0, rel_a)
        rel_b = min(window_s, rel_b)
    dur = rel_b - rel_a
    if dur > max_s:
        rel_a = rel_b - max_s
        if rel_a < 0:
            rel_a = 0.0
            rel_b = min(window_s, max_s)
    return rel_a, rel_b


def trim_from_row(
    row: dict,
    window_s: float = REVIEW_DUR,
    *,
    max_s: float = MAX_COOL_DUR,
) -> tuple[float, float]:
    start = parse_clip_time(row.get("trim_start_s", row.get("trim_start", row.get("start"))))
    end = parse_clip_time(row.get("trim_end_s", row.get("trim_end", row.get("end"))))
    dur = parse_clip_time(row.get("duration", row.get("length")))
    if start is not None and start > window_s:
        start = None
    if end is not None and end > window_s:
        end = None
    if start is not None and end is not None:
        return clamp_trim(start, end, window_s=window_s, max_s=max_s)
    if start is not None and dur is not None:
        return clamp_trim(start, start + dur, window_s=window_s, max_s=max_s)
    if end is not None and dur is not None:
        return clamp_trim(end - dur, end, window_s=window_s, max_s=max_s)
    if dur is not None:
        dur = min(max_s, max(MIN_CLIP_DUR, dur))
        start = max(0.0, window_s - dur)
        return clamp_trim(start, start + dur, window_s=window_s, max_s=max_s)
    return clamp_trim(*default_trim(window_s), window_s=window_s, max_s=max_s)


def final_source_span(
    window_start: float,
    window_dur: float,
    trim_start: float | None,
    trim_end: float | None,
    *,
    video_end: float | None = None,
    max_s: float = MAX_CLIP_DUR,
) -> tuple[float, float]:
    if trim_start is None or trim_end is None:
        rel_a, rel_b = default_trim(window_dur)
        rel_a, rel_b = clamp_trim(rel_a, rel_b, window_s=window_dur, max_s=max_s)
    else:
        rel_a, rel_b = clamp_trim(
            trim_start, trim_end, window_s=window_dur, max_s=max_s
        )
    start = window_start + rel_a
    dur = rel_b - rel_a
    if start < 0:
        dur = max(0.0, dur + start)
        start = 0.0
    if video_end is not None and start + dur > video_end:
        dur = max(0.0, video_end - start)
    return start, dur
