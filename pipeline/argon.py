"""Third step: Gemini 4 Argon turns the 16:9 reel into a 9:16 TikTok cut."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

from google import genai
from google.genai import types

from gemini_defaults import ARGON_MODEL, ARGON_THINKING, VERTEX_LOCATION, VERTEX_PROJECT
from media import INLINE_MAX_BYTES, ffmpeg_bin

HTTP_TIMEOUT_MS = 600_000
OUT_W = 1080
OUT_H = 1920
SPORTS = ("auto", "volleyball", "basketball")
LAYOUTS = ("smart-crop", "blurred-bg")
THINKING = ("LOW", "MEDIUM", "HIGH")

SPORT_GUIDE = {
    "volleyball": "Keep spikes, blocks, digs, aces, and long rallies. Drop walks back to serve, ball chase, and huddles.",
    "basketball": "Keep dunks, threes, steals, blocks, and fast breaks. Drop free-throw waits, walk-ups, and timeouts.",
    "auto": "Detect volleyball or basketball. Keep the explosive plays. Drop dead time.",
}

HIGHLIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "sport_detected": {"type": "string", "enum": ["volleyball", "basketball", "other"]},
        "game_summary": {"type": "string"},
        "tiktok_caption": {"type": "string"},
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clip_id": {"type": "integer"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "play_type": {"type": "string"},
                    "excitement_score": {"type": "number"},
                    "action_center_x": {"type": "number"},
                    "on_screen_hook": {"type": "string"},
                    "is_hook_candidate": {"type": "boolean"},
                    "reasoning": {"type": "string"},
                },
                "required": [
                    "clip_id",
                    "start_time",
                    "end_time",
                    "play_type",
                    "excitement_score",
                    "action_center_x",
                    "on_screen_hook",
                    "is_hook_candidate",
                    "reasoning",
                ],
            },
        },
    },
    "required": ["sport_detected", "game_summary", "tiktok_caption", "clips"],
}


def parse_timestamp(ts: str) -> float:
    text = str(ts).strip()
    if ":" not in text:
        return float(text)
    parts = text.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60.0 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)
    raise ValueError(f"Unrecognized timestamp format: {text}")


def hook_text(raw: str) -> str:
    chars = [ch for ch in str(raw).upper() if ch.isalnum() or ch == " "]
    text = "".join(chars).strip()
    return (text[:48] or "HIGHLIGHT")


def ffmpeg_font() -> str:
    path = Path(r"C:\Windows\Fonts\arialbd.ttf")
    if not path.is_file():
        return ""
    return str(path).replace("\\", "/").replace(":", "\\:")


def probe_video(path: Path) -> dict:
    proc = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    )
    err = proc.stderr or ""
    duration = 0.0
    width, height = 1920, 1080
    for line in err.splitlines():
        if "Duration:" in line and duration == 0.0:
            clock = line.split("Duration:")[1].split(",")[0].strip()
            hours, minutes, seconds = clock.split(":")
            duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if "Video:" in line:
            match = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if match:
                width, height = int(match.group(1)), int(match.group(2))
    return {"duration": duration, "width": width, "height": height}


def _run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(tail[-1] if tail else "ffmpeg failed")


def make_argon_proxy(source: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_bin()
    for fps, crf in (("8", "28"), ("8", "36"), ("4", "32"), ("2", "36")):
        _run_ffmpeg(
            [
                ff, "-y", "-i", str(source),
                "-map", "0:v:0", "-map", "0:a:0?",
                "-vf", f"scale=854:480,fps={fps}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", crf,
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k", "-ar", "48000", "-ac", "2",
                str(dest),
            ]
        )
        size = dest.stat().st_size
        print(f"ARGON proxy fps={fps} crf={crf} bytes={size}", flush=True)
        if size <= INLINE_MAX_BYTES:
            return dest
    raise RuntimeError(f"proxy too large for Vertex inline: {dest.name} {dest.stat().st_size} bytes")


def build_prompt(sport: str, target_s: float, video_s: float) -> str:
    guide = SPORT_GUIDE.get(sport, SPORT_GUIDE["auto"])
    low = target_s * 1.15
    high = target_s * 1.35
    return (
        "You edit an already-cut 16:9 sports highlight reel into a vertical TikTok. "
        f"This file is {video_s:.1f} seconds. It is not the full match. "
        "Times are MM:SS.s from the start of THIS file.\n"
        f"{guide}\n"
        "Rules:\n"
        "1. Each play is 3.5 to 9.0 seconds.\n"
        f"2. Offer enough plays to fill about {low:.0f} to {high:.0f} seconds.\n"
        "3. Start 0.5 seconds before the hit. End 0.8 seconds after the point.\n"
        "4. action_center_x is 0.0 at the left edge and 1.0 at the right edge.\n"
        "5. Mark one play with is_hook_candidate true.\n"
        "6. Return JSON only."
    )


def pack_timeline(
    edl: dict,
    video_s: float,
    target_s: float = 60.0,
    *,
    hook: bool = True,
    min_score: float = 6.5,
) -> list[dict]:
    """Pack Argon clips to the target length. The best play can open the cut."""
    normalized = []
    for clip in edl.get("clips") or []:
        start_s = max(0.0, parse_timestamp(clip["start_time"]))
        end_s = min(video_s, parse_timestamp(clip["end_time"]))
        dur = round(end_s - start_s, 2)
        if dur < 1.2:
            continue
        if dur > 9.5:
            start_s = end_s - 9.0
            dur = 9.0
        item = dict(clip)
        item["start_sec"] = round(start_s, 2)
        item["end_sec"] = round(end_s, 2)
        item["duration_sec"] = round(dur, 2)
        item["action_center_x"] = max(0.15, min(0.85, float(clip.get("action_center_x", 0.5))))
        normalized.append(item)
    qualified = [c for c in normalized if float(c.get("excitement_score") or 0) >= min_score]
    if not qualified:
        qualified = normalized
    ranked = sorted(qualified, key=lambda row: float(row.get("excitement_score") or 0), reverse=True)
    selected: list[dict] = []
    used = 0.0
    opener = None
    if hook and ranked:
        best = ranked[0]
        hook_dur = min(3.0, best["duration_sec"])
        hook_start = max(best["start_sec"], best["end_sec"] - hook_dur)
        opener = dict(best)
        opener["clip_id"] = 0
        opener["start_sec"] = round(hook_start, 2)
        opener["end_sec"] = round(hook_start + hook_dur, 2)
        opener["duration_sec"] = round(hook_dur, 2)
        opener["on_screen_hook"] = f"WAIT FOR THIS {hook_text(best.get('on_screen_hook', ''))}"
        opener["is_opening_teaser"] = True
        used += opener["duration_sec"]
    for candidate in ranked:
        if used >= target_s:
            break
        overlaps = any(
            not (candidate["end_sec"] <= row["start_sec"] or candidate["start_sec"] >= row["end_sec"])
            for row in selected
        )
        if overlaps:
            continue
        remain = round(target_s - used, 2)
        if candidate["duration_sec"] <= remain:
            selected.append(candidate)
            used = round(used + candidate["duration_sec"], 2)
        elif remain >= 2.5:
            trimmed = dict(candidate)
            trimmed["start_sec"] = round(trimmed["end_sec"] - remain, 2)
            trimmed["duration_sec"] = remain
            selected.append(trimmed)
            used = round(used + remain, 2)
            break
    selected.sort(key=lambda row: row["start_sec"])
    if opener:
        return [opener] + selected
    return selected


def _crop_filter(width: int, height: int, center_x: float) -> str:
    crop_h = height - (height % 2)
    crop_w = min(width, int(round(crop_h * 9.0 / 16.0)))
    crop_w -= crop_w % 2
    if crop_w < 2 or crop_h < 2:
        raise RuntimeError("frame is too small to crop")
    max_x = max(0, width - crop_w)
    desired = int(round(center_x * width - crop_w / 2.0))
    crop_x = max(0, min(max_x, desired))
    crop_x -= crop_x % 2
    return f"crop={crop_w}:{crop_h}:{crop_x}:0,scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1"


def _blur_filter() -> str:
    return (
        f"split[bg][fg];"
        f"[bg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},boxblur=25:5[bgblur];"
        f"[fg]scale={OUT_W}:-2:force_original_aspect_ratio=decrease[fgscale];"
        f"[bgblur][fgscale]overlay=(W-w)/2:(H-h)/2,setsar=1[vout]"
    )


def _caption_filter(text: str) -> str:
    font = ffmpeg_font()
    if not font:
        return ""
    safe = hook_text(text)
    return (
        f"drawtext=fontfile='{font}':text='{safe}':fontcolor=white:fontsize=52:"
        f"borderw=4:bordercolor=black:x=(w-text_w)/2:y=h*0.14"
    )


def render_tiktok(
    source: Path,
    timeline: list[dict],
    dest: Path,
    *,
    layout: str = "smart-crop",
    captions: bool = True,
) -> None:
    if not timeline:
        raise RuntimeError("Argon returned no clips")
    if layout not in LAYOUTS:
        raise ValueError("Layout must be smart-crop or blurred-bg.")
    meta = probe_video(source)
    ff = ffmpeg_bin()
    print(f"ARGON render n={len(timeline)} {layout}", flush=True)
    with tempfile.TemporaryDirectory(prefix="argon_tiktok_") as tmp:
        tmp_path = Path(tmp)
        segments = []
        for idx, clip in enumerate(timeline):
            seg = tmp_path / f"seg_{idx:03d}.mp4"
            center = float(clip.get("action_center_x", 0.5))
            if layout == "smart-crop":
                graph = _crop_filter(meta["width"], meta["height"], center)
                if captions and clip.get("on_screen_hook"):
                    extra = _caption_filter(str(clip["on_screen_hook"]))
                    if extra:
                        graph += "," + extra
                cmd = [
                    ff, "-y", "-ss", f"{clip['start_sec']:.3f}", "-i", str(source),
                    "-t", f"{clip['duration_sec']:.3f}",
                    "-vf", graph,
                    "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                    "-r", "60", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    str(seg),
                ]
            else:
                graph = _blur_filter()
                if captions and clip.get("on_screen_hook"):
                    extra = _caption_filter(str(clip["on_screen_hook"]))
                    if extra:
                        graph = graph.replace("[vout]", f"[vraw];[vraw]{extra}[vout]")
                cmd = [
                    ff, "-y", "-ss", f"{clip['start_sec']:.3f}", "-i", str(source),
                    "-t", f"{clip['duration_sec']:.3f}",
                    "-filter_complex", graph, "-map", "[vout]", "-map", "0:a:0?",
                    "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                    "-r", "60", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    str(seg),
                ]
            _run_ffmpeg(cmd)
            segments.append(seg)
        listing = tmp_path / "list.txt"
        listing.write_text("".join(f"file '{seg.as_posix()}'\n" for seg in segments), encoding="utf-8")
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(
            [
                ff, "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", "-movflags", "+faststart", str(dest),
            ]
        )
    total = sum(float(clip["duration_sec"]) for clip in timeline)
    print(f"ARGON {dest.name} {total:.1f}s clips={len(timeline)}", flush=True)


def analyze_reel(
    proxy: Path,
    *,
    model: str,
    sport: str,
    thinking: str,
    target_s: float,
    video_s: float,
) -> dict:
    client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    video = types.Part.from_bytes(data=proxy.read_bytes(), mime_type="video/mp4")
    prompt = build_prompt(sport, target_s, video_s)
    print(f"ARGON call {model} thinking={thinking} {video_s:.1f}s", flush=True)

    def _run():
        return client.models.generate_content(
            model=model,
            contents=[video, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=HIGHLIGHT_SCHEMA,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=types.ThinkingConfig(thinking_level=thinking),
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                max_output_tokens=65536,
                http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS),
            ),
        )

    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_run)
    try:
        response = future.result(timeout=HTTP_TIMEOUT_MS / 1000.0)
    except FuturesTimeout as exc:
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError("local timeout argon") from exc
    else:
        pool.shutdown(wait=True, cancel_futures=False)
    print(f"ARGON ok {round(time.time() - t0, 1)}s", flush=True)
    return json.loads(response.text or "{}")


HIT_PRE = 0.20
HIT_POST = 0.25
MIN_PRE = 0.45
MIN_POST = 0.55
SLOW = 2.0

LOCK_SCHEMA = {
    "type": "object",
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "impact_s": {"type": "number"},
                    "in_s": {"type": "number"},
                    "out_s": {"type": "number"},
                    "center_x": {"type": "number"},
                    "label": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["index", "impact_s", "in_s", "out_s", "center_x", "label", "score"],
            },
        },
    },
    "required": ["shots"],
}


def label_text(raw: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", str(raw or ""))[:2]
    return " ".join(word.capitalize() for word in words)


def spans_from_keeps(path: Path) -> list[tuple[float, float]]:
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    spans = []
    t = 0.0
    for clip in meta.get("clips") or []:
        dur = float(clip["dur_s"])
        if dur <= 0:
            continue
        spans.append((round(t, 4), round(t + dur, 4)))
        t += dur
    return spans


def fit_spans(spans: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    rows = [(float(a), float(b)) for a, b in spans if float(b) > float(a) + 0.05]
    if not rows:
        return [(0.0, float(duration))]
    if rows[0][0] > 0.05:
        rows[0] = (0.0, rows[0][1])
    start, end = rows[-1]
    if duration > start and abs(duration - end) < 2.0:
        rows[-1] = (start, float(duration))
    return [(round(a, 4), round(b, 4)) for a, b in rows]


def shot_spans(path: Path, duration: float) -> list[tuple[float, float]]:
    ff = ffmpeg_bin()
    cuts: list[float] = []
    for thresh in ("0.30", "0.18", "0.12"):
        proc = subprocess.run(
            [ff, "-hide_banner", "-i", str(path), "-filter:v", f"select='gt(scene,{thresh})',showinfo", "-an", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        found = []
        for line in (proc.stderr or "").splitlines():
            if "pts_time:" not in line:
                continue
            for part in line.replace("\t", " ").split():
                if part.startswith("pts_time:"):
                    found.append(float(part.split(":", 1)[1]))
        if len(found) >= 2:
            cuts = found
            break
    edges = [0.0]
    for t in cuts:
        if t <= 0.05 or t >= duration - 0.05:
            continue
        if t - edges[-1] >= 1.0:
            edges.append(t)
    if duration - edges[-1] >= 0.4:
        edges.append(duration)
    else:
        edges[-1] = duration
    return [(round(edges[i], 4), round(edges[i + 1], 4)) for i in range(len(edges) - 1)]


def screen_of(start: float, end: float, impact: float) -> float:
    if end <= start:
        return 0.0
    hit_a = min(max(impact - HIT_PRE, start), end)
    hit_b = min(max(impact + HIT_POST, start), end)
    if hit_b < hit_a:
        hit_b = hit_a
    return (hit_a - start) + (end - hit_b) + (hit_b - hit_a) * SLOW


def hit_slices(start: float, end: float, impact: float) -> list[tuple[float, float, bool]]:
    hit_a = min(max(impact - HIT_PRE, start), end)
    hit_b = min(max(impact + HIT_POST, start), end)
    if hit_b - hit_a < 0.12:
        return [(start, end, False)]
    left = (start, hit_a, False) if hit_a - start >= 0.04 else None
    mid_a = hit_a if left else start
    right = (hit_b, end, False) if end - hit_b >= 0.04 else None
    mid_b = hit_b if right else end
    rows = [row for row in (left, (mid_a, mid_b, True), right) if row and row[1] > row[0] + 0.01]
    return rows or [(start, end, False)]


def _budgets(scores: list[float], target: float) -> list[float]:
    floors = []
    for score in scores:
        if score >= 8:
            floors.append(11.0)
        elif score >= 7:
            floors.append(9.0)
        else:
            floors.append(5.0)
    if sum(floors) >= target:
        scale = target / sum(floors)
        return [floor * scale for floor in floors]
    extra = target - sum(floors)
    weights = [max(score, 1.0) for score in scores]
    weight_sum = sum(weights)
    out = [floors[i] + extra * weights[i] / weight_sum for i in range(len(scores))]
    out[-1] += target - sum(out)
    return out


def fit_window(span_a: float, span_b: float, impact: float, budget: float, pre_share: float = 0.58) -> tuple[float, float]:
    hit_a = min(max(impact - HIT_PRE, span_a), span_b)
    hit_b = min(max(impact + HIT_POST, span_a), span_b)
    if hit_b < hit_a:
        hit_b = hit_a
    hit_len = hit_b - hit_a
    span_len = span_b - span_a
    want_source = min(span_len, max(hit_len, budget - hit_len))
    extra = max(0.0, want_source - hit_len)
    pre_max = hit_a - span_a
    post_max = span_b - hit_b
    share = min(0.8, max(0.25, pre_share))
    pre = min(pre_max, extra * share)
    post = min(post_max, extra - pre)
    if pre + post < extra - 0.0001:
        pre = min(pre_max, extra - post)
    if pre + post < extra - 0.0001:
        post = min(post_max, extra - pre)
    return hit_a - pre, hit_b + post


def _min_screen(shot: dict) -> float:
    hit_a = min(max(shot["impact_s"] - HIT_PRE, shot["span_a"]), shot["span_b"])
    hit_b = min(max(shot["impact_s"] + HIT_POST, shot["span_a"]), shot["span_b"])
    return max(0.2, (hit_b - hit_a) * SLOW)


def _drop_until_fit(shots: list[dict], target: float) -> list[dict]:
    rows = list(shots)
    while len(rows) > 1 and sum(_min_screen(shot) for shot in rows) > target + 0.05:
        idx = min(range(len(rows)), key=lambda i: (rows[i]["score"], -i))
        del rows[idx]
    return rows


def _nudge(shots: list[dict], target: float) -> None:
    for _ in range(len(shots) * 3 + 2):
        total = sum(screen_of(shot["in_sec"], shot["out_sec"], shot["impact_s"]) for shot in shots)
        delta = target - total
        if abs(delta) < 0.0008:
            return
        if delta > 0:
            order = sorted(range(len(shots)), key=lambda i: shots[i]["score"], reverse=True)
        else:
            order = sorted(range(len(shots)), key=lambda i: shots[i]["score"])
        moved = False
        for i in order:
            shot = shots[i]
            if delta > 0:
                room_out = shot["span_b"] - shot["out_sec"]
                room_in = shot["in_sec"] - shot["span_a"]
                take = min(delta, room_out + room_in)
                if take <= 0.0001:
                    continue
                grow_out = min(room_out, take)
                shot["out_sec"] += grow_out
                shot["in_sec"] -= min(room_in, take - grow_out)
                moved = True
                break
            hit_a = min(max(shot["impact_s"] - HIT_PRE, shot["in_sec"]), shot["out_sec"])
            hit_b = min(max(shot["impact_s"] + HIT_POST, shot["in_sec"]), shot["out_sec"])
            pre_room = max(0.0, (hit_a - shot["in_sec"]) - MIN_PRE)
            post_room = max(0.0, (shot["out_sec"] - hit_b) - MIN_POST)
            take = min(-delta, pre_room + post_room)
            if take <= 0.0001:
                continue
            cut_pre = min(pre_room, take)
            shot["in_sec"] += cut_pre
            shot["out_sec"] -= min(post_room, take - cut_pre)
            moved = True
            break
        if not moved:
            return


def restore_openings(shots: list[dict]) -> None:
    """Pull a high-score play back to its marked start. Take a little time from each lower play."""
    needy = sorted(
        (shot for shot in shots if shot.get("prefer_in") is not None and shot["score"] >= 8 and shot["in_sec"] > float(shot["prefer_in"]) + 0.05),
        key=lambda shot: shot["score"],
        reverse=True,
    )
    given = {id(shot): 0.0 for shot in shots}
    for shot in needy:
        miss = shot["in_sec"] - max(float(shot["prefer_in"]), shot["span_a"])
        if miss > 1.0:
            continue
        want = miss
        donors = sorted((row for row in shots if row is not shot and row["score"] + 1 < shot["score"]), key=lambda row: row["score"])
        for donor in donors:
            if want <= 0.01:
                break
            hit_a = min(max(donor["impact_s"] - HIT_PRE, donor["in_sec"]), donor["out_sec"])
            spare = max(0.0, (hit_a - donor["in_sec"]) - MIN_PRE)
            spare = min(spare, 0.45 - given[id(donor)])
            take = min(spare, want / max(1, len(donors)))
            if take <= 0.005:
                continue
            donor["in_sec"] += take
            shot["in_sec"] -= take
            given[id(donor)] += take
            want -= take
        for donor in donors:
            if want <= 0.01:
                break
            hit_a = min(max(donor["impact_s"] - HIT_PRE, donor["in_sec"]), donor["out_sec"])
            spare = max(0.0, (hit_a - donor["in_sec"]) - MIN_PRE)
            spare = min(spare, 0.45 - given[id(donor)])
            take = min(spare, want)
            if take <= 0.005:
                continue
            donor["in_sec"] += take
            shot["in_sec"] -= take
            given[id(donor)] += take
            want -= take


def _prepare(spans: list[tuple[float, float]], notes: dict) -> list[dict]:
    shots = []
    for index, (span_a, span_b) in enumerate(spans):
        note = notes.get(index) or {}
        impact = parse_timestamp(note.get("impact_s", (span_a + span_b) / 2.0))
        impact = min(max(impact, span_a + 0.12), span_b - 0.12)
        prefer_a = min(max(parse_timestamp(note.get("in_s", span_a)), span_a), impact)
        prefer_b = max(min(parse_timestamp(note.get("out_s", span_b)), span_b), impact)
        pre = max(0.0, impact - prefer_a)
        post = max(0.0, prefer_b - impact)
        share = pre / (pre + post) if pre + post > 0.05 else 0.58
        score = min(10.0, max(1.0, float(note.get("score") or 4)))
        shots.append({
            "index": index,
            "span_a": span_a,
            "span_b": span_b,
            "impact_s": impact,
            "pre_share": min(0.8, max(0.25, share)),
            "prefer_in": prefer_a if "in_s" in note else None,
            "score": score,
            "center_x": min(1.0, max(0.0, float(note.get("center_x", 0.5) or 0.5))),
            "label": label_text(note.get("label", "")),
        })
    return shots


def fit_to_target(spans: list[tuple[float, float]], notes: dict, target_s: float) -> list[dict]:
    """Give each locked shot a window. Screen time sums to target_s."""
    if target_s <= 0:
        raise ValueError("Target length must be positive.")
    shots = _drop_until_fit(_prepare(spans, notes), target_s)
    if not shots:
        raise RuntimeError("No shots to fit.")
    budgets = _budgets([shot["score"] for shot in shots], target_s)
    for _ in range(8):
        actuals = []
        for shot, budget in zip(shots, budgets):
            start, end = fit_window(shot["span_a"], shot["span_b"], shot["impact_s"], budget, shot["pre_share"])
            shot["in_sec"] = start
            shot["out_sec"] = end
            actuals.append(screen_of(start, end, shot["impact_s"]))
        gap = target_s - sum(actuals)
        if abs(gap) < 0.01:
            break
        if gap > 0:
            rooms = []
            for shot, actual in zip(shots, actuals):
                rooms.append(max(0.0, screen_of(shot["span_a"], shot["span_b"], shot["impact_s"]) - actual - 0.02))
            room = sum(rooms)
            if room < 0.01:
                break
            for i, extra in enumerate(rooms):
                budgets[i] += gap * (extra / room)
            continue
        order = sorted(range(len(shots)), key=lambda i: shots[i]["score"])
        cut = -gap
        for i in order:
            floor = _min_screen(shots[i]) + MIN_PRE + MIN_POST
            can = max(0.0, actuals[i] - floor)
            take = min(can, cut)
            budgets[i] = actuals[i] - take
            cut -= take
            if cut <= 0.0001:
                break
    _nudge(shots, target_s)
    for shot in shots:
        shot["in_sec"] = round(min(shot["span_b"], max(shot["span_a"], shot["in_sec"])), 4)
        shot["out_sec"] = round(min(shot["span_b"], max(shot["in_sec"] + 0.2, shot["out_sec"])), 4)
        shot["impact_s"] = round(min(shot["out_sec"] - 0.05, max(shot["in_sec"] + 0.05, shot["impact_s"])), 4)
        shot["screen_s"] = round(screen_of(shot["in_sec"], shot["out_sec"], shot["impact_s"]), 4)
    _nudge(shots, target_s)
    restore_openings(shots)
    _nudge(shots, target_s)
    real = sum(screen_of(shot["in_sec"], shot["out_sec"], shot["impact_s"]) for shot in shots)
    if abs(real - target_s) > 0.25:
        raise RuntimeError(f"Cut is {real:.2f}s. Target is {target_s:.2f}s.")
    for shot in shots:
        shot["screen_s"] = round(screen_of(shot["in_sec"], shot["out_sec"], shot["impact_s"]), 3)
    return shots


PACE_SCHEMA = {
    "type": "object",
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "keep": {"type": "boolean"},
                    "in_s": {"type": "number"},
                    "out_s": {"type": "number"},
                    "center_x": {"type": "number"},
                    "energy": {"type": "number"},
                },
                "required": ["index", "keep", "in_s", "out_s", "center_x", "energy"],
            },
        },
    },
    "required": ["shots"],
}


def build_pace_prompt(sport: str, video_s: float, spans: list[tuple[float, float]], target_s: float) -> str:
    guide = SPORT_GUIDE.get(sport, SPORT_GUIDE["auto"])
    lines = [f"{index} {start:.2f} {end:.2f}" for index, (start, end) in enumerate(spans)]
    return (
        "You are the editor of a 60 second vertical sports reel. "
        f"This file is the locked 16:9 cut. It is {video_s:.1f} seconds. "
        f"Your cut plays at normal speed and lasts {target_s:.0f} seconds. "
        "You cannot slow a play. You cannot speed a play. "
        "The sum of out_s minus in_s on the plays you keep must be 59 to 61 seconds.\n"
        f"{guide}\n"
        "Shots, one per line, as index start end:\n"
        + "\n".join(lines)
        + "\nFor every shot return index, keep, in_s, out_s, center_x, and energy.\n"
        "keep is false only when the play is dead and would make the reel boring.\n"
        "in_s is the frame where the point becomes worth watching. Cut the walk and the waiting.\n"
        "out_s is just after the ball is down. Keep the result. Cut the next walk.\n"
        "A routine finish can be 3.5 to 6 seconds. A rally that is actually good can be 8 to 14 seconds.\n"
        "Vary the length. Do not make every play the same length.\n"
        "Keep the shot order. Do not move a later play in front.\n"
        "center_x is 0 at the left edge and 1 at the right edge, on the ball at the finish.\n"
        "energy is 1 to 10. A routine touch is 4. A dive, a stuff, or a long rally is 8 or more.\n"
        "No titles. No slow motion. JSON only."
    )


def fit_pace(spans: list[tuple[float, float]], notes: dict, target_s: float) -> list[dict]:
    """Use Argon's 1x windows. Trim or extend only enough to land on target_s."""
    shots = []
    for index, (span_a, span_b) in enumerate(spans):
        note = notes.get(index) or {}
        if note.get("keep") is False:
            continue
        energy = min(10.0, max(1.0, float(note.get("energy") or note.get("score") or 5)))
        inn = min(max(parse_timestamp(note.get("in_s", span_a)), span_a), span_b - 0.8)
        out = min(max(parse_timestamp(note.get("out_s", span_b)), inn + 0.8), span_b)
        shots.append({
            "index": index,
            "span_a": span_a,
            "span_b": span_b,
            "in_sec": inn,
            "out_sec": out,
            "impact_s": inn,
            "energy": energy,
            "score": energy,
            "center_x": min(1.0, max(0.0, float(note.get("center_x", 0.5) or 0.5))),
            "label": "",
        })
    if not shots:
        raise RuntimeError("Argon kept no plays")

    def total() -> float:
        return sum(shot["out_sec"] - shot["in_sec"] for shot in shots)

    def floor_for(shot: dict) -> float:
        return 3.2 if shot["energy"] >= 8 else 2.6

    for _ in range(len(shots) * 8):
        gap = target_s - total()
        if abs(gap) < 0.02:
            break
        if gap < 0:
            order = sorted(range(len(shots)), key=lambda i: (shots[i]["energy"], -(shots[i]["out_sec"] - shots[i]["in_sec"])))
            moved = False
            for i in order:
                shot = shots[i]
                can = (shot["out_sec"] - shot["in_sec"]) - floor_for(shot)
                take = min(max(0.0, can), -gap)
                if take <= 0.01:
                    continue
                shot["in_sec"] += take
                moved = True
                break
            if not moved:
                break
        else:
            order = sorted(range(len(shots)), key=lambda i: shots[i]["energy"], reverse=True)
            moved = False
            for i in order:
                shot = shots[i]
                room = shot["span_b"] - shot["out_sec"]
                take = min(room, gap)
                if take > 0.01:
                    shot["out_sec"] += take
                    moved = True
                    break
                room_in = shot["in_sec"] - shot["span_a"]
                take = min(room_in, gap)
                if take <= 0.01:
                    continue
                shot["in_sec"] -= take
                moved = True
                break
            if not moved:
                break
    delta = target_s - total()
    order = sorted(range(len(shots)), key=lambda i: shots[i]["energy"], reverse=True)
    for i in order:
        if abs(delta) < 0.01:
            break
        shot = shots[i]
        if delta > 0:
            take = min(shot["span_b"] - shot["out_sec"], delta)
            shot["out_sec"] += take
            delta -= take
            take = min(shot["in_sec"] - shot["span_a"], delta)
            shot["in_sec"] -= take
            delta -= take
        else:
            can = max(0.0, (shot["out_sec"] - shot["in_sec"]) - 2.2)
            take = min(can, -delta)
            shot["in_sec"] += take
            delta += take
    real = total()
    if abs(real - target_s) > 0.25:
        raise RuntimeError(f"Cut is {real:.2f}s. Target is {target_s:.2f}s.")
    for shot in shots:
        shot["in_sec"] = round(shot["in_sec"], 4)
        shot["out_sec"] = round(shot["out_sec"], 4)
        shot["screen_s"] = round(shot["out_sec"] - shot["in_sec"], 3)
        shot["impact_s"] = shot["in_sec"]
    return shots


def impact_screen_times(shots: list[dict]) -> list[float]:
    clock = 0.0
    marks = []
    for shot in shots:
        for start, end, slow in hit_slices(shot["in_sec"], shot["out_sec"], shot["impact_s"]):
            if slow:
                local = min(max(shot["impact_s"] - start, 0.0), end - start)
                marks.append(round(clock + local * SLOW, 3))
            clock += (end - start) * (SLOW if slow else 1.0)
    return marks


def notes_by_index(edl: dict, count: int) -> dict:
    by = {}
    for row in edl.get("shots") or []:
        by[int(row["index"])] = row
    if count and 0 not in by and 1 in by:
        by = {key - 1: value for key, value in by.items()}
    return by


def build_lock_prompt(sport: str, video_s: float, spans: list[tuple[float, float]], target_s: float) -> str:
    guide = SPORT_GUIDE.get(sport, SPORT_GUIDE["auto"])
    lines = [f"{index} {start:.2f} {end:.2f}" for index, (start, end) in enumerate(spans)]
    return (
        "You time a locked sports edit for a vertical cut. "
        f"The file is {video_s:.1f} seconds. The finished cut is exactly {target_s:.0f} seconds. "
        "Do not add shots. Do not drop shots. Do not reorder shots.\n"
        f"{guide}\n"
        "Shots, one per line, as index start end:\n"
        + "\n".join(lines)
        + "\nFor every shot return index, impact_s, in_s, out_s, center_x, label, and score.\n"
        "impact_s is the contact. That is the spike, the block, the dig, or the ball on the floor.\n"
        "in_s is where the play becomes the point. Keep a dig that starts the rally. Cut the walk back.\n"
        "out_s is just after the ball is down. Keep a short look at the result. Cut the next walk.\n"
        "Keep in_s, impact_s, and out_s inside that shot. Keep impact_s between in_s and out_s.\n"
        "center_x is 0 at the left edge and 1 at the right edge, on the ball at impact.\n"
        "label is two words for the play. No emoji.\n"
        "score is 1 to 10. A routine kill is 4. A long rally or a dive is 8 or more.\n"
        "Use a short window on a routine play. Leave air on the best rally. "
        f"The editor fits the windows to {target_s:.0f} seconds.\n"
        "Return one object per shot, same order. JSON only."
    )


def ask_argon(proxy: Path, prompt: str, schema: dict, model: str, thinking: str, timeout_ms: int = HTTP_TIMEOUT_MS) -> dict:
    client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    video = types.Part.from_bytes(data=proxy.read_bytes(), mime_type="video/mp4")
    print(f"ARGON call {model} thinking={thinking}", flush=True)

    def _run():
        return client.models.generate_content(
            model=model,
            contents=[video, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=types.ThinkingConfig(thinking_level=thinking),
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                max_output_tokens=65536,
                http_options=types.HttpOptions(timeout=timeout_ms),
            ),
        )

    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_run)
    try:
        response = future.result(timeout=timeout_ms / 1000.0)
    except FuturesTimeout as exc:
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError("local timeout argon") from exc
    else:
        pool.shutdown(wait=True, cancel_futures=False)
    print(f"ARGON ok {round(time.time() - t0, 1)}s", flush=True)
    text = response.text or ""
    if not text.strip():
        raise RuntimeError("Argon returned no JSON")
    return json.loads(text)


def _lower_caption(text: str) -> str:
    font = ffmpeg_font()
    safe = label_text(text)
    if not font or not safe:
        return ""
    return (
        f"drawtext=fontfile='{font}':text='{safe}':fontcolor=white:fontsize=42:"
        f"box=1:boxcolor=black@0.45:boxborderw=12:x=(w-text_w)/2:y=h*0.78"
    )


def _pace_graph(center_x: float) -> str:
    cx = min(1.0, max(0.0, float(center_x)))
    wide = int(round(OUT_W * 2.15))
    wide -= wide % 2
    tall = int(round(wide * 9 / 16))
    tall -= tall % 2
    crop_h = int(round(tall * 0.90))
    crop_h -= crop_h % 2
    return (
        "split[bg][fg];"
        f"[bg]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=8:2,scale={OUT_W}:{OUT_H}[bgb];"
        f"[fg]scale={wide}:{tall}:flags=lanczos,crop={OUT_W}:{crop_h}:(iw-{OUT_W})*{cx:.4f}:(ih-{crop_h})*0.72,"
        f"eq=contrast=1.10:saturation=1.18:gamma=0.98,setpts=PTS-STARTPTS,fps=60[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)*0.18,setsar=1[vout]"
    )


def _stage_graph(center_x: float, punch: bool, caption: str, slow: bool) -> str:
    fg = "scale=1080:-2:flags=lanczos,eq=contrast=1.06:saturation=1.12:gamma=0.98"
    if punch:
        cx = min(1.0, max(0.0, float(center_x)))
        fg += f",scale=1146:-2:flags=lanczos,crop=1080:ih:(iw-1080)*{cx:.4f}:0"
    cap = ""
    if caption:
        extra = _lower_caption(caption)
        if extra:
            cap = "," + extra
    tail = "setpts=2*(PTS-STARTPTS),fps=60" if slow else "setpts=PTS-STARTPTS,fps=60"
    return (
        "split[bg][fg];"
        f"[bg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},boxblur=18:3[bgb];"
        f"[fg]{fg}[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1{cap},{tail}[vout]"
    )


def _audio_filter(dur: float, slow: bool) -> str:
    out = dur * (SLOW if slow else 1.0)
    tempo = "atempo=0.5," if slow else ""
    if out < 0.08:
        return f"{tempo}aresample=48000,asetpts=PTS-STARTPTS"
    fade = min(0.012, out / 5.0)
    end = max(0.0, out - fade)
    return f"{tempo}afade=t=in:st=0:d={fade:.3f},afade=t=out:st={end:.3f}:d={fade:.3f},aresample=48000,asetpts=PTS-STARTPTS"


def stream_seconds(path: Path, selector: str) -> float:
    proc = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-i", str(path), "-map", selector, "-c", "copy", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    clock = ""
    for line in (proc.stderr or "").splitlines():
        if "time=" in line:
            clock = line.split("time=")[1].split()[0]
    if not clock:
        raise RuntimeError(f"no stream length for {selector}")
    hours, minutes, seconds = clock.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _conform(src: Path, dest: Path, target_s: float) -> None:
    video_s = stream_seconds(src, "0:v:0")
    audio_s = stream_seconds(src, "0:a:0")
    if video_s <= 0:
        raise RuntimeError("render has no duration")
    factor = video_s / target_s
    ff = ffmpeg_bin()
    if abs(video_s - target_s) <= 0.02 and abs(audio_s - video_s) <= 0.02:
        _run_ffmpeg([ff, "-y", "-i", str(src), "-t", f"{target_s:.3f}", "-c", "copy", "-movflags", "+faststart", str(dest)])
        return
    if factor < 0.5 or factor > 2.0:
        raise RuntimeError(f"Cut is {video_s:.2f}s. Target is {target_s:.2f}s.")
    _run_ffmpeg(
        [
            ff, "-y", "-i", str(src),
            "-vf", f"setpts={target_s / video_s:.8f}*(PTS-STARTPTS),fps=60",
            "-af", f"asetpts=PTS-STARTPTS,atempo={factor:.8f},atrim=0:{target_s:.3f}",
            "-t", f"{target_s:.3f}",
            "-shortest",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            str(dest),
        ]
    )


def render_stage(
    source: Path,
    shots: list[dict],
    dest: Path,
    *,
    layout: str = "blurred-bg",
    captions: bool = True,
    target_s: float = 60.0,
    pace: bool = False,
) -> None:
    if not shots:
        raise RuntimeError("Argon returned no shots")
    if layout not in LAYOUTS:
        raise ValueError("Layout must be smart-crop or blurred-bg.")
    punch_hits = layout == "smart-crop"
    ff = ffmpeg_bin()
    pieces = []
    if pace:
        for shot in shots:
            pieces.append((shot["in_sec"], shot["out_sec"], False, shot))
    else:
        for shot in shots:
            for start, end, slow in hit_slices(shot["in_sec"], shot["out_sec"], shot["impact_s"]):
                pieces.append((start, end, slow, shot))
    print(f"ARGON render n={len(shots)} segs={len(pieces)} {layout} target={target_s:.2f}", flush=True)
    with tempfile.TemporaryDirectory(prefix="argon_stage_") as tmp:
        tmp_path = Path(tmp)
        segments = []
        for idx, (start, end, slow, shot) in enumerate(pieces):
            dur = end - start
            screen = dur * (SLOW if slow else 1.0)
            seg = tmp_path / f"seg_{idx:03d}.mp4"
            caption = "" if pace else (shot.get("label") if captions and slow else "")
            if pace:
                graph = _pace_graph(float(shot.get("center_x", 0.5)))
            else:
                graph = _stage_graph(float(shot.get("center_x", 0.5)), punch_hits and slow, caption or "", slow)
            print(f"ARGON seg {idx + 1}/{len(pieces)} {start:.2f}-{end:.2f} slow={int(slow)}", flush=True)
            _run_ffmpeg(
                [
                    ff, "-y", "-ss", f"{start:.4f}", "-t", f"{dur:.4f}", "-i", str(source),
                    "-filter_complex", graph, "-map", "[vout]", "-map", "0:a:0?",
                    "-af", _audio_filter(dur, slow),
                    "-t", f"{screen:.4f}",
                    "-shortest",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                    str(seg),
                ]
            )
            segments.append(seg)
        listing = tmp_path / "list.txt"
        listing.write_text("".join(f"file '{seg.as_posix()}'\n" for seg in segments), encoding="utf-8")
        joined = tmp_path / "joined.mp4"
        _run_ffmpeg([ff, "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(joined)])
        graded = joined
        if stream_seconds(joined, "0:v:0") >= 3.0:
            graded = tmp_path / "loud.mp4"
            picture_s = stream_seconds(joined, "0:v:0")
            _run_ffmpeg(
                [
                    ff, "-y", "-i", str(joined),
                    "-c:v", "copy",
                    "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,atrim=0:{picture_s:.3f},asetpts=PTS-STARTPTS",
                    "-t", f"{picture_s:.3f}",
                    "-shortest",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                    "-movflags", "+faststart",
                    str(graded),
                ]
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        _conform(graded, dest, target_s)
    picture_s = stream_seconds(dest, "0:v:0")
    sound_s = stream_seconds(dest, "0:a:0")
    if abs(picture_s - target_s) > 0.05 or abs(sound_s - picture_s) > 0.05:
        raise RuntimeError(f"TikTok picture is {picture_s:.2f}s and sound is {sound_s:.2f}s. Target is {target_s:.2f}s.")
    print(f"ARGON {dest.name} {picture_s:.2f}s target={target_s:.2f} clips={len(shots)}", flush=True)


def polish_reel(
    reel: Path,
    dest: Path,
    *,
    model: str | None = None,
    sport: str = "volleyball",
    thinking: str = ARGON_THINKING,
    layout: str = "blurred-bg",
    captions: bool = True,
    hook: bool = True,
    target_s: float = 60.0,
    proxy_dir: Path | None = None,
    keeps: Path | None = None,
    spans: list | None = None,
) -> dict:
    """Time the locked 16:9 reel and write a 60 second 9:16 cut."""
    use = model or ARGON_MODEL
    level = (thinking or ARGON_THINKING).upper()
    if level not in THINKING:
        raise ValueError("Thinking must be LOW, MEDIUM, or HIGH.")
    if sport not in SPORTS:
        raise ValueError("Sport must be volleyball, basketball, or auto.")
    if layout not in LAYOUTS:
        raise ValueError("Layout must be smart-crop or blurred-bg.")
    if target_s < 15 or target_s > 180:
        raise ValueError("Target length must be 15 to 180 seconds.")
    meta = probe_video(reel)
    if meta["duration"] <= 0:
        raise RuntimeError("The reel has no duration.")
    if spans is None and keeps is not None and Path(keeps).is_file():
        spans = spans_from_keeps(keeps)
    if spans is None and (reel.parent / "keeps.json").is_file():
        spans = spans_from_keeps(reel.parent / "keeps.json")
    if spans is None:
        spans = shot_spans(reel, meta["duration"])
    spans = fit_spans(list(spans), meta["duration"])
    print(f"ARGON pace shots={len(spans)} target={target_s:.2f}s", flush=True)
    proxies = proxy_dir if proxy_dir is not None else dest.parent / "argon"
    proxy = proxies / "reel_proxy.mp4"
    if proxy.is_file() and proxy.stat().st_size <= INLINE_MAX_BYTES:
        print(f"ARGON proxy reuse bytes={proxy.stat().st_size}", flush=True)
    else:
        proxy = make_argon_proxy(reel, proxy)
    edl = ask_argon(
        proxy,
        build_pace_prompt(sport, meta["duration"], spans, target_s),
        PACE_SCHEMA,
        use,
        level,
        timeout_ms=900_000,
    )
    shots = fit_pace(spans, notes_by_index(edl, len(spans)), target_s)
    for shot in shots:
        print(
            f"ARGON shot {shot['index']} {shot['in_sec']:.2f}-{shot['out_sec']:.2f} {shot['screen_s']:.2f}s energy={shot['energy']:.0f}",
            flush=True,
        )
    edl["shots_locked"] = shots
    edl["model"] = use
    edl["thinking"] = level
    edl["target_s"] = target_s
    edl["hook"] = bool(hook)
    edl["pace"] = True
    note_path = dest.with_name(dest.stem + "_argon.json")
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(json.dumps(edl, indent=2), encoding="utf-8")
    render_stage(reel, shots, dest, layout=layout, captions=False, target_s=target_s, pace=True)
    return edl
