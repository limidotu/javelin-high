"""Virtual 9:16 camera: constant zoom, two-pass ball path, no live jiggle."""

from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from clip_window import clip_window
from detect_highlights import ffmpeg_bin

ZOOM_MIN = 1.12
ZOOM_MAX = 1.85
HOLD_DIST = 0.16
PLAY_SPEED = 0.32
ZOOM_SERVER = 1.62
ZOOM_BALL = 1.52
LOCK_ZOOM = 1.52
OUT_FPS = 30.0
OUT_W = 1080
OUT_H = 1920
Y_BIAS = 0.10
DETECT_FPS = 15.0
TRACK_SIGMA = 3.0
CAM_SIGMA = 1.2
DEFAULT_CY = 0.36
MIN_TRACK_Q = 58.0
YOLO_EVERY = 0
WEIGHTS = Path(__file__).resolve().parent / "yolov8n.pt"

_YOLO = None


def even(n: int) -> int:
    return max(2, int(n) // 2 * 2)


def crop_rect(
    frame_w: int,
    frame_h: int,
    cx: float,
    cy: float,
    zoom: float,
) -> tuple[int, int, int, int]:
    """9:16 crop box. cx/cy are 0-1."""
    zoom = max(1.05, min(2.4, zoom))
    crop_h = even(int(frame_h / zoom))
    crop_w = even(int(crop_h * 9 / 16))
    if crop_w > frame_w:
        crop_w = even(frame_w)
        crop_h = even(int(crop_w * 16 / 9))
    px = cx * frame_w
    py = cy * frame_h
    x = int(round(px - crop_w / 2))
    y = int(round(py - crop_h / 2))
    x = max(0, min(frame_w - crop_w, x))
    y = max(0, min(frame_h - crop_h, y))
    return even(x), even(y), crop_w, crop_h


def damp(
    pos: float,
    vel: float,
    target: float,
    dt: float,
    freq: float = 1.35,
    zeta: float = 1.05,
) -> tuple[float, float]:
    omega = 2.0 * math.pi * freq
    acc = omega * omega * (target - pos) - 2.0 * zeta * omega * vel
    vel = vel + acc * dt
    pos = pos + vel * dt
    return pos, vel


@dataclass
class Target:
    cx: float
    cy: float
    zoom: float
    has_ball: bool
    has_person: bool


def _clamp_target(cx: float, cy: float, zoom: float, has_ball: bool, has_person: bool) -> Target:
    return Target(
        max(0.05, min(0.95, cx)),
        max(0.08, min(0.85, cy)),
        max(ZOOM_MIN, min(ZOOM_MAX, zoom)),
        has_ball,
        has_person,
    )


def _nearest_person(
    xy: tuple[float, float], persons: list[tuple[float, float, float, float]]
) -> tuple[tuple[float, float, float, float], float] | None:
    if not persons:
        return None
    best = min(persons, key=lambda p: (p[0] - xy[0]) ** 2 + (p[1] - xy[1]) ** 2)
    dist = math.hypot(best[0] - xy[0], best[1] - xy[1])
    return best, dist


def aim(
    ball: tuple[float, float] | None,
    ball_speed: float,
    persons: list[tuple[float, float, float, float]],
    event: str,
    in_play: bool,
    motion: tuple[float, float] = (0.5, 0.4),
    lead: tuple[float, float] = (0.0, 0.0),
) -> tuple[Target, bool]:
    """Always aim at the ball. A ball in hand still counts."""
    del ball_speed, event
    if ball is not None:
        return (
            _clamp_target(ball[0] + lead[0], ball[1] + lead[1], ZOOM_BALL, True, bool(persons)),
            True,
        )
    return _clamp_target(motion[0], motion[1], ZOOM_MIN + 0.18, False, bool(persons)), in_play


class SmoothCam:
    """Fixed zoom. Slow pan. Hold still inside the dead zone."""

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        zoom: float = LOCK_ZOOM,
        cx: float = 0.5,
        cy: float = 0.55,
        alpha: float = 0.08,
        dead: float = 0.02,
    ) -> None:
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.zoom = zoom
        self.cx = cx
        self.cy = cy
        self.alpha = alpha
        self.dead = dead
        _, _, self.cw, self.ch = crop_rect(frame_w, frame_h, cx, cy, zoom)

    def follow(self, tx: float, ty: float) -> None:
        dx = tx - self.cx
        dy = ty - self.cy
        if math.hypot(dx, dy) < self.dead:
            return
        self.cx = max(0.05, min(0.95, self.cx + dx * self.alpha))
        self.cy = max(0.08, min(0.88, self.cy + dy * self.alpha))

    def box(self) -> tuple[int, int, int, int]:
        x = even(int(round(self.cx * self.frame_w - self.cw / 2)))
        y = even(int(round(self.cy * self.frame_h - self.ch / 2)))
        x = max(0, min(self.frame_w - self.cw, x))
        y = max(0, min(self.frame_h - self.ch, y))
        return x, y, self.cw, self.ch


def yellow_ball_cands(
    bgr: np.ndarray,
    prev_gray: np.ndarray | None,
) -> list[tuple[float, float, float]]:
    """Yellow circular blobs with motion. Each item is (score, cx, cy)."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 80, 90), (40, 255, 255))
    y1 = int(h * 0.30)
    high = cv2.inRange(hsv[:y1], (12, 35, 70), (48, 255, 255))
    yellow[:y1] = cv2.bitwise_or(yellow[:y1], high)
    yellow = cv2.medianBlur(yellow, 5)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mot = None
    if prev_gray is not None:
        delta = cv2.absdiff(gray, prev_gray)
        _, mot = cv2.threshold(delta, 11, 255, cv2.THRESH_BINARY)
        mot = cv2.dilate(mot, np.ones((3, 3), np.uint8))
        mask = cv2.bitwise_and(yellow, mot)
        if int(cv2.countNonZero(mask)) < 5:
            mask = yellow
    else:
        mask = yellow
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[float, float, float]] = []
    for c in cnts:
        area = cv2.contourArea(c)
        scale = (w * h) / (1280.0 * 720.0)
        max_a = 700.0 * scale
        if area < 4.0 * scale or area > max_a:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circ = 4.0 * math.pi * area / (peri * peri)
        if circ < 0.48:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        max_side = 36.0 * w / 1280.0
        if bw > max_side or bh > max_side:
            continue
        ar = bw / max(bh, 1)
        if ar < 0.55 or ar > 1.85:
            continue
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        if cy < 0.02 or cy > 0.50:
            continue
        if cy >= 0.22 and area < 8.0 * scale:
            continue
        mfrac = 0.4
        if mot is not None:
            patch = mot[y : y + bh, x : x + bw]
            mfrac = float(cv2.countNonZero(patch)) / max(patch.size, 1)
        if mfrac < 0.08 and circ < 0.75:
            continue
        if cy > 0.50 and mfrac < 0.16:
            continue
        if cy > 0.42:
            close_max_a = 180.0 * (w * h) / (1280.0 * 720.0)
            close_side = 22.0 * w / 1280.0
            if area > close_max_a or bw > close_side or bh > close_side:
                continue
        score = circ * math.sqrt(area) + 8.0 * mfrac
        if 0.14 < cy < 0.48:
            score += 3.0
        if area > 320.0 * (w * h) / (1280.0 * 720.0):
            score -= 2.5
        out.append((score, cx, cy))
    out.sort(reverse=True)
    return out[:8]


def choose_detection(
    cands: list[tuple[float, float, float]],
    last: tuple[float, float] | None,
    dt: float,
    speed: float,
    hint_cx: float = 0.5,
) -> tuple[float, float] | None:
    """Pick the blob that continues the track. Ignore far false hits."""
    if not cands:
        return None
    if last is None:
        air = [c for c in cands if 0.12 < c[2] < 0.52 and c[0] >= 7.0]
        if not air:
            return None
        air.sort(key=lambda c: c[0] - 1.4 * abs(c[1] - hint_cx), reverse=True)
        return (air[0][1], air[0][2])
    gate = min(0.14, 0.04 + 1.3 * speed * dt + 0.10 * dt)
    near: list[tuple[float, float, float, float]] = []
    for score, cx, cy in cands:
        dist = math.hypot(cx - last[0], cy - last[1])
        if dist <= gate:
            near.append((dist, score, cx, cy))
    if near:
        near.sort(key=lambda r: r[0] - 0.015 * r[1])
        return (near[0][2], near[0][3])
    return None


def best_play_path(
    frame_cands: list[list[tuple[float, float, float]]],
    hint_cx: float,
    court_tol: float = 0.36,
) -> tuple[list[float | None], list[float | None]]:
    """Link blobs into tracks. Keep the moving airborne path. Drop a spare ball."""
    n = len(frame_cands)
    tracks: list[list[tuple[int, float, float, float]]] = []
    for i, cands in enumerate(frame_cands):
        pool = [c for c in cands if abs(c[1] - hint_cx) <= court_tol]
        used: set[int] = set()
        for tr in tracks:
            li, lx, ly, _sc = tr[-1]
            gap = i - li
            if gap > 14:
                continue
            best_j = None
            best_d = 9.0
            gate = min(0.24, 0.07 + 0.025 * gap)
            for j, cand in enumerate(pool):
                if j in used:
                    continue
                if cand[2] > 0.50:
                    continue
                dist = math.hypot(cand[1] - lx, cand[2] - ly)
                if dist <= gate and dist < best_d:
                    best_d = dist
                    best_j = j
            if best_j is not None:
                cand = pool[best_j]
                tr.append((i, cand[1], cand[2], cand[0]))
                used.add(best_j)
        for j, cand in enumerate(pool):
            if j in used or cand[0] < 7.0:
                continue
            if cand[2] > 0.52 or cand[2] < 0.10:
                continue
            tracks.append([(i, cand[1], cand[2], cand[0])])
    xs: list[float | None] = [None] * n
    ys: list[float | None] = [None] * n
    if not tracks:
        return xs, ys

    def track_score(tr: list[tuple[int, float, float, float]]) -> float:
        if len(tr) < 4:
            return -1.0
        if _carried_ball(tr):
            return -1.0
        mean_x = float(np.mean([p[1] for p in tr]))
        if mean_x < 0.20 or mean_x > 0.80:
            return -1.0
        air = sum(1.0 for _i, _x, cy, _s in tr if cy < 0.50) / len(tr)
        dist = 0.0
        for a, b in zip(tr, tr[1:]):
            dist += math.hypot(b[1] - a[1], b[2] - a[2])
        cys = [p[2] for p in tr]
        bounce = float(np.std(np.array(cys, dtype=np.float64)))
        return air * 4.0 + dist * 3.0 + bounce * 6.0 + 0.04 * len(tr)

    winner = max(tracks, key=track_score)
    if track_score(winner) < 0:
        return xs, ys
    for i, cx, cy, _s in winner:
        xs[i] = cx
        ys[i] = cy
    return extend_rising_set(xs, ys)


def extend_rising_set(
    xs: list[float | None],
    ys: list[float | None],
    extra: int = 8,
) -> tuple[list[float | None], list[float | None]]:
    """Keep panning up when a set disappears against the roof."""
    hits = [(i, x, y) for i, (x, y) in enumerate(zip(xs, ys)) if x is not None and y is not None]
    if len(hits) < 3:
        return xs, ys
    tail = hits[-4:]
    dys = [tail[i][2] - tail[i - 1][2] for i in range(1, len(tail))]
    mean_dy = sum(dys) / len(dys)
    if mean_dy >= -0.008:
        return xs, ys
    dx = (tail[-1][1] - tail[0][1]) / max(len(tail) - 1, 1)
    dy = mean_dy
    i0 = tail[-1][0]
    x, y = tail[-1][1], tail[-1][2]
    xs2 = list(xs)
    ys2 = list(ys)
    for k in range(1, extra + 1):
        j = i0 + k
        if j >= len(xs2) or xs2[j] is not None:
            break
        x = max(0.05, min(0.95, x + dx))
        y = max(0.04, min(0.70, y + dy))
        dy += 0.006
        xs2[j] = x
        ys2[j] = y
    return xs2, ys2


def track_quality(xs: list[float | None], ys: list[float | None]) -> float:
    """High when the track is a moving airborne rally. Near 0 for walkers."""
    air_i = [
        i
        for i, (x, y) in enumerate(zip(xs, ys))
        if x is not None and y is not None and y < 0.48
    ]
    if len(air_i) < 6:
        return 0.0
    air = [(xs[i], ys[i]) for i in air_i]
    dist = 0.0
    for a, b in zip(air, air[1:]):
        dist += math.hypot(b[0] - a[0], b[1] - a[1])
    if dist < 0.08:
        return 0.0
    bounce = float(np.std(np.array([p[1] for p in air], dtype=np.float64)))
    q = float(len(air) + 12.0 * dist + 20.0 * bounce)
    n = max(len(xs), 1)
    med = float(np.median(np.array(air_i, dtype=np.float64))) / n
    if med < 0.42:
        q *= 0.35
    last_frac = float(max(air_i)) / n
    if last_frac < 0.58:
        q *= 0.40
    return q


def _carried_ball(tr: list[tuple[int, float, float, float]]) -> bool:
    """True when the blob is a held spare walked across the camera."""
    if len(tr) < 4:
        return False
    cys = np.array([p[2] for p in tr], dtype=np.float64)
    mean_cy = float(cys.mean())
    std_cy = float(cys.std())
    dy = float(np.abs(np.diff(cys)).mean()) if cys.size > 1 else 0.0
    return mean_cy > 0.44 and std_cy < 0.045 and dy < 0.012


def lock_play_court(
    frame_cands: list[list[tuple[float, float, float]]],
    hint_cx: float,
) -> tuple[float, float]:
    """Return (court_cx, court_tol) for the in-play court only."""
    xs, _ys = best_play_path(frame_cands, hint_cx=hint_cx, court_tol=0.40)
    hit = [x for x in xs if x is not None]
    if len(hit) < 5:
        return hint_cx, 0.22
    med = float(np.median(np.array(hit, dtype=np.float64)))
    court_cx = 0.7 * med + 0.3 * hint_cx
    return court_cx, 0.16


def _gauss1d(vals: np.ndarray, sigma: float) -> np.ndarray:
    if vals.size == 0:
        return vals
    radius = max(1, int(round(sigma * 3)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / max(sigma, 0.5)) ** 2)
    kernel /= kernel.sum()
    ext = np.pad(vals.astype(np.float64), radius, mode="edge")
    return np.convolve(ext, kernel, mode="valid")


def _nan_hold_interp(vals: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill short holes by lerp. Hold the last value across long holes.

    Leave a leading gap as NaN so the camera can stay on the hint.
    """
    out = vals.astype(np.float64).copy()
    n = out.size
    good = np.isfinite(out)
    if not np.any(good):
        return out
    last_i: int | None = None
    last_v: float | None = None
    for i in range(n):
        if good[i]:
            if last_i is not None and last_v is not None:
                gap = i - last_i
                if gap > 1 and gap <= max_gap:
                    for k in range(1, gap):
                        t = k / gap
                        out[last_i + k] = last_v * (1.0 - t) + out[i] * t
                elif gap > max_gap:
                    out[last_i + 1 : i] = last_v
            last_i = i
            last_v = float(out[i])
        elif last_v is not None:
            out[i] = last_v
    return out


def _drop_orphans(xs: np.ndarray, ys: np.ndarray, min_run: int = 2, max_dist: float = 0.16) -> None:
    n = xs.size
    i = 0
    while i < n:
        if not np.isfinite(xs[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and np.isfinite(xs[j + 1]):
            if math.hypot(xs[j + 1] - xs[j], ys[j + 1] - ys[j]) > max_dist:
                break
            j += 1
        if (j - i + 1) < min_run:
            xs[i : j + 1] = np.nan
            ys[i : j + 1] = np.nan
        i = j + 1


def smooth_track(
    xs: list[float | None],
    ys: list[float | None],
    *,
    sigma: float = 14.0,
    max_gap: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn sparse detections into a held, interpolated, Gaussian path."""
    ax = np.array([np.nan if v is None else float(v) for v in xs], dtype=np.float64)
    ay = np.array([np.nan if v is None else float(v) for v in ys], dtype=np.float64)
    _drop_orphans(ax, ay)
    if not np.any(np.isfinite(ax)):
        return ax, ay
    ax = _nan_hold_interp(ax, max_gap)
    ay = _nan_hold_interp(ay, max_gap)
    ax = np.clip(_gauss1d(ax, sigma), 0.05, 0.95)
    ay = np.clip(_gauss1d(ay, sigma), 0.08, 0.78)
    return ax, ay


def _yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO

        weights = str(WEIGHTS if WEIGHTS.exists() else "yolov8n.pt")
        model = YOLO(weights)
        model.fuse()
        _YOLO = model
    return _YOLO


def _yolo_hints(
    bgr: np.ndarray, want_people: bool
) -> tuple[tuple[float, float] | None, list[tuple[float, float, float, float]]]:
    classes = [0, 32] if want_people else [32]
    res = _yolo().predict(bgr, imgsz=960, verbose=False, conf=0.08, classes=classes)[0]
    names = res.names
    h, w = bgr.shape[:2]
    ball = None
    ball_conf = -1.0
    persons: list[tuple[float, float, float, float]] = []
    if res.boxes is None:
        return None, persons
    for box in res.boxes:
        cls = int(box.cls[0])
        label = names.get(cls, "") if isinstance(names, dict) else str(names[cls])
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        area = ((x2 - x1) * (y2 - y1)) / (w * h)
        conf = float(box.conf[0])
        if label == "sports ball" and area < 0.008 and 0.07 < cy < 0.68:
            if conf > ball_conf:
                ball = (cx, cy)
                ball_conf = conf
        elif label == "person":
            head_y = (y1 + 0.16 * (y2 - y1)) / h
            box_h = (y2 - y1) / h
            if area < 0.12 and box_h < 0.58:
                persons.append((cx, head_y, area, box_h))
    return ball, persons


def collect_track(
    path: Path,
    start: float,
    dur: float,
    hint_cx: float = 0.5,
) -> tuple[list[float | None], list[float | None], float]:
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start) * 1000.0)
    stride = max(1, int(round(src_fps / DETECT_FPS)))
    frame_cands: list[list[tuple[float, float, float]]] = []
    prev_gray = None
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / src_fps
            if t > dur + 0.15:
                break
            if idx % stride == 0:
                small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                cands = yellow_ball_cands(small, prev_gray)
                prev_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if YOLO_EVERY > 0 and idx % YOLO_EVERY == 0:
                    yball, _ypeople = _yolo_hints(frame, want_people=False)
                    if yball is not None and 0.08 < yball[1] < 0.52:
                        cands = [(16.0, yball[0], yball[1])] + cands
                frame_cands.append(cands)
            idx += 1
    finally:
        cap.release()
    court_cx, court_tol = lock_play_court(frame_cands, hint_cx)
    xs, ys = best_play_path(frame_cands, hint_cx=court_cx, court_tol=court_tol)
    return xs, ys, src_fps / stride


def hold_after_rally(
    xs: list[float | None],
    ys: list[float | None],
) -> tuple[list[float | None], list[float | None]]:
    """Keep the longest airborne run. After it, do not chase walkers."""
    n = len(xs)
    best_i = 0
    best_j = 0
    best_n = 0
    i = 0
    while i < n:
        yi = ys[i]
        if xs[i] is None or yi is None or yi > 0.48:
            i += 1
            continue
        j = i
        while j < n:
            yj = ys[j]
            if xs[j] is None or yj is None or yj > 0.48:
                break
            j += 1
        if j - i > best_n:
            best_i = i
            best_j = j
            best_n = j - i
        i = j
    if best_n < 3:
        return xs, ys
    while best_j - best_i > 3:
        y_prev = ys[best_j - 1]
        if y_prev is not None and y_prev > 0.38:
            best_j -= 1
            continue
        break
    peak_k = best_i
    peak_y = 1.0
    for k in range(best_i, best_j):
        yk = ys[k]
        if yk is not None and yk < peak_y:
            peak_y = yk
            peak_k = k
    peak_x = xs[peak_k]
    xs2 = list(xs)
    ys2 = list(ys)
    for k in range(best_j, n):
        xs2[k] = peak_x
        ys2[k] = peak_y
    return xs2, ys2


def plan_camera(
    xs: list[float | None],
    ys: list[float | None],
    hint_cx: float,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = hold_after_rally(xs, ys)
    n = len(xs)
    sx, sy = smooth_track(xs, ys, sigma=TRACK_SIGMA, max_gap=6)
    out_x = np.full(n, hint_cx, dtype=np.float64)
    out_y = np.full(n, DEFAULT_CY, dtype=np.float64)
    if n == 0:
        return out_x, out_y
    for i in range(n):
        if np.isfinite(sx[i]):
            lead = 0.0
            if i >= 2 and np.isfinite(sx[i - 2]):
                lead = 0.40 * (sx[i] - sx[i - 2])
            out_x[i] = sx[i] + lead
            lift = 0.02 if sy[i] < 0.26 else Y_BIAS
            out_y[i] = sy[i] + lift
    out_x = np.clip(_gauss1d(out_x, CAM_SIGMA), 0.10, 0.90)
    out_y = np.clip(_gauss1d(out_y, CAM_SIGMA), 0.10, 0.62)
    return out_x, out_y


def slice_clip(path: Path, start: float, dur: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_bin(),
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{dur:.3f}",
            "-i",
            str(path),
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-ac",
            "2",
            str(dest),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def render_follow(
    path: Path,
    start: float,
    dur: float,
    dest: Path,
    hint_cx: float = 0.5,
    event: str = "play",
) -> float:
    del event
    dest.parent.mkdir(parents=True, exist_ok=True)
    sliced = dest.with_suffix(".slice.mp4")
    cmd_file = dest.with_suffix(".crop.txt")
    t0 = time.time()
    slice_clip(path, start, dur, sliced)
    print(f"  slice {time.time() - t0:.1f}s", flush=True)
    quality = 0.0
    try:
        t1 = time.time()
        xs, ys, _det_fps = collect_track(sliced, 0.0, dur, hint_cx=hint_cx)
        quality = track_quality(xs, ys)
        cam_x, cam_y = plan_camera(xs, ys, hint_cx=hint_cx)
        print(f"  track {time.time() - t1:.1f}s quality={quality:.1f}", flush=True)
        if quality < MIN_TRACK_Q:
            return quality
        cap = cv2.VideoCapture(str(sliced))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        _, _, cw, ch = crop_rect(fw, fh, hint_cx, DEFAULT_CY, LOCK_ZOOM)
        n_out = max(1, int(round(dur * OUT_FPS)))
        rx, ry = _resample_path(cam_x, cam_y, n_out)
        _write_crop_cmd(cmd_file, rx, ry, OUT_FPS, fw, fh, cw, ch)
        vf = (
            f"sendcmd=f={cmd_file.name},"
            f"crop={cw}:{ch}:0:0,"
            f"scale={OUT_W}:{OUT_H}:flags=bilinear,"
            f"fps={OUT_FPS:.3f}"
        )
        t2 = time.time()
        proc = subprocess.run(
            [
                ffmpeg_bin(),
                "-y",
                "-i",
                sliced.name,
                "-vf",
                vf,
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-shortest",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-af",
                "loudnorm",
                "-c:a",
                "aac",
                dest.name,
            ],
            cwd=str(dest.parent),
            capture_output=True,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")[-2500:]
            raise RuntimeError(err)
        print(f"  crop {time.time() - t2:.1f}s", flush=True)
        return quality
    finally:
        sliced.unlink(missing_ok=True)
        cmd_file.unlink(missing_ok=True)


def _resample_path(
    path_x: np.ndarray, path_y: np.ndarray, n_out: int
) -> tuple[np.ndarray, np.ndarray]:
    n = int(path_x.size)
    if n == 0:
        return np.full(n_out, 0.5), np.full(n_out, DEFAULT_CY)
    if n == 1:
        return np.full(n_out, float(path_x[0])), np.full(n_out, float(path_y[0]))
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, n_out)
    return np.interp(dst, src, path_x), np.interp(dst, src, path_y)


def _write_crop_cmd(
    path: Path,
    cam_x: np.ndarray,
    cam_y: np.ndarray,
    fps: float,
    fw: int,
    fh: int,
    cw: int,
    ch: int,
) -> None:
    lines: list[str] = []
    for i, (cx, cy) in enumerate(zip(cam_x, cam_y)):
        t = i / max(fps, 1e-6)
        x = even(int(round(float(cx) * fw - cw / 2)))
        y = even(int(round(float(cy) * fh - ch / 2)))
        x = max(0, min(fw - cw, x))
        y = max(0, min(fh - ch, y))
        lines.append(f"{t:.4f} crop x {x};")
        lines.append(f"{t:.4f} crop y {y};")
    path.write_text("\n".join(lines), encoding="utf-8")
