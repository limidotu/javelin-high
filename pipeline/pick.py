"""Pick 2.5 candidates, 3.8 reel clips, and a complete drop-folder file."""

from __future__ import annotations

import time
from pathlib import Path

SCAN_CAP = 75
REVIEW_BATCH = 10
REEL_MIN = 8
REEL_KEEP = 10
MIN_REVIEW_SCORE = 8
STABLE_S = 8.0
POLL_S = 1.0
WAIT_TIMEOUT_S = 7200.0
GENS_NAME = "generations"


def cap_by_score(rows: list[dict], limit: int = SCAN_CAP) -> list[dict]:
    """Highest 2.5 score first. Earlier start wins a tie. Never pad."""
    if limit <= 0 or not rows:
        return []
    ordered = sorted(
        rows,
        key=lambda r: (-int(r["score"]), float(r["start_s"])),
    )
    return ordered[: min(limit, len(ordered))]


def batches(items: list, n: int = REVIEW_BATCH) -> list[list]:
    if n <= 0:
        raise ValueError("batch size must be > 0")
    return [items[i : i + n] for i in range(0, len(items), n)]


def child_review_groups(group: list) -> list[list]:
    """Next smaller 3.8 batches after a failed 10-clip or 5-clip call."""
    n = len(group)
    if n <= 1:
        return []
    if n > 5:
        mid = (n + 1) // 2
        return [group[:mid], group[mid:]]
    return [[item] for item in group]


def pick_reel(
    reviews: list[dict],
    n: int = REEL_KEEP,
    *,
    min_n: int = REEL_MIN,
    min_score: int = MIN_REVIEW_SCORE,
) -> list[dict]:
    """Take 8 to 10 clips. Prefer 3.8 keep with score >= 8. Fill from high scores."""
    n = max(0, int(n))
    min_n = max(0, min(int(min_n), n))
    kept = [
        r
        for r in reviews
        if r.get("keep") and int(r.get("score") or 0) >= min_score
    ]
    kept.sort(key=lambda r: (-int(r.get("score") or 0), float(r["start_s"])))
    top = kept[:n]
    if len(top) < min_n:
        used = {id(r) for r in top}
        rest = [r for r in reviews if id(r) not in used]
        rest.sort(key=lambda r: (-int(r.get("score") or 0), float(r["start_s"])))
        top.extend(rest[: min_n - len(top)])
    top.sort(key=lambda r: float(r["start_s"]))
    return top[:n]


def fallback_reel(scan_rows: list[dict], n: int = REEL_KEEP) -> list[dict]:
    """Use 2.5 ranks when 3.8 returns no keep."""
    top = cap_by_score(scan_rows, limit=n)
    return sorted(top, key=lambda r: float(r["start_s"]))


def fill_partial_reel(
    reviewed: list[dict],
    candidates: list[dict],
    *,
    partial: bool,
    n: int = REEL_KEEP,
) -> list[dict]:
    """Keep 3.8 keeps. If review stopped early, fill from unreviewed 2.5 clips."""
    kept = pick_reel(reviewed, n=n)
    if not partial or len(kept) >= n:
        return kept
    seen = {r.get("file") for r in reviewed if r.get("file")}
    unused = [c for c in candidates if c.get("file") not in seen]
    extra = fallback_reel(unused, n=n - len(kept))
    combined = kept + extra
    combined.sort(key=lambda r: float(r["start_s"]))
    return combined


def gens_dir(root: Path) -> Path:
    return root / GENS_NAME


def ensure_gens_dir(root: Path) -> Path:
    path = gens_dir(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def next_gen_n(root: Path) -> int:
    nums: list[int] = []
    folder = gens_dir(root)
    if not folder.is_dir():
        return 1
    for path in folder.glob("gen_*"):
        if not path.is_dir():
            continue
        try:
            nums.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return (max(nums) if nums else 0) + 1


def inbox_mp4s(inbox: Path) -> list[Path]:
    if not inbox.is_dir():
        return []
    return sorted(
        (p for p in inbox.glob("*.mp4") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )


def pick_source_mp4(inbox: Path, done: Path, source: Path) -> tuple[Path, bool]:
    """Inbox first, then done, then source.mp4. True means move to done after the reel."""
    dropped = inbox_mp4s(inbox)
    if dropped:
        return dropped[0], True
    archived = inbox_mp4s(done)
    if archived:
        return archived[0], False
    if source.is_file():
        return source, False
    raise FileNotFoundError(f"drop an MP4 in {inbox} or pass --source")


def wait_until_complete(
    path: Path,
    *,
    stable_s: float = STABLE_S,
    poll_s: float = POLL_S,
    timeout_s: float = WAIT_TIMEOUT_S,
) -> Path:
    """Return when size stays the same and the file opens for read."""
    deadline = time.time() + timeout_s
    last = -1
    stable_since = time.time()
    while time.time() < deadline:
        if not path.exists():
            time.sleep(poll_s)
            last = -1
            stable_since = time.time()
            continue
        size = path.stat().st_size
        if size == last and size > 0:
            if time.time() - stable_since >= stable_s:
                with path.open("rb"):
                    pass
                return path
        else:
            last = size
            stable_since = time.time()
        time.sleep(poll_s)
    raise TimeoutError(f"file not complete: {path}")
