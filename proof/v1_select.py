"""Pick 2.5 candidates, 3.8 reel clips, and a complete drop-folder file."""

from __future__ import annotations

import time
from pathlib import Path

SCAN_CAP = 75
REVIEW_BATCH = 10
REEL_KEEP = 8
STABLE_S = 8.0
POLL_S = 1.0
WAIT_TIMEOUT_S = 7200.0


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


def pick_reel(reviews: list[dict], n: int = REEL_KEEP) -> list[dict]:
    """Take up to n clips 3.8 marked keep. Concat order is chrono. Never pad."""
    kept = [r for r in reviews if r.get("keep")]
    kept.sort(key=lambda r: (-int(r.get("score") or 0), float(r["start_s"])))
    top = kept[: max(0, n)]
    top.sort(key=lambda r: float(r["start_s"]))
    return top


def fallback_reel(scan_rows: list[dict], n: int = REEL_KEEP) -> list[dict]:
    """Use 2.5 ranks when 3.8 returns no keep."""
    top = cap_by_score(scan_rows, limit=n)
    return sorted(top, key=lambda r: float(r["start_s"]))


def next_gen_n(root: Path) -> int:
    nums: list[int] = []
    for path in root.glob("gen_*"):
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
