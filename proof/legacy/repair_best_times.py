"""Rewrite source_* clocks on vertex_best JSON. Models mix relative and match time."""

from __future__ import annotations

import json
from pathlib import Path

from scan_full_match import VIDEO_END_S, clock, parse_rel, to_source

OUT = Path(__file__).resolve().parent / "out"


def main() -> int:
    n_ok = 0
    n_drop = 0
    for path in sorted(OUT.glob("vertex_best_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        w = payload["requested_window"]
        ws = int(str(w["start"]).rstrip("s"))
        we = int(str(w["end"]).rstrip("s"))
        fixed = []
        for h in payload.get("highlights", []):
            a = parse_rel(h["start"])
            b = parse_rel(h["end"])
            sa = to_source(a, ws, we)
            sb = to_source(b, ws, we)
            if sa is None or sb is None:
                n_drop += 1
                print(f"drop {path.name} {h['start']}")
                continue
            if sb < sa:
                sa, sb = sb, sa
            if sa >= VIDEO_END_S:
                n_drop += 1
                continue
            sb = min(sb, float(VIDEO_END_S))
            h["source_start"] = clock(sa)
            h["source_end"] = clock(sb)
            fixed.append(h)
            n_ok += 1
        payload["highlights"] = fixed
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"ok={n_ok} drop={n_drop}")
    rows = []
    for path in sorted(OUT.glob("vertex_best_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for h in payload.get("highlights", []):
            if h.get("keep"):
                rows.append(h)
    rows.sort(key=lambda r: (-int(r["score"]), r["source_start"]))
    n10 = sum(1 for r in rows if int(r["score"]) >= 10)
    n9 = sum(1 for r in rows if int(r["score"]) >= 9)
    print(f"keeps={len(rows)} ge10={n10} ge9={n9}")
    for h in rows[:24]:
        print(
            f"{h['score']} {h['source_start']}-{h['source_end']} {h['event']} {h['reason'][:55]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
