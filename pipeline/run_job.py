"""Drop-folder job: 2.5 scan, 16:9 cuts, 3.8 review, concat reel."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from clip_window import (
    REVIEW_DUR,
    cool_review_window,
    default_trim,
    final_source_span,
    one_cool_caps,
    review_window,
)
from gemini_defaults import ARGON_MODEL, ARGON_THINKING, REVIEW_MODEL, SCAN_MODEL
from review_clips import LONG_PROMPT, review_clips
from scan_io import load_all_keeps, merge_windows, slug
from scan_local import scan_local
from media import concat_wide, cut_wide, probe_secs
from pick import (
    REEL_KEEP,
    cap_by_score,
    ensure_gens_dir,
    fallback_reel,
    fill_partial_reel,
    gens_dir,
    next_gen_n,
    pick_reel,
    pick_source_mp4,
    wait_until_complete,
)

ROOT = Path(__file__).resolve().parent
INBOX = ROOT / "inbox"
DONE = INBOX / "done"
SOURCE = ROOT / "source.mp4"
STILL_GAP = 15


def dump_stills(reel: Path, dest_dir: Path, secs: float) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    from media import ffmpeg_bin
    import subprocess

    ff = ffmpeg_bin()
    t = 3
    while t < secs:
        out = dest_dir / f"t{int(t):03d}.jpg"
        subprocess.run(
            [ff, "-y", "-ss", str(t), "-i", str(reel), "-frames:v", "1", str(out)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        t += STILL_GAP


def resolve_source(arg: Path | None) -> tuple[Path, bool]:
    """Return (mp4, from_inbox)."""
    if arg is not None:
        path = arg if arg.is_absolute() else (ROOT / arg).resolve()
        wait_until_complete(path)
        return path, False
    path, from_inbox = pick_source_mp4(INBOX, DONE, SOURCE)
    wait_until_complete(path)
    return path, from_inbox


def slice_candidates(
    source: Path,
    rows: list[dict],
    clips_dir: Path,
    video_end: float,
) -> list[dict]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old in clips_dir.glob("*.mp4"):
        old.unlink()
    windows = merge_windows(rows)
    picked = cap_by_score(windows)
    out: list[dict] = []
    for i, w in enumerate(picked, start=1):
        start, dur = review_window(w["start_s"], w["end_s"], video_end=video_end)
        name = f"c{i:02d}_{slug(w['event'])}.mp4"
        dest = clips_dir / name
        print(f"cut {name} {start:.1f}s +{dur:.1f}s score={w['score']}", flush=True)
        cut_wide(source, start, dur, dest)
        rec = {
            "file": name,
            "path": str(dest),
            "window_start_s": start,
            "window_dur_s": dur,
            "start_s": start,
            "end_s": start + dur,
            "dur_s": dur,
            "event_start_s": w["start_s"],
            "event_end_s": w["end_s"],
            "event": w["event"],
            "score": int(w["score"]),
        }
        out.append(rec)
    return out


def fill_missing_trims(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for rec in rows:
        rec = dict(rec)
        win_d = float(rec.get("window_dur_s", rec.get("dur_s", REVIEW_DUR)))
        if rec.get("trim_start_s") is None or rec.get("trim_end_s") is None:
            a, b = default_trim(win_d)
            rec["trim_start_s"] = a
            rec["trim_end_s"] = b
        out.append(rec)
    return out


def slice_from_keeps(
    source: Path,
    keeps: list[dict],
    clips_dir: Path,
    video_end: float,
) -> list[dict]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old in clips_dir.glob("*.mp4"):
        old.unlink()
    out: list[dict] = []
    for i, w in enumerate(keeps, start=1):
        start, dur = review_window(w["start_s"], w["end_s"], video_end=video_end)
        name = f"c{i:02d}_{slug(w['event'])}.mp4"
        dest = clips_dir / name
        print(f"cut {name} {start:.1f}s +{dur:.1f}s score={w.get('score', 0)}", flush=True)
        cut_wide(source, start, dur, dest)
        rec = {
            "file": name,
            "path": str(dest),
            "window_start_s": start,
            "window_dur_s": dur,
            "start_s": start,
            "end_s": start + dur,
            "dur_s": dur,
            "event_start_s": w["start_s"],
            "event_end_s": w["end_s"],
            "event": w["event"],
            "score": int(w.get("score") or 0),
        }
        out.append(rec)
    return out


def merge_trims(candidates: list[dict], reviewed: list[dict]) -> list[dict]:
    by_file = {r.get("file"): r for r in reviewed if r.get("file")}
    out: list[dict] = []
    for rec in candidates:
        rec = dict(rec)
        hit = by_file.get(rec["file"])
        if hit is None:
            a, b = default_trim(float(rec.get("window_dur_s", REVIEW_DUR)))
            rec["trim_start_s"] = a
            rec["trim_end_s"] = b
            out.append(rec)
            continue
        rec["trim_start_s"] = hit.get("trim_start_s")
        rec["trim_end_s"] = hit.get("trim_end_s")
        rec["keep"] = hit.get("keep", True)
        rec["review_score"] = hit.get("score")
        rec["reason"] = hit.get("reason")
        out.append(rec)
    return fill_missing_trims(out)


def recut_finals(
    source: Path,
    picked: list[dict],
    clips_dir: Path,
    video_end: float,
) -> list[dict]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    caps = one_cool_caps(picked)
    spans: list[tuple[float, float, dict]] = []
    for rec, cap in zip(picked, caps):
        win_s = float(rec.get("window_start_s", rec["start_s"]))
        win_d = float(rec.get("window_dur_s", rec.get("dur_s", REVIEW_DUR)))
        trim_a = rec.get("trim_start_s")
        trim_b = rec.get("trim_end_s")
        start, dur = final_source_span(
            win_s,
            win_d,
            None if trim_a is None else float(trim_a),
            None if trim_b is None else float(trim_b),
            video_end=video_end,
            max_s=cap,
        )
        spans.append((start, dur, rec))
    spans.sort(key=lambda item: item[0])
    out: list[dict] = []
    for i, (start, dur, rec) in enumerate(spans, start=1):
        name = f"reel_{i:02d}_{slug(rec['event'])}.mp4"
        dest = clips_dir / name
        print(f"final {name} {start:.1f}s +{dur:.1f}s", flush=True)
        cut_wide(source, start, dur, dest)
        rec = dict(rec)
        rec["file"] = name
        rec["path"] = str(dest)
        rec["start_s"] = start
        rec["end_s"] = start + dur
        rec["dur_s"] = dur
        rec["path"] = str(dest)
        out.append(rec)
    return out


def pick_coolest_index(clips: list[dict]) -> int:
    return max(
        range(len(clips)),
        key=lambda i: (
            int(clips[i].get("score") or 0),
            float(clips[i].get("dur_s") or 0),
        ),
    )


def run_job(
    *,
    source: Path | None = None,
    gen_n: int | None = None,
    scan_dir: Path | None = None,
    skip_scan: bool = False,
    skip_review: bool = False,
    from_gen: int | None = None,
    extend_cool: bool = False,
    scan_model: str | None = None,
    review_model: str | None = None,
    argon: bool = True,
    argon_model: str | None = None,
    sport: str = "volleyball",
    thinking: str = ARGON_THINKING,
    layout: str = "blurred-bg",
    captions: bool = True,
    hook: bool = True,
    target_s: float = 60.0,
) -> int:
    scan_name = scan_model or SCAN_MODEL
    review_name = review_model or REVIEW_MODEL
    src, from_inbox = resolve_source(source)
    gens = ensure_gens_dir(ROOT)
    n = gen_n if gen_n is not None else next_gen_n(ROOT)
    dest = gens / f"gen_{n}"
    dest.mkdir(parents=True, exist_ok=True)
    clips_dir = dest / "clips"
    stills_dir = dest / "stills"
    job_scan = dest / "scan"
    job_scan.mkdir(parents=True, exist_ok=True)
    video_end = probe_secs(src)
    scan_result = {
        "keeps": 0,
        "partial": False,
        "scan_dir": str(job_scan),
        "video_end": int(round(video_end)),
    }
    used_scan = job_scan
    candidates: list[dict]
    keep_all = False
    extend_done = False
    review_payload: dict | None = None
    if from_gen is not None:
        prior_path = gens_dir(ROOT) / f"gen_{from_gen}" / "keeps.json"
        if not prior_path.exists():
            print(f"missing {prior_path}", file=sys.stderr)
            return 1
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        used_scan = Path(prior.get("scan_dir") or ROOT / "scan")
        skip_scan = True
        if extend_cool:
            clips_dir.mkdir(parents=True, exist_ok=True)
            candidates = [dict(c) for c in prior["clips"]]
            idx = pick_coolest_index(candidates)
            cool = dict(candidates[idx])
            ev_a = float(cool.get("event_start_s", cool["start_s"]))
            ev_b = float(cool.get("event_end_s", cool["end_s"]))
            start, dur = cool_review_window(ev_a, ev_b, video_end=video_end)
            name = f"long_{idx + 1:02d}.mp4"
            dest_clip = clips_dir / name
            print(
                f"extend clip {idx + 1} {cool.get('event')} {start:.1f}s +{dur:.1f}s",
                flush=True,
            )
            cut_wide(src, start, dur, dest_clip)
            cool["file"] = name
            cool["path"] = str(dest_clip)
            cool["window_start_s"] = start
            cool["window_dur_s"] = dur
            review_payload = review_clips(
                [cool],
                dest / "review_long.json",
                proxy_dir=dest / "review",
                prompt=LONG_PROMPT,
                model=review_name,
            )
            candidates[idx] = merge_trims([cool], review_payload["clips"])[0]
            keep_all = True
            extend_done = True
        else:
            print(f"trim from gen_{from_gen}; n={len(prior['clips'])}", flush=True)
            candidates = slice_from_keeps(src, prior["clips"], clips_dir, video_end)
            keep_all = True
    elif skip_scan:
        used_scan = scan_dir if scan_dir is not None else job_scan
        if not any(used_scan.glob("vertex_best_*.json")):
            used_scan = ROOT / "scan"
        print(f"skip scan; use {used_scan}", flush=True)
        rows = load_all_keeps(used_scan)
        if not rows:
            print("no scan keeps", file=sys.stderr)
            return 1
        candidates = slice_candidates(src, rows, clips_dir, video_end)
    else:
        scan_result = scan_local(
            src,
            job_scan,
            video_end=int(round(video_end)),
            proxy_dir=dest / "windows",
            model=scan_name,
        )
        used_scan = job_scan
        rows = load_all_keeps(used_scan)
        if not rows:
            print("no scan keeps", file=sys.stderr)
            return 1
        candidates = slice_candidates(src, rows, clips_dir, video_end)
    (dest / "candidates.json").write_text(
        json.dumps({"clips": candidates, "scan_dir": str(used_scan)}, indent=2),
        encoding="utf-8",
    )
    picker = "review"
    picked: list[dict]
    if extend_done:
        picker = "extend-cool"
        picked = fill_missing_trims(candidates)
        print(f"extend-cool n={len(picked)}", flush=True)
    elif skip_review:
        picker = "scan"
        picked = candidates if keep_all else fallback_reel(candidates, n=REEL_KEEP)
        picked = fill_missing_trims(picked)
        print(f"skip review; n={len(picked)}", flush=True)
    else:
        review_payload = review_clips(
            candidates,
            dest / "review.json",
            proxy_dir=dest / "review",
            model=review_name,
        )
        if keep_all:
            picker = "trim"
            picked = merge_trims(candidates, review_payload["clips"])
        else:
            picked = pick_reel(review_payload["clips"])
            if review_payload.get("partial") and len(picked) < REEL_KEEP:
                filled = fill_partial_reel(
                    review_payload["clips"],
                    candidates,
                    partial=True,
                    n=REEL_KEEP,
                )
                if len(filled) > len(picked):
                    picker = "review-partial-fill"
                    picked = filled
                    print(f"3.8 partial; fill n={len(picked)}", flush=True)
            if not picked:
                picker = "scan-fallback"
                picked = fallback_reel(candidates, n=REEL_KEEP)
                picked = merge_trims(picked, review_payload["clips"])
                print(f"3.8 keep=0; 2.5 fallback n={len(picked)}", flush=True)
            picked = fill_missing_trims(picked)
    if not picked:
        print("reel has 0 clips", file=sys.stderr)
        return 1
    picked = recut_finals(src, picked, clips_dir, video_end)
    reel = dest / "reel.mp4"
    concat_wide([Path(r["path"]) for r in picked], reel)
    dump_stills(reel, stills_dir, probe_secs(reel))
    secs = probe_secs(reel)
    meta = {
        "gen": n,
        "aspect": "16:9",
        "scan_model": scan_name,
        "review_model": review_name,
        "argon": bool(argon),
        "argon_model": (argon_model or ARGON_MODEL) if argon else None,
        "source": str(src),
        "scan_dir": str(used_scan),
        "scan_partial": bool(scan_result.get("partial")),
        "review_partial": bool(review_payload["partial"]) if review_payload else False,
        "picker": picker,
        "from_gen": from_gen,
        "n_candidates": len(candidates),
        "n_clips": len(picked),
        "seconds": round(secs, 2),
        "reel": str(reel),
        "clips": [{k: v for k, v in r.items() if k != "path"} for r in picked],
    }
    (dest / "keeps.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{reel} {secs:.1f}s clips={len(picked)}", flush=True)
    if argon:
        from argon import polish_reel

        tiktok = dest / "tiktok.mp4"
        polish_reel(
            reel,
            tiktok,
            model=argon_model or ARGON_MODEL,
            sport=sport,
            thinking=thinking,
            layout=layout,
            captions=captions,
            hook=hook,
            target_s=target_s,
            proxy_dir=dest / "argon",
        )
        meta["tiktok"] = str(tiktok)
        meta["seconds_tiktok"] = round(probe_secs(tiktok), 2)
        (dest / "keeps.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if from_inbox:
        DONE.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(DONE / src.name))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a 16:9 highlight reel from one MP4.")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--gen", type=int, default=None)
    parser.add_argument("--scan-dir", type=Path, default=None)
    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--from-gen", type=int, default=None)
    parser.add_argument("--extend-cool", action="store_true")
    parser.add_argument("--scan-model", default=None)
    parser.add_argument("--review-model", default=None)
    parser.add_argument("--argon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--argon-model", default=None)
    parser.add_argument("--sport", default="volleyball")
    parser.add_argument("--thinking", default=ARGON_THINKING)
    parser.add_argument("--layout", default="blurred-bg")
    parser.add_argument("--no-captions", action="store_true")
    parser.add_argument("--no-hook", action="store_true")
    parser.add_argument("--target-s", type=float, default=60.0)
    args = parser.parse_args()
    scan_dir = args.scan_dir
    if scan_dir is not None and not scan_dir.is_absolute():
        scan_dir = ROOT / scan_dir
    try:
        return run_job(
            source=args.source,
            gen_n=args.gen,
            scan_dir=scan_dir,
            skip_scan=args.skip_scan,
            skip_review=args.skip_review,
            from_gen=args.from_gen,
            extend_cool=args.extend_cool,
            scan_model=args.scan_model,
            review_model=args.review_model,
            argon=args.argon,
            argon_model=args.argon_model,
            sport=args.sport,
            thinking=args.thinking,
            layout=args.layout,
            captions=not args.no_captions,
            hook=not args.no_hook,
            target_s=args.target_s,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
