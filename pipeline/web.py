"""Upload one MP4. Return the 16:9 reel, and a 9:16 TikTok cut when polish is on."""

from __future__ import annotations

import builtins
import threading
from pathlib import Path

from gemini_defaults import ARGON_MODEL, REVIEW_MODEL, SCAN_MODEL, clean_model
from pick import next_gen_n
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

ROOT = Path(__file__).resolve().parent
PAGE = Path(__file__).with_name("web.html")
MAX_BYTES = 10 * 1024 * 1024 * 1024
RUN_IN_THREAD = True

ERR_MP4 = "Use an MP4 file."
ERR_SIZE = "File is larger than 10 GB."
ERR_BUSY = "A job is in progress."
ERR_JOB = "The job failed."
ERR_REEL = "The reel is not ready."

_lock = threading.Lock()
_state = {"status": "idle", "detail": "", "reel": None, "tiktok": None}


def upload_dir() -> Path:
    return ROOT / "web_uploads"


def upload_path() -> Path:
    return upload_dir() / "source.mp4"


def looks_like_mp4(head: bytes) -> bool:
    if len(head) < 8:
        return False
    return b"ftyp" in head[:128]


def filename_is_mp4(name: str) -> bool:
    return Path(name).suffix.lower() == ".mp4"


def get_state() -> dict:
    with _lock:
        return dict(_state)


def set_state(**kwargs) -> None:
    with _lock:
        _state.update(kwargs)


def reset_state() -> None:
    set_state(status="idle", detail="", reel=None, tiktok=None)


def claim_upload() -> bool:
    with _lock:
        if _state["status"] in ("uploading", "working"):
            return False
        _state.update(status="uploading", detail="Upload", reel=None, tiktok=None)
        return True


def _flag(headers, name: str, default: bool) -> bool:
    raw = headers.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() not in ("0", "false", "off", "no")


def job_options(headers) -> dict:
    sport = (headers.get("x-sport") or "volleyball").strip().lower()
    if sport not in ("auto", "volleyball", "basketball"):
        raise ValueError("Sport must be volleyball, basketball, or auto.")
    thinking = (headers.get("x-thinking") or "MEDIUM").strip().upper()
    if thinking not in ("LOW", "MEDIUM", "HIGH"):
        raise ValueError("Thinking must be LOW, MEDIUM, or HIGH.")
    layout = (headers.get("x-layout") or "blurred-bg").strip().lower()
    if layout not in ("smart-crop", "blurred-bg"):
        raise ValueError("Layout must be smart-crop or blurred-bg.")
    try:
        target_s = float(headers.get("x-target-s") or 60)
    except ValueError:
        raise ValueError("Target length must be 15 to 180 seconds.") from None
    if target_s < 15 or target_s > 180:
        raise ValueError("Target length must be 15 to 180 seconds.")
    return {
        "scan_model": clean_model(headers.get("x-scan-model"), SCAN_MODEL),
        "review_model": clean_model(headers.get("x-review-model"), REVIEW_MODEL),
        "argon": _flag(headers, "x-argon", True),
        "argon_model": clean_model(headers.get("x-argon-model"), ARGON_MODEL),
        "sport": sport,
        "thinking": thinking,
        "layout": layout,
        "captions": _flag(headers, "x-captions", True),
        "hook": _flag(headers, "x-hook", True),
        "target_s": target_s,
    }


def run_pipeline(source: Path, gen_n: int, options: dict) -> int:
    from run_job import run_job

    return run_job(source=source, gen_n=gen_n, **options)


def _finish(code: int, reel: Path, err: str) -> None:
    detail = (err or get_state().get("detail") or "").strip()
    if code == 0:
        detail = ""
    elif detail in ("", "Working", "Upload"):
        detail = ERR_JOB
    tiktok = reel.parent / "tiktok.mp4"
    if reel.is_file():
        set_state(
            status="done",
            detail=detail[:300],
            reel=str(reel),
            tiktok=str(tiktok) if tiktok.is_file() else None,
        )
        return
    set_state(status="error", detail=(detail or ERR_JOB)[:300], reel=None, tiktok=None)


def _worker(source: Path, gen_n: int, reel: Path, options: dict) -> None:
    set_state(status="working", detail="Working", reel=None, tiktok=None)
    real = builtins.print

    def hooked(*args, **kwargs):
        real(*args, **kwargs)
        text = " ".join(str(item) for item in args).strip()
        if text:
            set_state(detail=text[:300])

    builtins.print = hooked
    code = 0
    err = ""
    try:
        code = run_pipeline(source, gen_n, options)
    except Exception as exc:
        code = 1
        err = str(exc).strip() or ERR_JOB
        real(err, flush=True)
    finally:
        builtins.print = real
    _finish(code, reel, err)


def start_pipeline(source: Path, options: dict) -> None:
    gen_n = next_gen_n(ROOT)
    reel = ROOT / "generations" / f"gen_{gen_n}" / "reel.mp4"
    if RUN_IN_THREAD:
        threading.Thread(
            target=_worker,
            args=(source, gen_n, reel, options),
            daemon=True,
        ).start()
        return
    _worker(source, gen_n, reel, options)


async def save_upload(request: Request, dest: Path) -> None:
    length = request.headers.get("content-length")
    if length is not None:
        try:
            declared = int(length)
        except ValueError:
            declared = -1
        if declared > MAX_BYTES:
            raise ValueError("size")
        if declared <= 0:
            raise ValueError("mp4")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    if part.exists():
        part.unlink()
    size = 0
    head = bytearray()
    checked = False
    try:
        with part.open("wb") as fh:
            async for chunk in request.stream():
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("size")
                if not checked:
                    need = 128 - len(head)
                    if need > 0:
                        head.extend(chunk[:need])
                    if len(head) >= 8:
                        if not looks_like_mp4(bytes(head)):
                            raise ValueError("mp4")
                        checked = True
                fh.write(chunk)
        if size <= 0 or not checked or not looks_like_mp4(bytes(head)):
            raise ValueError("mp4")
        if dest.exists():
            dest.unlink()
        part.replace(dest)
    except BaseException:
        if part.exists():
            part.unlink()
        raise


def json_error(detail: str, status: int) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


async def page(_request: Request) -> HTMLResponse:
    return HTMLResponse(PAGE.read_text(encoding="utf-8"))


def _media(state: dict, key: str, missing: str) -> Response:
    path = state.get(key)
    if state["status"] != "done" or not path:
        return json_error(missing, 404)
    file_path = Path(path)
    if not file_path.is_file():
        return json_error(missing, 404)
    name = "tiktok.mp4" if key == "tiktok" else "reel.mp4"
    return FileResponse(file_path, media_type="video/mp4", filename=name)


async def status(_request: Request) -> JSONResponse:
    state = get_state()
    return JSONResponse(
        {
            "status": state["status"],
            "detail": state["detail"],
            "reel": bool(state.get("reel")),
            "tiktok": bool(state.get("tiktok")),
        }
    )


async def reel(request: Request) -> Response:
    return _media(get_state(), "reel", ERR_REEL)


async def tiktok(request: Request) -> Response:
    return _media(get_state(), "tiktok", ERR_REEL)


async def upload(request: Request) -> Response:
    name = request.headers.get("x-filename") or "upload.mp4"
    if not filename_is_mp4(name):
        return json_error(ERR_MP4, 400)
    try:
        options = job_options(request.headers)
    except ValueError as exc:
        return json_error(str(exc), 400)
    if not claim_upload():
        return json_error(ERR_BUSY, 409)
    dest = upload_path()
    try:
        await save_upload(request, dest)
    except ClientDisconnect:
        reset_state()
        return json_error(ERR_JOB, 400)
    except ValueError as exc:
        reset_state()
        if str(exc) == "size":
            return json_error(ERR_SIZE, 413)
        return json_error(ERR_MP4, 400)
    except Exception:
        reset_state()
        return json_error(ERR_JOB, 500)
    start_pipeline(dest, options)
    return JSONResponse({"ok": True})


app = Starlette(
    routes=[
        Route("/", page),
        Route("/upload", upload, methods=["POST"]),
        Route("/status", status),
        Route("/reel", reel),
        Route("/tiktok", tiktok),
    ]
)


def main() -> None:
    import uvicorn

    print("Open http://127.0.0.1:8000", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8000, timeout_keep_alive=75)


if __name__ == "__main__":
    main()
