"""Supply the original reel to Vertex through private Cloud Storage."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from google.api_core.exceptions import Conflict, PreconditionFailed
from google.cloud import storage

from gemini_defaults import VERTEX_PROJECT


def upload_reel(reel: Path) -> dict:
    """Reuse the same object when the reel's bytes are unchanged."""
    digest = hashlib.sha256()
    with reel.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    name = os.environ.get("ARGON_VIDEO_BUCKET", f"{VERTEX_PROJECT}-argon-reels").strip()
    if not name or "/" in name:
        raise ValueError("ARGON_VIDEO_BUCKET must be a bucket name without gs:// or a path.")
    with storage.Client(project=VERTEX_PROJECT) as client:
        bucket = client.lookup_bucket(name, timeout=60)
        if bucket is None:
            bucket = client.bucket(name)
            bucket.iam_configuration.uniform_bucket_level_access_enabled = True
            bucket.iam_configuration.public_access_prevention = "enforced"
            try:
                bucket = client.create_bucket(bucket, location="us-central1", timeout=60)
            except Conflict:
                bucket = client.get_bucket(name, timeout=60)
        blob = bucket.blob(f"reels/{sha256}.mp4")
        if not blob.exists(timeout=60):
            print(f"ARGON upload original bytes={reel.stat().st_size}", flush=True)
            try:
                blob.upload_from_filename(
                    str(reel), content_type="video/mp4", if_generation_match=0, timeout=300,
                )
            except PreconditionFailed:
                # Another job uploaded the same content during this request.
                blob.reload(timeout=60)
        else:
            print("ARGON reuse original upload", flush=True)
    return {"video_uri": f"gs://{name}/{blob.name}", "sha256": sha256, "bytes": reel.stat().st_size}
