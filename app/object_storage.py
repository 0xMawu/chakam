"""
Phase 7 §10: S3-compatible object storage for the two on-disk thumbnail
caches that previously lived only on the app server's local filesystem
-- `data/thumb_cache/` (whole-photo thumbnails) and
`data/face_thumb_cache/` (per-face crop thumbnails), built and read by
`main._build_or_get_cached_photo_thumbnail` and `main.admin_face_thumbnail`.

Why this matters at scale: those two caches were correct for a single,
long-lived process with a persistent disk, but break down as soon as a
deployment has more than one thing touching them:
  - most container/PaaS platforms (a plain Docker deploy without a
    mounted volume, many "git push to deploy" hosts, etc.) give each
    instance/redeploy a fresh, ephemeral filesystem -- a redeploy
    silently throws away every cached thumbnail, and the very next photo
    view re-downloads the original from Drive and re-resizes it, even
    for a photo that was already cached seconds before the redeploy.
  - running more than one app instance behind a load balancer (to spread
    out /find/capture's face-detection load, say) means each instance
    independently builds and caches its own copy of the same thumbnail
    -- N redundant Drive downloads + resizes instead of one, and a
    member's second /find visit hitting a different instance gets a
    fresh rebuild instead of the other instance's already-cached copy.

Additive, matching this project's own pattern for §6 (Postgres) and §9
(Redis): if `S3_BUCKET` isn't set, `get`/`put` fall back to exactly the
old local-disk-cache behavior, byte for byte -- an existing small/
single-instance deployment (this README's stated target) needs to
change nothing. Setting `S3_BUCKET` (plus `S3_ENDPOINT_URL` for anything
that isn't AWS S3 itself -- MinIO, Cloudflare R2, Backblaze B2,
DigitalOcean Spaces, ...) switches both caches over to that bucket
instead, shared across every app instance and durable across redeploys.

Scope note: `PHASE7_SCALE_ARCHITECTURE.md` (the planning doc README.md's
Phase 7 section cites for the full §10 spec) isn't present in this
checkout, so this is inferred directly from what actually exists in the
codebase -- the two disk caches above -- rather than from that doc's own
wording. If the real spec calls for something broader (e.g. also moving
the *original* Drive-downloaded photo bytes to S3, to cut repeat Drive
API calls beyond what thumbnail caching already avoids), that's a
reasonable follow-up but isn't implemented here: `app/ingestion.py` and
`app/drive_client.py` are untouched, and originals still come from Drive
on demand every time, same as before this module existed.

boto3 is only imported lazily, inside `_get_client()`, not at module
import time -- so a deployment that never sets `S3_BUCKET` doesn't need
a working boto3 install (credentials configured, network reachable,
etc.) just to import this module and fall through to the local-disk
path, even though `boto3` is unconditionally listed in requirements.txt
(same convention this project already uses for `psycopg2-binary`, which
`database_pg.py` needs but the live app doesn't import by default).
"""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

S3_BUCKET = os.environ.get("S3_BUCKET")
# Unset -> real AWS S3. Set for anything S3-compatible that isn't AWS
# itself (MinIO, Cloudflare R2, Backblaze B2, DigitalOcean Spaces, ...).
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
# Optional namespace prefix, useful for sharing one bucket across
# environments/apps (e.g. "church-photo-finder/thumbs/").
S3_KEY_PREFIX = os.environ.get("S3_KEY_PREFIX", "")

# Whether S3-backed caching is active at all. Checked instead of a
# one-time import-time decision so tests can monkeypatch S3_BUCKET
# without reloading the module.
def enabled() -> bool:
    return bool(S3_BUCKET)


_client = None


def _get_client():
    global _client
    if _client is None:
        import boto3  # lazy -- see module docstring

        _client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            region_name=S3_REGION,
        )
    return _client


def _full_key(key: str) -> str:
    return f"{S3_KEY_PREFIX}{key}" if S3_KEY_PREFIX else key


def get(key: str, local_dir: Path) -> Optional[bytes]:
    """Returns the cached object's bytes, or None if it isn't cached yet
    (or S3 is unreachable). None is deliberately indistinguishable from
    "not cached" to callers -- both mean "rebuild from Drive" -- so a
    transient S3 outage degrades to a slower response instead of a 500,
    the same tolerance the old direct-disk code had for a cold cache."""
    if enabled():
        try:
            resp = _get_client().get_object(Bucket=S3_BUCKET, Key=_full_key(key))
            return resp["Body"].read()
        except _get_client().exceptions.NoSuchKey:
            return None
        except Exception:
            logger.exception("S3 get failed for key=%s -- falling back to rebuild", key)
            return None
    else:
        path = local_dir / key
        if path.is_file():
            return path.read_bytes()
        return None


def put(key: str, data: bytes, local_dir: Path, content_type: str = "image/jpeg") -> None:
    """Best-effort: a failed cache write must never fail the request that
    triggered it -- the caller already has the bytes it needs to respond
    with; this call is purely a future-request optimization. Mirrors the
    old code's tolerance for a local disk write failing (it caught
    OSError around cache_path.write_bytes() and just logged a warning)."""
    if enabled():
        try:
            _get_client().put_object(
                Bucket=S3_BUCKET,
                Key=_full_key(key),
                Body=data,
                ContentType=content_type,
                CacheControl="public, max-age=604800, immutable",
            )
        except Exception:
            logger.exception("S3 put failed for key=%s -- thumbnail will be rebuilt next request", key)
    else:
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / key).write_bytes(data)
        except OSError as e:
            logger.warning("Could not cache %s to local disk: %s", key, e)
