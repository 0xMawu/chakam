"""
Church Photo Finder — Phases 1-6, plus real-world hardening.

Scope so far (see Section 8 of the spec for the full build order):
  Phase 1 — Admin folder-link intake.
  Phase 2 — Ingestion pipeline (Drive download + face detection/embedding).
  Phase 3 — Clustering (DBSCAN over all face embeddings).
  Phase 4 — Live selfie capture, member-facing. A camera-only page at
    /find that captures a single frame from the browser's live video
    stream (no <input type="file"> anywhere in this flow — hard
    requirement per spec Section 7) and posts it to /find/capture.
  Phase 5 — Matching endpoint (backend). /find/capture detects a face in
    the captured frame, embeds it (reusing face_processing, the same code
    path ingestion uses), and compares it against every current cluster
    centroid (app/matching.py). The photos belonging to the single
    closest cluster are returned if that cluster is within
    MATCH_THRESHOLD — otherwise a "no match" response, same
    precision-over-recall stance as clustering (spec Section 7).
  Phase 6 — Gallery view. Matched photos render as an actual thumbnail
    gallery in find.html rather than plain Drive links, served through
    the signed, short-lived tokens in /find/photo/{token} (see
    app/tokens.py) so the browser never sees a raw, guessable Drive file
    id for a photo it hasn't actually matched.

Hardening added alongside Phase 6, since a publicly-reachable biometric
matching endpoint and an admin panel with a single shared password both
needed more than "it works" before pointing this at real users:
  - Rate limiting (app/security.py) on /find/capture (expensive,
    unauthenticated) and /admin/login (brute-forceable password).
  - Constant-time password comparison on login.
  - /healthz for load balancer / uptime-monitor checks.
  - An audit_log table + /admin/audit recording folder/cluster admin
    actions.
  - /admin/backup for an on-demand, consistency-safe DB snapshot
    download.
  - Indices on faces.cluster_id / faces.photo_id / photos.folder_id so
    the query patterns used throughout database.py don't degrade to full
    table scans as the photo library grows.
"""
import base64
import binascii
import hmac
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app import auth, clustering, database, drive_watcher, ingest_queue, ingestion, matching, object_storage, security, tokens
from app.drive_utils import InvalidDriveFolderLink, extract_folder_id
from app.face_processing import ImageDecodeError, detect_faces

logger = logging.getLogger(__name__)

# A single JPEG/PNG frame from a phone camera is comfortably under this;
# anything bigger is either a much higher resolution than needed or not a
# single-frame capture, so it's rejected rather than processed.
MAX_SELFIE_BYTES = 8 * 1024 * 1024

# Plain Drive "view" link for a file id — enough for a member to open/save
# their own matched photo. Doesn't require any extra Drive API scope beyond
# what ingestion already has, since it's just a URL template.
DRIVE_VIEW_URL = "https://drive.google.com/file/d/{file_id}/view"

logging.basicConfig(level=logging.INFO, force=True)

# Phase 7 §9: whether a folder is currently processing/queued, and
# pause/stop signaling, both now live in Redis via app.ingest_queue
# (RedisControl/is_active) instead of the in-memory set + dict that used
# to live here — see ingest_queue.py's module docstring for why.

# Both limiters are in-memory / per-process — see security.RateLimiter's
# docstring for what that means for a multi-instance deployment.
# /find/capture: unauthenticated and runs real face detection per
# request, so it's both the most abuse-prone and the most expensive route
# in the app. /admin/login: a shared password with no per-user lockout,
# so this is the only thing slowing down a brute-force attempt.
FIND_CAPTURE_LIMITER = security.RateLimiter(max_requests=8, window_seconds=5 * 60)
LOGIN_LIMITER = security.RateLimiter(max_requests=10, window_seconds=15 * 60)

BASE_DIR = os.path.dirname(__file__)
THUMB_CACHE_DIR = Path(BASE_DIR).parent / "data" / "thumb_cache"

app = FastAPI(title="Church Photo Finder")

# SESSION_SECRET must be set to a real random value in production (see README).
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "dev-only-insecure-secret"),
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.on_event("startup")
def on_startup():
    # uvicorn --reload re-configures logging on every worker (re)start,
    # which can wipe out the logging.basicConfig() call above (that's why
    # the terminal only ever showed uvicorn's own log lines, never
    # app.ingestion's "Listed N image(s)..."/"Zero images found..." lines,
    # even though process_folder() was running fine and hitting those
    # exact log calls). Re-apply it here so it survives a reload.
    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger("app").setLevel(logging.INFO)
    database.init_db()
    # Phase 7 §9: only resets folders with no active RQ job, not every
    # processing/paused row unconditionally (see ingest_queue's docstring
    # for why the old database.reset_stuck_processing_folders() was wrong
    # once ingestion could outlive a web-process restart). If Redis isn't
    # reachable yet at startup, skip reconciliation rather than crashing
    # the whole app over it -- folders will just show their current
    # (possibly stale) status until the next successful startup or a
    # manual retry.
    try:
        ingest_queue.reconcile_stuck_folders()
    except Exception:
        logger.exception("Skipping stuck-folder reconciliation -- couldn't reach Redis.")

    drive_watcher.start()   # ← add this


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return RedirectResponse(url="/admin")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    """Liveness/readiness check for a load balancer or uptime monitor.
    Also confirms the DB file is actually reachable, not just that the
    process is up — a stuck/corrupt DB should show as unhealthy, not as
    a false "ok"."""
    try:
        with database.get_connection() as conn:
            conn.execute("SELECT 1")
        return JSONResponse({"status": "ok"})
    except Exception as e:
        logger.exception("Health check failed")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=503)


@app.get("/admin/login", response_class=HTMLResponse)
def login_form(request: Request):
    if auth.is_authed(request):
        return RedirectResponse(url="/admin", status_code=303)
    return templates.TemplateResponse(
        "login.html", {"request": request, "error": None}
    )


@app.post("/admin/login", response_class=HTMLResponse)
def login_submit(request: Request, password: str = Form(...)):
    ip = security.client_ip(request)
    allowed, retry_after = LOGIN_LIMITER.check(ip)
    if not allowed:
        logger.warning("Login rate limit hit for %s", ip)
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": f"Too many attempts. Try again in {int(retry_after // 60) + 1} minute(s).",
            },
            status_code=429,
        )

    # Constant-time comparison so response timing can't be used to guess
    # the password one character at a time — meaningless protection for a
    # password this short in practice, but it's a free, standard
    # precaution and the rate limiter above is the real defense.
    if hmac.compare_digest(password, auth.ADMIN_PASSWORD):
        auth.log_in(request)
        return RedirectResponse(url="/admin", status_code=303)
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Incorrect password. Try again."},
        status_code=401,
    )


@app.post("/admin/logout")
def logout(request: Request):
    auth.log_out(request)
    return RedirectResponse(url="/admin/login", status_code=303)


# ---------------------------------------------------------------------------
# Admin folder intake
# ---------------------------------------------------------------------------

@app.get("/admin", response_class=HTMLResponse)
def admin_home(request: Request):
    redirect = auth.require_admin(request)
    if redirect:
        return redirect
    folders = database.list_folders()
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "folders": folders,
            "cluster_count": database.count_clusters(),
            "error": None,
            "success": None,
        },
    )


@app.post("/admin/folders", response_class=HTMLResponse)
def submit_folder(request: Request, drive_link: str = Form(...), label: str = Form(...)):
    redirect = auth.require_admin(request)
    if redirect:
        return redirect

    error = None
    success = None

    label_clean = label.strip()
    if not label_clean:
        error = "Please give this week a label (e.g. \"Week of Jan 12\")."
    else:
        try:
            folder_id = extract_folder_id(drive_link)
            if database.folder_exists(folder_id):
                error = "That folder has already been added."
            else:
                database.add_folder(folder_id, drive_link.strip(), label_clean)
                database.record_audit_log("folder_added", f"{label_clean} ({folder_id})")
                success = f"Folder added as \u201c{label_clean}\u201d. Folder ID: {folder_id}"
        except InvalidDriveFolderLink as e:
            error = str(e)

    folders = database.list_folders()
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "folders": folders,
            "cluster_count": database.count_clusters(),
            "error": error,
            "success": success,
        },
    )


# ---------------------------------------------------------------------------
# Ingestion (Phase 2)
# ---------------------------------------------------------------------------

@app.post("/admin/folders/{folder_id}/process", response_class=HTMLResponse)
def process_folder(request: Request, folder_id: int):
    redirect = auth.require_admin(request)
    if redirect:
        return redirect

    error = None
    success = None

    folder = database.get_folder(folder_id)
    if folder is None:
        error = "That folder no longer exists."
    elif folder["status"] == "processing":
        error = f"\u201c{folder['label']}\u201d is already processing."
    else:
        try:
            import threading
            from app import ingestion as _ingestion
            from app.ingest_queue import RedisControl
            def _run():
                control = RedisControl(folder_id=folder_id)
                _ingestion.process_folder(folder_id, control=control)
            t = threading.Thread(target=_run, daemon=True)
            t.start()
        except Exception:
            logger.exception("Couldn't start ingestion for folder id=%s", folder_id)
            error = "Couldn't start processing."
        else:
            database.record_audit_log("folder_process_started", folder["label"])
            success = f"Started processing \u201c{folder['label']}\u201d."

    folders = database.list_folders()
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "folders": folders,
            "cluster_count": database.count_clusters(),
            "error": error,
            "success": success,
        },
    )


@app.post("/admin/folders/{folder_id}/pause")
def pause_folder(request: Request, folder_id: int):
    redirect = auth.require_admin(request)
    if redirect:
        return redirect
    if not ingest_queue.is_active(folder_id):
        return JSONResponse({"error": "That folder isn't currently processing."}, status_code=400)
    ingest_queue.request_pause(folder_id)
    database.update_folder_status(folder_id, "paused")
    return JSONResponse({"status": "paused"})


@app.post("/admin/folders/{folder_id}/resume")
def resume_folder(request: Request, folder_id: int):
    redirect = auth.require_admin(request)
    if redirect:
        return redirect
    if not ingest_queue.is_active(folder_id):
        return JSONResponse({"error": "That folder isn't currently processing."}, status_code=400)
    database.update_folder_status(folder_id, "processing")
    ingest_queue.request_resume(folder_id)
    return JSONResponse({"status": "processing"})


@app.post("/admin/folders/{folder_id}/stop")
def stop_folder(request: Request, folder_id: int):
    redirect = auth.require_admin(request)
    if redirect:
        return redirect
    if not ingest_queue.is_active(folder_id):
        return JSONResponse({"error": "That folder isn't currently processing."}, status_code=400)
    # request_stop also wakes a currently-paused run so it notices the
    # stop rather than staying blocked on control.paused.wait() forever.
    ingest_queue.request_stop(folder_id)
    return JSONResponse({"status": "stopping"})


@app.get("/admin/folders/{folder_id}/status")
def folder_status(request: Request, folder_id: int):
    """Lightweight JSON polled by the admin page while a folder is
    processing/paused, so the progress bar and status stamp update live
    without a full page reload."""
    redirect = auth.require_admin(request)
    if redirect:
        return redirect
    folder = database.get_folder(folder_id)
    if folder is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(
        {
            "status": folder["status"],
            "processed_count": folder["processed_count"],
            "total_count": folder["total_count"],
        }
    )


@app.post("/admin/folders/{folder_id}/delete", response_class=HTMLResponse)
def delete_folder(request: Request, folder_id: int):
    redirect = auth.require_admin(request)
    if redirect:
        return redirect

    error = None
    success = None

    folder = database.get_folder(folder_id)
    if folder is None:
        error = "That folder no longer exists."
    elif ingest_queue.is_active(folder_id) or folder["status"] in ("processing", "paused"):
        error = f"Stop processing \u201c{folder['label']}\u201d before deleting it."
    else:
        label = folder["label"]
        database.delete_folder(folder_id)
        database.record_audit_log("folder_deleted", label)
        success = (
            f"Deleted \u201c{label}\u201d and its photos/faces. "
            f"Run \u201cRecluster all faces\u201d to update people clusters."
        )

    folders = database.list_folders()
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "folders": folders,
            "cluster_count": database.count_clusters(),
            "error": error,
            "success": success,
        },
    )


@app.get("/admin/audit", response_class=HTMLResponse)
def admin_audit(request: Request):
    """
    Read-only audit trail of admin actions (folder add/process/delete,
    recluster runs, and manual cluster corrections). Exists so more than
    one admin — or one admin checking their own past changes — can see
    who did what and when, which the single shared-password auth model
    (see app/auth.py) otherwise has no record of on its own beyond
    `actor` always being "admin" for now.
    """
    redirect = auth.require_admin(request)
    if redirect:
        return redirect
    return templates.TemplateResponse(
        "admin_audit.html",
        {"request": request, "entries": database.list_audit_log()},
    )


@app.get("/admin/backup")
def admin_backup(request: Request):
    """
    Download a consistent snapshot of the SQLite database. Uses sqlite3's
    online backup API (Connection.backup) rather than copying the file
    directly, since a plain file copy taken mid-write (e.g. while
    ingestion or clustering is running) can capture a torn, inconsistent
    snapshot — the backup API produces a correct copy even against a live
    database. This is a manual/on-demand safety net, not a replacement
    for a real scheduled backup job in production (e.g. a cron/systemd
    timer hitting this route, or copying the resulting file to off-box
    storage) — wire one of those up before relying on this alone.
    """
    redirect = auth.require_admin(request)
    if redirect:
        return redirect

    import tempfile
    from datetime import datetime, timezone

    snapshot_path = Path(tempfile.gettempdir()) / f"church_photos_backup_{os.getpid()}.db"
    try:
        source = sqlite3.connect(database.DB_PATH)
        dest = sqlite3.connect(snapshot_path)
        with dest:
            source.backup(dest)
        source.close()
        dest.close()
    except sqlite3.Error as e:
        logger.exception("Database backup failed")
        return JSONResponse({"error": f"Backup failed: {e}"}, status_code=500)

    database.record_audit_log("db_backup_downloaded")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return FileResponse(
        snapshot_path,
        media_type="application/octet-stream",
        filename=f"church_photos_{stamp}.db",
    )


@app.get("/admin/clusters", response_class=HTMLResponse)
def admin_clusters(request: Request):
    """
    Cluster viewer + manual correction tool: lets an admin see who got
    grouped with whom before trusting matching in front of real members,
    and fix mistakes directly rather than only being able to re-run
    clustering from scratch.

    Clusters are grouped by folder and numbered 1, 2, 3... *within each
    folder* (see database.list_clusters_grouped_by_folder) instead of by
    their raw database id, which otherwise climbs forever across every
    "Recluster all faces" run and every new folder processed, and
    quickly stops meaning anything to an admin. A cluster that spans
    several folders (the same person across multiple weeks) gets its own
    number under each folder it appears in.

    Every card — a real multi-face cluster or a single still-unclustered
    "missed match" face — shows one clickable representative thumbnail;
    clicking it expands a lightbox with every photo for that person,
    fetched on demand, plus a control to move that face into a different
    cluster or out to unclustered.
    """
    redirect = auth.require_admin(request)
    if redirect:
        return redirect
    folders = database.list_clusters_grouped_by_folder()
    # Move-dropdown options are derived from the exact same per-folder
    # "Person #" numbers shown on the cards above, so the dropdown never
    # shows a number that disagrees with what the admin is looking at.
    cluster_options = [
        {
            "cluster_id": item["cluster_id"],
            "label": f"{folder['folder_label']} \u00b7 Person #{item['number']}",
        }
        for folder in folders
        for item in folder["cards"]
        if item["type"] == "cluster"
    ]
    return templates.TemplateResponse(
        "clusters.html",
        {
            "request": request,
            "folders": folders,
            "cluster_options_json": json.dumps(cluster_options),
        },
    )


@app.post("/admin/clusters/group")
async def admin_group_faces(request: Request):
    """
    Group two or more selected faces into one brand-new cluster — mainly
    for the "not yet matched" cards: once an admin can see by eye that
    several of them are the same person (often the same person across a
    few different weeks' folders, each only appearing once so far), this
    merges them into a single person in one action instead of needing an
    existing cluster to move each one into individually.

    Works on faces from any current cluster (or none), so it also
    doubles as a quick way to merge two clusters the app split apart —
    select one face from each and group them; their whole clusters don't
    merge automatically, but a full "Recluster all faces" or further
    individual moves can finish that. Every face's old cluster (if any)
    has its centroid recomputed afterward, same as a single-face move,
    and is deleted if the move left it empty.
    """
    redirect = auth.require_admin(request)
    if redirect:
        return redirect

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid request body."}, status_code=400)

    raw_ids = body.get("face_ids")
    if not isinstance(raw_ids, list) or len(raw_ids) < 2:
        return JSONResponse({"error": "Select at least two faces to group."}, status_code=400)
    try:
        face_ids = [int(f) for f in raw_ids]
    except (TypeError, ValueError):
        return JSONResponse({"error": "face_ids must be integers."}, status_code=400)

    old_cluster_ids = set()
    for face_id in face_ids:
        face = database.get_face_with_photo(face_id)
        if face is None:
            return JSONResponse({"error": f"Face {face_id} not found."}, status_code=404)
        old_cluster_ids.add(database.get_face_cluster(face_id))

    new_cluster_id = database.create_cluster_from_faces(face_ids)
    clustering.recompute_cluster_centroid(new_cluster_id)

    for old_id in old_cluster_ids:
        if old_id is not None and old_id != new_cluster_id:
            clustering.recompute_cluster_centroid(old_id)

    database.record_audit_log(
        "faces_grouped", f"faces {face_ids} \u2192 new cluster {new_cluster_id}"
    )
    return JSONResponse({"success": True, "cluster_id": new_cluster_id})


@app.post("/admin/face/{face_id}/cluster")
async def admin_set_face_cluster(request: Request, face_id: int):
    """
    Manually move one face into a different cluster, or out to
    unclustered ("missed match"), per the cluster viewer's per-face move
    control. Body: {"cluster_id": <int>} to move into an existing
    cluster, or {"cluster_id": null} to pull it out of whatever cluster
    it's currently in.

    Both the face's old and new cluster (whichever apply) have their
    centroid recomputed immediately from their remaining/new members, so
    matching reflects the correction right away rather than only after
    the next full "Recluster all faces" run. A cluster left with zero
    members after the move is deleted (see
    clustering.recompute_cluster_centroid).
    """
    redirect = auth.require_admin(request)
    if redirect:
        return redirect

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid request body."}, status_code=400)

    raw_target = body.get("cluster_id")
    if raw_target in (None, "", "null"):
        target_cluster_id = None
    else:
        try:
            target_cluster_id = int(raw_target)
        except (TypeError, ValueError):
            return JSONResponse({"error": "cluster_id must be an integer or null."}, status_code=400)

    face = database.get_face_with_photo(face_id)
    if face is None:
        return JSONResponse({"error": "Face not found."}, status_code=404)

    if target_cluster_id is not None and not database.cluster_exists(target_cluster_id):
        return JSONResponse({"error": "Target cluster not found."}, status_code=400)

    old_cluster_id = database.get_face_cluster(face_id)
    if old_cluster_id == target_cluster_id:
        return JSONResponse({"success": True, "cluster_id": target_cluster_id, "unchanged": True})

    database.assign_face_cluster(face_id, target_cluster_id)

    if old_cluster_id is not None:
        clustering.recompute_cluster_centroid(old_cluster_id)
    if target_cluster_id is not None:
        clustering.recompute_cluster_centroid(target_cluster_id)

    database.record_audit_log(
        "face_moved", f"face {face_id}: cluster {old_cluster_id} \u2192 {target_cluster_id}"
    )
    return JSONResponse({"success": True, "cluster_id": target_cluster_id})


@app.get("/admin/clusters/{cluster_id}/faces")
def admin_cluster_faces(request: Request, cluster_id: int):
    """JSON: every face in one cluster, for the cluster viewer's
    expand-on-click lightbox. Fetched only when a card is opened."""
    redirect = auth.require_admin(request)
    if redirect:
        return redirect
    faces = database.list_faces_for_cluster(cluster_id)
    return JSONResponse({"faces": faces})


@app.get("/admin/unclustered/{face_id}/faces")
def admin_unclustered_face(request: Request, face_id: int):
    """JSON: wraps a single still-unclustered face in the same {faces: [...]}
    shape admin_cluster_faces returns, so the clusters page's lightbox can
    treat a single-photo card exactly like a multi-face cluster card
    without a separate code path in the frontend."""
    redirect = auth.require_admin(request)
    if redirect:
        return redirect
    face = database.get_face_with_photo(face_id)
    if face is None:
        return JSONResponse({"faces": []}, status_code=404)
    return JSONResponse(
        {
            "faces": [
                {
                    "face_id": face["face_id"],
                    "drive_file_id": face["drive_file_id"],
                    "folder_label": face["folder_label"],
                }
            ]
        }
    )


FACE_THUMB_CACHE_DIR = Path(BASE_DIR).parent / "data" / "face_thumb_cache"


@app.get("/admin/face/{face_id}/thumb")
def admin_face_thumbnail(request: Request, face_id: int):
    """
    Streams a thumbnail cropped to one detected face (plus a small margin
    for context), rather than the whole photo. The clusters page shows
    this instead of a whole-photo thumbnail because a cluster's claim is
    "these individual faces look alike" — with more than one person in a
    photo, a whole-photo thumbnail makes it impossible to tell which face
    is the one that actually matched, so an admin eyeballing cluster
    quality ends up judging the wrong thing.

    Crops in the same downscaled coordinate space face_processing.detect_faces
    used to compute the bounding box (MAX_DETECTION_DIMENSION), since that's
    the pixel space the stored (top, right, bottom, left) values are in —
    cropping against the original full-resolution image would cut the
    wrong region for any photo larger than that limit.

    Admin-only, same reasoning as admin_photo_thumbnail. Cached (to
    S3-compatible object storage if configured, else local disk -- see
    app/object_storage.py, Phase 7 §10) by face_id (a face's bounding box
    never changes without a re-ingest, which always creates new face
    rows) so repeat views don't re-download and re-crop.
    """
    redirect = auth.require_admin(request)
    if redirect:
        return redirect

    cache_key = f"{face_id}.jpg"
    cached = object_storage.get(cache_key, local_dir=FACE_THUMB_CACHE_DIR)
    if cached is not None:
        return Response(
            content=cached,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=604800, immutable"},
        )

    import json
    from io import BytesIO

    from PIL import Image, ImageOps

    from app.drive_client import download_image_bytes
    from app.face_processing import MAX_DETECTION_DIMENSION

    face = database.get_face_with_photo(face_id)
    if face is None or not face["bounding_box"]:
        return Response(status_code=404)

    try:
        raw = download_image_bytes(face["drive_file_id"])
        img = Image.open(BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        if max(img.size) > MAX_DETECTION_DIMENSION:
            img.thumbnail((MAX_DETECTION_DIMENSION, MAX_DETECTION_DIMENSION), Image.LANCZOS)

        top, right, bottom, left = json.loads(face["bounding_box"])
        # Pad ~35% of the face's own size on each side so the crop reads
        # as a recognizable headshot rather than a tight, disorienting box.
        face_h, face_w = bottom - top, right - left
        pad_y, pad_x = int(face_h * 0.35), int(face_w * 0.35)
        crop_box = (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(img.width, right + pad_x),
            min(img.height, bottom + pad_y),
        )
        crop = img.crop(crop_box)
        crop.thumbnail((220, 220), Image.LANCZOS)

        buf = BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        thumb_bytes = buf.getvalue()

        object_storage.put(cache_key, thumb_bytes, local_dir=FACE_THUMB_CACHE_DIR)

        return Response(
            content=thumb_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=604800, immutable"},
        )
    except Exception as e:
        logger.warning("Failed to build face thumbnail for face %s: %s", face_id, e)
        return Response(status_code=404)


def _build_or_get_cached_photo_thumbnail(drive_file_id: str) -> bytes | None:
    """
    Shared by the admin-only whole-photo thumbnail route and the public,
    token-gated gallery route (Phase 6) — both want the exact same
    "download from Drive, downscale to 300px, cache, serve" behavior, and
    keeping one implementation means a caching bug or Drive-error edge
    case only needs fixing once. Returns None (never raises) on any
    failure so both callers can turn that into their own 404 without
    duplicating error handling.

    Caching is via app/object_storage.py (Phase 7 §10): S3-compatible
    object storage if S3_BUCKET is configured, else local disk under
    THUMB_CACHE_DIR exactly as before that module existed.
    """
    safe_id = "".join(c for c in drive_file_id if c.isalnum() or c in "-_")
    cache_key = f"{safe_id}.jpg"

    cached = object_storage.get(cache_key, local_dir=THUMB_CACHE_DIR)
    if cached is not None:
        return cached

    from io import BytesIO

    from PIL import Image

    from app.drive_client import download_image_bytes

    try:
        raw = download_image_bytes(drive_file_id)
        img = Image.open(BytesIO(raw))
        img.thumbnail((300, 300))
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        thumb_bytes = buf.getvalue()

        object_storage.put(cache_key, thumb_bytes, local_dir=THUMB_CACHE_DIR)

        return thumb_bytes
    except Exception as e:
        logger.warning("Failed to load thumbnail for %s: %s", drive_file_id, e)
        return None


@app.get("/admin/photo/{drive_file_id}")
def admin_photo_thumbnail(request: Request, drive_file_id: str):
    """
    Streams a downscaled thumbnail of a Drive photo through our own
    server. Necessary because these Drive files are only ever shared with
    the service account (see drive_client.py's module docstring on folder
    isolation) — a member's or admin's browser has no Drive permission of
    its own to load them directly, so <img src="https://drive.google.com/...">
    would just fail.

    Admin-only (auth.require_admin) since this takes a raw Drive file id
    straight from the URL with no further check — anyone who could guess
    or enumerate ids could otherwise pull any church photo through this
    route. The member-facing /find flow never uses this route; it gets
    signed, short-lived tokens instead (see /find/photo/{token} and
    app/tokens.py) that only ever decode back to a photo that specific
    selfie actually matched.

    Speed: see _build_or_get_cached_photo_thumbnail — resized JPEGs are
    cached to disk by file id, and the response also carries a long-lived
    Cache-Control header so a browser that already loaded a thumbnail
    doesn't refetch it at all.
    """
    redirect = auth.require_admin(request)
    if redirect:
        return redirect

    thumb_bytes = _build_or_get_cached_photo_thumbnail(drive_file_id)
    if thumb_bytes is None:
        return Response(status_code=404)
    return Response(
        content=thumb_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=604800, immutable"},
    )


# ---------------------------------------------------------------------------
# Member-facing live selfie capture (Phase 4)
# ---------------------------------------------------------------------------
# No auth per spec Section 4 ("no auth for the member-facing selfie flow").

@app.get("/find/photo/{token}")
def find_photo(token: str):
    """
    Phase 6: the public, token-gated counterpart to /admin/photo/{id}.
    Serves a downscaled thumbnail for the member-facing gallery, but only
    for a token /find/capture actually minted for a photo that specific
    selfie matched — a raw, guessable Drive file id is never exposed to
    the browser (see app/tokens.py for why). An invalid or expired token
    (tokens live 15 minutes — plenty for one /find visit) gets a plain
    404, same as a nonexistent photo, so this endpoint can't be used to
    distinguish "wrong token" from "right token, gone photo".
    """
    drive_file_id = tokens.verify_photo_token(token)
    if drive_file_id is None:
        return Response(status_code=404)

    thumb_bytes = _build_or_get_cached_photo_thumbnail(drive_file_id)
    if thumb_bytes is None:
        return Response(status_code=404)
    return Response(
        content=thumb_bytes,
        media_type="image/jpeg",
        # Short-lived, private cache only — unlike the admin thumbnail
        # route, the URL itself expires, so nothing should cache this
        # past that point, and no shared/CDN cache should store a
        # member's matched-photo thumbnail at all.
        headers={"Cache-Control": "private, max-age=60"},
    )


@app.get("/find", response_class=HTMLResponse)
def find_menu(request: Request):
    """Member-facing landing menu: choose Photo Finder or Reaction Finder.
    Kept as a separate page (rather than folding a toggle into one form)
    per the split requested between the two — Photo Finder is live,
    Reaction Finder is a stub pointing at docs/REACTION_FINDER.md until
    it's built."""
    return templates.TemplateResponse("find_menu.html", {"request": request})


@app.get("/find/photos", response_class=HTMLResponse)
def find_photos_page(request: Request):
    return templates.TemplateResponse("find_photos.html", {"request": request})


@app.get("/find/reactions", response_class=HTMLResponse)
def find_reactions_page(request: Request):
    """Stub page for the not-yet-built Reaction Finder (see
    docs/REACTION_FINDER.md). No capture flow here yet — this exists so
    the menu on /find has somewhere real to link to, and so the page is
    already in place (title, copy, inside-photo slideshow) once the
    feature itself is built."""
    return templates.TemplateResponse("find_reactions.html", {"request": request})


@app.post("/find/capture")
async def find_capture(request: Request):
    """
    Phase 5: receives one captured video frame from the /find page,
    validates it (same checks as the Phase 4 stub), then runs real
    matching: detect a face in the frame, embed it, compare against every
    current cluster centroid, and return the matched photos (if any).

    Body: {"image": "data:image/jpeg;base64,...."}

    Nothing about the selfie — not the image bytes, not the embedding — is
    written to the DB or disk anywhere in this path. `raw` and the
    embedding derived from it only ever live in this request's memory.

    Rate-limited per client IP (see security.RateLimiter): this is the
    one unauthenticated route that runs real, non-trivial face-detection
    work, so it's the obvious target for either abuse (hammering it to
    find out who's in the photo library) or simple resource exhaustion.
    """
    ip = security.client_ip(request)
    allowed, retry_after = FIND_CAPTURE_LIMITER.check(ip)
    if not allowed:
        logger.warning("find/capture rate limit hit for %s", ip)
        return JSONResponse(
            {
                "status": "rate_limited",
                "message": "Too many attempts — please wait a few minutes and try again.",
            },
            status_code=429,
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    body = await request.json()
    data_url = body.get("image", "") if isinstance(body, dict) else ""

    if not data_url or "," not in data_url:
        return JSONResponse({"error": "No image received."}, status_code=400)

    header, _, b64_payload = data_url.partition(",")
    if "image/" not in header:
        return JSONResponse({"error": "Expected an image capture."}, status_code=400)

    try:
        raw = base64.b64decode(b64_payload, validate=True)
    except (binascii.Error, ValueError):
        return JSONResponse({"error": "Couldn't read that capture. Try again."}, status_code=400)

    if not raw:
        return JSONResponse({"error": "Empty capture. Try again."}, status_code=400)
    if len(raw) > MAX_SELFIE_BYTES:
        return JSONResponse(
            {"error": "That capture looks too large to be a single frame."},
            status_code=400,
        )

    try:
        faces = detect_faces(raw)
    except ImageDecodeError as e:
        logger.warning("Selfie capture didn't decode as an image: %s", e)
        return JSONResponse(
            {"error": "Couldn't read that capture. Try again."}, status_code=400
        )

    if not faces:
        return JSONResponse(
            {
                "status": "no_face",
                "message": "Didn't spot a face in that frame — try again with more light, closer to the camera.",
            }
        )
    if len(faces) > 1:
        # Ambiguous which face is "you" — ask for a retake rather than
        # guessing (e.g. picking the largest box), consistent with
        # favoring precision over recall per spec Section 7.
        return JSONResponse(
            {
                "status": "multiple_faces",
                "message": "Looks like more than one face in frame — make sure it's just you, then try again.",
            }
        )

    logger.info("Received selfie capture (%d bytes), 1 face detected — matching.", len(raw))

    matched_photos = matching.find_matching_photos(faces[0].embedding)

    if not matched_photos:
        return JSONResponse(
            {
                "status": "no_match",
                "message": "No matching photos found yet. New weeks get added regularly — check back soon.",
            }
        )

    return JSONResponse(
        {
            "status": "match",
            "message": f"Found {len(matched_photos)} photo(s) of you!",
            "photos": [
                {
                    # Phase 6: an actual thumbnail the gallery can render
                    # inline, via a token scoped to this one matched photo
                    # rather than the raw Drive file id (see /find/photo).
                    "thumb_url": f"/find/photo/{tokens.sign_photo_token(p.drive_file_id)}",
                    "drive_view_url": DRIVE_VIEW_URL.format(file_id=p.drive_file_id),
                    "folder_label": p.folder_label,
                }
                for p in matched_photos
            ],
        }
    )


# ---------------------------------------------------------------------------
# Clustering (Phase 3)
# ---------------------------------------------------------------------------

@app.post("/admin/cluster", response_class=HTMLResponse)
def recluster(request: Request):
    redirect = auth.require_admin(request)
    if redirect:
        return redirect

    error = None
    success = None
    try:
        summary = clustering.run_clustering()
        database.record_audit_log(
            "recluster_all",
            f"{summary['faces']} faces \u2192 {summary['clusters']} clusters, {summary['unclustered']} unclustered",
        )
        success = (
            f"Reclustered {summary['faces']} faces into {summary['clusters']} "
            f"people ({summary['unclustered']} not yet grouped)."
        )
    except Exception as e:
        logging.getLogger(__name__).exception("Clustering failed")
        error = f"Clustering failed: {e}"

    folders = database.list_folders()
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "folders": folders,
            "cluster_count": database.count_clusters(),
            "error": error,
            "success": success,
        },
    )