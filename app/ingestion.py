"""
Orchestrates Phase 2: for one approved folder, list its Drive images,
download any not already ingested, run face detection + embeddings, and
persist photos/faces rows. Per spec Section 8 step 2.

Called from a FastAPI background task (see main.py); safe to re-run on an
`error` or already-`processed` folder — already-ingested photos are
skipped via the `drive_file_id` uniqueness check in the DB layer, so a
partial prior failure is cheap to retry.

Performance: downloading from Drive is the slow part (network round
trip per photo), while face detection is CPU-bound. Both are done
concurrently across a small worker pool (INGEST_WORKERS, default 4) so
a folder of N photos takes roughly N/workers times as long as doing
them one at a time, instead of paying the full network+CPU cost for
every photo in serial. Results are still written to the DB from a single
place (the main loop, not the worker threads) so writes stay ordered and
SQLite never sees concurrent writers.

Pause/stop: `control` is an optional object (see main.PauseStopControl)
exposing `.stopped` and `.paused` (threading.Event-like: `.is_set()` /
`.wait()`). Checked between photos so an admin can pause a long run
(resume later, nothing lost) or stop it outright (already-ingested
photos stay ingested; the folder is left in a resumable state).
"""
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from app import clustering, database
from app.drive_client import DriveConfigError, download_image_bytes, list_images_in_folder
from app.face_processing import ImageDecodeError, detect_faces, embedding_to_blob

logger = logging.getLogger(__name__)

INGEST_WORKERS = int(os.environ.get("INGEST_WORKERS", "4"))

# How often (in processed photos) to write progress counters to the DB.
# Every single photo would be a lot of extra writes on a big folder; this
# keeps the admin progress bar responsive without hammering SQLite.
_PROGRESS_WRITE_EVERY = 1

def _pregenerate_face_thumb(face_id: int, face, image_bytes: bytes) -> None:
    """Build and cache the face thumbnail during ingestion while image_bytes
    is already in memory. Errors are logged and swallowed — a missing
    thumbnail is cosmetic; it will be generated lazily on first cluster view."""
    try:
        import json
        from io import BytesIO
        from PIL import Image, ImageOps
        from app import object_storage
        from app.face_processing import MAX_DETECTION_DIMENSION
        from pathlib import Path

        cache_key = f"{face_id}.jpg"
        face_thumb_cache_dir = Path(__file__).parent.parent / "data" / "face_thumb_cache"

        img = Image.open(BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        if max(img.size) > MAX_DETECTION_DIMENSION:
            img.thumbnail((MAX_DETECTION_DIMENSION, MAX_DETECTION_DIMENSION), Image.LANCZOS)

        top, right, bottom, left = face.bounding_box
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
        object_storage.put(cache_key, buf.getvalue(), local_dir=face_thumb_cache_dir)
    except Exception as e:
        logger.warning("Failed to pre-generate face thumb for face %s: %s", face_id, e)

def _fetch_and_detect(image):
    try:
        image_bytes = download_image_bytes(image.file_id)
        faces = detect_faces(image_bytes)
        return image, faces, image_bytes, None   # ← add image_bytes
    except BaseException as e:
        return image, None, None, e              # ← add None


def process_folder(folder_id: int, control=None) -> None:
    folder = database.get_folder(folder_id)
    if folder is None:
        logger.error("process_folder called for missing folder id=%s", folder_id)
        return

    database.update_folder_status(folder_id, "processing")
    drive_folder_id = folder["drive_folder_id"]

    try:
        images = list_images_in_folder(drive_folder_id)
    except DriveConfigError as e:
        logger.error("Drive config error for folder %s: %s", folder_id, e)
        database.update_folder_status(folder_id, "error")
        return
    except Exception as e:
        logger.error("Failed to list images for folder %s: %s", folder_id, e)
        database.update_folder_status(folder_id, "error")
        return

    logger.info(
        "Listed %d image(s) in Drive folder %s for folder id=%s",
        len(images), drive_folder_id, folder_id,
    )
    if not images:
        logger.warning(
            "Zero images found in Drive folder %s (folder id=%s). Common "
            "causes: folder is empty, the service account isn't shared on "
            "this exact folder, images live in a subfolder (not recursed), "
            "or the folder is in a Shared Drive missing supportsAllDrives.",
            drive_folder_id, folder_id,
        )

    to_process = [img for img in images if not database.photo_exists(img.file_id)]
    skipped_already_ingested = len(images) - len(to_process)
    database.update_folder_progress(folder_id, skipped_already_ingested, len(images))

    had_failure = False
    stopped = False
    processed_so_far = skipped_already_ingested
    # Phase 7 §8.2/§9: faces added in this batch get incrementally
    # clustered once the batch finishes, instead of paying for a full
    # DBSCAN re-cluster on every ingestion run (see clustering.py).
    new_face_ids: list[int] = []

    if to_process:
        with ThreadPoolExecutor(max_workers=INGEST_WORKERS) as pool:
            futures = {pool.submit(_fetch_and_detect, img): img for img in to_process}
            try:
                for future in as_completed(futures):
                    if control is not None and control.stopped.is_set():
                        stopped = True
                        for f in futures:
                            f.cancel()
                        break

                    if control is not None:
                        # Blocks the main (writer) thread while paused; worker
                        # threads already in flight finish and queue up, but
                        # nothing new gets written to the DB until resumed.
                        control.paused.wait()

                    image, faces, image_bytes, error = future.result()

                    if error is not None:
                        if isinstance(error, ImageDecodeError):
                            logger.warning("Skipping undecodable image %s (%s): %s", image.file_id, image.name, error)
                        else:
                            logger.warning("Skipping image %s (%s) after download/detect error: %s", image.file_id, image.name, error)
                        had_failure = True
                        processed_so_far += 1
                        if processed_so_far % _PROGRESS_WRITE_EVERY == 0:
                            database.update_folder_progress(folder_id, processed_so_far, len(images))
                        continue

                    photo_id = database.add_photo(image.file_id, folder_id)
                    for face in faces:
                        new_face_id = database.add_face(
                            photo_id=photo_id,
                            embedding=embedding_to_blob(face.embedding),
                            bounding_box=json.dumps(list(face.bounding_box)),
                        )
                        new_face_ids.append(new_face_id)
                        # Pre-generate the face thumbnail now while image_bytes
                        # is already in memory, so /admin/clusters doesn't
                        # trigger a burst of concurrent Drive re-downloads on
                        # first page view.
                        _pregenerate_face_thumb(new_face_id, face, image_bytes)

                    processed_so_far += 1
                    logger.info(
                        "[%d/%d] Processed %s (%s) — %d face(s) found",
                        processed_so_far, len(images), image.name, image.file_id, len(faces),
                    )
                    if processed_so_far % _PROGRESS_WRITE_EVERY == 0:
                        database.update_folder_progress(folder_id, processed_so_far, len(images))
            finally:
                # Ensure no half-finished futures keep running past a stop.
                for f in futures:
                    f.cancel()

    database.update_folder_progress(folder_id, processed_so_far, len(images))

    if new_face_ids:
        try:
            clustering.assign_new_faces_incrementally(new_face_ids)
        except Exception:
            # Best-effort: a clustering hiccup shouldn't mark an otherwise
            # successful ingestion run as failed. Worst case these faces
            # stay unclustered until the next incremental batch or full
            # "Recluster all faces" run picks them up — matching (§7)
            # doesn't depend on clustering having run at all, so this
            # never blocks member-facing search either way.
            logger.exception(
                "Incremental clustering failed for folder id=%s (%d new faces); "
                "faces are ingested and searchable regardless, will be picked up "
                "by the next recluster.",
                folder_id, len(new_face_ids),
            )

    if skipped_already_ingested:
        logger.info(
            "Skipped %d image(s) already ingested in a prior run for folder id=%s",
            skipped_already_ingested, folder_id,
        )

    if stopped:
        database.update_folder_status(folder_id, "cancelled")
        logger.info("Folder id=%s processing stopped by admin request.", folder_id)
        return

    database.update_folder_status(folder_id, "error" if had_failure else "processed")
    logger.info(
        "Finished processing folder id=%s: status=%s",
        folder_id, "error" if had_failure else "processed",
    )
