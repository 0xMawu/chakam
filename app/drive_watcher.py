"""
Auto-watcher for the church photo pipeline.

Polls a configured Google Drive parent folder at a regular interval,
finds any subfolders named "outside" (case-insensitive), registers them
in the `folders` table if not already known, and kicks off ingestion —
exactly as if an admin had pasted the link and hit "Process" in the UI.

Configuration (all via environment variables):
  DRIVE_WATCH_FOLDER_ID   — the Drive folder ID of the parent to watch.
                            Required; watcher is disabled when absent.
  DRIVE_WATCH_INTERVAL_SECONDS — poll interval (default 300, i.e. 5 min).
  DRIVE_WATCH_LABEL_PREFIX     — prefix prepended to the auto-label for
                                 each discovered folder, e.g. "Auto –".
                                 Default: "Auto –".

How it fits into the existing pipeline:
  - Uses _get_service() from drive_client (same service account, same
    read-only scope) — no new credentials needed.
  - Calls database.folder_exists() / database.add_folder() / database.record_audit_log()
    exactly as the admin submit_folder route does.
  - Spawns ingestion in a daemon thread the same way /admin/folders/{id}/process does.
  - Stores last-checked time in the `watcher_state` DB table (added by
    database._run_migrations) so each poll only asks Drive for subfolders
    modified after the previous check, instead of re-listing everything.

The watcher runs as a single daemon thread started in main.py's on_startup.
A daemon thread means it is killed automatically when the main process
exits — no separate cleanup needed.
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Subfolder name filter — only subfolders with this name (case-insensitive)
# are registered and ingested.
_TARGET_FOLDER_NAME = "outside"

# Drive mime type for a folder
_FOLDER_MIME = "application/vnd.google-apps.folder"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z"


def _list_outside_subfolders(service, parent_id: str, modified_since: str | None) -> list[dict]:
    """
    List all subfolders of `parent_id` named "outside" (case-insensitive).

    If `modified_since` is an RFC 3339 timestamp string, only folders
    modified after that time are returned — this keeps each poll cheap
    for a large parent with many subfolders.  On the very first run
    (modified_since=None) all matching subfolders are returned.

    Returns a list of dicts: {"id": str, "name": str}.
    """
    # Build the query
    # Note: Drive's `name contains` is case-insensitive for ASCII, but to
    # be explicit we filter by exact name in Python after the API call so
    # we don't accidentally pick up "outside_extra" or "outside-photos".
    query_parts = [
        f"'{parent_id}' in parents",
        "trashed = false",
        f"mimeType = '{_FOLDER_MIME}'",
    ]
    if modified_since:
        query_parts.append(f"modifiedTime > '{modified_since}'")

    query = " and ".join(query_parts)

    results: list[dict] = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name)",
                pageToken=page_token,
                pageSize=100,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

        for f in response.get("files", []):
            if f["name"].strip().lower() == _TARGET_FOLDER_NAME:
                results.append({"id": f["id"], "name": f["name"]})

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return results


def _build_folder_url(folder_id: str) -> str:
    """Canonical Drive URL for a folder — same format the admin pastes."""
    return f"https://drive.google.com/drive/folders/{folder_id}"


def _start_ingestion(db_folder_id: int) -> None:
    """
    Kick off ingestion for a DB folder in a daemon thread.
    Mirrors the logic in main.py's /admin/folders/{id}/process route exactly.
    """
    import threading
    from app import ingestion as _ingestion
    from app.ingest_queue import RedisControl

    def _run():
        control = RedisControl(folder_id=db_folder_id)
        _ingestion.process_folder(db_folder_id, control=control)

    t = threading.Thread(target=_run, daemon=True, name=f"ingest-watcher-{db_folder_id}")
    t.start()
    logger.info("Watcher: started ingestion thread for folder id=%s", db_folder_id)


def _poll_once(parent_folder_id: str, label_prefix: str) -> None:
    """
    One poll cycle:
      1. Read last-checked timestamp from DB.
      2. Ask Drive for 'outside' subfolders modified since then.
      3. Register any new ones and kick off ingestion.
      4. Update last-checked timestamp in DB.
    """
    # Import here (not at module level) so this module can be imported
    # without the full app environment initialised — same lazy-import
    # pattern used throughout drive_client.py.
    from app import database
    from app.drive_client import DriveConfigError, _get_service  # noqa: PLC2701

    last_checked = database.get_watcher_state("last_checked")
    check_started_at = _now_utc_iso()

    logger.info(
        "Watcher: polling parent folder %s for 'outside' subfolders "
        "(modified since %s)",
        parent_folder_id,
        last_checked or "the beginning of time",
    )

    try:
        service = _get_service()
    except DriveConfigError as e:
        logger.error("Watcher: Drive config error — %s", e)
        return

    try:
        subfolders = _list_outside_subfolders(service, parent_folder_id, last_checked)
    except Exception as e:
        logger.error("Watcher: failed to list subfolders of %s: %s", parent_folder_id, e)
        return

    logger.info(
        "Watcher: found %d 'outside' subfolder(s) modified since last check",
        len(subfolders),
    )

    for subfolder in subfolders:
        drive_id = subfolder["id"]
        drive_url = _build_folder_url(drive_id)

        if database.folder_exists(drive_id):
            logger.debug("Watcher: folder %s already registered, skipping", drive_id)
            continue

        # Build an auto-label: e.g. "Auto – outside (2026-10-08T14:30:00Z)"
        label = f"{label_prefix} {check_started_at}"
        db_folder_id = database.add_folder(drive_id, drive_url, label)
        database.record_audit_log(
            "watcher_folder_added",
            f"{label} ({drive_id})",
        )
        logger.info(
            "Watcher: registered new 'outside' subfolder %s as folder id=%s (%s)",
            drive_id,
            db_folder_id,
            label,
        )

        _start_ingestion(db_folder_id)

    # Always advance the timestamp even when nothing was found — this is
    # what keeps each subsequent poll cheap (only new changes are fetched).
    database.set_watcher_state("last_checked", check_started_at)


def _watcher_loop(parent_folder_id: str, interval: int, label_prefix: str) -> None:
    """Main loop — runs forever in its own daemon thread."""
    logger.info(
        "Watcher: started. Parent folder: %s | Interval: %ds | Target subfolder name: '%s'",
        parent_folder_id,
        interval,
        _TARGET_FOLDER_NAME,
    )

    while True:
        try:
            _poll_once(parent_folder_id, label_prefix)
        except Exception:
            # Never let an unhandled exception kill the watcher thread.
            # Individual poll failures are already logged inside _poll_once;
            # this is a last-resort catch for truly unexpected errors.
            logger.exception("Watcher: unexpected error in poll cycle — will retry after interval")

        time.sleep(interval)


def start(app_instance=None) -> threading.Thread | None:
    """
    Start the watcher background thread.

    Returns the Thread object if the watcher was started, or None if
    DRIVE_WATCH_FOLDER_ID is not set (watcher disabled).

    Call this from main.py's on_startup event.
    The `app_instance` parameter is accepted but unused — it exists so
    the call site can pass `app` for clarity without breaking if we later
    need it (e.g. to register a shutdown handler).
    """
    parent_folder_id = os.environ.get("DRIVE_WATCH_FOLDER_ID", "").strip()
    if not parent_folder_id:
        logger.info(
            "Watcher: DRIVE_WATCH_FOLDER_ID not set — auto-watcher disabled. "
            "Set it to the Drive folder ID of the parent you want to watch."
        )
        return None

    interval = int(os.environ.get("DRIVE_WATCH_INTERVAL_SECONDS", "300"))
    label_prefix = os.environ.get("DRIVE_WATCH_LABEL_PREFIX", "Auto –").strip()

    t = threading.Thread(
        target=_watcher_loop,
        args=(parent_folder_id, interval, label_prefix),
        daemon=True,
        name="drive-watcher",
    )
    t.start()
    return t
