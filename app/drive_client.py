"""
Google Drive read access, scoped per spec Section 7 ("Drive API scope"):
only ever reads folders the admin has explicitly approved (folder IDs
already sitting in the `folders` table). This module never lists or
walks anything else — there is intentionally no "list everything" or
"list parent" method. The actual access boundary is enforced by Google's
own sharing model: the service account has zero visibility into any
Drive folder until an admin shares that specific folder with it (see
README, "Google Drive API setup").

Auth: a service account JSON key, path given by GOOGLE_SERVICE_ACCOUNT_FILE.
Requires: google-api-python-client, google-auth.
"""
import io
import logging
import os
import socket
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# The googleapiclient/httplib2 stack has NO default timeout — a stalled
# network call (flaky wifi, a proxy or antivirus silently swallowing
# traffic, an expired/misconfigured credential that hangs instead of
# failing) hangs forever with zero error output, which looks exactly like
# ingestion being "stuck" with no log lines at all. This sets a global
# socket timeout so any such call fails loudly instead.
socket.setdefaulttimeout(30)

# Drive mime types we treat as processable images. Google Docs/Sheets/etc.
# living in the same folder are silently skipped, not errored.
_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/heic",
    "image/webp",
}


class DriveConfigError(RuntimeError):
    """Raised when the service account credentials are missing/invalid."""


@dataclass
class DriveImage:
    file_id: str
    name: str
    mime_type: str


def _get_service():
    """
    Imports the Google API client lazily (rather than at module load) so
    the rest of the app — including Phases 1 and 3, and even just booting
    the server — doesn't require google-api-python-client/google-auth to
    be installed. It's only needed once ingestion actually runs.

    Credentials are loaded from one of two sources (checked in order):
    1. GOOGLE_SERVICE_ACCOUNT_CREDENTIALS — the full service account JSON
       as a string (useful for Railway/Render/Heroku where you can't upload
       files but can set env vars).
    2. GOOGLE_SERVICE_ACCOUNT_FILE — path to a local JSON key file (the
       original method, still works for local dev).
    """
    import json
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    credentials = None

    # Try inline JSON string first (Railway / cloud hosting)
    inline_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_CREDENTIALS")
    if inline_json:
        try:
            info = json.loads(inline_json)
            credentials = service_account.Credentials.from_service_account_info(
                info, scopes=SCOPES
            )
        except (ValueError, KeyError) as e:
            raise DriveConfigError(
                f"GOOGLE_SERVICE_ACCOUNT_CREDENTIALS is set but could not be "
                f"parsed as a valid service account JSON: {e}"
            ) from e

    # Fall back to file path (local dev)
    if credentials is None:
        key_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
        if not key_path or not os.path.isfile(key_path):
            raise DriveConfigError(
                "Neither GOOGLE_SERVICE_ACCOUNT_CREDENTIALS nor "
                "GOOGLE_SERVICE_ACCOUNT_FILE is set (or the file path doesn't "
                "exist). See README > Google Drive API setup."
            )
        try:
            credentials = service_account.Credentials.from_service_account_file(
                key_path, scopes=SCOPES
            )
        except (ValueError, OSError) as e:
            raise DriveConfigError(f"Could not load service account credentials: {e}") from e

    logger.debug("Building Drive service client")
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def list_images_in_folder(folder_id: str) -> list[DriveImage]:
    """
    List image files directly inside `folder_id`.

    Deliberately does NOT recurse into subfolders and does NOT accept a
    "list everything" mode — the only folder ID this function ever touches
    is the one explicitly passed in by the ingestion pipeline, which in
    turn only ever pulls IDs out of the `folders` table (see Section 7).
    """
    service = _get_service()
    images: list[DriveImage] = []
    page_token = None
    mime_clause = " or ".join(f"mimeType='{m}'" for m in _IMAGE_MIME_TYPES)
    query = f"'{folder_id}' in parents and trashed = false and ({mime_clause})"

    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
                pageSize=100,
                # Without these two flags, files.list silently returns zero
                # results for folders that live in a Shared Drive (a Team
                # Drive) — even with correct sharing/permissions and no
                # error raised. Harmless to set for a personal "My Drive"
                # folder too, so always pass them.
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for f in response.get("files", []):
            images.append(DriveImage(file_id=f["id"], name=f["name"], mime_type=f["mimeType"]))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return images


def download_image_bytes(file_id: str) -> bytes:
    """Download a single file's raw bytes into memory. No local copy is kept."""
    from googleapiclient.http import MediaIoBaseDownload

    logger.info("Starting download of Drive file %s", file_id)
    service = _get_service()
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    logger.info("Finished download of Drive file %s (%d bytes)", file_id, buffer.tell())
    return buffer.getvalue()
