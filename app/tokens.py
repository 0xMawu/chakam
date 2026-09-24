"""
Short-lived signed tokens standing in for a Drive file id on the
member-facing gallery (Phase 6). GET /find/photo/{token} is deliberately
NOT just GET /find/photo/{drive_file_id}: an admin-style route that takes
a raw file id would let anyone who guesses/enumerates ids pull any
church photo through the server, with no relationship to having actually
taken a matching selfie. Instead, /find/capture mints one of these tokens
per matched photo and only that token (not the underlying id) goes to
the browser; the token expires quickly and only ever decodes back to the
specific file id it was issued for.

Reuses itsdangerous (already a dependency via Starlette's session
middleware) rather than adding a new one, and reuses SESSION_SECRET so
there's only one secret to manage/rotate.
"""
import os

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

_SECRET = os.environ.get("SESSION_SECRET", "dev-only-insecure-secret")
_SALT = "find-photo-gallery"

# Long enough to cover an admin/member actually browsing the results of a
# single /find visit, short enough that a token that leaked (e.g. via a
# shared screenshot URL) doesn't stay valid for long.
_MAX_AGE_SECONDS = 15 * 60

_serializer = URLSafeTimedSerializer(_SECRET, salt=_SALT)


def sign_photo_token(drive_file_id: str) -> str:
    return _serializer.dumps(drive_file_id)


def verify_photo_token(token: str) -> str | None:
    """Returns the drive_file_id if the token is valid and unexpired,
    else None. Never raises — every caller treats an invalid token as
    "not found", not a 500."""
    try:
        return _serializer.loads(token, max_age=_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
