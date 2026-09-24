"""
Minimal shared-password auth for /admin, per Section 4 ("Basic shared
password / simple auth for the admin panel only; no auth for the
member-facing selfie flow"). Uses Starlette's signed session cookie —
no separate user table needed for a single shared password.
"""
import os

from fastapi import Request
from fastapi.responses import RedirectResponse

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme")
SESSION_KEY = "admin_authed"


def is_authed(request: Request) -> bool:
    return request.session.get(SESSION_KEY) is True


def log_in(request: Request) -> None:
    request.session[SESSION_KEY] = True


def log_out(request: Request) -> None:
    request.session.pop(SESSION_KEY, None)


def require_admin(request: Request):
    """Call at the top of a protected route; returns a redirect or None."""
    if not is_authed(request):
        return RedirectResponse(url="/admin/login", status_code=303)
    return None
