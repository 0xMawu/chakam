"""
Small, dependency-free building blocks for hardening the two endpoints
that face the open internet without admin auth in front of them:
POST /find/capture (unauthenticated, runs real face detection — expensive
and a biometric-matching oracle if left unthrottled) and POST /admin/login
(a brute-forceable shared password).

In-memory sliding-window rate limiting. This is deliberately simple and
has one real limitation worth being explicit about: it's per-process
state, so it only limits a single uvicorn worker. Running this behind a
load balancer with multiple app instances/workers needs the counters
moved to a shared store (Redis is the standard choice) so limits apply
across the whole fleet, not per-instance. Fine for the single-process
deployment this app currently targets; flagged here so it's not
mistaken for a scalable-by-default solution when that changes.
"""
import time
from collections import defaultdict
from threading import Lock

from fastapi import Request


class RateLimiter:
    """Sliding-window limiter: at most `max_requests` per `window_seconds`
    per key (typically client IP). Thread-safe for uvicorn's default
    threadpool-backed sync routes and safe to call from async routes too."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, key: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds). Records the attempt
        immediately if allowed, so a request that proceeds counts against
        the window right away rather than only on completion."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = self._hits[key]
            # Drop expired timestamps so the dict doesn't grow forever.
            while hits and hits[0] < cutoff:
                hits.pop(0)
            if len(hits) >= self.max_requests:
                retry_after = hits[0] + self.window_seconds - now
                return False, max(retry_after, 0.0)
            hits.append(now)
            return True, 0.0


def client_ip(request: Request) -> str:
    """Best-effort client IP. Trusts X-Forwarded-For's first hop only when
    present, since a real deployment behind a reverse proxy (nginx,
    Cloudflare, a cloud load balancer) replaces request.client with the
    proxy's own address otherwise — without this, every request would
    share one rate-limit bucket. NOTE: only safe to trust this header once
    the reverse proxy is configured to strip/overwrite any client-supplied
    X-Forwarded-For before it reaches this app; otherwise a client could
    forge it to dodge the limiter. Document that requirement alongside
    whatever reverse proxy config ships with the real deployment."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
