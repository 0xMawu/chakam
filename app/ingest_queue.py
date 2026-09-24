"""
Phase 7 §9: durable ingestion queue.

Why this exists: before this module, ingestion ran as a FastAPI
BackgroundTask — an in-process coroutine kicked off by whichever
uvicorn worker handled the "Process" click (see git history / the
Phase 2 section of README.md). That meant:
  - a server restart/redeploy while a folder was mid-ingest silently
    killed it, leaving the DB row stuck at status="processing" forever.
    `database.reset_stuck_processing_folders()` existed purely to paper
    over this by marking *every* such row "error" on the next startup —
    including a folder some other, still-legitimately-running process
    happened to be partway through, since there was no way to tell the
    difference.
  - pause/resume/stop only worked because `main._controls` held a real
    in-memory `threading.Event` per folder_id, in the same process that
    was running the ingestion loop. That state couldn't survive (or be
    reached from) a second uvicorn worker, a redeploy, or a
    horizontally-scaled deployment.

This module replaces both with RQ (Redis Queue): `enqueue_folder_ingestion`
puts a durable job on a Redis-backed queue, and a separate `rq worker`
process (see README's "Running the ingestion worker" section) picks it
up and runs `run_ingestion_job`, a thin wrapper around the existing
`app.ingestion.process_folder`. Pause/stop/resume, which used to live on
an in-memory `threading.Event`, now live as small keys in the same
Redis instance (`RedisControl`/`RedisEvent` below) — process boundary
doesn't matter anymore, since the web process and the worker process
both just talk to Redis.

Chose RQ over Celery: this app is a single small task type (folder
ingestion) with no need for Celery's routing/chaining/multi-broker
machinery, and RQ's only infrastructure dependency is the Redis this
project would need for the durable queue anyway — one moving part
instead of two.

`app/ingestion.py` itself needed *no changes*: `process_folder`'s
`control` parameter was already duck-typed against a
`threading.Event`-shaped `.paused`/`.stopped` pair (see its own
docstring), so `RedisControl` is a drop-in for the old
`main.PauseStopControl` — `process_folder` doesn't know or care which
one it's holding.

Scope note: this buys durability across *this app's* process
restarts/redeploys, which was the actual problem
`reset_stuck_processing_folders` was working around. It does not by
itself protect against Redis itself losing data (this project doesn't
configure Redis persistence/AOF) — that's an infrastructure decision
for whoever deploys this, same as this project doesn't configure
Postgres backups either. Worker crash-detection/dead-job requeueing
beyond what's below is also out of scope — a real deployment would
want to run `rq worker` under something that restarts it (systemd,
supervisord, a container restart policy) and would want RQ's own
`Worker` monitoring/heartbeat for jobs whose worker died mid-run
without a chance to update status; `reconcile_stuck_folders()` below
only catches that case at the *next app startup*, not immediately.
"""
import logging
import os
import time

import redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app import database, ingestion

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = "ingestion"

# A folder's ingestion can legitimately take a long time (thousands of
# photos, each a Drive download + face detection). RQ's own default job
# timeout (180s) would kill a real run partway through, so this is
# generous by default; override via env if a deployment's folders are
# small enough that failing fast is more useful.
INGEST_JOB_TIMEOUT = int(os.environ.get("INGEST_JOB_TIMEOUT_SECONDS", str(6 * 60 * 60)))

# How long a finished/failed job's RQ bookkeeping sticks around in
# Redis. Long enough to be inspectable during that admin session, short
# enough not to accumulate forever — folders.status in the DB is the
# durable source of truth for outcome, not the RQ job object.
_RESULT_TTL_SECONDS = 24 * 60 * 60
_FAILURE_TTL_SECONDS = 24 * 60 * 60

_ACTIVE_STATUSES = {"queued", "started", "deferred", "scheduled"}

_redis_conn = None


def get_redis():
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = redis.from_url(REDIS_URL)
    return _redis_conn


def _queue() -> Queue:
    return Queue(QUEUE_NAME, connection=get_redis())


def _job_id(folder_id: int) -> str:
    return f"ingest-folder-{folder_id}"


def get_job(folder_id: int) -> Job | None:
    try:
        return Job.fetch(_job_id(folder_id), connection=get_redis())
    except NoSuchJobError:
        return None


def is_active(folder_id: int) -> bool:
    """True if a job for this folder is queued or currently running.

    Doesn't attempt to distinguish "genuinely active" from "the worker
    that had this job died without RQ's own monitoring noticing yet" —
    that's what RQ's Worker heartbeat/monitoring is for at the
    infrastructure level (out of scope here; see module docstring)."""
    job = get_job(folder_id)
    if job is None:
        return False
    try:
        return job.get_status(refresh=True) in _ACTIVE_STATUSES
    except Exception:
        # Redis hiccup mid-check -- fail closed (report *not* active) so
        # a flaky Redis connection can't permanently block an admin from
        # retrying a folder. Worst case this causes a duplicate job,
        # which ingestion.process_folder already tolerates (already-
        # ingested photos are skipped via the drive_file_id check).
        logger.exception("Couldn't check job status for folder id=%s", folder_id)
        return False


def enqueue_folder_ingestion(folder_id: int) -> None:
    """Enqueues a durable ingestion job for this folder. Caller (main.py)
    is responsible for checking is_active() first if it wants to refuse
    a duplicate submit with a friendly admin-facing error instead of
    silently replacing whatever job currently holds this job id."""
    _clear_control_flags(folder_id)
    _queue().enqueue(
        run_ingestion_job,
        folder_id,
        job_id=_job_id(folder_id),
        job_timeout=INGEST_JOB_TIMEOUT,
        result_ttl=_RESULT_TTL_SECONDS,
        failure_ttl=_FAILURE_TTL_SECONDS,
    )


def run_ingestion_job(folder_id: int) -> None:
    """The actual RQ job body — runs inside the `rq worker` process, not
    the web process. Builds its RedisControl here (rather than having
    main.py construct one and pass it into enqueue()) so nothing
    Redis-connection-shaped needs to survive pickling across the
    enqueue() call; RQ only needs to pickle the plain int folder_id."""
    control = RedisControl(folder_id)
    try:
        ingestion.process_folder(folder_id, control=control)
    finally:
        _clear_control_flags(folder_id)


# ---------------------------------------------------------------------------
# Pause/stop, now backed by Redis instead of an in-process threading.Event
# ---------------------------------------------------------------------------

class RedisEvent:
    """threading.Event-alike (is_set/set/clear/wait), backed by a Redis
    key instead of process memory, so a value written by the web process
    (an admin clicking Pause) is visible to the worker process (blocked
    in ingestion.process_folder's per-photo loop) and vice versa.
    `default` mirrors threading.Event's constructor argument: what the
    event reads as before anything has ever written the key (i.e. a
    fresh job that hasn't been paused/stopped yet)."""

    _POLL_INTERVAL_SECONDS = 0.5

    def __init__(self, key: str, default: bool):
        self._key = key
        self._default = default

    def is_set(self) -> bool:
        try:
            value = get_redis().get(self._key)
        except Exception:
            logger.exception("Redis unreachable checking %s -- assuming %s", self._key, self._default)
            return self._default
        if value is None:
            return self._default
        return value == b"1"

    def set(self) -> None:
        get_redis().set(self._key, "1")

    def clear(self) -> None:
        get_redis().set(self._key, "0")

    def wait(self, timeout: float | None = None) -> bool:
        """Polls rather than blocking natively on a Redis primitive —
        there's no wait-for-key-change primitive as simple as this needs
        without a pubsub channel per folder, and a pause is a human
        clicking a button (not a hot loop), so a half-second poll
        interval is invisible in practice while much cheaper than a
        tight spin."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.is_set():
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(self._POLL_INTERVAL_SECONDS)


class RedisControl:
    """Drop-in replacement for the old main.PauseStopControl (see that
    class's docstring for the inverted-paused convention this mirrors
    exactly: `paused` starts *set*, meaning "not paused, go ahead")."""

    def __init__(self, folder_id: int):
        self.paused = RedisEvent(f"ingest:paused:{folder_id}", default=True)
        self.stopped = RedisEvent(f"ingest:stopped:{folder_id}", default=False)


def _clear_control_flags(folder_id: int) -> None:
    """Resets both flags to their just-started defaults. Called both when
    a new job is enqueued (in case a stale pause/stop flag from a prior
    run of this same folder_id never got cleaned up -- e.g. Redis kept
    the key but the worker that would have cleared it was killed) and
    when a job finishes (tidy up rather than leaving keys around
    forever)."""
    try:
        conn = get_redis()
        conn.delete(f"ingest:paused:{folder_id}")
        conn.delete(f"ingest:stopped:{folder_id}")
    except Exception:
        logger.exception("Couldn't clear control flags for folder id=%s", folder_id)


def request_pause(folder_id: int) -> None:
    RedisControl(folder_id).paused.clear()


def request_resume(folder_id: int) -> None:
    RedisControl(folder_id).paused.set()


def request_stop(folder_id: int) -> None:
    control = RedisControl(folder_id)
    control.stopped.set()
    control.paused.set()  # wake a currently-paused run so it notices the stop


def reconcile_stuck_folders() -> None:
    """Replaces the old database.reset_stuck_processing_folders(), which
    ran on every startup and unconditionally marked *every*
    processing/paused folder "error" -- correct when ingestion only ever
    lived in the same process as the web server (a restart really did
    mean it was gone), wrong now: a folder can legitimately still be
    processing in a separate `rq worker` process while the web process
    restarts for an unrelated redeploy. Only resets folders that are
    *actually* stuck -- no active RQ job -- e.g. because the worker
    process itself crashed or was killed."""
    for folder in database.list_folders():
        if folder["status"] not in ("processing", "paused"):
            continue
        if is_active(folder["id"]):
            continue
        logger.warning(
            "Folder id=%s stuck at status=%s with no active ingestion job -- "
            "marking error so it can be retried.",
            folder["id"], folder["status"],
        )
        database.update_folder_status(folder["id"], "error")
