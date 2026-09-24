## Phase 7 progress (current, read this first)

Build plan: `PHASE7_SCALE_ARCHITECTURE.md` (planning doc, written against
this codebase as audited 2026-09-22). Status of each piece:

| § | What | Status |
|---|---|---|
| §7 | k-NN matching (search individual face embeddings, not cluster centroids) | **Done** — `app/matching.py`, runs against SQLite today |
| §8 | Incremental clustering (fast path between full re-clusters) | **Done** — `app/clustering.assign_new_faces_incrementally`, wired into `app/ingestion.py` after every batch |
| §6 | Postgres + pgvector schema + migration | **Done, not cut over** — see below |
| §4 | Detector/embedder swap (SCRFD/ArcFace-class) | **Done, not re-embedded** — see below |
| §9 | Durable ingestion queue (Celery/RQ) | **Done, now smoke-tested against real Redis/RQ** — see below |
| §10 | S3-compatible object storage | **Done** — `app/object_storage.py`; not smoke-tested against real S3 (see below) |
| §13 | Labeled eval set / threshold tuning | **Tooling built and run for real; thresholds updated to a provisional interim value** — see below, this is not the same as "validated" |

**§4 details:** `app/face_processing.py` now uses `insightface`'s
`buffalo_l` pack (SCRFD detector + ArcFace-class 512-d recognition
model) instead of dlib/`face_recognition`'s HOG/CNN detector + 128-d
embedding. `face_recognition`, `face_recognition_models`, and the
`setuptools` pin that existed only for them are removed from
`requirements.txt`; `insightface`+`onnxruntime` (CPU wheels, no
cmake/C++ toolchain needed) replace them. `DetectedFace.embedding` is
now `(512,)` float64 instead of `(128,)`; `DetectedFace.bounding_box`
stays in the same `(top, right, bottom, left)` order every caller
already expects, so nothing in `database.py`, `ingestion.py`, or
`main.py` needed to change. `embedding_to_blob`/`blob_to_embedding` are
dimension-agnostic and also needed no change.

**Important: not yet re-embedded against the real library, and threshold
constants are now known-wrong.** A real smoke test in this environment
confirmed the swap works end-to-end: `insightface`'s `buffalo_l` weights
downloaded successfully (via GitHub release assets — reachable even in
this sandbox's restricted network), and `detect_faces()` ran against
several of this repo's actual bundled photos in `data/thumb_cache/` and
`data/face_thumb_cache/`, correctly finding 0-3 faces per photo with
512-d float64 embeddings and sane bounding boxes. **What that same test
also showed:** two different people in one group photo landed at
Euclidean distance ~1.26-1.41 apart in the new embedding space — well
above `MATCH_THRESHOLD`'s and `CLUSTER_EPS`'s current defaults (0.4 /
0.38, both tuned for the old dlib space), which means those defaults
would currently reject essentially everything, including real same-
person matches, not just correctly reject different people. **Before
this goes anywhere near real users:** (1) re-run ingestion (or a
dedicated re-embed pass — not yet built) over every already-ingested
photo, since the `data/church_photos.db` bundled with this repo still
holds old 128-d embeddings from before this swap (confirmed directly:
every row in `faces.embedding` is still 1024 bytes = 128 × float64, not
512's 4096), which are **not comparable** to the new 512-d ones and will
either get skipped by `migrate_to_postgres.py`'s dimension check or
silently mixed with new ones if that check isn't heeded — this still
needs real Drive access to do, which this environment doesn't have; (2)
retune `MATCH_THRESHOLD` and `CLUSTER_EPS` for the new embedding space —
§13 tooling now exists and was run for real (see §13 details below), and
both constants were updated to a conservative interim value, but that is
explicitly **not** the same as a validated threshold — see §13 details
for exactly what's still missing before that's true.

**§9 details:** `app/ingest_queue.py` replaces the old FastAPI
`BackgroundTasks` ingestion (in-process, gone on restart, pause/stop only
reachable from the same process) with a durable RQ (Redis Queue) job.
Clicking "Process" in the admin panel now enqueues `run_ingestion_job`
instead of running it inline; a separate `rq worker ingestion` process
(see "Running the ingestion worker" below) actually executes it, calling
the *same* `app/ingestion.process_folder` as before — that function
needed **no changes**, since its `control` parameter was already
duck-typed against a `threading.Event`-shaped pair rather than hard-coded
to the old in-process `PauseStopControl`. Pause/resume/stop now set small
keys in Redis (`RedisControl`/`RedisEvent`) instead of an in-memory
`threading.Event`, so they work correctly across the web process and
worker process being different processes (or different machines). The
old `database.reset_stuck_processing_folders()` — which ran on every
startup and unconditionally marked *every* `processing`/`paused` folder
`error`, correct only when ingestion couldn't outlive the web process —
is replaced by `ingest_queue.reconcile_stuck_folders()`, which only
resets a folder if there's genuinely no active RQ job for it, so a
folder a worker is still legitimately chewing through survives a web
process restart/redeploy instead of getting yanked out from under it.

**Update — now smoke-tested against a real Redis/RQ.** A later
environment for this repo did have network access to install and run
real Redis/RQ (a change from the note below, kept for history). What was
verified for real: `RedisEvent`/`RedisControl` state transitions
(default states, pause→resume→stop, `wait()`'s timeout path) against an
actual local `redis-server`, not the fake used in the original unit
tests; and a full `enqueue_folder_ingestion` → separate `rq worker
ingestion --burst` process → `run_ingestion_job` → `ingestion.process_folder`
→ job status `FINISHED` → control flags cleared round trip, confirming
the web process and worker process really do coordinate through Redis
as designed rather than just in a hand-written fake. **Still not
verified:** pause/stop clicked *while* `process_folder` is actually
mid-loop against real Drive photos (this environment still has no Drive
API access — see §2's Google Drive API setup note — so the job above
was run against a nonexistent folder id specifically to exercise the
queue/worker machinery without needing real photos; `process_folder`
logged and returned immediately rather than looping) and the admin UI's
progress bar/button states during a real multi-photo run. The original
note below is kept for what prompted this:

**Original note (network-restricted environment):** this sandbox had no
network access, so `pip install redis rq` couldn't run, and the real
end-to-end path (enqueue from FastAPI → `rq worker` picks it up →
`process_folder` runs → pause/stop round-trips through Redis) hadn't
been exercised for real. What *was* verified there: all edited files
parse (`py_compile`); and the pause/stop/enqueue/`is_active`/`reconcile`
logic in `RedisEvent`/`RedisControl`/`ingest_queue` was exercised
end-to-end against a hand-written fake `redis`/`rq` (in-memory dict
standing in for Redis, a dict-backed fake `Queue`/`Job` standing in for
RQ). **Before this goes anywhere near real users:** with real Drive
access available, re-run the Phase 2/7 §9 ingestion smoke tests against
a real folder — including actually clicking Pause/Resume/Stop mid-run
against real multi-photo ingestion and confirming the admin UI's
progress bar and button states still update correctly with ingestion
now running in a separate process.

**Running the ingestion worker:** set `REDIS_URL` in `.env` (defaults to
`redis://localhost:6379/0` if unset), then run, alongside `uvicorn`:

```bash
rq worker ingestion --url redis://localhost:6379/0
```

Without this process running, "Process" still enqueues the job (Redis
durably holds it) but nothing executes it until a worker is started —
by design, not a bug: restarting/redeploying the *worker* no longer
loses queued work either, since RQ picked it back up off Redis. The
admin folder list will just show `pending` until a worker is running to
pick the job up.

**§10 details:** `app/object_storage.py` gives the two existing on-disk
thumbnail caches (`data/thumb_cache/` for whole-photo thumbnails,
`data/face_thumb_cache/` for per-face crops — both already existed
before §10, see the Changelog section below) an optional S3-compatible
backend. `main._build_or_get_cached_photo_thumbnail` and
`main.admin_face_thumbnail` now call `object_storage.get`/`.put` instead
of touching `Path`/`cache_path` directly; unset `S3_BUCKET` and the
behavior is byte-for-byte the old local-disk caching, so an existing
small/single-instance deployment needs to change nothing. Set
`S3_BUCKET` (and `S3_ENDPOINT_URL` for anything that isn't AWS S3
itself) and both caches move to that bucket instead — this is what
actually matters at scale: local disk caches don't survive a redeploy
on most container/PaaS platforms (ephemeral filesystem) and don't get
shared across more than one app instance, so either one means silently
falling back to "re-download from Drive and re-resize" far more often
than the cache was meant to allow.

**Scope note:** `PHASE7_SCALE_ARCHITECTURE.md` (the planning doc this
README's Phase 7 section cites for the full §10 spec) isn't present in
this checkout — I couldn't find it in the delivered files. This
implementation is inferred directly from the two disk caches that
actually exist in the codebase, not from that doc's own wording. It
does **not** move the *original* Drive-downloaded photo bytes to S3 —
`app/ingestion.py` and `app/drive_client.py` are untouched, and
originals are still fetched from Drive on demand every time, exactly as
before. If the real §10 spec calls for that too (to cut repeat Drive API
calls further, e.g. during a re-embed pass), that's a reasonable
follow-up on top of this, not something this delivery does.

**Important: not smoke-tested against a real S3-compatible bucket** —
same sandbox limitation as §9 (no network access, so `pip install
boto3` couldn't run against a real package, let alone a real bucket).
What *was* verified: `py_compile`/AST parse on every edited file; and
`get`/`put`/the local-disk-fallback path/the S3-enabled path/key
prefixing were all exercised against a hand-written fake `boto3` (an
in-memory dict standing in for a bucket) — confirming the control flow
is sound, but **not** a substitute for testing against a real bucket.
**Before this goes anywhere near real users:** run `pip install -r
requirements.txt` with real network access, point `S3_BUCKET` (and
`S3_ENDPOINT_URL` if not using AWS proper) at a real bucket — a local
MinIO container is the fastest way to get one for testing
(`docker run -p 9000:9000 minio/minio server /data`) — and confirm both
thumbnail routes actually populate it and serve back what they wrote,
including after clearing local disk entirely (to catch any place this
delivery missed and still silently depends on the local cache
directories existing).

**§6 details:** `app/database_pg.py` defines the Postgres+pgvector schema
(folders/photos/faces/clusters/audit_log, `faces.embedding` as a native
`vector` column) and `scripts/migrate_to_postgres.py` copies the current
SQLite library into it, preserving ids/foreign keys. Verified end-to-end
against a real Postgres+pgvector instance using this repo's actual bundled
`data/church_photos.db` (19 photos, 22 faces, 6 clusters) — migration,
`find_matching_photos_pg`'s pgvector k-NN query, `create_ann_indexes()`
(HNSW), and idempotent re-runs (`ON CONFLICT DO NOTHING` + sequence reset)
all passed. That verification run predates §4 and used the old 128-d
embeddings/`EMBEDDING_DIM` default — it has not been re-run since §4
landed or against real 512-d vectors.

**Important: this is additive, not a cutover.** `main.py` and the live app
still run entirely on SQLite (`app/database.py`) — `database_pg.py` isn't
imported anywhere in the request path yet. Per §12's migration plan, the
Postgres path only gets cut over to (step 4) after:
- every embedding is actually re-generated at the new dimension now that
  §4 has shipped (`EMBEDDING_DIM` in `.env` now defaults to `512` to
  match `app/face_processing.py`'s output — the schema and migration
  script both read this from one env var, no code change needed, but the
  *data* still needs a real re-embed pass, see the §4 note above)
- the labeled eval set (§13) verifies the new threshold before real
  members depend on it

To try the Postgres path yourself: set `DATABASE_URL` (and optionally
`EMBEDDING_DIM`, default `512` now) in `.env`, `pip install
psycopg2-binary` (already in `requirements.txt`), then `python
scripts/migrate_to_postgres.py` — but see the re-embed caveat above
before doing this against real data.

**§13 details:** `scripts/build_eval_set.py` + `scripts/tune_threshold.py`
build and consume a labeled (same-person / different-person) pair set to
recommend `MATCH_THRESHOLD`/`CLUSTER_EPS`. Labels come from this app's
own admin-reviewed clusters (`faces.cluster_id`) — two faces in the same
cluster are labeled same-person, two faces in different clusters are
labeled different-person, unclustered faces are excluded as unreviewed.
Distances use the real `app/face_processing.detect_faces` pipeline
(buffalo_l), re-run against the cached crops in `data/face_thumb_cache/`
— no Drive access needed for this part, which is why it was runnable
here.

Run it (already run once against this repo's bundled data; re-run after
real clusters exist):

```bash
python scripts/build_eval_set.py     # writes data/eval_pairs.json
python scripts/tune_threshold.py     # prints a recommended threshold
```

**What running it here actually showed:** this repo's bundled
`data/church_photos.db` has 6 clusters, but `data/face_thumb_cache/`
only has a cached crop for one face in most of them — so
`build_eval_set.py` could only re-embed 3 faces, all in *different*
clusters. Result: 3 different-person pairs (distances 1.274-1.378,
consistent with the §4 smoke test above), **zero same-person pairs**.
`tune_threshold.py` correctly refuses to invent a validated threshold
from that — see its output for exactly what it can and can't conclude.
`MATCH_THRESHOLD`/`CLUSTER_EPS` were still updated (0.4/0.38 → 0.9 for
both) on the reasoning documented in each constant's own comment in
`matching.py`/`clustering.py`: 0.9 is a deliberately conservative
interim value comfortably below the observed different-person floor,
chosen so a real deployment starts out too strict rather than silently
matching everyone — **not** a validated number. §13 is not actually
finished: it needs an admin to confirm/correct real clusters (via the
People view) against real re-ingested photos before there's any
same-person data to tune against. Re-run both scripts at that point.


## Setup — Python version matters

**Use Python 3.11 or 3.12.** Do not install this project's dependencies
into Python 3.13 or newer (including 3.14). `scikit-learn` and `numpy`
only ship pre-built wheels for 3.11/3.12 at the time of writing —
installing on a newer interpreter forces `pip` to compile them from
source, which needs a C/C++ toolchain and, on Windows in particular,
frequently fails outright (e.g. Ninja/MSYS2 build errors when your
Windows username contains a space).

**As of Phase 7 §4, this project no longer depends on `dlib`/
`face_recognition`.** Those are gone from `requirements.txt`, replaced
by `insightface`+`onnxruntime` (see the §4 note above) — both ship
pre-built CPU wheels for 3.11/3.12, so the cmake/C++-toolchain/Visual-
Studio install pain the old "Installing dlib on Windows" section below
described no longer applies to a normal `pip install -r
requirements.txt`. That section is kept for anyone still on a
pre-§4 checkout, or troubleshooting an old venv that still has `dlib`
installed from before this swap.

Windows, if you have multiple Python versions installed via the official
installer, use the `py` launcher to pick 3.12 explicitly rather than
whatever `python`/`pip` currently point to:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you don't have 3.12 installed, grab it from
https://www.python.org/downloads/ (pick 3.12.x, not the newest release)
before running the commands above.

Note: `insightface`'s `buffalo_l` model weights (SCRFD + ArcFace-class,
a few hundred MB total) download on first use to
`~/.insightface/models`, not at `pip install` time — the first call to
`detect_faces()` (ingesting a folder, or a member's first selfie) needs
real internet access to insightface's model host (GitHub release
assets), separate from needing internet for `pip install` itself. This
was confirmed working in the sandbox that built §4 (see the §4 note
above) — a one-time download, well under a minute on that connection —
but budget time for it before the first real ingestion run on a new
machine, and note that fully offline/air-gapped environments will need
the model files pre-staged into `~/.insightface/models` some other way.

### Installing dlib on Windows (pre-§4 checkouts only)

`dlib` (pulled in by `face_recognition`) doesn't ship a pre-built wheel
on PyPI for recent Python versions, so `pip install -r requirements.txt`
compiles it from source via CMake. If that step fails with an error like
`You must use Visual Studio to build a python extension on windows`,
you're missing the C++ compiler CMake needs (a plain "C++ isn't
installed" error, nothing to do with this app's code):

1. Install **"Build Tools for Visual Studio"** (free, no full IDE
   needed): https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. In the installer, check the **"Desktop development with C++"**
   workload and install it.
3. Close and reopen your terminal so the new PATH takes effect,
   reactivate the venv, and re-run `pip install -r requirements.txt`.

If you'd rather not install several GB of build tools, install a
pre-built `dlib` wheel instead of compiling it — search
"dlib prebuilt wheel windows" for a wheel matching your Python version
(e.g. `cp312`) and architecture, download the `.whl` file, then:

```powershell
pip install path\to\the\downloaded\dlib‑*.whl
pip install -r requirements.txt   # face_recognition now finds dlib already installed
```

## Changelog — performance & clustering-quality update

- **Faster photo loading.** Thumbnails served through `/admin/photo/{id}`
  are now cached to disk (`data/thumb_cache/`) after the first request and
  served with a long-lived `Cache-Control` header, so repeat views of the
  cluster viewer / admin pages are a local file read instead of a fresh
  Drive download + resize every time.
- **Faster, concurrent ingestion.** `ingestion.process_folder` now downloads
  and face-detects up to `INGEST_WORKERS` (default 4) photos at once via a
  thread pool instead of one at a time, so a folder's processing time scales
  down with concurrency instead of growing linearly with photo count.
- **Tighter clustering.** `CLUSTER_EPS` (DBSCAN) and `MATCH_THRESHOLD`
  (selfie matching) were both tightened from `0.5` to `0.42` by default, so
  visibly different faces are far less likely to be grouped into the same
  cluster. Both remain configurable via env var. See `app/clustering.py`
  for the reasoning.
- **Pause / stop / delete controls**, all on `/admin`:
  - A running folder now shows a live progress bar (processed/total,
    polling every 2s — no manual refresh needed) with **Pause**, **Resume**,
    and **Stop** buttons. Pausing blocks between photos (nothing lost, safe
    to resume); stopping cancels the run (`status = cancelled`, safe to
    retry — already-ingested photos are kept).
  - Any folder not currently processing/paused can be **deleted**, which
    removes the folder and its photos/faces (with a confirmation prompt).
    Re-run "Recluster all faces" afterward to update cluster membership.

# Church Photo Finder — Build Handoff

Spec: `faceID.md` (Section 8 defines the build order). **This delivery covers
Phase 1 ("Admin folder-link intake"), Phase 2 ("Ingestion pipeline"),
Phase 3 ("Clustering"), Phase 4 ("Live selfie capture (frontend)"), and
Phase 5 ("Matching endpoint (backend)"). Do not start Phase 6 (gallery
view) until asked.**

## What's built

**Phase 1 — Admin folder-link intake**
- FastAPI app with a password-protected `/admin` panel.
- Admin can paste a Google Drive **folder** link + a week label (e.g. "Week
  of Jan 12"). Backend extracts the folder ID, stores it in SQLite as an
  "approved folder" with `status = pending`. Duplicates and unparseable
  links are rejected with an inline error.

**Phase 2 — Ingestion pipeline**
- Each folder with status `pending` or `error` gets a Process/Retry button.
- Clicking it runs a FastAPI background task (`ingestion.process_folder`)
  that lists images in that Drive folder via **read-only** Drive API access
  scoped to explicitly-shared folders, downloads each new image, runs face
  detection + 128-d embedding extraction (`face_recognition`/dlib), and
  stores one `photos` row per image and one `faces` row per detected face.
  No local copies of images are kept after processing.

**Phase 3 — Clustering**
- A "Recluster all faces" button on `/admin` runs `clustering.run_clustering()`,
  which reads every face embedding in the library, groups them with
  scikit-learn's DBSCAN (full re-cluster each run, not incremental — see
  the comment block in `app/clustering.py` for why), and rewrites the
  `clusters` table + every face's `cluster_id`. Tuned for precision over
  recall per spec Section 7; unmatched faces are left unclustered rather
  than forced into a group.

**Phase 4 — Live selfie capture, member-facing (new in this delivery)**
- A new, unauthenticated page at `GET /find` (spec Section 4: "no auth for
  the member-facing selfie flow"):
  - Shows the privacy notice from spec Section 7 before the camera
    activates ("This uses facial recognition to match you to your own
    photos...").
  - Requests camera access via `navigator.mediaDevices.getUserMedia` and
    shows a live, mirrored preview. **There is no `<input type="file">`
    anywhere in this flow** — the only way to supply a face is the live
    camera stream, per spec Section 7's hard requirement (the main
    safeguard against someone searching using another person's photo).
  - "Find my photos" captures exactly one frame from the live stream onto
    a canvas, encodes it as a JPEG data URL, and POSTs it as JSON to
    `/find/capture`. The camera stream is stopped immediately after
    capture (and on page unload) rather than left running.
  - Handles the real-world camera failure modes: browser without
    `getUserMedia` support, permission denied, no camera found, and a
    "capture before the stream is ready" race.
- **`POST /find/capture` is a wiring stub, not real matching.** It parses
  the posted data URL, validates it decodes to a non-empty image under a
  size ceiling (`MAX_SELFIE_BYTES`, 8 MB — generous for a single phone-camera
  frame), logs the byte count, and always responds with
  `{"status": "not_implemented", ...}`. It does **not** generate a face
  embedding or touch the `clusters` table yet.
  - This split is deliberate, same spirit as Phase 1 standing up the full
    DB schema before later phases needed it: it lets the capture UI
    (camera permissions, HTTPS/`getUserMedia` quirks, mobile layout,
    error states) get exercised end-to-end now, before the actual
    matching logic — which needs a real `face_recognition` embedding call
    and a similarity comparison against cluster centroids — is built in
    Phase 5.
- No changes to the `folders`/`photos`/`faces`/`clusters` schema or to any
  Phase 1–3 route/behavior.

**Phase 5 — Matching endpoint, backend (new in this delivery)**
- `POST /find/capture` now does real matching instead of the Phase 4 stub.
  Same validation as before (real, non-empty, size-bounded image), then:
  1. Runs `face_processing.detect_faces` (the same detector ingestion
     uses) on the captured frame.
  2. **Zero faces** → `{"status": "no_face", ...}`. **More than one face**
     → `{"status": "multiple_faces", ...}` (ambiguous which face is "you",
     so it asks for a retake rather than guessing — precision over
     recall, spec Section 7). Neither is a `4xx`; both are normal,
     expected outcomes shown as an inline message, not an error banner.
  3. Exactly one face → embeds it and compares against every current
     cluster centroid in `app/matching.py` (Euclidean distance, per spec
     Section 5.2). Only the **single closest** cluster is considered (not
     "every cluster within threshold") — a real face should correspond to
     one person, and returning a second, merely-close-enough cluster's
     photos would be a recall-over-precision mistake.
  4. If that closest cluster is within `MATCH_THRESHOLD` (env var,
     default `0.5`, same value as `clustering.CLUSTER_EPS`) →
     `{"status": "match", "photos": [{"url": ..., "folder_label": ...}]}`
     with a plain Drive "view" link per photo. Otherwise →
     `{"status": "no_match", ...}`.
- **Nothing about the selfie is persisted.** The captured bytes and the
  embedding derived from them live only in the request's memory
  (`app/matching.py` and the `/find/capture` handler); no DB write, no
  disk write, no `photos`/`faces` row is created for a member's selfie.
  Only pre-existing (Phase 2/3) photos and cluster centroids are read.
- `find.html`'s JS now renders `status: "match"` as a list of links (one
  per matched photo, labeled with the week/folder label) instead of just
  the plain-text `message` banner from Phase 4; the other statuses
  (`no_face`, `multiple_faces`, `no_match`) still just show `message`.
- New module `app/matching.py`; new `database.py` helpers
  `list_clusters_with_centroid()` and `get_photos_for_cluster()`. No
  changes to `folders`/`photos`/`faces`/`clusters` schema, and no changes
  to any Phase 1–4 route/behavior.

**Also added (not part of the original phase plan, added on request) —
admin cluster viewer**
- `GET /admin/clusters`: a read-only page listing every cluster with a
  handful of sample photo thumbnails, so an admin can eyeball whether
  clustering actually grouped the right people together before trusting
  it in front of real members. Also surfaces a count of faces not in any
  cluster yet (either clustering hasn't been re-run since a folder was
  processed, or DBSCAN treated them as noise).
- `GET /admin/photo/{drive_file_id}`: streams a downscaled (300px,
  JPEG-normalized) thumbnail of a single Drive photo through the server.
  Necessary — not optional — because these Drive files are only ever
  shared with the service account (see `drive_client.py`'s docstring on
  folder isolation), so a browser has no Drive permission of its own to
  load them directly; this route re-uses the same
  `drive_client.download_image_bytes()` ingestion already calls. **Both
  routes are admin-only** (`auth.require_admin`) — the member-facing
  `/find` flow still only ever gets plain Drive "view" links, never
  proxied bytes through this route.
- New `database.py` helpers `list_clusters_summary()` and
  `count_unclustered_faces()`; new template `templates/clusters.html`; a
  link to it added from `admin.html`'s "People clusters" panel; new
  `.cluster-*` CSS rules appended to `static/style.css`.

## File structure

```
church-photo-finder/
├── app/
│   ├── main.py            # FastAPI routes: login, /admin, folders, /admin/cluster, /find, /find/capture
│   ├── auth.py             # shared-password + session cookie auth (admin only)
│   ├── database.py         # SQLite schema + folder/photo/face/cluster CRUD
│   ├── drive_utils.py       # extract_folder_id() — parses Drive links (Phase 1)
│   ├── drive_client.py      # Drive API: list/download images in a folder (Phase 2)
│   ├── face_processing.py   # face detection + embedding extraction (Phase 2)
│   ├── ingestion.py         # orchestrates drive_client + face_processing -> DB (Phase 2)
│   ├── clustering.py        # DBSCAN clustering of all face embeddings (Phase 3)
│   ├── matching.py          # compares one selfie embedding vs cluster centroids (Phase 5)
│   ├── templates/           # Jinja2: login.html, admin.html, find.html (Phase 4/5)
│   └── static/style.css
├── requirements.txt
├── .env.example
└── README.md               # this file
```

## Running it

```bash
cd church-photo-finder
pip install -r requirements.txt   # note: face_recognition pulls in dlib, which
                                   # compiles from source — needs cmake and a
                                   # C++ toolchain, and can take several minutes
cp .env.example .env              # then edit ADMIN_PASSWORD, SESSION_SECRET,
                                   # and GOOGLE_SERVICE_ACCOUNT_FILE
export $(cat .env | xargs)
python -m uvicorn app.main:app --reload --port 8000
```

**As of Phase 7 §9, ingestion also needs Redis + an `rq worker` process**
running alongside the command above, or clicking "Process" on a folder
will enqueue the job but nothing will ever run it:

```bash
redis-server &                                 # or: docker run -p 6379:6379 redis
rq worker ingestion --url redis://localhost:6379/0
```

See the §9 details above for what this replaced and what's/isn't been
verified about it.

- Admin panel: `http://localhost:8000/admin` — log in with `ADMIN_PASSWORD`,
  paste a Drive folder link and a label, click Process, then "Recluster all
  faces" once ingestion finishes.
- Member selfie page (new): `http://localhost:8000/find` — no login. Note:
  `getUserMedia` requires a **secure context**. `localhost` is exempted, so
  it works over plain `http://localhost:8000` in dev, but any non-localhost
  hostname (a LAN IP, a staging domain, etc.) will need HTTPS for the
  camera to be allowed — worth confirming against spec Section 10's still-
  open hosting-target question before Phase 5 goes near a real deployment.

### Google Drive API setup (needed for Phase 2 to work)

1. In Google Cloud Console, create (or reuse) a project and enable the
   **Google Drive API**.
2. Create a **service account** and download its JSON key. Point
   `GOOGLE_SERVICE_ACCOUNT_FILE` at that file.
3. For each folder an admin approves in `/admin`, **share that Drive folder**
   with the service account's email as a Viewer. The service account has
   zero Drive access until a folder is explicitly shared with it — that's
   what enforces the folder-isolation requirement in spec Section 7,
   independent of anything in this app's code.

A fresh `data/church_photos.db` SQLite file is created automatically on
first run (the `data/` dir isn't committed — it's gitignore-able).

## Verified working

**Phases 1–3**: unchanged and still passing (see `readme3.md` for the
itemized list) — nothing in this delivery touches their routes, templates,
or DB helpers.

**Phase 4** (tested against the FastAPI `TestClient`; camera/`getUserMedia`
itself can only be exercised in a real browser, not this harness):
- `GET /find` renders with no auth required, includes the privacy notice
  and the `getUserMedia` capture script. Manually inspected: the only place
  the string `type="file"` appears in the rendered page is inside an HTML
  comment describing the requirement, not an actual form element. ✅
- `POST /find/capture` with a real small JPEG (base64 data URL) → `200`,
  `{"status": "not_implemented", ...}`, and a log line with the byte count. ✅
- Missing `image` field → `400` with an inline-style error message. ✅
- Malformed base64 payload → `400`, doesn't crash the request. ✅
- Non-image MIME type in the data URL header → `400`. ✅
- Oversized payload (>8 MB) → `400`, rejected before any decode work past
  the size check. ✅
- `/admin` and `/admin/login` behavior unchanged (redirect-when-unauthed
  still verified). ✅

**Phase 5** (smoke-tested against the FastAPI `TestClient` with
`detect_faces` monkeypatched — a real `face_recognition`/dlib install
wasn't available in this environment, see note below):
- No face detected in the captured frame → `200`,
  `{"status": "no_face", ...}`. ✅
- More than one face detected → `200`, `{"status": "multiple_faces", ...}`. ✅
- One face, zero clusters exist yet → `200`, `{"status": "no_match", ...}`. ✅
- One face, embedding within `MATCH_THRESHOLD` of a cluster centroid →
  `200`, `{"status": "match", "photos": [{"url": "https://drive.google.com/file/d/<id>/view", "folder_label": ...}]}`,
  built from that cluster's actual member photos via
  `database.get_photos_for_cluster()`. ✅
- Existing Phase 4 validation (missing/empty/malformed/oversized/non-image
  payload → `400`) unchanged and still passing. ✅
- `/admin`, `/admin/login`, ingestion, and clustering routes unaffected
  (redirect-when-unauthed still verified). ✅

**Important caveat — not yet run against the real `face_recognition`
library**: this sandbox couldn't install `face_recognition`/`dlib` (native
build, needs cmake + a C++ toolchain + several minutes — see
requirements.txt comment), so Phase 5's tests monkeypatch
`face_processing.detect_faces` with fixed embeddings rather than running
real detection/embedding end-to-end. The matching *logic* (distance
comparison, threshold, single-closest-cluster selection, photo lookup) is
exercised for real; the actual dlib detector/encoder is not. **Before this
goes anywhere near real users, run it locally with the real dependency
installed and test with actual photos** — ideally including a same-person
photo taken at a different angle/lighting than what's in the library, to
sanity-check `MATCH_THRESHOLD`'s default of `0.5` against real embedding
distances, not just the zero-distance case the automated test uses.

**Not yet tested**: real camera permission flows (grant/deny/no-camera) in
an actual browser, and behavior over a real non-localhost HTTPS
deployment. Worth a manual pass on a phone, since camera UX quirks (iOS
Safari's autoplay/muted requirements, Android Chrome permission prompts,
etc.) are exactly the kind of thing that doesn't show up in a
server-side test client. This carries over from Phase 4 — Phase 5 doesn't
touch the capture UI, only what happens after a frame is posted.

## Notes / decisions for the next phase's builder

- **Response shape actually shipped in Phase 5**: `{"status": "match",
  "photos": [{"url": ..., "folder_label": ...}]}`,
  `{"status": "no_match", "message": ...}`, `{"status": "no_face", ...}`,
  or `{"status": "multiple_faces", ...}` — all `200`s, since each is a
  normal outcome, not a request error. `find.html` renders `match`
  specially (a list of photo links); everything else just shows
  `message`. Phase 6 (gallery view) will presumably want to replace the
  plain link list with an actual `<img>` gallery — the `photos` array's
  shape (`url` + `folder_label` per photo) is the contract to extend.
- **`MATCH_THRESHOLD` (env var, default `0.5`) is untuned against real
  faces** — see the caveat above about not having a working
  `face_recognition` install in this environment. Treat the default as a
  starting point, not a validated value; Phase 6 (or before) should
  budget time for tuning it against real church-photo test data, same as
  `clustering.CLUSTER_EPS`.
- **Only the single closest cluster is ever considered a match** —
  `app/matching.py` doesn't return photos from a second cluster even if
  it's also within threshold. If real-world testing shows people
  legitimately end up split across two clusters (e.g. a haircut or
  glasses shifted their embedding enough to land in a second DBSCAN
  cluster), that's a clustering-quality problem to fix via
  `CLUSTER_EPS`/`CLUSTER_MIN_SAMPLES` tuning or better source photos, not
  something to paper over by matching multiple clusters here — doing the
  latter would quietly trade away the precision-over-recall stance spec
  Section 7 asks for.
- **Preview is mirrored, capture is not.** The `<video>` element is
  CSS-mirrored (`transform: scaleX(-1)`) so the live preview feels like
  looking in a mirror, which is what people expect from a selfie camera.
  The canvas capture draws from the underlying `<video>` frame directly, so
  the JPEG sent to the backend is **not** mirrored — just the on-screen
  preview. This shouldn't matter for face embeddings (most extractors are
  orientation-tolerant for a simple horizontal flip, but worth a sanity
  check in Phase 5 with a real asymmetric test face if matching accuracy
  ever looks off).
- **No liveness/anti-spoofing beyond camera-only capture**, per spec
  Section 3 (non-goal for v1) — someone could still point their live
  camera at a photo of someone else's face on another screen or printout.
  Spec Section 9 lists blink/motion-based liveness detection as a stretch
  goal, not required for launch; flagging again here since Phase 4 is the
  first phase where this non-goal becomes user-facing rather than
  theoretical.
- **Minors and biometric data** (carried over from `readme3.md`, still
  unresolved): church after-service photo sets will very likely include
  children, and the ingestion pipeline extracts biometric face embeddings
  for everyone in every approved photo, not just adults who could consent
  themselves. Phase 4 makes the *searching* side member-facing but doesn't
  change this — it's still worth the product owner deciding, before
  Phase 5 makes matching actually work, whether (a) parents/guardians are
  notified, (b) there's an opt-out for specific individuals or families,
  and (c) any local regulations on biometric data of minors apply. This is
  a product and consent question, not a code one.
- **Open questions from spec Section 10** (photo volume/week, single vs
  multi-admin, hosting target) are still unanswered. The hosting-target
  answer now also matters for Phase 4/5 specifically, since `getUserMedia`
  needs HTTPS outside of `localhost`.

## Next phase (do NOT start without explicit go-ahead)

Phase 6 per Section 8: **Gallery view** — replace the current plain
Drive-link list in `find.html`'s "match" state with an actual photo
gallery (thumbnails, not just text links), rendering the `photos` array
`/find/capture` already returns. Worth deciding whether thumbnails are
fetched live from Drive per-view or something is cached, and revisiting
`MATCH_THRESHOLD` against real test photos (see caveat above) before or
alongside this phase, since a real UI in front of matching is when a
too-loose or too-tight threshold will actually get noticed.

## Still-open product questions (unchanged by Phase 5, worth resolving soon)

Carried over from earlier phases, **not addressed by this delivery**:
- **Minors and biometric data**: church photo sets will likely include
  children, and every face in every approved photo gets a biometric
  embedding, not just adults who could consent themselves. Phase 5 makes
  *matching* real, which makes this less theoretical than it was in
  Phase 4 — worth the product owner deciding, ideally before Phase 6
  ships to real members, whether (a) parents/guardians are notified,
  (b) there's an opt-out for specific individuals/families, and (c) any
  local regulations on biometric data of minors apply. Still a product
  and consent question, not a code one — nothing in Phase 5 changes what
  data is collected, only what happens with a selfie once it's been
  provided.
- **Hosting target / HTTPS** (spec Section 10): `getUserMedia` needs a
  secure context outside `localhost`, so this still needs an answer
  before any non-dev deployment.
- **Photo volume/week and single vs multi-admin** (spec Section 10): also
  still open.
