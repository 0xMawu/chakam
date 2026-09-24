"""
Phase 7 §6: PostgreSQL + pgvector, standing up ALONGSIDE the existing
SQLite DB (app/database.py) per §12 step 1 — nothing in main.py has been
cut over yet, and this module is not imported from anywhere in the live
request path. It exists so the schema, the pgvector query shape, and the
migration script (scripts/migrate_to_postgres.py) can be built and tested
for real before step 4 of §12 ("cut member-facing /find/capture over ...
only after step 2-3 are verified").

Why leave SQLite for this: see PHASE7_SCALE_ARCHITECTURE.md §6.1
(single-writer model, no native vector index — matching.py's brute-force
numpy scan stands in for that today — no horizontal read scaling).

Embedding dimension is intentionally NOT hardcoded — it's read from
EMBEDDING_DIM (default now 512, matching §4's ArcFace-class model, which
has now shipped in app/face_processing.py) rather than baked into the
schema, since §4 landing doesn't by itself change what's sitting in the
live SQLite library.

**Migration note (post-§4):** any face rows already in `data/church_photos.db`
from before §4 shipped still hold old 128-d dlib/face_recognition
embeddings (this module's own docstring history noted this as "today's
real embeddings" when §6 was first built) — those are NOT comparable to
new 512-d embeddings (see face_processing.py's module docstring, §4.2)
and must not be mixed in a single `vector(EMBEDDING_DIM)` column. Before
re-running scripts/migrate_to_postgres.py against a library that has any
pre-§4 faces in it, re-run ingestion (or a dedicated re-embed pass, not
yet built) so every face's embedding is actually 512-d; the migration
script's own EMBEDDING_DIM mismatch check (see its module docstring)
will skip — not silently corrupt — any row whose embedding doesn't match,
but a library with a mix of old and new dimensions needs cleaning up
before migrating, not just skipping around.
"""
import logging
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# Standard libpq connection string, e.g.
# postgresql://user:password@host:5432/church_photos
# Falls back to individual PG* env vars (PGHOST/PGPORT/PGUSER/PGPASSWORD/
# PGDATABASE) via psycopg2's own defaults if DATABASE_URL isn't set, same
# as any other libpq-based tool.
DATABASE_URL = os.environ.get("DATABASE_URL")

# See module docstring: 512 now that §4's ArcFace-class swap has shipped
# in face_processing.py (was 128, dlib/face_recognition, before §4). A
# single env var, not a code change, so the migration script and this
# schema always agree with whatever face_processing.py is actually
# producing — but see the module docstring's migration note about not
# mixing pre-§4 and post-§4 embeddings in one migration.
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "512"))

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS folders (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'drive',   -- 'drive' | 's3'
    source_folder_id TEXT NOT NULL,
    source_folder_url TEXT NOT NULL,
    label TEXT NOT NULL,
    date_added TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL DEFAULT 'pending',
    processed_count INT NOT NULL DEFAULT 0,
    total_count INT NOT NULL DEFAULT 0,
    UNIQUE (source, source_folder_id)
);

CREATE TABLE IF NOT EXISTS photos (
    id BIGSERIAL PRIMARY KEY,
    folder_id INT NOT NULL REFERENCES folders(id),
    source_file_id TEXT NOT NULL,           -- Drive file id, or S3 key
    date_added TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (folder_id, source_file_id)
);

CREATE TABLE IF NOT EXISTS clusters (
    id BIGSERIAL PRIMARY KEY,
    centroid vector({dim}),
    face_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- faces.embedding is a native pgvector column, not a BLOB (§6.2) — the
-- core scalability fix. Matching (see find_matching_photos_pg below)
-- searches THIS directly, per face, not cluster centroids (§3, §7).
CREATE TABLE IF NOT EXISTS faces (
    id BIGSERIAL PRIMARY KEY,
    photo_id BIGINT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    embedding vector({dim}) NOT NULL,
    bounding_box JSONB,
    quality_score REAL,
    cluster_id BIGINT REFERENCES clusters(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor TEXT NOT NULL DEFAULT 'admin',
    action TEXT NOT NULL,
    detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_photos_folder_id ON photos(folder_id);
CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces(cluster_id);

-- §6.3: below roughly 1M rows an exact scan is often fast enough and
-- gives perfect recall. Deliberately NOT creating the HNSW index here by
-- default — add it once query latency on real data is a measured
-- problem, not preemptively (same reasoning as CLUSTER_EPS/
-- MATCH_THRESHOLD not being guessed either). See create_ann_indexes()
-- below for the opt-in version of this, run manually once that point is
-- reached.
"""

# Deliberately separate from SCHEMA (see the comment above it): building
# an HNSW index is the expensive, opt-in step, so it's its own function
# rather than always running on init_db().
_ANN_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS faces_embedding_hnsw_idx
    ON faces USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS clusters_centroid_hnsw_idx
    ON clusters USING hnsw (centroid vector_cosine_ops);
"""


def _require_connection_configured() -> None:
    if not DATABASE_URL and not os.environ.get("PGHOST") and not os.environ.get("PGDATABASE"):
        raise RuntimeError(
            "Postgres isn't configured: set DATABASE_URL (or PGHOST/PGDATABASE/"
            "PGUSER/PGPASSWORD) before using app.database_pg. This module is "
            "separate from the SQLite path (app/database.py) and only needed "
            "once you're standing up the Phase 7 §6 migration."
        )


@contextmanager
def get_connection():
    _require_connection_configured()
    conn = psycopg2.connect(DATABASE_URL) if DATABASE_URL else psycopg2.connect()
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create the schema (idempotent — CREATE ... IF NOT EXISTS
    throughout) at EMBEDDING_DIM. Does NOT create the HNSW ANN indexes;
    call create_ann_indexes() separately once that's actually warranted
    (see §6.3 and the comment on _ANN_INDEX_SQL above)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA.format(dim=EMBEDDING_DIM))
        conn.commit()
    logger.info("Postgres schema ready (embedding dim=%d)", EMBEDDING_DIM)


def create_ann_indexes() -> None:
    """Opt-in: build the HNSW indexes over faces.embedding and
    clusters.centroid. Slow to build and memory-hungry (§6.3) — run this
    once query latency on real data volume is a measured problem, e.g.
    as a one-off admin/ops command, not automatically on every startup."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_ANN_INDEX_SQL)
        conn.commit()
    logger.info("HNSW indexes built on faces.embedding and clusters.centroid")


# ---------------------------------------------------------------------------
# Write helpers used by scripts/migrate_to_postgres.py. These accept an
# explicit id from the source SQLite row so that photos.folder_id and
# faces.photo_id/cluster_id foreign keys carry over unchanged instead of
# needing an id-remapping pass.
# ---------------------------------------------------------------------------

def insert_folder(id: int, source: str, source_folder_id: str, source_folder_url: str,
                   label: str, status: str, processed_count: int, total_count: int) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO folders (id, source, source_folder_id, source_folder_url,
                                      label, status, processed_count, total_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (id, source, source_folder_id, source_folder_url, label, status,
                 processed_count, total_count),
            )
        conn.commit()


def insert_photo(id: int, folder_id: int, source_file_id: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO photos (id, folder_id, source_file_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (id, folder_id, source_file_id),
            )
        conn.commit()


def insert_cluster(id: int, centroid: list[float] | None, face_count: int) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO clusters (id, centroid, face_count)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (id, centroid, face_count),
            )
        conn.commit()


def insert_face(id: int, photo_id: int, embedding: list[float], bounding_box: str | None,
                 cluster_id: int | None) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO faces (id, photo_id, embedding, bounding_box, cluster_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (id, photo_id, embedding, bounding_box, cluster_id),
            )
        conn.commit()


def insert_audit_log(id: int, created_at, actor: str, action: str, detail: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_log (id, created_at, actor, action, detail)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (id, created_at, actor, action, detail),
            )
        conn.commit()


def reset_id_sequences() -> None:
    """After migrating rows with explicit ids (see the insert_* helpers
    above), every SERIAL sequence is still at 1 and the next INSERT
    without an explicit id would collide. Bump each sequence to
    max(id)+1, the standard Postgres fix after an id-preserving bulk
    load."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            for table in ("folders", "photos", "faces", "clusters", "audit_log"):
                cur.execute(
                    f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM {table}), 1), "
                    f"(SELECT MAX(id) FROM {table}) IS NOT NULL)"
                )
        conn.commit()


# ---------------------------------------------------------------------------
# §7 matching, as an actual pgvector k-NN query -- the real version of
# matching.py's brute-force numpy scan, once cut over. Kept here (not
# wired into main.py yet) so the query shape from §7.1 is proven against
# a real pgvector index rather than just sketched in the architecture
# doc. Selects individual face embeddings, not cluster centroids, same
# reasoning as matching.py.
# ---------------------------------------------------------------------------

def find_matching_photos_pg(embedding: list[float], threshold: float, limit: int = 200) -> list[dict]:
    """
    k-NN query per §7.1: order every face by cosine distance to the
    selfie embedding, keep the ones within `threshold`, group by photo
    and keep each photo's best (closest) face, return ranked closest-
    first and capped at `limit`. `embedding <=> %s` is pgvector's cosine
    distance operator (vector_cosine_ops, matching the HNSW index type in
    the schema above).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (p.id)
                    p.id AS photo_id, p.source_file_id, f.label AS folder_label,
                    fc.embedding <=> %s::vector AS distance
                FROM faces fc
                JOIN photos p ON p.id = fc.photo_id
                JOIN folders f ON f.id = p.folder_id
                WHERE fc.embedding <=> %s::vector <= %s
                ORDER BY p.id, distance
                """,
                (embedding, embedding, threshold),
            )
            rows = cur.fetchall()
    ranked = sorted(rows, key=lambda r: r["distance"])
    return ranked[:limit]
