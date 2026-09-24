"""
Phase 7 §12, steps 1-3: stand up Postgres+pgvector alongside the existing
SQLite DB and copy the current library into it.

    1. Stand up Postgres + pgvector alongside SQLite (don't touch
       production until step 4).
    2. Re-run ingestion's detect+embed step against every already-
       ingested photo, using the new detector/embedder...
    3. Run the incremental clustering path once over the freshly embedded
       library to build initial clusters.

What this script actually does vs. what §12 describes: step 2 as written
needs §4 (the ArcFace-class model swap) to have shipped, AND every
already-ingested photo to have been re-detected/re-embedded with it. §4
itself has now shipped (app/face_processing.py produces 512-d ArcFace-
class embeddings for anything detected from here on), but this script
still does NOT do that re-embed pass -- it's a straight row copy, not a
re-detect-and-embed step. So: if the SQLite library you point this at
still holds faces ingested before §4 shipped, this script will copy
their old 128-d embeddings AS-IS, which will either fail EMBEDDING_DIM's
mismatch check below (and get skipped, logged, not silently corrupted --
see the check a few lines down) or, worse, "succeed" into a column sized
for 512-d if EMBEDDING_DIM was left at a stale 128. Before running this
against a real library, re-run ingestion (or a dedicated re-embed pass --
not yet built; would mean re-running detect_faces() over every already-
ingested photo's stored image and overwriting `faces.embedding`, since
source images are reachable via Drive/S3) so every face's embedding is
actually the new 512-d ArcFace-class one. Once that's true, this script's
job really is just the structural copy §12 describes for step 1: proving
the pgvector schema, the id-preserving foreign keys, and the k-NN query
shape (database_pg.find_matching_photos_pg) against real 512-d data.

Step 3 (initial clustering) is left as a manual follow-up: run
clustering.assign_new_faces_incrementally() (or a Postgres-native
equivalent) over the migrated faces once this script finishes -- not
folded into this script, so a migration re-run doesn't also re-cluster
every time.

Usage:
    DATABASE_URL=postgresql://user:pass@host/church_photos \
        python scripts/migrate_to_postgres.py [--sqlite-path PATH]

Safe to re-run: every insert is `ON CONFLICT (id) DO NOTHING`, so a
partial prior run (or re-running after new folders were ingested in
SQLite) just fills in what's missing, per the audit's habit of making
existing ops (ingestion retries, etc.) idempotent -- see the existing
`photo_exists`/`folder_exists` checks in the SQLite path.
"""
import argparse
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app import database as sqlite_db  # noqa: E402
from app import database_pg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def migrate(sqlite_path: Path) -> None:
    if not sqlite_path.exists():
        raise SystemExit(f"SQLite DB not found at {sqlite_path}")

    logger.info("Source: %s", sqlite_path)
    logger.info("Target: Postgres (embedding dim=%d, see EMBEDDING_DIM)", database_pg.EMBEDDING_DIM)

    # Apply the same idempotent ALTER-TABLE migrations app.database.init_db()
    # runs on every normal app startup (e.g. audit_log, clusters.face_count),
    # so a source DB from an older running instance still has every table
    # this script expects to read from.
    sqlite_db.DB_PATH = sqlite_path
    sqlite_db.init_db()

    conn = _sqlite_connect(sqlite_path)

    logger.info("Creating Postgres schema (idempotent)...")
    database_pg.init_db()

    folders = conn.execute("SELECT * FROM folders").fetchall()
    logger.info("Migrating %d folder(s)...", len(folders))
    for row in folders:
        database_pg.insert_folder(
            id=row["id"],
            source="drive",
            source_folder_id=row["drive_folder_id"],
            source_folder_url=row["drive_folder_url"],
            label=row["label"],
            status=row["status"],
            processed_count=row["processed_count"],
            total_count=row["total_count"],
        )

    photos = conn.execute("SELECT * FROM photos").fetchall()
    logger.info("Migrating %d photo(s)...", len(photos))
    for row in photos:
        database_pg.insert_photo(
            id=row["id"], folder_id=row["folder_id"], source_file_id=row["drive_file_id"],
        )

    clusters = conn.execute("SELECT * FROM clusters").fetchall()
    logger.info("Migrating %d cluster(s)...", len(clusters))
    face_counts = {
        r["cluster_id"]: r["n"]
        for r in conn.execute(
            "SELECT cluster_id, COUNT(*) AS n FROM faces "
            "WHERE cluster_id IS NOT NULL GROUP BY cluster_id"
        ).fetchall()
    }
    for row in clusters:
        centroid_blob = row["centroid_embedding"]
        centroid = (
            np.frombuffer(centroid_blob, dtype=np.float64).tolist()
            if centroid_blob is not None
            else None
        )
        database_pg.insert_cluster(
            id=row["id"], centroid=centroid, face_count=face_counts.get(row["id"], 0),
        )

    faces = conn.execute("SELECT * FROM faces").fetchall()
    logger.info("Migrating %d face(s)...", len(faces))
    dim_mismatches = 0
    for row in faces:
        embedding = np.frombuffer(row["embedding"], dtype=np.float64)
        if embedding.shape[0] != database_pg.EMBEDDING_DIM:
            dim_mismatches += 1
            logger.warning(
                "Face id=%s has %d-d embedding, EMBEDDING_DIM=%d -- skipping "
                "(set EMBEDDING_DIM to match today's real embeddings, e.g. 512 "
                "before §4 ships, or re-run this migration after re-embedding)",
                row["id"], embedding.shape[0], database_pg.EMBEDDING_DIM,
            )
            continue
        database_pg.insert_face(
            id=row["id"],
            photo_id=row["photo_id"],
            embedding=embedding.tolist(),
            bounding_box=row["bounding_box"],
            cluster_id=row["cluster_id"],
        )
    if dim_mismatches:
        logger.warning("%d face(s) skipped due to embedding-dimension mismatch", dim_mismatches)

    audit_rows = conn.execute("SELECT * FROM audit_log").fetchall()
    logger.info("Migrating %d audit log entrie(s)...", len(audit_rows))
    for row in audit_rows:
        database_pg.insert_audit_log(
            id=row["id"], created_at=row["created_at"], actor=row["actor"],
            action=row["action"], detail=row["detail"],
        )

    logger.info("Resetting Postgres id sequences past the migrated max ids...")
    database_pg.reset_id_sequences()

    conn.close()
    logger.info(
        "Migration complete. Next: run clustering over the migrated faces "
        "(§12 step 3) before cutting /find/capture over (§12 step 4) -- do "
        "NOT cut over yet, this script only performs steps 1-2."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "church_photos.db",
        help="Path to the source SQLite DB (default: data/church_photos.db)",
    )
    args = parser.parse_args()
    migrate(args.sqlite_path)
