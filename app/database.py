"""
SQLite setup for the Church Photo Finder app.

Phase 1 only writes to the `folders` table. The other tables (photos, faces,
clusters) are created now, matching the data model in Section 6 of the spec,
so later phases (ingestion, clustering, matching) don't need a migration step.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "church_photos.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drive_folder_id TEXT NOT NULL UNIQUE,
    drive_folder_url TEXT NOT NULL,
    label TEXT NOT NULL,
    date_added TEXT NOT NULL DEFAULT (datetime('now')),
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | processing | paused | processed | error | cancelled
    processed_count INTEGER NOT NULL DEFAULT 0,
    total_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drive_file_id TEXT NOT NULL UNIQUE,
    folder_id INTEGER NOT NULL,
    date_added TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (folder_id) REFERENCES folders(id)
);

CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    bounding_box TEXT,  -- JSON-encoded [top, right, bottom, left]
    cluster_id INTEGER,
    FOREIGN KEY (photo_id) REFERENCES photos(id),
    FOREIGN KEY (cluster_id) REFERENCES clusters(id)
);

CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    centroid_embedding BLOB,
    face_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Who did what, for accountability once more than one admin touches
-- cluster corrections/folder management. `actor` is currently always
-- "admin" (single shared password, see app/auth.py) but the column
-- exists now so multi-admin auth can populate a real username later
-- without an audit_log migration.
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    actor TEXT NOT NULL DEFAULT 'admin',
    action TEXT NOT NULL,
    detail TEXT
);

-- Every one of these mirrors a query pattern already used throughout
-- this file (faces by cluster, faces by photo, photos by folder); with
-- no index, each of those degrades to a full table scan as the library
-- grows past a season or two of weekly photos.
CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces(cluster_id);
CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_photos_folder_id ON photos(folder_id);
"""


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        conn.commit()
        _run_migrations(conn)


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Lightweight, idempotent ALTER TABLE migrations for columns added
    after initial release. Older DB files created before `processed_count`
    /`total_count` existed just get them added on next startup; a fresh DB
    already has them from SCHEMA, so the ALTER is skipped either way
    without needing a separate migrations table."""
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(folders)").fetchall()}
    if "processed_count" not in existing_cols:
        conn.execute("ALTER TABLE folders ADD COLUMN processed_count INTEGER NOT NULL DEFAULT 0")
    if "total_count" not in existing_cols:
        conn.execute("ALTER TABLE folders ADD COLUMN total_count INTEGER NOT NULL DEFAULT 0")

    # Phase 7 §8.2: incremental clustering needs each cluster's current
    # member count to do a streaming centroid update (new_centroid =
    # (old_centroid * n + new_embedding) / (n + 1)) without re-reading
    # every member on each new face. A DB created before this existed
    # gets the column added and backfilled from the real face counts;
    # a fresh DB already has the column (default 0) from SCHEMA, and the
    # backfill below is a no-op since there are no clusters yet either way.
    cluster_cols = {row["name"] for row in conn.execute("PRAGMA table_info(clusters)").fetchall()}
    if "face_count" not in cluster_cols:
        conn.execute("ALTER TABLE clusters ADD COLUMN face_count INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            """
            UPDATE clusters SET face_count = (
                SELECT COUNT(*) FROM faces WHERE faces.cluster_id = clusters.id
            )
            """
        )
    conn.commit()


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def add_folder(drive_folder_id: str, drive_folder_url: str, label: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO folders (drive_folder_id, drive_folder_url, label) "
            "VALUES (?, ?, ?)",
            (drive_folder_id, drive_folder_url, label),
        )
        conn.commit()
        return cur.lastrowid


def folder_exists(drive_folder_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM folders WHERE drive_folder_id = ?", (drive_folder_id,)
        ).fetchone()
        return row is not None


def list_folders() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT
                f.*,
                COUNT(DISTINCT p.id) AS photo_count,
                COUNT(fc.id) AS face_count
            FROM folders f
            LEFT JOIN photos p ON p.folder_id = f.id
            LEFT JOIN faces fc ON fc.photo_id = p.id
            GROUP BY f.id
            ORDER BY f.date_added DESC
            """
        ).fetchall()


def get_folder(folder_id: int) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM folders WHERE id = ?", (folder_id,)
        ).fetchone()


def update_folder_status(folder_id: int, status: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE folders SET status = ? WHERE id = ?", (status, folder_id)
        )
        conn.commit()


def update_folder_progress(folder_id: int, processed_count: int, total_count: int) -> None:
    """Cheap progress counters so the admin UI can poll and show live
    progress/a progress bar during a long ingestion run, instead of the
    page looking frozen until the whole folder finishes."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE folders SET processed_count = ?, total_count = ? WHERE id = ?",
            (processed_count, total_count, folder_id),
        )
        conn.commit()


def reset_stuck_processing_folders() -> None:
    """Deprecated as of Phase 7 §9 — main.py's startup hook now calls
    app.ingest_queue.reconcile_stuck_folders() instead, which only resets
    a folder if there's no active RQ job for it (see that function's
    docstring). This unconditional version is *wrong* now that ingestion
    can legitimately outlive a web-process restart in a separate
    `rq worker` process: it would mark a folder "error" out from under a
    job that's still actually running. Left here only for any existing
    script/test that imports it directly; not called from the app
    anymore."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE folders SET status = 'error' WHERE status IN ('processing', 'paused')"
        )
        conn.commit()


def delete_folder(folder_id: int) -> None:
    """Delete a folder and everything under it: its photos, the faces
    detected in those photos, and any cluster assignments those faces
    held. Clusters themselves are left alone (a cluster can also contain
    faces from other folders); an admin should re-run "Recluster all
    faces" after a delete so cluster membership/centroids reflect the
    removal. Done as a single transaction so a crash mid-delete can't
    leave photos/faces orphaned from a folder row that no longer exists."""
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM faces WHERE photo_id IN (SELECT id FROM photos WHERE folder_id = ?)",
            (folder_id,),
        )
        conn.execute("DELETE FROM photos WHERE folder_id = ?", (folder_id,))
        conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
        conn.commit()


def photo_exists(drive_file_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM photos WHERE drive_file_id = ?", (drive_file_id,)
        ).fetchone()
        return row is not None


def add_photo(drive_file_id: str, folder_id: int) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO photos (drive_file_id, folder_id) VALUES (?, ?)",
            (drive_file_id, folder_id),
        )
        conn.commit()
        return cur.lastrowid


def add_face(photo_id: int, embedding: bytes, bounding_box: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO faces (photo_id, embedding, bounding_box) VALUES (?, ?, ?)",
            (photo_id, embedding, bounding_box),
        )
        conn.commit()
        return cur.lastrowid


def count_photos_for_folder(folder_id: int) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM photos WHERE folder_id = ?", (folder_id,)
        ).fetchone()
        return row["n"]


def count_faces_for_folder(folder_id: int) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM faces f "
            "JOIN photos p ON f.photo_id = p.id "
            "WHERE p.folder_id = ?",
            (folder_id,),
        ).fetchone()
        return row["n"]


# ---------------------------------------------------------------------------
# Clustering (Phase 3)
# ---------------------------------------------------------------------------

def all_faces_with_embeddings() -> list[sqlite3.Row]:
    """Every face row, for (re)clustering across the whole library."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, photo_id, embedding, cluster_id FROM faces"
        ).fetchall()


def create_cluster(centroid_embedding: bytes, face_count: int = 0) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO clusters (centroid_embedding, face_count) VALUES (?, ?)",
            (centroid_embedding, face_count),
        )
        conn.commit()
        return cur.lastrowid


def update_cluster_centroid(cluster_id: int, centroid_embedding: bytes, face_count: int | None = None) -> None:
    """Update a cluster's centroid, and optionally its face_count in the
    same write when the caller already knows the new true count (e.g.
    clustering.recompute_cluster_centroid, which reads every member
    anyway). face_count is left untouched when omitted."""
    with get_connection() as conn:
        if face_count is None:
            conn.execute(
                "UPDATE clusters SET centroid_embedding = ? WHERE id = ?",
                (centroid_embedding, cluster_id),
            )
        else:
            conn.execute(
                "UPDATE clusters SET centroid_embedding = ?, face_count = ? WHERE id = ?",
                (centroid_embedding, face_count, cluster_id),
            )
        conn.commit()


def delete_cluster(cluster_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM clusters WHERE id = ?", (cluster_id,))
        conn.commit()


def clear_all_face_cluster_assignments() -> None:
    """Null out every face's cluster_id. Must be called before deleting
    cluster rows (faces.cluster_id has a foreign key to clusters.id, so
    deleting a still-referenced cluster raises under PRAGMA foreign_keys=ON)."""
    with get_connection() as conn:
        conn.execute("UPDATE faces SET cluster_id = NULL")
        conn.commit()


def list_cluster_ids() -> list[int]:
    with get_connection() as conn:
        rows = conn.execute("SELECT id FROM clusters").fetchall()
        return [r["id"] for r in rows]


def record_audit_log(action: str, detail: str = "", actor: str = "admin") -> None:
    """Append one audit trail entry. Best-effort: a logging failure should
    never block the admin action it's describing, so callers don't need
    to wrap this in their own try/except."""
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO audit_log (actor, action, detail) VALUES (?, ?, ?)",
                (actor, action, detail),
            )
            conn.commit()
    except sqlite3.Error:
        pass


def list_audit_log(limit: int = 200) -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


def assign_face_cluster(face_id: int, cluster_id: int | None) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE faces SET cluster_id = ? WHERE id = ?", (cluster_id, face_id)
        )
        conn.commit()


def get_face_cluster(face_id: int) -> int | None:
    """The cluster a face currently belongs to (None if unclustered/missing),
    used by the admin move-face endpoint to know what to recompute after a
    manual move."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT cluster_id FROM faces WHERE id = ?", (face_id,)
        ).fetchone()
        return row["cluster_id"] if row else None


def cluster_exists(cluster_id: int) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM clusters WHERE id = ?", (cluster_id,)
        ).fetchone()
        return row is not None


def faces_for_cluster_with_embeddings(cluster_id: int) -> list[sqlite3.Row]:
    """Every embedding currently in one cluster, for recomputing its
    centroid after an admin manually moves a face in or out (see
    clustering.recompute_cluster_centroid)."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT embedding FROM faces WHERE cluster_id = ?", (cluster_id,)
        ).fetchall()


def get_cluster_centroid(cluster_id: int) -> bytes | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT centroid_embedding FROM clusters WHERE id = ?", (cluster_id,)
        ).fetchone()
        return row["centroid_embedding"] if row else None


def count_clusters() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM clusters").fetchone()
        return row["n"]


# ---------------------------------------------------------------------------
# Matching (Phase 5)
# ---------------------------------------------------------------------------

def all_faces_with_photo_info() -> list[sqlite3.Row]:
    """Every face's embedding plus its originating photo/folder, for the
    Phase 7 §7 k-NN matching path. Matching now searches individual face
    embeddings directly rather than cluster centroids, so — unlike the
    old candidate pool (list_clusters_with_centroid +
    list_unclustered_faces_with_embeddings) — this deliberately ignores
    cluster_id entirely and returns every face regardless of clustering
    state; a face that hasn't been (or was never) clustered can still be
    matched."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT fc.id AS face_id, fc.embedding, p.id AS photo_id,
                   p.drive_file_id, f.label AS folder_label
            FROM faces fc
            JOIN photos p ON p.id = fc.photo_id
            JOIN folders f ON f.id = p.folder_id
            """
        ).fetchall()


def list_clusters_with_centroid_and_count() -> list[sqlite3.Row]:
    """Every cluster with a centroid plus its current face_count, for the
    incremental clustering nearest-centroid fast path (Phase 7 §8.2) —
    the count is needed to do a streaming mean update without re-reading
    every member on each new face."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, centroid_embedding, face_count FROM clusters "
            "WHERE centroid_embedding IS NOT NULL"
        ).fetchall()


def faces_by_ids_with_embeddings(face_ids: list[int]) -> list[sqlite3.Row]:
    """A specific set of faces (by id) with their embeddings, for
    incremental clustering, which only needs to process the faces just
    added by the latest ingestion batch, not the whole library."""
    if not face_ids:
        return []
    with get_connection() as conn:
        placeholders = ",".join("?" for _ in face_ids)
        return conn.execute(
            f"SELECT id, embedding FROM faces WHERE id IN ({placeholders})",
            face_ids,
        ).fetchall()


def list_clusters_with_centroid() -> list[sqlite3.Row]:
    """Every cluster that actually has a centroid yet (id + embedding blob),
    for comparing a live selfie embedding against. A cluster row is only
    ever created by clustering.run_clustering() with a centroid already
    set, so in practice this is just 'all clusters' — the NOT NULL filter
    is defensive."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, centroid_embedding FROM clusters "
            "WHERE centroid_embedding IS NOT NULL"
        ).fetchall()


def get_photos_for_cluster(cluster_id: int) -> list[sqlite3.Row]:
    """Distinct photos (drive_file_id + originating folder label) that have
    at least one face assigned to this cluster. A person can appear more
    than once in the same photo (rare, but possible), hence DISTINCT."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT DISTINCT p.drive_file_id, p.id AS photo_id, f.label AS folder_label
            FROM faces fc
            JOIN photos p ON p.id = fc.photo_id
            JOIN folders f ON f.id = p.folder_id
            WHERE fc.cluster_id = ?
            ORDER BY p.id
            """,
            (cluster_id,),
        ).fetchall()


def list_clusters_summary() -> list[dict]:
    """
    For the admin cluster viewer: every cluster, how many distinct photos
    and faces it has, and one representative face (face id + drive_file_id
    + bounding_box + folder_label) to render as the card's clickable
    thumbnail. The full set of faces for a cluster is fetched on demand
    (see list_faces_for_cluster) only when that card is expanded, rather
    than loading every member's thumbnail up front for every cluster.

    The representative is per-face, not per-distinct-photo: a cluster's
    identity is "these individual faces look alike," and a whole-photo
    thumbnail can't show that when a photo has multiple people in it —
    cropping to the matched face itself (see /admin/face/{id}/thumb)
    makes the same-person claim actually checkable.
    """
    with get_connection() as conn:
        cluster_rows = conn.execute(
            "SELECT id FROM clusters ORDER BY id"
        ).fetchall()

        summaries = []
        for row in cluster_rows:
            cluster_id = row["id"]
            face_count = conn.execute(
                "SELECT COUNT(*) AS n FROM faces WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchone()["n"]
            photo_count = conn.execute(
                """
                SELECT COUNT(DISTINCT p.id) AS n
                FROM faces fc JOIN photos p ON p.id = fc.photo_id
                WHERE fc.cluster_id = ?
                """,
                (cluster_id,),
            ).fetchone()["n"]
            representative = conn.execute(
                """
                SELECT fc.id AS face_id, fc.bounding_box, p.drive_file_id, f.label AS folder_label
                FROM faces fc
                JOIN photos p ON p.id = fc.photo_id
                JOIN folders f ON f.id = p.folder_id
                WHERE fc.cluster_id = ?
                ORDER BY fc.id
                LIMIT 1
                """,
                (cluster_id,),
            ).fetchone()
            summaries.append(
                {
                    "id": cluster_id,
                    "face_count": face_count,
                    "photo_count": photo_count,
                    "representative": dict(representative) if representative else None,
                }
            )
        return summaries


def list_clusters_grouped_by_folder() -> list[dict]:
    """
    For the admin cluster viewer. Rather than one flat list of clusters
    numbered by their (continuously incrementing, never-reused) database
    id — which grows forever across every "Recluster all faces" run and
    is meaningless to an admin — this groups every face-group by the
    folder(s) it actually has faces in, and numbers them 1, 2, 3... *within
    each folder*, restarting for every folder.

    A folder's "Person #" numbering is a single sequence covering BOTH
    real multi-face clusters and still-unclustered ("missed match")
    faces together, ordered by first-seen face id — so every person an
    admin can see in a folder, matched or not, gets exactly one number
    and there's only one numbering system to keep in their head. A
    person who appears across several weeks' folders gets their own
    (possibly different) per-folder number under each folder they're in;
    `total_face_count` on a cluster entry says when it has members
    elsewhere too, so that isn't mistaken for the cluster's whole size.
    """
    with get_connection() as conn:
        folders = conn.execute(
            "SELECT id, label FROM folders ORDER BY date_added DESC"
        ).fetchall()

        result = []
        for folder in folders:
            folder_id = folder["id"]

            cluster_rows = conn.execute(
                """
                SELECT fc.cluster_id AS cluster_id, MIN(fc.id) AS first_face_id
                FROM faces fc
                JOIN photos p ON p.id = fc.photo_id
                WHERE p.folder_id = ? AND fc.cluster_id IS NOT NULL
                GROUP BY fc.cluster_id
                """,
                (folder_id,),
            ).fetchall()

            unclustered_rows = conn.execute(
                """
                SELECT fc.id AS face_id, fc.bounding_box, p.drive_file_id, f.label AS folder_label
                FROM faces fc
                JOIN photos p ON p.id = fc.photo_id
                JOIN folders f ON f.id = p.folder_id
                WHERE p.folder_id = ? AND fc.cluster_id IS NULL
                ORDER BY fc.id
                """,
                (folder_id,),
            ).fetchall()

            # One combined, first-seen-ordered sequence: real clusters and
            # still-unclustered faces interleaved by the face id that first
            # put them on this folder's radar, so "Person #N" advances the
            # same way an admin scrolling ingested photos in order would.
            pending = [
                {"kind": "cluster", "cluster_id": row["cluster_id"], "sort_key": row["first_face_id"]}
                for row in cluster_rows
            ] + [
                {"kind": "unclustered", "face_id": row["face_id"], "sort_key": row["face_id"]}
                for row in unclustered_rows
            ]
            pending.sort(key=lambda item: item["sort_key"])

            items = []
            for number, entry in enumerate(pending, start=1):
                if entry["kind"] == "cluster":
                    cluster_id = entry["cluster_id"]
                    face_count_in_folder = conn.execute(
                        """
                        SELECT COUNT(*) AS n FROM faces fc
                        JOIN photos p ON p.id = fc.photo_id
                        WHERE fc.cluster_id = ? AND p.folder_id = ?
                        """,
                        (cluster_id, folder_id),
                    ).fetchone()["n"]
                    total_face_count = conn.execute(
                        "SELECT COUNT(*) AS n FROM faces WHERE cluster_id = ?",
                        (cluster_id,),
                    ).fetchone()["n"]
                    representative = conn.execute(
                        """
                        SELECT fc.id AS face_id, fc.bounding_box, p.drive_file_id, f.label AS folder_label
                        FROM faces fc
                        JOIN photos p ON p.id = fc.photo_id
                        JOIN folders f ON f.id = p.folder_id
                        WHERE fc.cluster_id = ? AND p.folder_id = ?
                        ORDER BY fc.id
                        LIMIT 1
                        """,
                        (cluster_id, folder_id),
                    ).fetchone()
                    items.append(
                        {
                            "type": "cluster",
                            "number": number,
                            "cluster_id": cluster_id,
                            "face_count_in_folder": face_count_in_folder,
                            "total_face_count": total_face_count,
                            "representative": dict(representative) if representative else None,
                        }
                    )
                else:
                    face_row = next(r for r in unclustered_rows if r["face_id"] == entry["face_id"])
                    items.append(
                        {
                            "type": "unclustered",
                            "number": number,
                            "face_id": face_row["face_id"],
                            "bounding_box": face_row["bounding_box"],
                            "drive_file_id": face_row["drive_file_id"],
                            "folder_label": face_row["folder_label"],
                        }
                    )

            result.append(
                {
                    "folder_id": folder_id,
                    "folder_label": folder["label"],
                    "cards": items,
                }
            )
        return result


def create_cluster_from_faces(face_ids: list[int]) -> int:
    """Create a brand-new cluster containing exactly the given faces, for
    the admin "group these into one person" action (mainly for pulling
    several 'not yet matched' faces together once they're visibly the
    same person, without needing an existing cluster to move them into).
    Centroid is left NULL here; the caller recomputes it from the actual
    member embeddings (clustering.recompute_cluster_centroid) right
    after, same as a normal DBSCAN-formed cluster."""
    cluster_id = create_cluster(None)
    for face_id in face_ids:
        assign_face_cluster(face_id, cluster_id)
    return cluster_id


def list_faces_for_cluster(cluster_id: int) -> list[dict]:
    """Every face in one cluster (no limit), for the admin cluster viewer's
    expand-on-click lightbox — fetched only when a card is actually
    opened, so the initial page load stays light regardless of cluster
    size."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT fc.id AS face_id, fc.bounding_box, p.drive_file_id, f.label AS folder_label
            FROM faces fc
            JOIN photos p ON p.id = fc.photo_id
            JOIN folders f ON f.id = p.folder_id
            WHERE fc.cluster_id = ?
            ORDER BY fc.id
            """,
            (cluster_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_unclustered_faces(limit: int = 200) -> list[dict]:
    """Every face DBSCAN left as "noise" (cluster_id IS NULL), each shown
    on the admin cluster viewer as its own single-photo card — most
    commonly a person who only appears in one photo so far (DBSCAN can't
    form a cluster of one given min_samples=2), which is expected and not
    an error, but still worth being able to see rather than just a count."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT fc.id AS face_id, fc.bounding_box, p.drive_file_id, f.label AS folder_label
            FROM faces fc
            JOIN photos p ON p.id = fc.photo_id
            JOIN folders f ON f.id = p.folder_id
            WHERE fc.cluster_id IS NULL
            ORDER BY fc.id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_unclustered_faces_with_embeddings() -> list[sqlite3.Row]:
    """Every face DBSCAN left as "noise" (cluster_id IS NULL), with its
    embedding and originating photo, for the live-selfie matching path
    (Phase 5). These are people who only have a single ingested photo so
    far — DBSCAN can't form a real 2+-member cluster for them (see
    list_unclustered_faces), so they'd otherwise never appear in
    list_clusters_with_centroid() and could never be matched by
    /find/capture no matter how good the selfie is. matching.py treats
    each of these as its own singleton "cluster of one" candidate,
    compared directly by embedding distance (no centroid to average,
    since there's only the one embedding) alongside real clusters."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT fc.id AS face_id, fc.embedding, p.drive_file_id, f.label AS folder_label
            FROM faces fc
            JOIN photos p ON p.id = fc.photo_id
            JOIN folders f ON f.id = p.folder_id
            WHERE fc.cluster_id IS NULL
            ORDER BY fc.id
            """
        ).fetchall()


def count_unclustered_faces() -> int:
    """Faces detected during ingestion that don't belong to any cluster
    yet — either clustering hasn't been run since they were added, or
    DBSCAN treated them as noise (a face that didn't match closely enough
    with any other face to form/join a cluster). Surfaced in the admin
    cluster viewer so an admin isn't left wondering where those photos
    went."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM faces WHERE cluster_id IS NULL"
        ).fetchone()
        return row["n"]


def get_face_with_photo(face_id: int) -> sqlite3.Row | None:
    """One face's bounding_box plus its photo's drive_file_id and folder
    label, for cropping a face-only thumbnail (see main.admin_face_thumbnail)
    and for the unclustered-face lightbox endpoint."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT fc.id AS face_id, fc.bounding_box, p.drive_file_id, f.label AS folder_label
            FROM faces fc
            JOIN photos p ON p.id = fc.photo_id
            JOIN folders f ON f.id = p.folder_id
            WHERE fc.id = ?
            """,
            (face_id,),
        ).fetchone()