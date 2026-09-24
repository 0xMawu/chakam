"""
Phase 3 / Phase 7 §8: group `faces` rows into per-person clusters.

Two paths now exist, matching PHASE7_SCALE_ARCHITECTURE.md §8.2 (written
against this SQLite schema, not the future pgvector one — same algorithm,
just a numpy brute-force scan standing in for a real ANN index at this
scale):

- **`run_clustering()` — full re-cluster.** Reads every face embedding in
  the library, re-runs DBSCAN from scratch, and rewrites `clusters` +
  every face's `cluster_id` to match. Trivially correct (it's just "what
  would DBSCAN say given everything we know right now"), but O(library
  size) every time it runs — meant to run periodically (admin-triggered
  "Recluster all faces", or a scheduled job later), not on every
  ingestion batch.
- **`assign_new_faces_incrementally()` — cheap fast path.** Runs after
  every ingestion batch (see ingestion.process_folder). For each new
  face: nearest-centroid lookup against existing clusters (join if within
  CLUSTER_EPS, updating the centroid with a streaming mean); otherwise
  try pairing with another currently-unclustered face; otherwise left
  unclustered. Cheap, but can miss what a full DBSCAN pass would catch
  (e.g. a new face that should bridge — and merge — two previously-
  separate clusters, or split one that no longer belongs together). It's
  a fast approximation between full re-clusters, not a replacement for
  them.

Note this used to also matter for member-facing matching (Phase 4/5
compared a selfie against cluster centroids). As of Phase 7 §7, matching
searches individual face embeddings directly (see matching.py) and no
longer depends on clustering at all — clustering now only feeds the admin
"People" / cluster-correction view. A person split across two clusters
here no longer means they lose half their matched photos; it's purely an
admin-UI/grouping-quality question now, not a member-facing correctness
one.

Cluster **IDs are still not stable** across a full re-cluster (a person's
cluster id can change week to week) — same as before, and still fine:
spec Section 6 says clusters carry no name/identity field, and matching
no longer reads cluster ids at all.

Tuning: `eps` (max distance to be considered the same cluster) and
`min_samples` are the two DBSCAN knobs. Defaults below favor precision
over recall per spec Section 7 ("better to occasionally miss a photo
than show someone another person's photos") — a tighter `eps` means
faces need to look more alike to be grouped together. Both are
configurable via env vars so they can be tuned after real-world testing,
same spirit as spec Section 7's note on the matching threshold.
"""
import logging
import os

import numpy as np
from sklearn.cluster import DBSCAN

from app import database
from app.face_processing import blob_to_embedding, embedding_to_blob

logger = logging.getLogger(__name__)

# History: 0.5 -> 0.42 -> 0.38 tuned against the old 128-d dlib/
# face_recognition embedding space (see git history), each drop in
# response to real testing showing different people still merging into
# one cluster.
#
# **Stale default, needs re-tuning (Phase 7 §4/§13):** Phase 7 §4
# swapped in ArcFace-class 512-d embeddings (see face_processing.py's
# module docstring) — a different embedding space with different "same
# person" distance statistics than the dlib space this value was tuned
# against, so 0.38 is not just "maybe wrong" now. A real smoke test
# against this repo's bundled photos (buffalo_l's models downloaded
# fine in that environment) put two different people from the same
# group photo at distance ~1.26-1.41 apart, well above 0.38 — so at
# this default essentially nothing will cluster together, same-person
# faces included, since ArcFace's L2-normalized embeddings run in
# roughly a [0, 2] range rather than dlib's. Treat this as a
# placeholder until §13's labeled eval set verifies a real value
# against real church photos (same-person pairs specifically, not just
# the different-person distances noted above); tune the same way as
# before — small steps, re-run "Recluster all faces" to see the effect
# immediately, drop further if different people still merge, raise
# slightly if the same person keeps splitting into separate clusters.
# Configurable via env var for further tuning without a code change.
#
# **Updated by §13 tooling, still provisional — same caveat as
# matching.MATCH_THRESHOLD** (see that constant's comment for what was
# actually verified: real different-person distances of 1.274-1.378 from
# a real re-embed pass, but zero same-person pairs available in this
# repo's bundled data to validate a lower bound). 0.9 is the same
# deliberately-conservative interim value as MATCH_THRESHOLD, not a
# separately-tuned number — re-run scripts/tune_threshold.py once real
# same-person pairs exist and set this independently if the two ever
# need to diverge (see matching.py's module docstring for why they no
# longer have to move together).
CLUSTER_EPS = float(os.environ.get("CLUSTER_EPS", "0.9"))
CLUSTER_MIN_SAMPLES = int(os.environ.get("CLUSTER_MIN_SAMPLES", "2"))


def run_clustering() -> dict:
    """
    Re-cluster every face in the library. Returns a small summary dict
    (useful for the admin UI / logs): counts of faces, clusters formed,
    and faces left unclustered (DBSCAN "noise" — usually a face that only
    appears once so far, or one too dissimilar from everything else).
    """
    face_rows = database.all_faces_with_embeddings()

    if not face_rows:
        _clear_all_clusters()
        return {"faces": 0, "clusters": 0, "unclustered": 0}

    face_ids = [row["id"] for row in face_rows]
    embeddings = np.stack([blob_to_embedding(row["embedding"]) for row in face_rows])

    labels = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES, metric="euclidean").fit_predict(
        embeddings
    )

    _clear_all_clusters()

    unclustered = 0
    cluster_count = 0
    for label in sorted(set(labels)):
        member_mask = labels == label
        member_embeddings = embeddings[member_mask]
        member_face_ids = [fid for fid, is_member in zip(face_ids, member_mask) if is_member]

        if label == -1:
            # DBSCAN "noise": doesn't meet min_samples with any neighbor yet.
            # Left with cluster_id = NULL rather than forced into a cluster,
            # per the precision-over-recall preference in spec Section 7.
            for face_id in member_face_ids:
                database.assign_face_cluster(face_id, None)
            unclustered += len(member_face_ids)
            continue

        centroid = member_embeddings.mean(axis=0)
        cluster_id = database.create_cluster(embedding_to_blob(centroid), face_count=len(member_face_ids))
        for face_id in member_face_ids:
            database.assign_face_cluster(face_id, cluster_id)
        cluster_count += 1

    logger.info(
        "Clustering complete: %d faces -> %d clusters (%d unclustered)",
        len(face_ids),
        cluster_count,
        unclustered,
    )
    return {"faces": len(face_ids), "clusters": cluster_count, "unclustered": unclustered}


def recompute_cluster_centroid(cluster_id: int) -> None:
    """
    Recalculate one cluster's centroid from its current members. Called
    after an admin manually moves a face into or out of a cluster (see
    main.admin_set_face_cluster), so matching against that cluster stays
    accurate without needing a full "Recluster all faces" run.

    If the move emptied the cluster out entirely, the cluster row is
    deleted rather than left behind with a now-meaningless centroid and
    zero members — otherwise it would linger as a phantom empty card.
    """
    rows = database.faces_for_cluster_with_embeddings(cluster_id)
    if not rows:
        database.delete_cluster(cluster_id)
        return
    embeddings = np.stack([blob_to_embedding(row["embedding"]) for row in rows])
    centroid = embeddings.mean(axis=0)
    database.update_cluster_centroid(cluster_id, embedding_to_blob(centroid), face_count=len(rows))


def assign_new_faces_incrementally(face_ids: list[int]) -> dict:
    """
    Phase 7 §8.2 fast path: cluster only the faces a single ingestion
    batch just added, instead of re-running DBSCAN over the whole
    library. For each new face, in order:

      1. Nearest-centroid lookup against every existing cluster. If the
         closest centroid is within CLUSTER_EPS, join that cluster and
         update its centroid with a streaming mean —
         `(old_centroid * n + new_embedding) / (n + 1)` — using the
         cluster's stored face_count, so this never needs to re-read
         every existing member.
      2. Otherwise, check the current pool of unclustered ("noise")
         faces. If the closest one is within CLUSTER_EPS, the two form a
         brand-new 2-member cluster — this is what DBSCAN would do the
         moment a second sample lands near a previously-lonely point,
         and without it, two photos of a new person taken seconds apart
         in the same batch would otherwise sit unclustered indefinitely
         until the next full re-cluster.
      3. Otherwise, left unclustered (cluster_id stays NULL) and added to
         the in-batch unclustered pool, so a later face in this same
         batch can still pair with it per step 2.

    This intentionally is NOT a substitute for run_clustering(): it can't
    detect a new face that should merge two previously-separate clusters
    (a "bridging" point) or split one that shouldn't have been merged.
    Those cases are still caught by the periodic full re-cluster, which
    should keep running on a schedule (or via the existing admin
    "Recluster all faces" button) regardless of how often this runs.
    """
    if not face_ids:
        return {"assigned_to_existing": 0, "new_clusters": 0, "still_unclustered": 0}

    face_rows = database.faces_by_ids_with_embeddings(face_ids)

    cluster_pool = [
        {
            "id": row["id"],
            "centroid": blob_to_embedding(row["centroid_embedding"]),
            "face_count": row["face_count"],
        }
        for row in database.list_clusters_with_centroid_and_count()
    ]

    batch_face_id_set = set(face_ids)
    unclustered_pool = [
        {"face_id": row["face_id"], "embedding": blob_to_embedding(row["embedding"])}
        for row in database.list_unclustered_faces_with_embeddings()
        if row["face_id"] not in batch_face_id_set
    ]

    assigned_to_existing = 0
    new_clusters = 0
    still_unclustered = 0

    for face_row in face_rows:
        face_id = face_row["id"]
        embedding = blob_to_embedding(face_row["embedding"])

        # Step 1: nearest existing cluster centroid.
        best_cluster = None
        best_cluster_distance = None
        for candidate in cluster_pool:
            distance = float(np.linalg.norm(embedding - candidate["centroid"]))
            if best_cluster_distance is None or distance < best_cluster_distance:
                best_cluster_distance = distance
                best_cluster = candidate

        if best_cluster is not None and best_cluster_distance <= CLUSTER_EPS:
            n = best_cluster["face_count"]
            new_centroid = (best_cluster["centroid"] * n + embedding) / (n + 1)
            database.assign_face_cluster(face_id, best_cluster["id"])
            database.update_cluster_centroid(
                best_cluster["id"], embedding_to_blob(new_centroid), face_count=n + 1
            )
            best_cluster["centroid"] = new_centroid
            best_cluster["face_count"] = n + 1
            assigned_to_existing += 1
            continue

        # Step 2: pair with another currently-unclustered face.
        best_partner_idx = None
        best_partner_distance = None
        for idx, other in enumerate(unclustered_pool):
            distance = float(np.linalg.norm(embedding - other["embedding"]))
            if best_partner_distance is None or distance < best_partner_distance:
                best_partner_distance = distance
                best_partner_idx = idx

        if best_partner_idx is not None and best_partner_distance <= CLUSTER_EPS:
            partner = unclustered_pool.pop(best_partner_idx)
            centroid = (embedding + partner["embedding"]) / 2
            cluster_id = database.create_cluster(embedding_to_blob(centroid), face_count=2)
            database.assign_face_cluster(face_id, cluster_id)
            database.assign_face_cluster(partner["face_id"], cluster_id)
            cluster_pool.append({"id": cluster_id, "centroid": centroid, "face_count": 2})
            new_clusters += 1
            continue

        # Step 3: still unclustered, but available for a later face in
        # this same batch to pair with.
        unclustered_pool.append({"face_id": face_id, "embedding": embedding})
        still_unclustered += 1

    logger.info(
        "Incremental clustering: %d new face(s) -> %d joined existing clusters, "
        "%d new clusters formed, %d left unclustered",
        len(face_rows), assigned_to_existing, new_clusters, still_unclustered,
    )
    return {
        "assigned_to_existing": assigned_to_existing,
        "new_clusters": new_clusters,
        "still_unclustered": still_unclustered,
    }


def _clear_all_clusters() -> None:
    """Drop every existing cluster row so a run with fewer clusters than
    last time doesn't leave orphaned centroid rows behind. Faces'
    cluster_id must be nulled first: it's a foreign key into clusters,
    so deleting a still-referenced cluster row would violate the
    constraint under PRAGMA foreign_keys=ON."""
    database.clear_all_face_cluster_assignments()
    for cluster_id in database.list_cluster_ids():
        database.delete_cluster(cluster_id)
