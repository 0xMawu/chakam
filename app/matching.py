"""
Phase 7 §7: match a single live-selfie embedding against every individual
face embedding in the library via k-NN search, then group by photo — not
against cluster centroids (the old Phase 5 approach; see git history /
PHASE7_SCALE_ARCHITECTURE.md §3, §7 for why this changed).

Same invariants as before: deliberately stateless / single-embedding-in,
result-out. No DB writes happen here, only reads. Nothing about the
member's selfie or its embedding is ever persisted (see find_capture in
main.py — the embedding lives only for the duration of the request).

Why this replaced the old cluster-centroid approach:
  Matching against cluster centroids meant a person split across two
  clusters (a haircut, new glasses, a few photos with unusual lighting)
  would only ever match ONE of those clusters — the other cluster's
  photos were silently unreachable, no matter how good the selfie was.
  Searching individual face embeddings directly and then grouping by
  photo removes that failure mode entirely: matching no longer depends
  on clustering being correct, or even having run recently. Clustering
  (clustering.py) still exists and still matters for the admin "People"
  view, it's just off the matching critical path now.

This module still does a brute-force numpy scan over every face in the
library (SQLite has no native vector index) rather than a real ANN index
— see PHASE7_SCALE_ARCHITECTURE.md §6 for the pgvector/HNSW version of
this once the DB migrates. Vectorized with numpy rather than the old
Python per-candidate loop, which matters more now since the candidate
pool is every face, not just cluster count + singleton count.
"""
import logging
import os
from dataclasses import dataclass

import numpy as np

from app import database
from app.face_processing import blob_to_embedding

logger = logging.getLogger(__name__)

# Euclidean distance threshold for a match. This now compares directly
# to individual face embeddings (not cluster centroids), so — unlike
# before — it no longer needs to stay aligned with CLUSTER_EPS to make
# sense; it's simply "how close must two face embeddings be to call them
# the same person." Configurable via env var for post-launch tuning
# without a code change, same pattern as CLUSTER_EPS/CLUSTER_MIN_SAMPLES
# in clustering.py.
#
# **Stale default, needs re-tuning (Phase 7 §4/§13):** 0.4 was tuned
# against the old 128-d dlib/face_recognition embedding space. Phase 7
# §4 swapped in ArcFace-class 512-d embeddings (see
# face_processing.py's module docstring) — a different embedding space
# with different "same person" distance statistics. Confirmed by a real
# smoke test against this repo's bundled photos (buffalo_l's models
# downloaded fine in that environment): two different people in the
# same group photo landed at distance ~1.26-1.41, well above this
# threshold — so 0.4 is not just "maybe wrong," it would currently
# reject every real match, same-person pairs included, since ArcFace's
# L2-normalized vectors run in roughly a [0, 2] range rather than
# dlib's. Do not trust this default for real matching until §13's
# labeled eval set verifies a real threshold against real church
# photos (same-person pairs, not just the different-person distances
# noted above).
#
# **Updated by §13 tooling (scripts/build_eval_set.py +
# scripts/tune_threshold.py), still provisional.** Running that
# pipeline for real (real insightface buffalo_l model, real re-embed of
# this repo's cached face crops) confirms the distance range noted
# above: 3 different-person pairs at 1.274-1.378. But this repo's bundled
# data only has ONE cached face crop per cluster for every cluster with
# 2+ faces, so zero same-person pairs could be re-embedded — see
# tune_threshold.py's output for why that means no real threshold can be
# validated from this environment alone (an upper bound only, not a
# lower one). 0.9 below is a deliberately conservative interim value —
# comfortably under the observed different-person floor of ~1.27, on the
# assumption (not yet verified) that ArcFace same-person distances for
# reasonable photos land well under 1.0 — chosen so real deployment
# testing starts from "probably too strict" rather than "silently
# rejects everything," not because it's been validated. **Still do not
# trust this for real users**: re-run scripts/build_eval_set.py once an
# admin has confirmed/corrected real clusters in the People view (so
# same-person pairs actually exist), then scripts/tune_threshold.py, and
# update this value from that output.
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.9"))

# Cap on how many photos a single selfie match returns, per
# PHASE7_SCALE_ARCHITECTURE.md §7.1 step 4 ("cap the result count for UI
# sanity ... but don't cap it at one cluster's worth"). Configurable so
# it can be tuned independently of the distance threshold.
MAX_MATCHED_PHOTOS = int(os.environ.get("MAX_MATCHED_PHOTOS", "200"))


@dataclass
class MatchedPhoto:
    drive_file_id: str
    folder_label: str
    distance: float


def find_matching_photos(selfie_embedding: np.ndarray) -> list[MatchedPhoto]:
    """
    Compare one selfie embedding against every face embedding currently
    in the library, keep the faces within MATCH_THRESHOLD, then group by
    photo and keep each photo's single best (closest) face distance —
    per PHASE7_SCALE_ARCHITECTURE.md §7.1. Returns EVERY qualifying
    photo, ranked closest-first and capped at MAX_MATCHED_PHOTOS, not
    just one candidate's worth the way the old cluster-centroid approach
    did.

    Returns an empty list if the library has no faces yet, or none are
    within threshold.
    """
    face_rows = database.all_faces_with_photo_info()
    if not face_rows:
        return []

    embeddings = np.stack([blob_to_embedding(row["embedding"]) for row in face_rows])
    distances = np.linalg.norm(embeddings - selfie_embedding, axis=1)

    within_threshold = distances <= MATCH_THRESHOLD
    if not np.any(within_threshold):
        logger.info(
            "No face within threshold (closest distance=%.3f, threshold=%.3f)",
            float(distances.min()),
            MATCH_THRESHOLD,
        )
        return []

    # Group by photo_id, keeping each photo's closest (best) face
    # distance — a photo can have more than one face close enough (e.g.
    # the member appears twice, or two similar-looking people), and a
    # photo should only be returned once either way.
    best_per_photo: dict[int, dict] = {}
    for row, distance, is_match in zip(face_rows, distances, within_threshold):
        if not is_match:
            continue
        photo_id = row["photo_id"]
        existing = best_per_photo.get(photo_id)
        distance = float(distance)
        if existing is None or distance < existing["distance"]:
            best_per_photo[photo_id] = {
                "distance": distance,
                "drive_file_id": row["drive_file_id"],
                "folder_label": row["folder_label"],
            }

    ranked = sorted(best_per_photo.values(), key=lambda item: item["distance"])
    capped = ranked[:MAX_MATCHED_PHOTOS]

    logger.info(
        "Selfie matched %d photo(s) out of %d candidate(s) within threshold "
        "(closest distance=%.3f, threshold=%.3f)%s",
        len(capped),
        len(ranked),
        ranked[0]["distance"],
        MATCH_THRESHOLD,
        " [capped]" if len(ranked) > MAX_MATCHED_PHOTOS else "",
    )

    return [
        MatchedPhoto(
            drive_file_id=item["drive_file_id"],
            folder_label=item["folder_label"],
            distance=item["distance"],
        )
        for item in capped
    ]
