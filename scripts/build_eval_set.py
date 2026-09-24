"""
Phase 7 §13: build a labeled eval set of (face_a, face_b, same_person)
pairs, re-embedded with the current app/face_processing.py pipeline
(insightface buffalo_l, 512-d), then hand it to tune_threshold.py.

Where the labels come from
---------------------------
There is no separately-collected "ground truth" dataset for this app --
the only existing signal is the admin's own cluster corrections in
data/church_photos.db (app/clustering.py + the admin "People" UI). This
script treats those as the labels:
  - two faces in the SAME cluster_id  -> same_person = True
  - two faces in DIFFERENT cluster_ids -> same_person = False
Unclustered faces (cluster_id IS NULL) are excluded -- they were never
reviewed by an admin, so they're not a trustworthy label either way.

This is provisional, not a real labeled eval set: those clusters were
formed by DBSCAN against the OLD 128-d dlib embedding space and, per the
admin UI, may never have been manually reviewed/corrected at all in a
given deployment. Treat this script's output as a smoke test that the
tuning pipeline works end-to-end and a *starting* threshold, not a
substitute for a real admin pass (confirm/merge/split clusters in the
"People" view) once real church photos have been ingested with the new
embedding pipeline. See tune_threshold.py's docstring and README §13 for
what's still missing.

Only faces that have a cached crop image (data/face_thumb_cache/{id}.jpg)
can be re-embedded -- the original Drive photos aren't available outside
a real deployment. Faces without a cached crop are skipped and reported.

Usage:
    python scripts/build_eval_set.py [--out data/eval_pairs.json]
"""
import argparse
import json
import sqlite3
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.face_processing import detect_faces  # noqa: E402

DB_PATH = Path(__file__).parent.parent / "data" / "church_photos.db"
FACE_THUMB_DIR = Path(__file__).parent.parent / "data" / "face_thumb_cache"


def load_clustered_faces() -> dict[int, list[int]]:
    """cluster_id -> [face_id, ...], excluding unclustered faces."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    rows = cur.execute(
        "SELECT id, cluster_id FROM faces WHERE cluster_id IS NOT NULL "
        "ORDER BY cluster_id, id"
    ).fetchall()
    con.close()
    by_cluster: dict[int, list[int]] = {}
    for face_id, cluster_id in rows:
        by_cluster.setdefault(cluster_id, []).append(face_id)
    return by_cluster


def re_embed_face(face_id: int) -> list[float] | None:
    """Re-run the *current* (buffalo_l) pipeline on a cached face crop.
    Returns None (and the caller logs it) if no crop is cached or no
    face is (re-)detected in it -- a crop tight enough for the old
    dlib pipeline can occasionally fail SCRFD's stricter detector."""
    path = FACE_THUMB_DIR / f"{face_id}.jpg"
    if not path.exists():
        return None
    faces = detect_faces(path.read_bytes())
    if not faces:
        return None
    # A face crop should have exactly one face; if SCRFD finds more than
    # one (e.g. a second person at the edge of a loose crop), take the
    # largest -- same tie-break ingestion.py uses.
    faces.sort(
        key=lambda f: (f.bounding_box[1] - f.bounding_box[3])
        * (f.bounding_box[2] - f.bounding_box[0]),
        reverse=True,
    )
    return faces[0].embedding.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(Path(__file__).parent.parent / "data" / "eval_pairs.json")
    )
    args = parser.parse_args()

    by_cluster = load_clustered_faces()
    all_face_ids = sorted({fid for ids in by_cluster.values() for fid in ids})

    print(f"Clustered faces in DB: {len(all_face_ids)} across {len(by_cluster)} clusters")

    embeddings: dict[int, list[float]] = {}
    skipped = []
    for face_id in all_face_ids:
        emb = re_embed_face(face_id)
        if emb is None:
            skipped.append(face_id)
        else:
            embeddings[face_id] = emb

    print(f"Re-embedded {len(embeddings)} face(s); skipped {len(skipped)} (no cached "
          f"crop or no face re-detected): {skipped}")

    pairs = []
    cluster_ids = list(by_cluster.keys())

    # Same-person pairs: every pair within a cluster.
    for cluster_id, face_ids in by_cluster.items():
        usable = [f for f in face_ids if f in embeddings]
        for a, b in combinations(usable, 2):
            pairs.append({"face_a": a, "face_b": b, "same_person": True,
                          "source": f"cluster {cluster_id}"})

    # Different-person pairs: every pair across two different clusters.
    for c1, c2 in combinations(cluster_ids, 2):
        usable1 = [f for f in by_cluster[c1] if f in embeddings]
        usable2 = [f for f in by_cluster[c2] if f in embeddings]
        for a in usable1:
            for b in usable2:
                pairs.append({"face_a": a, "face_b": b, "same_person": False,
                              "source": f"cluster {c1} vs cluster {c2}"})

    n_same = sum(1 for p in pairs if p["same_person"])
    n_diff = len(pairs) - n_same
    print(f"Built {len(pairs)} labeled pairs ({n_same} same-person, {n_diff} different-person)")

    out = {
        "embeddings": embeddings,
        "pairs": pairs,
        "caveat": (
            "Labels come from existing (possibly unreviewed) DBSCAN clusters, "
            "not a real admin-confirmed ground truth set. Treat thresholds "
            "derived from this file as provisional -- see script docstring."
        ),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")

    if n_same == 0:
        print(
            "\nWARNING: zero same-person pairs -- every cluster with a cached "
            "crop only has one usable face. Can't estimate a same-person "
            "distance distribution from this data alone (see output above "
            "for which faces were skipped and why). tune_threshold.py will "
            "still run on the different-person pairs, but the recommended "
            "MATCH_THRESHOLD will be a rough upper bound, not a real "
            "precision/recall-validated number."
        )


if __name__ == "__main__":
    main()
