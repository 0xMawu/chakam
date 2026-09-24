"""
Phase 7 §13: recommend MATCH_THRESHOLD (matching.py) and CLUSTER_EPS
(clustering.py) from a labeled eval set built by build_eval_set.py.

Method: Euclidean distance (same metric matching.py/clustering.py use)
between each labeled pair's re-embedded vectors. If both same-person and
different-person distances are present, the recommended threshold is the
midpoint between the largest same-person distance and the smallest
different-person distance (maximizes margin on this eval set -- a real
ROC/precision-recall sweep is worth doing once there are enough pairs for
that to be meaningful; with a handful of pairs it would just overfit).
If either class is missing, no threshold can be validated -- this prints
what it can (the distances it does have) and says so explicitly rather
than fabricating a number.

Usage:
    python scripts/tune_threshold.py [--eval-set data/eval_pairs.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-set", default=str(Path(__file__).parent.parent / "data" / "eval_pairs.json")
    )
    args = parser.parse_args()

    data = json.loads(Path(args.eval_set).read_text())
    embeddings = {int(k): np.array(v) for k, v in data["embeddings"].items()}
    pairs = data["pairs"]

    same_dists, diff_dists = [], []
    for p in pairs:
        a, b = embeddings[p["face_a"]], embeddings[p["face_b"]]
        d = float(np.linalg.norm(a - b))
        (same_dists if p["same_person"] else diff_dists).append(d)

    print(f"Eval set: {len(same_dists)} same-person pair(s), {len(diff_dists)} "
          f"different-person pair(s)\n")

    def summarize(name, vals):
        if not vals:
            print(f"{name}: none available")
            return
        arr = np.array(vals)
        print(f"{name}: n={len(arr)} min={arr.min():.3f} max={arr.max():.3f} "
              f"mean={arr.mean():.3f}")

    summarize("Same-person distances", same_dists)
    summarize("Different-person distances", diff_dists)
    print()

    if same_dists and diff_dists:
        same_max, diff_min = max(same_dists), min(diff_dists)
        if same_max < diff_min:
            recommended = (same_max + diff_min) / 2
            print(f"Classes are cleanly separated on this eval set (same-person max "
                  f"{same_max:.3f} < different-person min {diff_min:.3f}).")
            print(f"Recommended MATCH_THRESHOLD / CLUSTER_EPS: {recommended:.3f}")
        else:
            print(f"Classes OVERLAP on this eval set (same-person max {same_max:.3f} "
                  f">= different-person min {diff_min:.3f}) -- no threshold perfectly "
                  f"separates them here. This can be normal with real photos (lighting/"
                  f"angle variance), but with only {len(same_dists)} same-person pair(s) "
                  f"it's more likely just too little data. Do not treat a single number "
                  f"from this as validated; a naive Youden's-J pick would be:")
            best_j, best_t = -1, None
            candidates = sorted(set(same_dists + diff_dists))
            for t in candidates:
                tpr = sum(d <= t for d in same_dists) / len(same_dists)
                fpr = sum(d <= t for d in diff_dists) / len(diff_dists)
                j = tpr - fpr
                if j > best_j:
                    best_j, best_t = j, t
            print(f"  threshold={best_t:.3f} (Youden's J={best_j:.3f}, TPR="
                  f"{sum(d <= best_t for d in same_dists)/len(same_dists):.2f}, "
                  f"FPR={sum(d <= best_t for d in diff_dists)/len(diff_dists):.2f})")
    elif diff_dists and not same_dists:
        diff_min = min(diff_dists)
        print("No same-person pairs available -- can't validate a real threshold. "
              "The only thing this eval set supports is an UPPER BOUND: any "
              f"MATCH_THRESHOLD >= {diff_min:.3f} (smallest observed different-person "
              "distance) would start misclassifying different people as matches on "
              "this data. It says nothing about how LOW the threshold can safely go "
              "without also rejecting real same-person matches -- that needs same-"
              "person pairs, which needs either (a) an admin reviewing/confirming "
              "clusters in the People view so build_eval_set.py has real labels to "
              "work with, or (b) multiple photos of the same known person run through "
              "ingestion. Do not ship a threshold change from this run alone.")
    else:
        print("Not enough labeled data of either class to recommend anything. Run "
              "build_eval_set.py against a library with admin-reviewed clusters "
              "(2+ faces with cached crops per person) first.")


if __name__ == "__main__":
    main()
