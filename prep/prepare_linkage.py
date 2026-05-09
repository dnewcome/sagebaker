"""Build a pair-level dataset for the record-linkage problem.

Reads `<input>/training.parquet` + `<input>/ground_truth.parquet` from
a fuzzy_clickstream simulation, samples positive (same true_user_id)
and negative (different true_user_id) event pairs, and computes
pair-level features. Output: a single training.parquet ready for the
existing supervised harness, with a binary `target` column.

Usage::

    python prepare_linkage.py --input ./data_fuzzy --output ./data_linkage \\
        --n-pairs 20000

The output's `lineage.json` records the input directory's sha256, so
the linkage dataset is anchored back to the specific simulation run.

Pair sampling
-------------
By default, half positive / half negative — balanced for ROC-AUC. To
make the negative class harder, the `--ip-bucket-aware` flag biases
negatives toward same-IP-bucket pairs (where IP alone fails to
discriminate). This better matches the production deduplication
problem; off by default for the simplest first run.

Limitations / TODO
------------------
- Random-pair negatives are easy because device_fingerprint is
  near-unique-per-user in the current simulator. For a harder problem
  add fingerprint noise to the simulator (multi-device, fingerprint
  drift, shared family devices).
- Production systems also do candidate-window sampling (only consider
  pairs within ~N hours / shared session-cluster). Worth adding once
  the easy version is exercised.
"""
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _build_pair_features(a: pd.Series, b: pd.Series) -> dict:
    """Symmetric features for a single pair of events.

    Symmetric = same answer regardless of which event is "first" — the
    record-linkage problem is unordered. Non-symmetric features
    (e.g. signed time delta) would let the model learn order from
    sampling artifacts.

    `same_session` is intentionally NOT a feature: in this simulator
    every session belongs to a single user, so same_session=1 is a
    deterministic same_user=1, which collapses the problem. Linkage
    is the cross-session problem; within-session attribution is a
    different task.
    """
    a_uid = a["user_id"] if pd.notna(a["user_id"]) else None
    b_uid = b["user_id"] if pd.notna(b["user_id"]) else None
    return {
        "abs_time_diff_s": abs((a["timestamp"] - b["timestamp"]).total_seconds()),
        "same_fingerprint": int(a["device_fingerprint"] == b["device_fingerprint"]),
        "same_ip_bucket": int(a["ip_bucket"] == b["ip_bucket"]),
        "same_referrer": int(a["referrer"] == b["referrer"]),
        "same_event_type": int(a["event_type"] == b["event_type"]),
        "both_have_user_id": int(a_uid is not None and b_uid is not None),
        "both_same_user_id": int(
            a_uid is not None and b_uid is not None and a_uid == b_uid
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="directory with training.parquet + ground_truth.parquet")
    parser.add_argument("--output", required=True,
                        help="directory to write training.parquet + lineage.json")
    parser.add_argument("--n-pairs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ip-bucket-aware", action="store_true",
                        help="bias negatives toward same-IP-bucket pairs (harder)")
    args = parser.parse_args()

    events = pd.read_parquet(os.path.join(args.input, "training.parquet"))
    gt = pd.read_parquet(os.path.join(args.input, "ground_truth.parquet"))

    merged = events.merge(gt[["event_id", "true_user_id"]], on="event_id")
    rng = np.random.default_rng(args.seed)

    n_half = args.n_pairs // 2

    # For positives we need cross-session pairs of the same user — within-
    # session pairs are trivial since every session belongs to exactly one
    # user (same_session=1 ⇒ same_user=1). Cross-session is the realistic
    # linkage problem.
    user_session_indices: dict[int, dict[int, list[int]]] = {}
    for idx, row in merged.reset_index().iterrows():
        user_session_indices.setdefault(row["true_user_id"], {}) \
            .setdefault(row["session_id"], []).append(int(row["index"]))

    cross_session_users = [
        u for u, sessions in user_session_indices.items() if len(sessions) >= 2
    ]
    if not cross_session_users:
        raise SystemExit(
            "no users have events across multiple sessions — "
            "increase sessions_per_user in the simulator"
        )

    positives = []
    while len(positives) < n_half:
        u = cross_session_users[rng.integers(0, len(cross_session_users))]
        sessions = list(user_session_indices[u].keys())
        s_a, s_b = rng.choice(sessions, size=2, replace=False)
        i = rng.choice(user_session_indices[u][s_a])
        j = rng.choice(user_session_indices[u][s_b])
        positives.append((merged.iloc[i], merged.iloc[j]))

    # Negatives: random different-user pairs.
    negatives = []
    n_events = len(merged)
    while len(negatives) < n_half:
        i, j = int(rng.integers(0, n_events)), int(rng.integers(0, n_events))
        if i == j:
            continue
        a, b = merged.iloc[i], merged.iloc[j]
        if a["true_user_id"] == b["true_user_id"]:
            continue
        if args.ip_bucket_aware and a["ip_bucket"] != b["ip_bucket"]:
            # Reject ~80% of out-of-bucket negatives to bias the sample.
            if rng.random() < 0.8:
                continue
        negatives.append((a, b))

    rows = []
    for label, pairs in [(1, positives), (0, negatives)]:
        for a, b in pairs:
            row = _build_pair_features(a, b)
            row["target"] = label
            rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
    out_path = os.path.join(args.output, "training.parquet")
    df.to_parquet(out_path, index=False)

    # Anchor lineage to the input simulation's sha if available.
    src_lineage_path = os.path.join(args.input, "lineage.json")
    src_lineage = {}
    if os.path.exists(src_lineage_path):
        with open(src_lineage_path) as f:
            src_lineage = json.load(f)
    pair_sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()

    lineage = {
        "source": "prepare_linkage",
        "input_dir": os.path.abspath(args.input),
        "input_lineage": src_lineage,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(df),
        "positive_fraction": float(df["target"].mean()),
        "ip_bucket_aware": args.ip_bucket_aware,
        "seed": args.seed,
        "dataset_sha256": pair_sha,
        "feature_names": [c for c in df.columns if c != "target"],
    }
    with open(os.path.join(args.output, "lineage.json"), "w") as f:
        json.dump(lineage, f, indent=2)

    print(f"wrote {out_path} ({len(df)} rows × {len(df.columns)} cols, "
          f"positive_rate={df['target'].mean():.3f})")


if __name__ == "__main__":
    main()
