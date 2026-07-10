"""
MemoryWatch — pipeline test on the synthetic memory dataset
===========================================================
Loads memory_dataset.csv (produced by generate_memory_dataset.py),
runs the same Isolation Forest pipeline as the live monitor and
Deliverable 2 (normal-only training, 99th-percentile threshold), and
reports per-attack-category detection rates — mirroring Notebook 4's
evaluation methodology but on memory-domain features instead of
UNSW-NB15 network features.

This is the "sandboxed VM" validation: confirms that the detection
methodology, feature schema, and alert threshold work as expected on
realistic memory-monitoring data before deploying on a live host.

Usage:
  python3 generate_memory_dataset.py          # create the data first
  python3 test_on_memory_dataset.py           # then evaluate

Optional flags:
  --dataset PATH    (default: memory_dataset.csv in same directory)
  --threshold-pct N (default: 99  — change to explore the tradeoff)
  --seed N          (default: 42)
"""

import argparse
import csv
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from utils import IsolationForest, auroc

FEATURE_NAMES = [
    'vmrss_kb', 'vmrss_delta', 'fd_count', 'fd_proc_mem_open',
    'maps_total_regions', 'maps_exec_region_count', 'maps_anonymous_exec',
    'exec_region_growth_rate', 'syscall_entropy',
    'io_read_bytes_delta', 'io_write_bytes_delta',
]

DEFAULT_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'memory_dataset.csv')


def load_dataset(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def to_matrix(rows):
    X = np.array([[float(r[f]) for f in FEATURE_NAMES] for r in rows],
                  dtype=np.float64)
    y = np.array([int(r['is_attack']) for r in rows], dtype=int)
    cats = [r['attack_type'] for r in rows]
    return X, y, cats


def normalize(X_train, X_all):
    lo = X_train.min(axis=0)
    hi = X_train.max(axis=0)
    r = hi - lo
    r[r == 0] = 1
    Xn_train = np.clip((X_train - lo) / r, 0, 1)
    Xn_all   = np.clip((X_all   - lo) / r, 0, 1)
    return Xn_train, Xn_all, lo, hi


def print_metrics(label, y_true, preds, scores):
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    total = tp + fp + fn + tn
    acc   = (tp + tn) / total if total else 0
    prec  = tp / (tp + fp) if (tp + fp) else 0
    rec   = tp / (tp + fn) if (tp + fn) else 0
    fpr   = fp / (fp + tn) if (fp + tn) else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    auc   = auroc(y_true, scores)
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {acc:.4f}   (TP={tp} TN={tn} FP={fp} FN={fn})")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  FPR       : {fpr:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  AUROC     : {auc:.4f}")
    return dict(accuracy=acc, precision=prec, recall=rec, fpr=fpr, f1=f1, auroc=auc)


def per_category_detection(y_true, preds, cats):
    buckets = defaultdict(lambda: [0, 0])  # [flagged, total]
    for yi, pi, cat in zip(y_true, preds, cats):
        if yi == 1:
            buckets[cat][1] += 1
            if pi == 1:
                buckets[cat][0] += 1
    print(f"\n  {'Attack type':<20} {'Detected':>8} {'Total':>8} {'Rate':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8}")
    for cat in sorted(buckets):
        flagged, total = buckets[cat]
        rate = flagged / total if total else 0
        bar = '█' * int(rate * 20)
        print(f"  {cat:<20} {flagged:>8} {total:>8} {rate:>7.1%}  {bar}")


def top_features_by_deviation(X_normal_n, X_attacks_n, attack_cats):
    """For each attack type, show which features deviate most from the
    normal baseline mean — useful for building intuition about why the
    model does or doesn't flag them."""
    normal_mean = X_normal_n.mean(axis=0)
    by_cat = defaultdict(list)
    for vec, cat in zip(X_attacks_n, attack_cats):
        by_cat[cat].append(vec)

    print(f"\n  Top deviating features vs. normal baseline mean:")
    for cat in sorted(by_cat):
        vecs = np.array(by_cat[cat])
        dev = np.abs(vecs.mean(axis=0) - normal_mean)
        idx = np.argsort(dev)[::-1][:3]
        parts = [f"{FEATURE_NAMES[i]}={vecs.mean(axis=0)[i]:.2f}" for i in idx]
        print(f"  {cat:<20} {', '.join(parts)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset',       default=DEFAULT_DATASET)
    ap.add_argument('--threshold-pct', type=float, default=99.0)
    ap.add_argument('--seed',          type=int,   default=42)
    args = ap.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}")
        print("Run:  python3 generate_memory_dataset.py  first.")
        sys.exit(1)

    print(f"Loading dataset: {args.dataset}")
    rows = load_dataset(args.dataset)
    X, y, cats = to_matrix(rows)

    # Normal rows only → training baseline (same as Deliverable 2's
    # normal-traffic-only Isolation Forest training)
    normal_mask  = (y == 0)
    attack_mask  = (y == 1)
    X_normal     = X[normal_mask]
    X_attacks    = X[attack_mask]
    attack_cats  = [c for c, m in zip(cats, attack_mask) if m]

    print(f"  {normal_mask.sum()} normal rows  |  "
          f"{attack_mask.sum()} attack rows  |  "
          f"{len(FEATURE_NAMES)} features")

    # Normalise using only training (normal) data — no leakage
    X_normal_n, X_all_n, lo, hi = normalize(X_normal, X)
    X_attacks_n = X_all_n[attack_mask]

    # Train
    max_samples = min(256, len(X_normal_n))
    print(f"\nTraining Isolation Forest "
          f"(100 trees, max_samples={max_samples}, seed={args.seed})...")
    model = IsolationForest(n_estimators=100, max_samples=max_samples,
                             seed=args.seed)
    model.fit(X_normal_n, verbose=True)

    # Score everything
    scores = model.anomaly_scores(X_all_n)
    normal_scores = scores[normal_mask]

    threshold = float(np.percentile(normal_scores, args.threshold_pct))
    print(f"\nThreshold ({args.threshold_pct:.0f}th pct of normal scores): "
          f"{threshold:.4f}  "
          f"(normal score range: {normal_scores.min():.4f}–{normal_scores.max():.4f})")

    preds = (scores >= threshold).astype(int)

    # ── Overall metrics ───────────────────────────────────────────────
    print_metrics("Overall (all rows)", y, preds, scores)

    # ── Per-category detection ────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  Per-attack-type detection rates")
    print(f"{'─'*55}")
    per_category_detection(y, preds, cats)

    # ── Feature deviation analysis ────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  Feature deviation analysis (attack mean vs. normal mean)")
    print(f"{'─'*55}")
    top_features_by_deviation(X_normal_n, X_attacks_n, attack_cats)

    print(f"\n{'─'*55}")
    print("  Done.")


if __name__ == '__main__':
    main()
