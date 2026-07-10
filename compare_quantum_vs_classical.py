"""
MemoryWatch — D3 Three-Way Comparison
Isolation Forest  vs  Classical RBF Kernel  vs  Quantum ZZFeatureMap Kernel

Put this file in the same folder as:
  utils.py, utils_quantum.py, memory_dataset.csv

Run:
  python3 generate_memory_dataset.py           # only if memory_dataset.csv missing
  python3 compare_quantum_vs_classical.py

Only numpy required. Quantum kernel step takes ~30-60s.
"""

import os, sys, csv
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import IsolationForest, auroc
from utils_quantum import (
    pca_fit, pca_transform,
    angle_scale_fit, angle_scale_transform,
    zz_feature_map_amplitudes, quantum_kernel,
    rbf_kernel, oneclass_kernel_scores,
)

FEATURE_NAMES = [
    'vmrss_kb', 'vmrss_delta', 'fd_count', 'fd_proc_mem_open',
    'maps_total_regions', 'maps_exec_region_count', 'maps_anonymous_exec',
    'exec_region_growth_rate', 'syscall_entropy',
    'io_read_bytes_delta', 'io_write_bytes_delta',
]
DATASET       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'memory_dataset.csv')
N_QUBITS      = 4        # 2^4 = 16-dimensional Hilbert space
THRESHOLD_PCT = 99.0     # same as D2 — calibrated on normal scores only
SEED          = 42


def load(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    X    = np.array([[float(r[f]) for f in FEATURE_NAMES] for r in rows], dtype=np.float64)
    y    = np.array([int(r['is_attack']) for r in rows])
    cats = [r['attack_type'] for r in rows]
    return X, y, cats


def normalize(X_train, X_all):
    lo = X_train.min(axis=0)
    hi = X_train.max(axis=0)
    r  = np.where(hi - lo == 0, 1.0, hi - lo)
    return np.clip((X_train - lo) / r, 0, 1), np.clip((X_all - lo) / r, 0, 1)


def report(name, y_true, scores, cats, threshold_pct=99.0):
    normal_scores = scores[y_true == 0]
    thresh = float(np.percentile(normal_scores, threshold_pct))
    preds  = (scores >= thresh).astype(int)

    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    auc = auroc(y_true, scores)

    print(f"\n  {'─'*52}")
    print(f"  {name}")
    print(f"  {'─'*52}")
    print(f"  AUROC  : {auc:.4f}   Recall : {rec:.4f}   FPR : {fpr:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}  (threshold={thresh:.4f})")
    print()

    buckets = defaultdict(lambda: [0, 0])
    for yi, pi, cat in zip(y_true, preds, cats):
        if yi == 1:
            buckets[cat][1] += 1
            if pi == 1:
                buckets[cat][0] += 1
    for cat in sorted(buckets):
        flagged, total = buckets[cat]
        rate = flagged / total if total else 0.0
        bar  = '█' * int(rate * 20)
        print(f"  {cat:<22} {flagged:>3}/{total:<3}  {rate:>6.1%}  {bar}")

    return auc


if not os.path.exists(DATASET):
    print("memory_dataset.csv not found. Run: python3 generate_memory_dataset.py")
    sys.exit(1)

print("=" * 54)
print("  MemoryWatch — Quantum vs Classical Comparison")
print("=" * 54)

X, y, cats = load(DATASET)
normal_mask = y == 0
X_normal    = X[normal_mask]
print(f"\n  {normal_mask.sum()} normal  |  {(~normal_mask).sum()} attack  |  {len(FEATURE_NAMES)} features")

Xn_normal, Xn_all = normalize(X_normal, X)

# ── 1. Isolation Forest ───────────────────────────────────────────────────
print("\n[1/3] Training Isolation Forest ...")
iforest = IsolationForest(n_estimators=100, max_samples=min(256, len(Xn_normal)), seed=SEED)
iforest.fit(Xn_normal, verbose=False)
if_scores = iforest.anomaly_scores(Xn_all)
auc_if = report("Isolation Forest  (D2 baseline)", y, if_scores, cats, THRESHOLD_PCT)

# ── Shared PCA + angle encoding ───────────────────────────────────────────
print(f"\n[prep] PCA → {N_QUBITS} components ...")
pca_mean, pca_comps, pca_var = pca_fit(Xn_normal, N_QUBITS)
print(f"  Explained variance: {pca_var.sum()*100:.1f}%  "
      f"({', '.join(f'{v*100:.0f}%' for v in pca_var)})")

Z_normal = pca_transform(Xn_normal, pca_mean, pca_comps)
Z_all    = pca_transform(Xn_all,    pca_mean, pca_comps)
lo_ang, hi_ang = angle_scale_fit(Z_normal)
Za_normal = angle_scale_transform(Z_normal, lo_ang, hi_ang)
Za_all    = angle_scale_transform(Z_all,    lo_ang, hi_ang)

# ── 2. Classical RBF Kernel ───────────────────────────────────────────────
print("\n[2/3] Computing RBF kernel ...")
gamma = 1.0 / (N_QUBITS * Z_normal.var())
print(f"  gamma = {gamma:.4f}")
K_rbf      = rbf_kernel(Za_all, Za_normal, gamma)
rbf_scores = oneclass_kernel_scores(K_rbf)
auc_rbf = report("Classical RBF Kernel  (control)", y, rbf_scores, cats, THRESHOLD_PCT)

# ── 3. Quantum ZZFeatureMap Kernel ────────────────────────────────────────
print(f"\n[3/3] Quantum kernel  (n_qubits={N_QUBITS}, reps=1) ...")
print("  Computing statevectors for training set ...")
Phi_train = zz_feature_map_amplitudes(Za_normal, N_QUBITS)
print("  Computing statevectors for all samples ...")
Phi_all   = zz_feature_map_amplitudes(Za_all, N_QUBITS)
print("  Computing fidelity K(x,y) = |<phi(x)|phi(y)>|^2  (this may take a minute) ...")
K_qk      = quantum_kernel(Phi_all, Phi_train)
qk_scores = oneclass_kernel_scores(K_qk)
auc_qk = report("Quantum ZZFeatureMap Kernel  (D3)", y, qk_scores, cats, THRESHOLD_PCT)

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'═'*54}")
print("  SUMMARY")
print(f"{'═'*54}")
print(f"  Isolation Forest   AUROC : {auc_if:.4f}")
print(f"  RBF Kernel         AUROC : {auc_rbf:.4f}  ({auc_rbf - auc_if:+.4f} vs IF)")
print(f"  Quantum Kernel     AUROC : {auc_qk:.4f}  ({auc_qk - auc_if:+.4f} vs IF, "
      f"{auc_qk - auc_rbf:+.4f} vs RBF)")
print(f"{'═'*54}")
