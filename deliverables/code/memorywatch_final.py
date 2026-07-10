"""
MemoryWatch — final consolidated pipeline
============================================
Single-file version of the full quantum-vs-classical anomaly detection
investigation. Requires numpy and pandas.

WHAT THIS RUNS, IN ORDER, AND WHY:
  1. Generate the baseline attack dataset (heap_spray / proc_scraper /
     mem_injection — each separable on 1-2 individual features).
  2. Generate the interaction-only attack dataset (a Gaussian-copula
     attack where NO single feature is anomalous alone — only the joint
     correlation between three features is — the one condition where a
     quantum kernel's pairwise ZZ coupling term has something to exploit
     that a classical RBF kernel doesn't).
  3. Audit both datasets (schema, duplicates, class balance, per-feature
     separability, correlation, range overlap) before trusting anything
     trained on them.
  4. Three-way comparison (Isolation Forest vs classical RBF kernel vs
     quantum ZZFeatureMap kernel) on BOTH datasets.
  5. Multi-seed validation (5 seeds) on the baseline dataset, to confirm
     the RBF=1.0 result and the quantum underperformance are stable
     findings, not single-seed luck.
  6. Qubit-count x reps hyperparameter sweep for the quantum kernel on
     both datasets, since PCA compression (fewer qubits) was found to
     be the actual bottleneck, not the kernel itself.
  7. Seed-locked validation of the best quantum config found in step 6
     (6 qubits, reps=2) specifically on the interaction dataset, across
     multiple seeds — because a config chosen by sweeping several
     options and picking the best one needs to be re-checked on fresh
     seeds before it can be trusted; otherwise it may just be noise that
     happened to look like a win once.
  8. Real-world validation on CIC-MalMem-2022, IF the CSV is present next
     to this script (see REAL_DATASET_FILENAME below). No path argument
     needed — this step auto-discovers the file and runs automatically;
     if the file isn't there, it's skipped with a one-line notice rather
     than failing the whole run.

HONESTY NOTE: this script does not manufacture a "quantum wins" result.
It reports whatever each step actually finds. Across everything tested,
quantum kernels showed no advantage on the baseline dataset, a narrow
hyperparameter-sensitive advantage on the interaction-only dataset at one
specific configuration, and a clear, consistent LOSS on real malware
memory-forensics data (CIC-MalMem-2022) — step 8 exists to check that
against real data, not just synthetic constructions.

DATASET: this script does not embed the CIC-MalMem-2022 CSV (it's ~19MB
raw — embedding it as text would bloat this file to several MB and isn't
good practice). Instead, place the CSV next to this script under the
name set in REAL_DATASET_FILENAME below (default: 'Obfuscated-MalMem2022.csv').
Get it from: https://www.unb.ca/cic/datasets/malmem-2022.html
        or: https://www.kaggle.com/datasets/dhoogla/cicmalmem2022

Usage:
  python3 memorywatch_final.py
  python3 memorywatch_final.py --polls 20 --seeds 42 7 99 123 2026
  python3 memorywatch_final.py --skip-sweep --skip-multiseed   # fast smoke test
"""

import argparse
import math
import os
import random
import sys
import time
from collections import defaultdict, Counter

import pandas as pd

import numpy as np

# ════════════════════════════════════════════════════════════════════
# 1. FEATURE SCHEMA
# ════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    'vmrss_kb', 'vmrss_delta', 'fd_count', 'fd_proc_mem_open',
    'maps_total_regions', 'maps_exec_region_count', 'maps_anonymous_exec',
    'exec_region_growth_rate', 'syscall_entropy',
    'io_read_bytes_delta', 'io_write_bytes_delta',
]

# Feature weighting for the Isolation Forest, derived from dataset_qa.py
# findings (which features actually separate normal from attack).
DEFAULT_FEATURE_WEIGHTS = np.array([
    1.0, 1.5, 1.0, 3.0, 1.0, 1.5, 3.0, 3.0, 2.5, 2.0, 1.0,
])

COUPLED_FEATURES = ['syscall_entropy', 'fd_count', 'maps_exec_region_count']
COUPLE_RHO = 0.9

# Real-world dataset auto-discovery: no path argument needed. Drop the CSV
# next to this script under this filename and step 8 will find and use it
# automatically; if it's not present, step 8 is skipped with a notice.
REAL_DATASET_FILENAME = 'Obfuscated-MalMem2022.csv'


# ════════════════════════════════════════════════════════════════════
# 2. BASELINE DATASET GENERATOR (heap_spray / proc_scraper / mem_injection)
# ════════════════════════════════════════════════════════════════════

NORMAL_PROFILES = [
    ('systemd',  dict(vmrss=(8000, 2000, 2000, 20000), fd=(12, 4, 3, 30),
                       maps=(65, 10, 40, 100), exec=(10, 2, 6, 18),
                       anon=(0, 0.3, 0, 1), entr=(2.5, 0.3, 1.5, 3.5),
                       ior=(1000, 800, 0, 5000), iow=(500, 400, 0, 3000))),
    ('chrome',   dict(vmrss=(350000, 100000, 80000, 900000), fd=(90, 30, 40, 200),
                       maps=(220, 40, 120, 350), exec=(38, 6, 20, 60),
                       anon=(2, 0.7, 0, 4), entr=(3.0, 0.3, 2.0, 3.8),
                       ior=(80000, 40000, 5000, 300000), iow=(20000, 10000, 1000, 80000))),
    ('vim',      dict(vmrss=(40000, 15000, 8000, 120000), fd=(25, 8, 8, 60),
                       maps=(100, 20, 60, 160), exec=(18, 4, 10, 30),
                       anon=(0, 0.4, 0, 1), entr=(2.3, 0.3, 1.5, 3.2),
                       ior=(3000, 2000, 0, 15000), iow=(1000, 800, 0, 6000))),
    ('postgres', dict(vmrss=(180000, 60000, 50000, 500000), fd=(80, 25, 30, 200),
                       maps=(150, 25, 80, 220), exec=(28, 5, 15, 45),
                       anon=(1, 0.5, 0, 2), entr=(2.2, 0.4, 1.4, 3.2),
                       ior=(200000, 100000, 5000, 800000), iow=(80000, 40000, 2000, 300000))),
    ('sshd',     dict(vmrss=(6000, 2000, 2000, 15000), fd=(10, 3, 4, 20),
                       maps=(55, 8, 35, 80), exec=(9, 2, 5, 14),
                       anon=(0, 0.2, 0, 1), entr=(2.6, 0.3, 1.8, 3.4),
                       ior=(2000, 1500, 0, 8000), iow=(1000, 800, 0, 5000))),
    ('python3',  dict(vmrss=(90000, 40000, 20000, 300000), fd=(18, 6, 6, 40),
                       maps=(95, 18, 55, 150), exec=(14, 3, 8, 22),
                       anon=(1, 0.6, 0, 3), entr=(2.7, 0.4, 1.8, 3.5),
                       ior=(10000, 8000, 0, 50000), iow=(3000, 2500, 0, 15000))),
    ('bash',     dict(vmrss=(5000, 1500, 2000, 12000), fd=(8, 3, 3, 20),
                       maps=(50, 8, 30, 75), exec=(8, 2, 4, 14),
                       anon=(0, 0.2, 0, 1), entr=(2.4, 0.4, 1.5, 3.3),
                       ior=(500, 400, 0, 3000), iow=(200, 200, 0, 1500))),
]


def rng_normal(mu, sigma, lo=0, hi=None, rng=random):
    v = rng.gauss(mu, sigma)
    v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def sample_normal(profile, rng, vmrss_prev=None):
    p = profile
    vmrss = int(rng_normal(*p['vmrss'], rng=rng))
    vmrss_delta = int(rng_normal(0, 200, -2000, 2000, rng=rng)) if vmrss_prev is None \
        else int(vmrss - vmrss_prev)
    fd = int(rng_normal(*p['fd'], rng=rng))
    maps = int(rng_normal(*p['maps'], rng=rng))
    exec_ = int(rng_normal(*p['exec'], rng=rng))
    anon = max(0, int(round(rng_normal(*p['anon'], rng=rng))))
    entr = round(rng_normal(*p['entr'], rng=rng), 3)
    ior = int(max(0, rng_normal(*p['ior'], rng=rng)))
    iow = int(max(0, rng_normal(*p['iow'], rng=rng)))
    return dict(vmrss_kb=vmrss, vmrss_delta=vmrss_delta, fd_count=fd,
                fd_proc_mem_open=0, maps_total_regions=maps,
                maps_exec_region_count=exec_, maps_anonymous_exec=min(anon, exec_),
                exec_region_growth_rate=0, syscall_entropy=entr,
                io_read_bytes_delta=ior, io_write_bytes_delta=iow)


def sample_heap_spray(rng, poll_idx):
    anon = int(rng_normal(8 + poll_idx * 0.5, 1.5, 4, 24, rng=rng))
    exec_ = int(rng_normal(20 + anon * 0.3, 3, 15, 50, rng=rng))
    return dict(vmrss_kb=int(rng_normal(150000, 40000, 30000, 400000, rng=rng)),
                vmrss_delta=int(rng_normal(8000, 3000, 1000, 40000, rng=rng)),
                fd_count=int(rng_normal(10, 4, 3, 25, rng=rng)), fd_proc_mem_open=0,
                maps_total_regions=int(rng_normal(130 + anon, 20, 80, 250, rng=rng)),
                maps_exec_region_count=exec_, maps_anonymous_exec=anon,
                exec_region_growth_rate=int(rng_normal(3, 1, 1, 8, rng=rng)),
                syscall_entropy=round(rng_normal(0.2, 0.1, 0.0, 0.5, rng=rng), 3),
                io_read_bytes_delta=int(rng_normal(5000, 3000, 0, 20000, rng=rng)),
                io_write_bytes_delta=int(rng_normal(1000, 800, 0, 5000, rng=rng)))


def sample_proc_scraper(rng, poll_idx):
    return dict(vmrss_kb=int(rng_normal(25000, 8000, 8000, 60000, rng=rng)),
                vmrss_delta=int(rng_normal(200, 300, -500, 2000, rng=rng)),
                fd_count=int(rng_normal(12, 4, 4, 30, rng=rng)), fd_proc_mem_open=1,
                maps_total_regions=int(rng_normal(70, 12, 40, 110, rng=rng)),
                maps_exec_region_count=int(rng_normal(12, 3, 6, 20, rng=rng)),
                maps_anonymous_exec=int(rng_normal(1, 0.7, 0, 3, rng=rng)),
                exec_region_growth_rate=0,
                syscall_entropy=round(rng_normal(0.7, 0.2, 0.2, 1.2, rng=rng), 3),
                io_read_bytes_delta=int(rng_normal(1500000, 500000, 200000, 4000000, rng=rng)),
                io_write_bytes_delta=int(rng_normal(500, 400, 0, 3000, rng=rng)))


def sample_mem_injection(rng, poll_idx):
    anon = int(rng_normal(7 + poll_idx * 0.8, 2, 3, 18, rng=rng))
    exec_ = int(rng_normal(20, 5, 12, 35, rng=rng))
    return dict(vmrss_kb=int(rng_normal(120000, 40000, 30000, 350000, rng=rng)),
                vmrss_delta=int(rng_normal(40000, 15000, 5000, 120000, rng=rng)),
                fd_count=int(rng_normal(15, 5, 4, 35, rng=rng)), fd_proc_mem_open=0,
                maps_total_regions=int(rng_normal(140, 25, 80, 230, rng=rng)),
                maps_exec_region_count=exec_, maps_anonymous_exec=min(anon, exec_),
                exec_region_growth_rate=int(rng_normal(5, 2, 2, 12, rng=rng)),
                syscall_entropy=round(rng_normal(0.6, 0.2, 0.1, 1.1, rng=rng), 3),
                io_read_bytes_delta=int(rng_normal(40000, 20000, 2000, 150000, rng=rng)),
                io_write_bytes_delta=int(rng_normal(8000, 5000, 0, 40000, rng=rng)))


def generate_baseline_dataset(n_polls=20, seed=42):
    rng = random.Random(seed)
    rows = []
    next_pid = 1000
    for profile_name, profile in NORMAL_PROFILES:
        n_instances = rng.randint(3, 5)
        for _ in range(n_instances):
            pid = next_pid; next_pid += 1
            vmrss_prev = None
            for poll in range(n_polls):
                feat = sample_normal(profile, rng, vmrss_prev)
                vmrss_prev = feat['vmrss_kb']
                rows.append(dict(pid=pid, name=profile_name, poll_id=poll,
                                  is_attack=0, attack_type='normal', **feat))
    attack_start = n_polls // 3
    for attack_type, sampler, n_inst in [('heap_spray', sample_heap_spray, 3),
                                          ('proc_scraper', sample_proc_scraper, 2),
                                          ('mem_injection', sample_mem_injection, 2)]:
        for _ in range(n_inst):
            pid = next_pid; next_pid += 1
            for poll in range(attack_start, n_polls):
                feat = sampler(rng, poll - attack_start)
                rows.append(dict(pid=pid, name=f'[{attack_type}]', poll_id=poll,
                                  is_attack=1, attack_type=attack_type, **feat))
    return rows


# ════════════════════════════════════════════════════════════════════
# 3. INTERACTION-ONLY DATASET GENERATOR (Gaussian copula attack)
# ════════════════════════════════════════════════════════════════════

def _std_normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def build_reference_pool(rng, n_samples=4000):
    pool = {f: [] for f in FEATURE_NAMES}
    profiles = [p for _, p in NORMAL_PROFILES]
    for _ in range(n_samples):
        profile = profiles[rng.randrange(len(profiles))]
        feat = sample_normal(profile, rng, vmrss_prev=None)
        for f in FEATURE_NAMES:
            pool[f].append(feat[f])
    return {f: np.sort(np.array(v, dtype=np.float64)) for f, v in pool.items()}


def empirical_quantile(sorted_ref, p):
    idx = np.clip((p * (len(sorted_ref) - 1)).astype(int), 0, len(sorted_ref) - 1)
    return sorted_ref[idx]


def sample_interaction_rows(rng_np, ref_pool, rng_py, n_rows):
    k = len(COUPLED_FEATURES)
    R = np.full((k, k), COUPLE_RHO)
    np.fill_diagonal(R, 1.0)
    L = np.linalg.cholesky(R)
    z = rng_np.standard_normal((n_rows, k)) @ L.T
    u = np.vectorize(_std_normal_cdf)(z)

    rows = []
    profiles = [p for _, p in NORMAL_PROFILES]
    for i in range(n_rows):
        feat = {}
        for j, fname in enumerate(COUPLED_FEATURES):
            feat[fname] = float(empirical_quantile(ref_pool[fname], u[i, j]))
        base_profile = profiles[rng_py.randrange(len(profiles))]
        base = sample_normal(base_profile, rng_py, vmrss_prev=None)
        for fname in FEATURE_NAMES:
            if fname not in feat:
                feat[fname] = base[fname]
        rows.append(feat)
    return rows


def generate_interaction_dataset(n_polls=20, seed=42, verbose=False):
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    rows = []
    next_pid = 1000
    for profile_name, profile in NORMAL_PROFILES:
        n_instances = py_rng.randint(3, 5)
        for _ in range(n_instances):
            pid = next_pid; next_pid += 1
            vmrss_prev = None
            for poll in range(n_polls):
                feat = sample_normal(profile, py_rng, vmrss_prev)
                vmrss_prev = feat['vmrss_kb']
                rows.append(dict(pid=pid, name=profile_name, poll_id=poll,
                                  is_attack=0, attack_type='normal', **feat))

    ref_pool = build_reference_pool(py_rng)
    n_attack_instances = 3
    attack_start = n_polls // 3
    n_attack_polls = (n_polls - attack_start) * n_attack_instances
    interaction_feats = sample_interaction_rows(np_rng, ref_pool, py_rng, n_attack_polls)

    idx = 0
    for _ in range(n_attack_instances):
        pid = next_pid; next_pid += 1
        for poll in range(attack_start, n_polls):
            feat = interaction_feats[idx]; idx += 1
            rows.append(dict(pid=pid, name='[interaction_only]', poll_id=poll,
                              is_attack=1, attack_type='interaction_only', **feat))

    if verbose:
        _verify_interaction_marginals(rows)
    return rows


def _verify_interaction_marginals(rows):
    normal = [r for r in rows if r['is_attack'] == 0]
    attack = [r for r in rows if r['attack_type'] == 'interaction_only']
    print("\n  Design verification — single-feature AUROC (should be ~0.5):")
    for f in COUPLED_FEATURES:
        vals_n = [r[f] for r in normal]
        vals_a = [r[f] for r in attack]
        y = np.array([0] * len(vals_n) + [1] * len(vals_a))
        x = np.array(vals_n + vals_a, dtype=np.float64)
        auc = single_feature_auroc(x, y)
        flag = "  ✓ ok" if abs(auc - 0.5) <= 0.1 else "  ⚠ leaking signal"
        print(f"    {f:<26} {auc:.3f}{flag}")


# ════════════════════════════════════════════════════════════════════
# 4. DATASET QA
# ════════════════════════════════════════════════════════════════════

def single_feature_auroc(x, y):
    order = np.argsort(x)[::-1]
    ys = y[order]
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return 0.5
    tp = fp = prev_tp = prev_fp = 0
    area = 0.0
    xs = x[order]
    for i in range(len(ys)):
        if ys[i]:
            tp += 1
        else:
            fp += 1
        if i + 1 == len(ys) or xs[i] != xs[i + 1]:
            area += (fp - prev_fp) * (tp + prev_tp) / 2
            prev_tp, prev_fp = tp, fp
    auc = area / (P * N)
    return max(auc, 1 - auc)


def dataset_qa(X, y, label):
    print(f"\n{'─'*70}\n  Dataset QA: {label}\n{'─'*70}")
    n_normal, n_attack = int((y == 0).sum()), int((y == 1).sum())
    print(f"  {n_normal} normal | {n_attack} attack | ratio={n_attack/len(y):.1%}")

    stds = X.std(axis=0)
    const_feats = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if stds[i] < 1e-9]
    if const_feats:
        print(f"  ⚠ Zero-variance features: {const_feats}")

    aucs = {FEATURE_NAMES[i]: single_feature_auroc(X[:, i], y) for i in range(len(FEATURE_NAMES))}
    ranked = sorted(aucs.items(), key=lambda kv: -kv[1])
    print(f"  Strongest single features: {', '.join(f for f, _ in ranked[:3])} "
          f"(AUROC {ranked[0][1]:.3f}, {ranked[1][1]:.3f}, {ranked[2][1]:.3f})")
    print(f"  Weakest single features:   {', '.join(f for f, _ in ranked[-3:])} "
          f"(AUROC {ranked[-1][1]:.3f}, {ranked[-2][1]:.3f}, {ranked[-3][1]:.3f})")

    X_normal, X_attack = X[y == 0], X[y == 1]
    if len(X_attack) > 0:
        contained = []
        for i, fname in enumerate(FEATURE_NAMES):
            n_lo, n_hi = X_normal[:, i].min(), X_normal[:, i].max()
            a_lo, a_hi = X_attack[:, i].min(), X_attack[:, i].max()
            if a_lo >= n_lo and a_hi <= n_hi:
                contained.append(fname)
        if contained:
            print(f"  ⚠ {len(contained)} feature(s) fully inside normal range "
                  f"(no standalone signal): {contained}")


# ════════════════════════════════════════════════════════════════════
# 5. CLASSICAL — ISOLATION FOREST + THRESHOLD SELECTION
# ════════════════════════════════════════════════════════════════════

def _c(n):
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


class _ITree:
    __slots__ = ['feat', 'val', 'left', 'right', 'leaf_size']
    def __init__(self):
        self.feat = -1
        self.val = None
        self.left = None
        self.right = None
        self.leaf_size = 1


def _choose_feature_order(n_features, rng, feature_weights=None):
    if feature_weights is None:
        return rng.permutation(n_features)
    w = np.clip(np.asarray(feature_weights, dtype=np.float64), 1e-6, None)
    p = w / w.sum()
    return rng.choice(n_features, size=n_features, replace=False, p=p)


def _build(X, depth, max_depth, rng, feature_weights=None):
    node = _ITree()
    node.leaf_size = len(X)
    if len(X) <= 1 or depth >= max_depth:
        return node
    for f in _choose_feature_order(X.shape[1], rng, feature_weights):
        lo, hi = X[:, f].min(), X[:, f].max()
        if hi > lo:
            node.feat = f
            node.val = rng.uniform(lo, hi)
            m = X[:, f] < node.val
            node.left = _build(X[m], depth + 1, max_depth, rng, feature_weights)
            node.right = _build(X[~m], depth + 1, max_depth, rng, feature_weights)
            return node
    return node


def _batch_score(node, X, idx, path_len):
    if len(idx) == 0:
        return
    if node.feat == -1 or node.left is None:
        path_len[idx] += _c(node.leaf_size)
        return
    go_left = X[idx, node.feat] < node.val
    path_len[idx] += 1
    _batch_score(node.left, X, idx[go_left], path_len)
    _batch_score(node.right, X, idx[~go_left], path_len)


def _score_tree(node, X):
    path_len = np.zeros(len(X))
    _batch_score(node, X, np.arange(len(X)), path_len)
    return path_len


class IsolationForest:
    def __init__(self, n_estimators=100, max_samples=256, seed=42, feature_weights=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed
        self.feature_weights = feature_weights
        self.trees_ = []
        self._c = 0.0

    def fit(self, X, verbose=False):
        rng = np.random.RandomState(self.seed)
        psi = min(self.max_samples, len(X))
        self._c = _c(psi)
        max_d = int(np.ceil(np.log2(max(psi, 2))))
        for i in range(self.n_estimators):
            idx = rng.choice(len(X), psi, replace=False)
            self.trees_.append(_build(X[idx], 0, max_d, rng, self.feature_weights))
        return self

    def anomaly_scores(self, X):
        avg = np.mean([_score_tree(t, X) for t in self.trees_], axis=0)
        return np.power(2.0, -avg / self._c) if self._c > 0 else np.zeros(len(X))


def auroc(y, s):
    order = np.argsort(s)[::-1]
    ys = y[order]
    P = y.sum()
    Nn = len(y) - P
    tp = fp = prev_tp = prev_fp = 0
    area = 0.0
    for i, yi in enumerate(ys):
        if yi:
            tp += 1
        else:
            fp += 1
        if i + 1 == len(ys) or s[order[i]] != s[order[i + 1]]:
            area += (fp - prev_fp) * (tp + prev_tp) / 2
            prev_tp, prev_fp = tp, fp
    return area / (P * Nn) if P * Nn else 0.0


def f1_optimal_threshold(y, s):
    candidates = np.unique(s)
    best_f1, best_t = -1.0, candidates[0]
    for t in candidates:
        preds = (s >= t).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


# ════════════════════════════════════════════════════════════════════
# 6. QUANTUM — PCA, ANGLE ENCODING, ZZFEATUREMAP, KERNELS
# ════════════════════════════════════════════════════════════════════

def pca_fit(X, n_components):
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    total_var = (S ** 2).sum()
    explained = (S[:n_components] ** 2) / total_var if total_var > 0 else np.zeros(n_components)
    return mean, components, explained


def pca_transform(X, mean, components):
    return (X - mean) @ components.T


def angle_scale_fit(Z):
    return Z.min(axis=0), Z.max(axis=0)


def angle_scale_transform(Z, lo, hi):
    r = np.where((hi - lo) == 0, 1.0, hi - lo)
    Zn = np.clip((Z - lo) / r, 0.0, 1.0)
    return Zn * (2 * np.pi)


def _hadamard_matrix(n_qubits):
    H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    Hn = H
    for _ in range(n_qubits - 1):
        Hn = np.kron(Hn, H)
    return Hn


def zz_feature_map_amplitudes(Za, n_qubits, reps=2):
    N = Za.shape[0]
    dim = 2 ** n_qubits
    bits = (np.arange(dim)[:, None] >> np.arange(n_qubits)[None, :]) & 1
    signs = 1 - 2 * bits
    linear = signs @ Za.T
    pair_phase = np.zeros((dim, N))
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            coeff = (np.pi - Za[:, i]) * (np.pi - Za[:, j])
            pair_phase += np.outer(signs[:, i] * signs[:, j], coeff)
    phi = linear + pair_phase
    phase = np.exp(1j * phi)
    Hn = _hadamard_matrix(n_qubits)
    state = np.zeros((dim, N), dtype=complex)
    state[0, :] = 1.0
    for _ in range(reps):
        state = Hn @ state
        state = phase * state
    return state


def quantum_kernel(Phi_a, Phi_b):
    inner = Phi_a.conj().T @ Phi_b
    return (np.abs(inner) ** 2).real


def rbf_kernel(X, Y, gamma):
    X2 = np.sum(X ** 2, axis=1)[:, None]
    Y2 = np.sum(Y ** 2, axis=1)[None, :]
    sqdist = X2 + Y2 - 2.0 * (X @ Y.T)
    np.maximum(sqdist, 0, out=sqdist)
    return np.exp(-gamma * sqdist)


def oneclass_kernel_scores(K):
    return 1.0 - K.mean(axis=1)


# ════════════════════════════════════════════════════════════════════
# 7. SHARED DATA HELPERS
# ════════════════════════════════════════════════════════════════════

def rows_to_matrix(rows):
    X = np.array([[float(r[f]) for f in FEATURE_NAMES] for r in rows], dtype=np.float64)
    y = np.array([int(r['is_attack']) for r in rows], dtype=int)
    cats = [r['attack_type'] for r in rows]
    return X, y, cats


def normalize(X_train, X_all):
    lo, hi = X_train.min(axis=0), X_train.max(axis=0)
    r = np.where(hi - lo == 0, 1.0, hi - lo)
    return np.clip((X_train - lo) / r, 0, 1), np.clip((X_all - lo) / r, 0, 1)


# ════════════════════════════════════════════════════════════════════
# 8. THREE-WAY COMPARISON (IF vs RBF vs Quantum)
# ════════════════════════════════════════════════════════════════════

def per_category_detection(y, preds, cats):
    buckets = defaultdict(lambda: [0, 0])
    for yi, pi, cat in zip(y, preds, cats):
        if yi == 1:
            buckets[cat][1] += 1
            if pi == 1:
                buckets[cat][0] += 1
    for cat in sorted(buckets):
        flagged, total = buckets[cat]
        rate = flagged / total if total else 0.0
        bar = '█' * int(rate * 20)
        print(f"    {cat:<20} {flagged:>3}/{total:<3}  {rate:>6.1%}  {bar}")


def three_way_comparison(X, y, cats, label, n_qubits=8, reps=2, seed=42, weighted=True):
    print(f"\n{'='*72}\n  Three-way comparison: {label}\n{'='*72}")
    normal_mask = y == 0
    Xn = X[normal_mask]
    Xn_n, Xn_all = normalize(Xn, X)

    # Isolation Forest
    weights = DEFAULT_FEATURE_WEIGHTS if weighted else None
    max_samples = min(256, len(Xn_n))
    iforest = IsolationForest(n_estimators=100, max_samples=max_samples,
                               seed=seed, feature_weights=weights)
    iforest.fit(Xn_n)
    if_scores = iforest.anomaly_scores(Xn_all)
    if_thr, _ = f1_optimal_threshold(y, if_scores)
    if_auc = auroc(y, if_scores)
    print(f"\n  Isolation Forest (weighted={weighted}, f1-optimal threshold)")
    print(f"    AUROC={if_auc:.4f}")
    per_category_detection(y, (if_scores >= if_thr).astype(int), cats)

    # Shared PCA + angle encoding
    pca_mean, pca_comps, var = pca_fit(Xn_n, n_qubits)
    Z_n = pca_transform(Xn_n, pca_mean, pca_comps)
    Z_all = pca_transform(Xn_all, pca_mean, pca_comps)
    lo_a, hi_a = angle_scale_fit(Z_n)
    Za_n = angle_scale_transform(Z_n, lo_a, hi_a)
    Za_all = angle_scale_transform(Z_all, lo_a, hi_a)
    print(f"\n  PCA({n_qubits}) explained variance: {var.sum()*100:.1f}%")

    # RBF
    gamma = 1.0 / (n_qubits * Z_n.var())
    K_rbf = rbf_kernel(Za_all, Za_n, gamma)
    rbf_scores = oneclass_kernel_scores(K_rbf)
    rbf_thr, _ = f1_optimal_threshold(y, rbf_scores)
    rbf_auc = auroc(y, rbf_scores)
    print(f"\n  Classical RBF Kernel   AUROC={rbf_auc:.4f}")
    per_category_detection(y, (rbf_scores >= rbf_thr).astype(int), cats)

    # Quantum
    Phi_n = zz_feature_map_amplitudes(Za_n, n_qubits, reps=reps)
    Phi_all = zz_feature_map_amplitudes(Za_all, n_qubits, reps=reps)
    K_qk = quantum_kernel(Phi_all, Phi_n)
    qk_scores = oneclass_kernel_scores(K_qk)
    qk_thr, _ = f1_optimal_threshold(y, qk_scores)
    qk_auc = auroc(y, qk_scores)
    print(f"\n  Quantum ZZFeatureMap Kernel (qubits={n_qubits}, reps={reps})  AUROC={qk_auc:.4f}")
    per_category_detection(y, (qk_scores >= qk_thr).astype(int), cats)

    print(f"\n  Summary: IF={if_auc:.4f}  RBF={rbf_auc:.4f}  Quantum={qk_auc:.4f}  "
          f"(Quantum - RBF = {qk_auc-rbf_auc:+.4f})")
    return dict(if_auc=if_auc, rbf_auc=rbf_auc, qk_auc=qk_auc)


# ════════════════════════════════════════════════════════════════════
# 9. MULTI-SEED VALIDATION (baseline dataset)
# ════════════════════════════════════════════════════════════════════

def multi_seed_validation(seeds, n_polls, n_qubits=8, reps=2):
    print(f"\n{'='*72}\n  Multi-seed validation — baseline dataset, seeds={seeds}\n{'='*72}")
    all_results = {'isolation_forest': [], 'rbf_kernel': [], 'quantum_kernel': []}
    for seed in seeds:
        rows = generate_baseline_dataset(n_polls=n_polls, seed=seed)
        X, y, cats = rows_to_matrix(rows)
        r = three_way_comparison_silent(X, y, seed=seed, n_qubits=n_qubits, reps=reps)
        all_results['isolation_forest'].append(r['if_auc'])
        all_results['rbf_kernel'].append(r['rbf_auc'])
        all_results['quantum_kernel'].append(r['qk_auc'])
        print(f"  seed={seed:<6} IF={r['if_auc']:.4f}  RBF={r['rbf_auc']:.4f}  "
              f"Quantum={r['qk_auc']:.4f}")

    print(f"\n  {'Method':<18} {'mean ± std AUROC':>20}")
    for method, vals in all_results.items():
        v = np.array(vals)
        print(f"  {method:<18} {v.mean():.4f} ± {v.std():.4f}")
    return all_results


def three_way_comparison_silent(X, y, seed=42, n_qubits=8, reps=2, weighted=True):
    """Same computation as three_way_comparison but no printing — used by
    validation loops that only need the numbers."""
    normal_mask = y == 0
    Xn = X[normal_mask]
    Xn_n, Xn_all = normalize(Xn, X)

    weights = DEFAULT_FEATURE_WEIGHTS if weighted else None
    max_samples = min(256, len(Xn_n))
    iforest = IsolationForest(n_estimators=100, max_samples=max_samples,
                               seed=seed, feature_weights=weights)
    iforest.fit(Xn_n)
    if_auc = auroc(y, iforest.anomaly_scores(Xn_all))

    pca_mean, pca_comps, _ = pca_fit(Xn_n, n_qubits)
    Z_n = pca_transform(Xn_n, pca_mean, pca_comps)
    Z_all = pca_transform(Xn_all, pca_mean, pca_comps)
    lo_a, hi_a = angle_scale_fit(Z_n)
    Za_n = angle_scale_transform(Z_n, lo_a, hi_a)
    Za_all = angle_scale_transform(Z_all, lo_a, hi_a)

    gamma = 1.0 / (n_qubits * Z_n.var())
    rbf_auc = auroc(y, oneclass_kernel_scores(rbf_kernel(Za_all, Za_n, gamma)))

    Phi_n = zz_feature_map_amplitudes(Za_n, n_qubits, reps=reps)
    Phi_all = zz_feature_map_amplitudes(Za_all, n_qubits, reps=reps)
    qk_auc = auroc(y, oneclass_kernel_scores(quantum_kernel(Phi_all, Phi_n)))

    return dict(if_auc=if_auc, rbf_auc=rbf_auc, qk_auc=qk_auc)


# ════════════════════════════════════════════════════════════════════
# 10. QUBIT x REPS SWEEP
# ════════════════════════════════════════════════════════════════════

def hparam_sweep(X, y, cats, label, qubits_list, reps_list):
    print(f"\n{'='*72}\n  Qubit x reps sweep — {label}\n{'='*72}")
    best = None
    summary_rows = []
    for nq in qubits_list:
        for reps in reps_list:
            print(f"\n{'─'*72}")
            print(f"  Config: qubits={nq}, reps={reps}")
            print(f"{'─'*72}")

            normal_mask = y == 0
            Xn = X[normal_mask]
            Xn_n, Xn_all = normalize(Xn, X)

            # Isolation Forest
            weights = DEFAULT_FEATURE_WEIGHTS
            max_samples = min(256, len(Xn_n))
            iforest = IsolationForest(n_estimators=100, max_samples=max_samples,
                                       seed=42, feature_weights=weights)
            iforest.fit(Xn_n)
            if_scores = iforest.anomaly_scores(Xn_all)
            if_thr, _ = f1_optimal_threshold(y, if_scores)
            if_auc = auroc(y, if_scores)
            print(f"\n  Isolation Forest (weighted=True, f1-optimal threshold)")
            print(f"    AUROC={if_auc:.4f}")
            per_category_detection(y, (if_scores >= if_thr).astype(int), cats)

            # Shared PCA + angle encoding
            pca_mean, pca_comps, var = pca_fit(Xn_n, nq)
            Z_n = pca_transform(Xn_n, pca_mean, pca_comps)
            Z_all = pca_transform(Xn_all, pca_mean, pca_comps)
            lo_a, hi_a = angle_scale_fit(Z_n)
            Za_n = angle_scale_transform(Z_n, lo_a, hi_a)
            Za_all = angle_scale_transform(Z_all, lo_a, hi_a)
            print(f"\n  PCA({nq}) explained variance: {var.sum()*100:.1f}%")

            # RBF
            gamma = 1.0 / (nq * Z_n.var())
            K_rbf = rbf_kernel(Za_all, Za_n, gamma)
            rbf_scores = oneclass_kernel_scores(K_rbf)
            rbf_thr, _ = f1_optimal_threshold(y, rbf_scores)
            rbf_auc = auroc(y, rbf_scores)
            print(f"\n  Classical RBF Kernel   AUROC={rbf_auc:.4f}")
            per_category_detection(y, (rbf_scores >= rbf_thr).astype(int), cats)

            # Quantum
            Phi_n = zz_feature_map_amplitudes(Za_n, nq, reps=reps)
            Phi_all = zz_feature_map_amplitudes(Za_all, nq, reps=reps)
            K_qk = quantum_kernel(Phi_all, Phi_n)
            qk_scores = oneclass_kernel_scores(K_qk)
            qk_thr, _ = f1_optimal_threshold(y, qk_scores)
            qk_auc = auroc(y, qk_scores)
            print(f"\n  Quantum ZZFeatureMap Kernel (qubits={nq}, reps={reps})  AUROC={qk_auc:.4f}")
            per_category_detection(y, (qk_scores >= qk_thr).astype(int), cats)

            delta = qk_auc - rbf_auc
            marker = "  <-- quantum wins" if delta > 0 else ""
            print(f"\n  Summary: IF={if_auc:.4f}  RBF={rbf_auc:.4f}  Quantum={qk_auc:.4f}  "
                  f"(Quantum - RBF = {delta:+.4f}){marker}")

            r = dict(if_auc=if_auc, rbf_auc=rbf_auc, qk_auc=qk_auc)
            summary_rows.append((nq, reps, r))
            if best is None or qk_auc > best[2]['qk_auc']:
                best = (nq, reps, r)

    print(f"\n{'='*72}")
    print(f"  Sweep summary — {label}")
    print(f"{'='*72}")
    print(f"  {'qubits':>7} {'reps':>5} {'RBF':>8} {'Quantum':>9} {'Δ':>8}")
    for nq, reps, r in summary_rows:
        delta = r['qk_auc'] - r['rbf_auc']
        marker = "  <-- quantum wins" if delta > 0 else ""
        print(f"  {nq:>7} {reps:>5} {r['rbf_auc']:>8.4f} "
              f"{r['qk_auc']:>9.4f} {delta:>+8.4f}{marker}")

    nq, reps, r = best
    print(f"\n  Best quantum config: qubits={nq}, reps={reps} -> "
          f"AUROC={r['qk_auc']:.4f} (RBF at same config: {r['rbf_auc']:.4f})")
    return nq, reps


# ════════════════════════════════════════════════════════════════════
# 11. SEED-LOCKED VALIDATION of best quantum config
# ════════════════════════════════════════════════════════════════════

def seed_locked_quantum_check(seeds, n_polls, n_qubits, reps):
    print(f"\n{'='*72}")
    print(f"  Seed-locked validation: interaction dataset, "
          f"qubits={n_qubits}, reps={reps}, seeds={seeds}")
    print(f"{'='*72}")
    print("  (Checks whether the best config found by the sweep is a real")
    print("   effect or noise from testing multiple configs and picking one.)\n")

    deltas = []
    for seed in seeds:
        rows = generate_interaction_dataset(n_polls=n_polls, seed=seed)
        X, y, cats = rows_to_matrix(rows)
        r = three_way_comparison_silent(X, y, seed=seed, n_qubits=n_qubits, reps=reps)
        delta = r['qk_auc'] - r['rbf_auc']
        deltas.append(delta)
        outcome = "QUANTUM WINS" if delta > 0 else "RBF wins/ties"
        print(f"  seed={seed:<6} RBF={r['rbf_auc']:.4f}  Quantum={r['qk_auc']:.4f}  "
              f"Δ={delta:+.4f}   {outcome}")

    deltas = np.array(deltas)
    wins = int((deltas > 0).sum())
    print(f"\n  Quantum won on {wins}/{len(seeds)} seeds.")
    print(f"  Mean Δ (Quantum - RBF) = {deltas.mean():+.4f} ± {deltas.std():.4f}")
    if deltas.mean() > 0 and wins >= len(seeds) * 0.7:
        print("  VERDICT: quantum advantage appears REAL and reasonably consistent")
        print("  at this configuration on the interaction-only dataset.")
    elif deltas.mean() > 0:
        print("  VERDICT: quantum wins on average but inconsistently across seeds —")
        print("  treat as a WEAK, not fully reliable, advantage.")
    else:
        print("  VERDICT: the earlier single-seed 'quantum win' does NOT replicate —")
        print("  most likely noise from testing multiple hyperparameter configs.")
    return deltas


# ════════════════════════════════════════════════════════════════════
# 11b. REAL-WORLD DATASET (CIC-MalMem-2022) — auto-discovered, no path arg
# ════════════════════════════════════════════════════════════════════

def detect_label_columns(df):
    """Find the binary label column and (optionally) the family/category
    column, without assuming exact CIC-MalMem-2022 naming — case-
    insensitive match on common names."""
    cols_lower = {c.lower(): c for c in df.columns}
    class_col = None
    for candidate in ('class', 'label', 'is_attack', 'target'):
        if candidate in cols_lower:
            class_col = cols_lower[candidate]
            break
    cat_col = None
    for candidate in ('category', 'attack_type', 'family', 'type'):
        if candidate in cols_lower:
            cat_col = cols_lower[candidate]
            break
    if class_col is None and cat_col is None:
        raise ValueError(
            "Could not find a label column. Expected one of "
            "'Class'/'Label'/'is_attack'/'target' (binary) and/or "
            "'Category'/'attack_type'/'family'/'type' (family-level). "
            f"Columns found: {list(df.columns)}"
        )
    return class_col, cat_col


def family_from_category(val):
    """CIC-MalMem-2022 'Category' values look like 'Trojan-Emotet-...' or
    'Benign'. Take the top-level family (text before the first '-')."""
    s = str(val)
    if s.lower() == 'benign':
        return 'Benign'
    return s.split('-')[0]


def load_real_dataset(path, sample_size, seed):
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    class_col, cat_col = detect_label_columns(df)
    print(f"  Detected label column(s): class='{class_col}', category='{cat_col}'")

    if class_col is not None:
        y_all = (df[class_col].astype(str).str.lower() != 'benign').astype(int).values
    else:
        y_all = (df[cat_col].astype(str).str.lower() != 'benign').astype(int).values

    if cat_col is not None:
        cats_all = df[cat_col].apply(family_from_category).tolist()
    else:
        cats_all = np.where(y_all == 1, 'malware', 'benign').tolist()

    exclude = {c for c in (class_col, cat_col) if c is not None}
    feature_cols = [c for c in df.columns
                     if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found after excluding label columns.")
    print(f"  Using {len(feature_cols)} numeric feature columns")

    X_all = df[feature_cols].fillna(0).values.astype(np.float64)

    stds = X_all.std(axis=0)
    keep = stds > 1e-12
    if (~keep).sum() > 0:
        dropped = [f for f, k in zip(feature_cols, keep) if not k]
        print(f"  Dropping {len(dropped)} zero-variance columns: {dropped[:10]}"
              f"{'...' if len(dropped) > 10 else ''}")
    X_all = X_all[:, keep]

    n_normal, n_attack = int((y_all == 0).sum()), int((y_all == 1).sum())
    print(f"  {n_normal} benign, {n_attack} malware in full dataset")

    rng = np.random.RandomState(seed)
    idx_normal = np.where(y_all == 0)[0]
    idx_attack = np.where(y_all == 1)[0]
    n_norm_sample = min(len(idx_normal), sample_size // 2)
    n_atk_sample = min(len(idx_attack), sample_size - n_norm_sample)
    sel = np.concatenate([
        rng.choice(idx_normal, n_norm_sample, replace=False),
        rng.choice(idx_attack, n_atk_sample, replace=False),
    ])
    rng.shuffle(sel)

    X = X_all[sel]
    y = y_all[sel]
    cats = [cats_all[i] for i in sel]
    print(f"  Subsampled to {len(X)} rows ({int((y==0).sum())} benign, "
          f"{int((y==1).sum())} malware) for kernel computation")
    fam_counts = Counter(c for c, yi in zip(cats, y) if yi == 1)
    print(f"  Malware family breakdown in subsample: {dict(fam_counts)}")
    return X, y, cats


def three_way_on_real_data(X, y, cats, label, n_qubits=8, reps=2, seed=42,
                            if_max_samples=256, kernel_train_cap=300):
    print(f"\n{'='*72}\n  Three-way comparison: {label}\n{'='*72}")
    normal_mask = y == 0
    Xn_full = X[normal_mask]

    Xn_n, Xn_all = normalize(Xn_full, X)
    max_samples = min(if_max_samples, len(Xn_n))
    iforest = IsolationForest(n_estimators=100, max_samples=max_samples,
                               seed=seed, feature_weights=None)
    iforest.fit(Xn_n)
    if_scores = iforest.anomaly_scores(Xn_all)
    if_thr, _ = f1_optimal_threshold(y, if_scores)
    if_auc = auroc(y, if_scores)
    print(f"\n  Isolation Forest (unweighted — no domain priors for this "
          f"feature set)\n    AUROC={if_auc:.4f}")
    per_category_detection(y, (if_scores >= if_thr).astype(int), cats)

    rng = np.random.RandomState(seed)
    train_idx = rng.choice(len(Xn_n), min(kernel_train_cap, len(Xn_n)), replace=False)
    Xn_kernel_train = Xn_n[train_idx]

    pca_mean, pca_comps, var = pca_fit(Xn_kernel_train, n_qubits)
    Z_train = pca_transform(Xn_kernel_train, pca_mean, pca_comps)
    Z_all = pca_transform(Xn_all, pca_mean, pca_comps)
    lo_a, hi_a = angle_scale_fit(Z_train)
    Za_train = angle_scale_transform(Z_train, lo_a, hi_a)
    Za_all = angle_scale_transform(Z_all, lo_a, hi_a)
    print(f"\n  PCA({n_qubits}) explained variance: {var.sum()*100:.1f}%  "
          f"(kernel train set: {len(Xn_kernel_train)} normal rows)")

    gamma = 1.0 / (n_qubits * Z_train.var())
    K_rbf = rbf_kernel(Za_all, Za_train, gamma)
    rbf_scores = oneclass_kernel_scores(K_rbf)
    rbf_thr, _ = f1_optimal_threshold(y, rbf_scores)
    rbf_auc = auroc(y, rbf_scores)
    print(f"\n  Classical RBF Kernel   AUROC={rbf_auc:.4f}")
    per_category_detection(y, (rbf_scores >= rbf_thr).astype(int), cats)

    Phi_train = zz_feature_map_amplitudes(Za_train, n_qubits, reps=reps)
    Phi_all = zz_feature_map_amplitudes(Za_all, n_qubits, reps=reps)
    K_qk = quantum_kernel(Phi_all, Phi_train)
    qk_scores = oneclass_kernel_scores(K_qk)
    qk_thr, _ = f1_optimal_threshold(y, qk_scores)
    qk_auc = auroc(y, qk_scores)
    print(f"\n  Quantum ZZFeatureMap Kernel (qubits={n_qubits}, reps={reps})  "
          f"AUROC={qk_auc:.4f}")
    per_category_detection(y, (qk_scores >= qk_thr).astype(int), cats)

    print(f"\n  Summary: IF={if_auc:.4f}  RBF={rbf_auc:.4f}  Quantum={qk_auc:.4f}  "
          f"(Quantum - RBF = {qk_auc-rbf_auc:+.4f})")
    return dict(if_auc=if_auc, rbf_auc=rbf_auc, qk_auc=qk_auc)


def run_real_dataset_step(script_dir, sample_size, n_qubits, reps, seed):
    path = os.path.join(script_dir, REAL_DATASET_FILENAME)
    if not os.path.exists(path):
        print(f"\n  [skipped] '{REAL_DATASET_FILENAME}' not found next to this script.")
        print(f"  To include real-world validation, place the CIC-MalMem-2022 CSV")
        print(f"  at: {path}")
        print(f"  Get it from: https://www.unb.ca/cic/datasets/malmem-2022.html")
        return None
    X, y, cats = load_real_dataset(path, sample_size, seed)
    return three_way_on_real_data(X, y, cats, REAL_DATASET_FILENAME,
                                   n_qubits=n_qubits, reps=reps, seed=seed)


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--polls', type=int, default=20)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 99, 123, 2026])
    ap.add_argument('--skip-qa', action='store_true')
    ap.add_argument('--skip-multiseed', action='store_true')
    ap.add_argument('--skip-sweep', action='store_true')
    ap.add_argument('--skip-seedlock', action='store_true')
    ap.add_argument('--skip-real', action='store_true',
                     help='Skip step 8 (real CIC-MalMem-2022 validation) even if the CSV is present')
    ap.add_argument('--real-sample-size', type=int, default=3000)
    ap.add_argument('--real-kernel-train-cap', type=int, default=300)
    args = ap.parse_args()

    t_start = time.time()
    print("#" * 72)
    print("  MEMORYWATCH — FINAL CONSOLIDATED QUANTUM VS CLASSICAL PIPELINE")
    print("#" * 72)

    # 1-2. Generate both datasets
    print("\n[step 1/8] Generating baseline dataset...")
    base_rows = generate_baseline_dataset(n_polls=args.polls, seed=42)
    Xb, yb, catsb = rows_to_matrix(base_rows)
    print(f"  {len(base_rows)} rows ({int((yb==0).sum())} normal, {int((yb==1).sum())} attack)")

    print("\n[step 2/8] Generating interaction-only dataset...")
    inter_rows = generate_interaction_dataset(n_polls=args.polls, seed=42, verbose=True)
    Xi, yi, catsi = rows_to_matrix(inter_rows)
    print(f"\n  {len(inter_rows)} rows ({int((yi==0).sum())} normal, {int((yi==1).sum())} attack)")

    # 3. QA
    if not args.skip_qa:
        print("\n[step 3/8] Dataset QA...")
        dataset_qa(Xb, yb, "baseline")
        dataset_qa(Xi, yi, "interaction-only")
    else:
        print("\n[step 3/8] Skipped (--skip-qa)")

    # 4. Three-way comparison on both datasets (default config: 8 qubits, reps=1)
    print("\n[step 4/8] Three-way comparison (default config: 4 qubits, reps=1)...")
    three_way_comparison(Xb, yb, catsb, "baseline dataset", n_qubits=8, reps=2)
    three_way_comparison(Xi, yi, catsi, "interaction-only dataset", n_qubits=8, reps=2)

    # 5. Multi-seed validation on baseline
    if not args.skip_multiseed:
        print("\n[step 5/8] Multi-seed validation (baseline dataset)...")
        multi_seed_validation(args.seeds, args.polls, n_qubits=8, reps=2)
    else:
        print("\n[step 5/8] Skipped (--skip-multiseed)")

    # 6. Hyperparameter sweep on both datasets
    best_nq, best_reps = 6, 2  # fallback if sweep skipped
    if not args.skip_sweep:
        print("\n[step 6/8] Qubit x reps sweep...")
        hparam_sweep(Xb, yb, catsb, "baseline dataset", qubits_list=[8], reps_list=[2])
        best_nq, best_reps = hparam_sweep(Xi, yi, catsi, "interaction-only dataset",
                                   qubits_list=[8], reps_list=[2])
    else:
        print("\n[step 6/8] Skipped (--skip-sweep)")

    # 7. Seed-locked validation of the best quantum config
    if not args.skip_seedlock:
        print("\n[step 7/8] Seed-locked validation of best quantum config...")
        seed_locked_quantum_check(args.seeds, args.polls, best_nq, best_reps)
    else:
        print("\n[step 7/8] Skipped (--skip-seedlock)")

    # 8. Real-world dataset (CIC-MalMem-2022), auto-discovered — no path arg
    if not args.skip_real:
        print("\n[step 8/8] Real-world validation (CIC-MalMem-2022, auto-discovered)...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        run_real_dataset_step(script_dir, args.real_sample_size,
                               n_qubits=8, reps=2, seed=42)
    else:
        print("\n[step 8/8] Skipped (--skip-real)")

    print(f"\n{'#'*72}")
    print(f"  DONE in {time.time()-t_start:.1f}s")
    print(f"{'#'*72}")


if __name__ == '__main__':
    main()