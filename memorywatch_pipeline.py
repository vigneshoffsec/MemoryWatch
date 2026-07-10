"""
MemoryWatch — Deliverable 2: Classical ML Anomaly Detection Pipeline
Dataset : UNSW-NB15 (official training/testing split)
Model   : Isolation Forest (vectorised NumPy — Liu, Ting & Zhou, ICDM 2008)
Metrics : Accuracy, Precision, Recall, FPR, F1, AUROC
"""

import pandas as pd
import numpy as np
import json, time
import warnings; warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN_PATH = '/sessions/relaxed-pensive-shannon/mnt/uploads/UNSW_NB15_training-set.csv'
TEST_PATH  = '/sessions/relaxed-pensive-shannon/mnt/uploads/UNSW_NB15_testing-set.csv'
OUT_DIR    = '/sessions/relaxed-pensive-shannon/mnt/outputs/'

print("=" * 65)
print("MemoryWatch — Classical ML Anomaly Detection Pipeline")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════
# ISOLATION FOREST — vectorised NumPy implementation
# Trees stored as flat arrays; all samples scored simultaneously
# ══════════════════════════════════════════════════════════════════

def _c(n):
    """Expected path length in BST of n nodes (Liu 2008 eq.1)."""
    if n <= 1: return 0.0
    if n == 2: return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n

class _ITree:
    __slots__ = ['feat', 'val', 'left', 'right', 'leaf_size']
    def __init__(self): self.feat = -1; self.val = None; self.left = None; self.right = None; self.leaf_size = 1

def _build(X, depth, max_depth, rng):
    node = _ITree(); node.leaf_size = len(X)
    if len(X) <= 1 or depth >= max_depth: return node
    for f in rng.permutation(X.shape[1]):
        lo, hi = X[:, f].min(), X[:, f].max()
        if hi > lo:
            node.feat = f; node.val = rng.uniform(lo, hi)
            m = X[:, f] < node.val
            node.left  = _build(X[m],  depth+1, max_depth, rng)
            node.right = _build(X[~m], depth+1, max_depth, rng)
            return node
    return node  # constant columns → leaf

def _batch_score(node, X, idx, path_len):
    """Vectorised: push all samples left/right simultaneously."""
    if len(idx) == 0: return
    if node.feat == -1 or node.left is None:
        path_len[idx] += _c(node.leaf_size)
        return
    go_left = X[idx, node.feat] < node.val
    path_len[idx] += 1          # count this split for all samples
    _batch_score(node.left,  X, idx[go_left],  path_len)
    _batch_score(node.right, X, idx[~go_left], path_len)

def _score_tree(node, X):
    path_len = np.zeros(len(X))
    _batch_score(node, X, np.arange(len(X)), path_len)
    return path_len

class IsolationForest:
    def __init__(self, n_estimators=100, max_samples=256, seed=42):
        self.n_estimators = n_estimators
        self.max_samples  = max_samples
        self.seed         = seed
        self.trees_       = []
        self._c           = 0.0

    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        psi = min(self.max_samples, len(X))
        self._c = _c(psi)
        max_d   = int(np.ceil(np.log2(max(psi, 2))))
        for i in range(self.n_estimators):
            idx  = rng.choice(len(X), psi, replace=False)
            self.trees_.append(_build(X[idx], 0, max_d, rng))
            if (i+1) % 25 == 0: print(f"    trees: {i+1}/{self.n_estimators}")
        return self

    def anomaly_scores(self, X):
        """Higher = more anomalous."""
        avg = np.mean([_score_tree(t, X) for t in self.trees_], axis=0)
        return np.power(2.0, -avg / self._c) if self._c > 0 else np.zeros(len(X))


# ══════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════
print("\n[1/6] Loading dataset...")
train = pd.read_csv(TRAIN_PATH, encoding='utf-8-sig')
test  = pd.read_csv(TEST_PATH,  encoding='utf-8-sig')
print(f"  Train: {len(train):,} records | Test: {len(test):,} records")
print("\n  Label distribution (train):")
print(train['label'].value_counts().to_string())
print("\n  Attack categories (train):")
print(train['attack_cat'].value_counts().to_string())

# ══════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════
print("\n[2/6] Preprocessing...")

DROP    = ['id', 'attack_cat', 'label']
CATS    = ['proto', 'service', 'state']
NUMS    = [c for c in train.columns if c not in DROP + CATS]

def preprocess(df, vocab=None, lo=None, hi=None, fit=True):
    df = df.copy()
    y  = df['label'].values.astype(int)
    at = df['attack_cat'].values
    df = df.drop(columns=DROP)
    df[CATS] = df[CATS].fillna('unknown')
    for c in NUMS:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    if fit: vocab = {}
    for col in CATS:
        uniq = sorted(df[col].astype(str).unique()) if fit else list(vocab[col].keys())
        if fit: vocab[col] = {v: i for i, v in enumerate(uniq)}
        df[col] = df[col].astype(str).map(lambda x, v=vocab[col]: v.get(x, len(v)))
    X = df.values.astype(np.float32)
    if fit:
        lo = X.min(axis=0); hi = X.max(axis=0)
    r = hi - lo; r[r == 0] = 1
    return (X - lo) / r, y, at, vocab, lo, hi

X_tr, y_tr, at_tr, vocab, lo, hi = preprocess(train, fit=True)
X_te, y_te, at_te, *_            = preprocess(test, vocab=vocab, lo=lo, hi=hi, fit=False)
X_norm = X_tr[y_tr == 0]
print(f"  Feature shape  — train {X_tr.shape}, test {X_te.shape}")
print(f"  Normal (train) : {len(X_norm):,} | Attack (train): {(y_tr==1).sum():,}")

# ══════════════════════════════════════════════════════════════════
# 3. TRAIN
# ══════════════════════════════════════════════════════════════════
print("\n[3/6] Training Isolation Forest (normal-only, n=100 trees, ψ=256)...")
t0  = time.time()
clf = IsolationForest(n_estimators=100, max_samples=256, seed=42)
clf.fit(X_norm)
print(f"  Done in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════
# 4. DYNAMIC THRESHOLD
# ══════════════════════════════════════════════════════════════════
print("\n[4/6] Calibrating threshold on normal training scores...")
s_norm_train = clf.anomaly_scores(X_norm)
threshold    = np.percentile(s_norm_train, 99)
print(f"  Threshold (99th pct normal): {threshold:.6f}")

print("  Scoring test set...")
t0 = time.time()
s_test = clf.anomaly_scores(X_te)
print(f"  Scored {len(X_te):,} samples in {time.time()-t0:.1f}s")

y_pred = (s_test >= threshold).astype(int)

# ══════════════════════════════════════════════════════════════════
# 5. EVALUATION
# ══════════════════════════════════════════════════════════════════
print("\n[5/6] Evaluation results...")

TN = int(((y_pred==0)&(y_te==0)).sum())
FP = int(((y_pred==1)&(y_te==0)).sum())
FN = int(((y_pred==0)&(y_te==1)).sum())
TP = int(((y_pred==1)&(y_te==1)).sum())

N = len(y_te); P_neg = (y_te==0).sum(); P_pos = (y_te==1).sum()
acc  = (TP+TN)/N
prec = TP/(TP+FP) if (TP+FP) else 0
rec  = TP/(TP+FN) if (TP+FN) else 0
fpr  = FP/(FP+TN) if (FP+TN) else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0

# AUROC via sorted thresholds
def auroc(y, s):
    order  = np.argsort(s)[::-1]
    ys     = y[order]; P=y.sum(); Nn=len(y)-P
    tp=fp=0; prev_tp=prev_fp=0; area=0.0
    for i, yi in enumerate(ys):
        if yi: tp+=1
        else:  fp+=1
        if i+1 == len(ys) or s[order[i]] != s[order[i+1]]:
            area += (fp-prev_fp)*(tp+prev_tp)/2
            prev_tp, prev_fp = tp, fp
    return area/(P*Nn) if P*Nn else 0.0

auc = auroc(y_te, s_test)

print(f"\n  {'Metric':<30} {'Value':>10}")
print(f"  {'-'*42}")
print(f"  {'Accuracy':<30} {acc:>10.4f}")
print(f"  {'Precision':<30} {prec:>10.4f}")
print(f"  {'Recall (TPR)':<30} {rec:>10.4f}")
print(f"  {'False Positive Rate (FPR)':<30} {fpr:>10.4f}")
print(f"  {'F1 Score':<30} {f1:>10.4f}")
print(f"  {'AUROC':<30} {auc:>10.4f}")
print(f"\n  Confusion Matrix:")
print(f"  {'':>20}  Pred Normal  Pred Attack")
print(f"  {'Actual Normal':<20}  {TN:>11,}  {FP:>11,}")
print(f"  {'Actual Attack':<20}  {FN:>11,}  {TP:>11,}")

print("\n  Per-attack-category detection:")
cat_rows = []
for cat in sorted(set(at_te[y_te==1])):
    m = (at_te==cat) & (y_te==1); n=int(m.sum()); d=int((y_pred[m]==1).sum())
    dr = d/n if n else 0
    cat_rows.append({'category':str(cat),'n':n,'detected':d,'detection_rate':round(dr,4)})
    print(f"    {str(cat):<22} n={n:>5,}  detected={d:>5,}  {dr:.1%}")

# ══════════════════════════════════════════════════════════════════
# 6. PLOTS + SAVE
# ══════════════════════════════════════════════════════════════════
print("\n[6/6] Saving outputs...")

# ROC
thresholds_roc = np.percentile(s_test, np.linspace(0,100,300))
fpr_pts=[0.]; tpr_pts=[0.]
for t in sorted(set(thresholds_roc), reverse=True):
    p=(s_test>=t).astype(int)
    fpr_pts.append(((p==1)&(y_te==0)).sum()/P_neg)
    tpr_pts.append(((p==1)&(y_te==1)).sum()/P_pos)
fpr_pts.append(1.); tpr_pts.append(1.)

fig, axes = plt.subplots(1, 2, figsize=(12,5))
axes[0].plot(fpr_pts, tpr_pts, color='black', lw=2, label=f'AUROC = {auc:.4f}')
axes[0].plot([0,1],[0,1],'--',color='gray',lw=1)
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve — Isolation Forest (MemoryWatch)')
axes[0].legend(); axes[0].grid(alpha=0.3)

sn = s_test[y_te==0]; sa = s_test[y_te==1]
axes[1].hist(sn, bins=60, alpha=0.6, color='black', label='Normal', density=True)
axes[1].hist(sa, bins=60, alpha=0.5, color='gray',  label='Attack', density=True)
axes[1].axvline(threshold, color='red', linestyle='--', lw=1.5, label=f'Threshold={threshold:.4f}')
axes[1].set_xlabel('Anomaly Score'); axes[1].set_ylabel('Density')
axes[1].set_title('Anomaly Score Distribution')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR+'memorywatch_evaluation_plots.png', dpi=150, bbox_inches='tight')
plt.close()

results = {
    'dataset': 'UNSW-NB15 — official UNSW training/testing split (Moustafa & Slay, 2015)',
    'model': 'Isolation Forest (vectorised NumPy — Liu, Ting & Zhou, ICDM 2008)',
    'hyperparameters': {'n_estimators':100,'max_samples':256,'random_state':42},
    'threshold': {'method':'Dynamic — 99th percentile of normal training scores','value':round(float(threshold),6)},
    'test_set': {'total':N,'normal':int(P_neg),'attack':int(P_pos)},
    'confusion_matrix': {'TN':TN,'FP':FP,'FN':FN,'TP':TP},
    'metrics': {'accuracy':round(acc,4),'precision':round(prec,4),'recall':round(rec,4),
                'fpr':round(fpr,4),'f1':round(f1,4),'auroc':round(auc,4)},
    'per_category': cat_rows
}
with open(OUT_DIR+'memorywatch_results.json','w') as f:
    json.dump(results, f, indent=2)

print("  Plots → memorywatch_evaluation_plots.png")
print("  Results → memorywatch_results.json")
print("\n" + "=" * 65)
print("Pipeline complete.")
print("=" * 65)
