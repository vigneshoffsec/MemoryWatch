"""
MemoryWatch - Week 5 (Prachin)
Task: Feed classical scores to harness; tune threshold (FPR vs recall).
Dependency: W4 + W3 comparison scores (Vignesh's shared evaluation harness).

STATUS: Vignesh's real evaluation harness isn't in the repo yet (Week 2/3
tasks: "Implement metrics module: FPR, recall, F1" / "Wrap classical model
in evaluation harness"). This file is a STAND-IN with the same metric
surface (FPR, recall, F1, AUROC -- matching the thesis's stated evaluation
methodology and Layman & Roden's finding that accuracy alone is misleading).
Swap `evaluate()` below for Vignesh's real harness call once it exists --
the ClassicalDetector output shapes won't need to change.

Two things happen here:
  1. evaluate(y_true, y_pred, scores) -> dict of FPR/recall/F1/AUROC,
     mirroring the exact metric set from Section 5.6 of the thesis.
  2. tune_threshold(...) sweeps the dynamic-threshold sensitivity
     parameter k and reports the FPR/recall/F1/AUROC trade-off at each
     value, so a k can be chosen deliberately rather than guessed.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from classical_detector import ClassicalDetector


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    """
    Compute FPR, recall, F1, AUROC -- the four metrics the thesis commits to
    (Section 5.6). Accuracy is deliberately NOT included, following Layman
    and Roden (2023) / Hesford et al. (2024): high accuracy can hide a
    useless model on imbalanced attack/normal data.

    y_true: (n,) ground truth, 1 = attack, 0 = normal
    y_pred: (n,) predicted labels, 1 = flagged, 0 = normal
    scores: (n,) continuous anomaly scores (higher = more anomalous),
            used for AUROC since it's threshold-independent
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(scores)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # AUROC needs both classes present in y_true, and is threshold-independent
    # (uses scores, not y_pred) -- this is why it's reported separately.
    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = float("nan")  # only one class present in this slice

    return {"FPR": fpr, "Recall": recall, "F1": f1, "AUROC": auroc, "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def tune_threshold(X_train_normal, groups_train, X_test, y_test, groups_test, k_values=None):
    """
    Sweep the dynamic-threshold sensitivity parameter k (std multiplier
    from Week 3's fit_dynamic_threshold) and report metrics at each value.
    Lower k -> more sensitive -> higher recall, higher FPR.
    Higher k -> stricter -> lower recall, lower FPR.
    """
    if k_values is None:
        k_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    results = []
    for k in k_values:
        model = ClassicalDetector(random_state=42).fit(X_train_normal)
        model.fit_dynamic_threshold(X_train_normal, groups_train, k=k)

        scores = model.score(X_test)
        y_pred = model.predict_dynamic(X_test, groups_test)

        metrics = evaluate(y_test, y_pred, scores)
        metrics["k"] = k
        results.append(metrics)

    return results


def print_tuning_table(results):
    print(f"{'k':>5} | {'FPR':>7} | {'Recall':>7} | {'F1':>7} | {'AUROC':>7}")
    print("-" * 45)
    for r in results:
        print(f"{r['k']:>5.1f} | {r['FPR']:>7.3f} | {r['Recall']:>7.3f} | {r['F1']:>7.3f} | {r['AUROC']:>7.3f}")


def _demo():
    """
    Toy stand-in for real UNSW-NB15 data (Ghita's Week 2 preprocessing
    output isn't in the repo yet). Same shape as the real pipeline:
    train on normal-only data, evaluate on a held-out mix of normal +
    attack-like points, grouped by a placeholder 'process class'.
    """
    rng = np.random.default_rng(0)

    quiet_normal = rng.normal(loc=0, scale=0.4, size=(150, 4))
    noisy_normal = rng.normal(loc=0, scale=2.0, size=(150, 4))
    X_train = np.vstack([quiet_normal, noisy_normal])
    groups_train = np.array(["quiet"] * 150 + ["noisy"] * 150)

    quiet_test_normal = rng.normal(loc=0, scale=0.4, size=(60, 4))
    noisy_test_normal = rng.normal(loc=0, scale=2.0, size=(60, 4))
    quiet_attacks = rng.uniform(-10, 10, size=(15, 4))
    noisy_attacks = rng.uniform(-15, 15, size=(15, 4))

    X_test = np.vstack([quiet_test_normal, noisy_test_normal, quiet_attacks, noisy_attacks])
    groups_test = np.array(["quiet"] * 60 + ["noisy"] * 60 + ["quiet"] * 15 + ["noisy"] * 15)
    y_test = np.array([0] * 120 + [1] * 30)

    print("=== Week 5: threshold tuning (FPR vs Recall trade-off) ===\n")
    results = tune_threshold(X_train, groups_train, X_test, y_test, groups_test)
    print_tuning_table(results)

    # pick the k with the best F1 as a starting recommendation -- but flag
    # that the real choice depends on operational priorities (Layman &
    # Roden: high FPR degrades analyst performance more than it seems)
    best = max(results, key=lambda r: r["F1"])
    print(f"\nBest F1 at k={best['k']}: FPR={best['FPR']:.3f}, Recall={best['Recall']:.3f}, "
          f"F1={best['F1']:.3f}, AUROC={best['AUROC']:.3f}")
    print("\nNOTE: this is a toy synthetic dataset, not UNSW-NB15. Re-run this")
    print("exact sweep once Ghita's preprocessed data lands to get real numbers.")


if __name__ == "__main__":
    _demo()