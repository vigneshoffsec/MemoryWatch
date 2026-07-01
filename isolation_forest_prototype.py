"""
MemoryWatch - Week 1 (Prachin)
Task: Prototype minimal Isolation Forest on toy data to validate sklearn setup.

This script does NOT touch UNSW-NB15. It only proves that:
  1. scikit-learn's IsolationForest trains and scores correctly in this env.
  2. Unsupervised "fit on normal, flag anomalies" pattern behaves as expected
     (this is the same pattern MemoryWatch will use on syscall/proc features).
  3. The output shape/API matches what the shared evaluation harness (Vignesh,
     Week 2+) will expect: fit(), decision_function(), predict().

Toy data: 2D Gaussian blob = "normal" behaviour, plus a handful of far-out
points = "attack-like" behaviour. This mirrors the real setup where
IsolationForest is trained on normal process behaviour and has to flag
memory-access anomalies (heap spray bursts, /proc reads) as outliers.
"""

import numpy as np
from sklearn.ensemble import IsolationForest

RANDOM_STATE = 42


def make_toy_data(n_normal: int = 200, n_anomalies: int = 10):
    """Synthetic stand-in for 'normal process behaviour' vs 'attack-like' points."""
    rng = np.random.default_rng(RANDOM_STATE)

    # "Normal" cluster: tight Gaussian blob (e.g. typical syscall-rate / /proc-access-rate pair)
    normal = rng.normal(loc=[0, 0], scale=1.0, size=(n_normal, 2))

    # "Anomalies": scattered far from the normal cluster (e.g. heap-spray burst,
    # unexpected /proc/[pid]/mem read pattern)
    anomalies = rng.uniform(low=-10, high=10, size=(n_anomalies, 2))
    # push them away from the origin so they're unambiguously outliers
    anomalies += np.sign(anomalies) * 6

    X = np.vstack([normal, anomalies])
    y_true = np.array([1] * n_normal + [-1] * n_anomalies)  # sklearn convention: 1=normal, -1=anomaly
    return X, y_true


def main():
    X, y_true = make_toy_data()

    # Train UNSUPERVISED (labels never touch .fit()) — same as the real pipeline
    # will do: model.fit() sees only normal behaviour data, contamination is
    # an assumption, not a learned quantity.
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=RANDOM_STATE,
    )
    model.fit(X)

    # decision_function: higher = more normal, lower/negative = more anomalous
    scores = model.decision_function(X)
    # predict: 1 = normal (inlier), -1 = anomaly (outlier)
    preds = model.predict(X)

    n_flagged = int((preds == -1).sum())
    true_anomalies = int((y_true == -1).sum())
    caught = int(((preds == -1) & (y_true == -1)).sum())

    print("=== IsolationForest sanity check ===")
    print(f"sklearn setup OK. Samples: {len(X)}")
    print(f"Score range: min={scores.min():.3f}, max={scores.max():.3f}")
    print(f"Flagged as anomaly: {n_flagged} / {len(X)}")
    print(f"True anomalies in toy set: {true_anomalies}")
    print(f"Correctly flagged: {caught} / {true_anomalies}")

    # Basic assertions -> fail loudly if the environment/API doesn't behave
    # the way the rest of the pipeline will assume.
    assert scores.shape == (len(X),), "decision_function output shape mismatch"
    assert set(np.unique(preds)).issubset({1, -1}), "predict() labels not in {1, -1}"
    assert n_flagged > 0, "Isolation Forest flagged zero anomalies on an obvious toy set"
    assert caught >= true_anomalies * 0.5, "Recall on an easy toy set is suspiciously low"

    print("\nAll checks passed. sklearn + IsolationForest setup is validated.")
    print("Interface confirmed for Week 2: fit(X) -> decision_function(X) -> scores, predict(X) -> {1,-1}.")


if __name__ == "__main__":
    main()