"""
MemoryWatch - Week 2 (Prachin)
Task: Agree array interface; stub model class against contract.
Dependency: W2 model interface contract (Esala).

This stubs the classical detector as a class so it can plug into the shared
evaluation harness (Vignesh, Week 3+) and the entry point (Esala) the same
way the quantum layer eventually will. No real training logic yet — that's
Week 3. This file exists to lock the SHAPE of inputs/outputs early so
nobody downstream has to guess.

Assumed interface (numpy-array in, numpy-array out — matches what Week 1's
prototype already validated against sklearn):

    model = ClassicalDetector(**hyperparams)
    model.fit(X_train)                # X_train: (n_samples, n_features) normal-only data
    scores = model.score(X)           # -> (n_samples,) float array, higher = more anomalous
    labels = model.predict(X)         # -> (n_samples,) int array, {0, 1}: 1 = flagged as attack

NOTE: sklearn's IsolationForest natively returns 1=normal/-1=anomaly and
"lower score = more anomalous". This stub flips both so the class's public
API matches what the evaluation harness will expect everywhere in the
codebase (0/1 labels, higher score = more anomalous is more intuitive for
FPR/recall/F1/AUROC reporting). This mapping is the actual "contract"
decision below -- flag it in review if Esala's harness expects sklearn's
raw convention instead.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class ClassicalDetector:
    """Thin wrapper around sklearn's IsolationForest with a MemoryWatch-shaped API."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self._model: IsolationForest | None = None
        self._threshold: float | None = None  # set in Week 3 (dynamic threshold, Shamim et al.)

    def fit(self, X: np.ndarray) -> "ClassicalDetector":
        """Fit on normal-only behaviour data. X shape: (n_samples, n_features)."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {X.shape}")

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score per sample. Higher = more anomalous (flipped from sklearn's raw sign)."""
        self._check_fitted()
        X = np.asarray(X)
        # sklearn: higher decision_function = more normal -> flip sign so
        # higher = more anomalous, which is the convention the rest of the
        # pipeline (dynamic threshold, FPR/recall/AUROC) will use.
        return -self._model.decision_function(X)

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """
        Binary labels: 1 = flagged as attack, 0 = normal.
        threshold: score cutoff. If None, uses self._threshold (set by Week 3's
        dynamic thresholding) and falls back to sklearn's own contamination-based
        cutoff (score > 0) if no threshold has been set yet.
        """
        self._check_fitted()
        scores = self.score(X)
        cutoff = threshold if threshold is not None else (self._threshold if self._threshold is not None else 0.0)
        return (scores > cutoff).astype(int)

    def _check_fitted(self):
        if self._model is None:
            raise RuntimeError("Call fit(X) before score()/predict().")


def _smoke_test():
    """Quick self-check that the stub's shapes/types behave as declared. Not the real Week 3 test suite."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(100, 4))
    X_test = np.vstack([rng.normal(size=(20, 4)), rng.uniform(-8, 8, size=(5, 4))])

    model = ClassicalDetector().fit(X_train)
    scores = model.score(X_test)
    labels = model.predict(X_test)

    assert scores.shape == (25,), f"score() shape wrong: {scores.shape}"
    assert labels.shape == (25,), f"predict() shape wrong: {labels.shape}"
    assert set(np.unique(labels)).issubset({0, 1}), "predict() must return {0, 1} labels"

    print("ClassicalDetector stub OK.")
    print(f"  score() range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  predict() flagged: {int(labels.sum())} / {len(labels)}")


if __name__ == "__main__":
    _smoke_test()