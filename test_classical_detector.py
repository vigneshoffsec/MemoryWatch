"""
MemoryWatch - Week 4 (Prachin)
Task: Stabilise baseline; add unit tests; document score ranges.
Dependency: W3 baseline (classical_detector.py).

Run with:
    pytest test_classical_detector.py -v

These tests cover:
  - the plain fit/score/predict path (Week 2 interface)
  - the dynamic per-group threshold path (Week 3)
  - error handling for misuse (unfitted model, bad shapes, mismatched arrays)
  - reproducibility (same random_state -> same output)
  - documented score-range expectations (see SCORE_RANGE_NOTES below)
"""

import numpy as np
import pytest

from classical_detector import ClassicalDetector


# ---------------------------------------------------------------------------
# Score range documentation (part of the Week 4 task, not just a comment).
#
# ClassicalDetector.score() = -sklearn_IsolationForest.decision_function(X)
#
# sklearn's decision_function() is centred so that:
#   - 0 is roughly the boundary between "normal" and "anomalous" learned
#     from the training contamination assumption
#   - typical values fall in roughly [-0.5, 0.5], though this is NOT a hard
#     mathematical bound -- it comes from how path-length averaging works
#     across the forest, and can exceed that range for very extreme outliers
#     or very small/unusual trees.
#
# Because ClassicalDetector flips the sign, in OUR convention:
#   - higher score  = more anomalous
#   - lower score   = more normal
#   - values are NOT guaranteed to be in [0, 1] or any fixed interval --
#     do not assume a fixed range when setting a manual threshold; use
#     fit_dynamic_threshold() or calibrate against a validation split.
# ---------------------------------------------------------------------------
SCORE_RANGE_NOTES = """
score() range is data-dependent (NOT a fixed [0,1] or [-1,1] interval).
Empirically, well-behaved datasets fall roughly within [-0.5, 0.5] before
the sign flip. Do not hardcode a threshold based on this range -- always
compute it from the normal-training-data distribution
(fit_dynamic_threshold) or validate empirically on held-out data.
"""


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def normal_data(rng):
    return rng.normal(size=(100, 4))


@pytest.fixture
def fitted_model(normal_data):
    return ClassicalDetector(random_state=42).fit(normal_data)


# --- fit / score / predict (Week 2 interface) -------------------------------

def test_fit_returns_self(normal_data):
    model = ClassicalDetector()
    result = model.fit(normal_data)
    assert result is model


def test_fit_rejects_1d_input():
    model = ClassicalDetector()
    with pytest.raises(ValueError):
        model.fit(np.array([1, 2, 3]))


def test_score_shape_matches_input(fitted_model, rng):
    X_test = rng.normal(size=(15, 4))
    scores = fitted_model.score(X_test)
    assert scores.shape == (15,)
    assert scores.dtype.kind == "f"


def test_predict_returns_binary_labels(fitted_model, rng):
    X_test = rng.normal(size=(15, 4))
    labels = fitted_model.predict(X_test)
    assert labels.shape == (15,)
    assert set(np.unique(labels)).issubset({0, 1})


def test_score_before_fit_raises():
    model = ClassicalDetector()
    with pytest.raises(RuntimeError):
        model.score(np.zeros((5, 4)))


def test_predict_before_fit_raises():
    model = ClassicalDetector()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 4)))


def test_obvious_outliers_score_higher_than_normal(fitted_model, rng):
    """Attack-like points should score as MORE anomalous than in-distribution points."""
    normal_like = rng.normal(size=(20, 4))
    far_outliers = rng.uniform(-20, 20, size=(20, 4))

    normal_scores = fitted_model.score(normal_like)
    outlier_scores = fitted_model.score(far_outliers)

    assert outlier_scores.mean() > normal_scores.mean()


# --- dynamic per-group threshold (Week 3) -----------------------------------

def test_dynamic_threshold_mismatched_lengths_raises(fitted_model, normal_data):
    bad_groups = np.array(["a"] * (len(normal_data) - 1))  # one short
    with pytest.raises(ValueError):
        fitted_model.fit_dynamic_threshold(normal_data, bad_groups)


def test_predict_dynamic_without_fit_dynamic_raises(fitted_model, rng):
    X_test = rng.normal(size=(10, 4))
    groups = np.array(["a"] * 10)
    with pytest.raises(RuntimeError):
        fitted_model.predict_dynamic(X_test, groups)


def test_dynamic_thresholds_differ_across_groups(rng):
    """Two process classes with different natural variance should get
    different thresholds -- this is the whole point of per-group
    thresholding (Shamim et al.) over a single global cutoff."""
    quiet = rng.normal(loc=0, scale=0.3, size=(60, 4))
    noisy = rng.normal(loc=0, scale=3.0, size=(60, 4))
    X = np.vstack([quiet, noisy])
    groups = np.array(["quiet"] * 60 + ["noisy"] * 60)

    model = ClassicalDetector(random_state=1).fit(X)
    model.fit_dynamic_threshold(X, groups, k=3.0)

    assert model._group_thresholds["quiet"] != model._group_thresholds["noisy"]
    # the noisier class should tolerate a higher raw score before flagging
    assert model._group_thresholds["noisy"] > model._group_thresholds["quiet"]


def test_predict_dynamic_low_false_positive_rate_on_normal_data(rng):
    """On in-distribution data from the SAME groups used to fit the
    threshold, false positive rate should be low (not necessarily zero --
    a k=3 std threshold still has some tail probability)."""
    quiet = rng.normal(loc=0, scale=0.3, size=(100, 4))
    noisy = rng.normal(loc=0, scale=3.0, size=(100, 4))
    X_train = np.vstack([quiet, noisy])
    groups_train = np.array(["quiet"] * 100 + ["noisy"] * 100)

    model = ClassicalDetector(random_state=1).fit(X_train)
    model.fit_dynamic_threshold(X_train, groups_train, k=3.0)

    # fresh in-distribution test data, same groups
    quiet_test = rng.normal(loc=0, scale=0.3, size=(50, 4))
    noisy_test = rng.normal(loc=0, scale=3.0, size=(50, 4))
    X_test = np.vstack([quiet_test, noisy_test])
    groups_test = np.array(["quiet"] * 50 + ["noisy"] * 50)

    labels = model.predict_dynamic(X_test, groups_test)
    false_positive_rate = labels.mean()
    assert false_positive_rate < 0.15  # generous bound for a toy random dataset


def test_predict_dynamic_unseen_group_falls_back_to_global(rng):
    X_train = rng.normal(size=(80, 4))
    groups_train = np.array(["known"] * 80)

    model = ClassicalDetector(random_state=1).fit(X_train)
    model.fit_dynamic_threshold(X_train, groups_train, k=3.0)

    X_test = rng.normal(size=(5, 4))
    groups_test = np.array(["never_seen_before"] * 5)

    # should not raise -- falls back to self._threshold
    labels = model.predict_dynamic(X_test, groups_test)
    assert labels.shape == (5,)


# --- reproducibility ---------------------------------------------------------

def test_same_random_state_gives_same_scores(normal_data, rng):
    X_test = rng.normal(size=(10, 4))
    model_a = ClassicalDetector(random_state=7).fit(normal_data)
    model_b = ClassicalDetector(random_state=7).fit(normal_data)
    np.testing.assert_array_equal(model_a.score(X_test), model_b.score(X_test))


def test_different_random_state_can_differ(normal_data, rng):
    X_test = rng.normal(size=(10, 4))
    model_a = ClassicalDetector(random_state=1).fit(normal_data)
    model_b = ClassicalDetector(random_state=2).fit(normal_data)
    # not asserting inequality strictly (could coincidentally match), just
    # documenting that random_state controls reproducibility, not a fixed
    # deterministic-regardless-of-seed algorithm
    assert model_a.score(X_test).shape == model_b.score(X_test).shape