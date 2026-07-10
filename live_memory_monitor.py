"""
MemoryWatch — lightweight real-time memory monitoring module
================================================================
Captures live process/memory behavioural features via /proc polling
(memory_features.py) and feeds them into a trained Isolation Forest
(utils.IsolationForest — the same from-scratch implementation used in
Deliverable 2) for live anomaly inference. This is the concrete,
runnable version of the "Data Collection Layer" + "Classical Anomaly
Detection Pipeline" described in Sections 4.2 and 4.4 of the threat
model document, built directly against the live host instead of a
static benchmark CSV.

Two phases, same methodology as Deliverable 2:
  1. BASELINE  — poll the live system for `--baseline-duration` seconds,
     treat everything observed as "normal", min-max normalise, train an
     Isolation Forest on it, and calibrate a 99th-percentile anomaly
     threshold from the model's own scores on that baseline (identical
     approach to Notebook 3, just against live data instead of the
     normal-traffic subset of UNSW-NB15).
  2. LIVE      — keep polling at the same interval, score every observed
     process against the frozen baseline normalisation + threshold, and
     print/log an alert whenever a process crosses it.

Run:  python3 live_memory_monitor.py
Optional: --spawn-demo spawns demo_workload.py partway through the live
phase so you can watch the monitor actually catch something, instead of
just trusting an idle system to never alert.
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from memory_features import ProcMonitor, FEATURE_NAMES, featurevec
from utils import IsolationForest


def normalize(X, lo, hi):
    r = hi - lo
    r = np.where(r == 0, 1, r)
    Xn = (X - lo) / r
    return np.clip(Xn, 0.0, 1.0)


def run_baseline(monitor, duration, interval, verbose=True, threshold_pct=99.0):
    if verbose:
        print(f"[baseline] observing live system for {duration:.0f}s "
              f"(interval={interval}s)...")
    rows = []
    n_polls = max(1, int(duration / interval))
    for i in range(n_polls):
        snap = monitor.poll()
        for feat in snap.values():
            rows.append(featurevec(feat))
        time.sleep(interval)
    X = np.array(rows, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    Xn = normalize(X, lo, hi)

    n_trees = 100
    max_samples = min(256, len(Xn))
    model = IsolationForest(n_estimators=n_trees, max_samples=max_samples, seed=42)
    model.fit(Xn, verbose=False)
    base_scores = model.anomaly_scores(Xn)
    threshold = float(np.percentile(base_scores, threshold_pct))
    baseline_mean = Xn.mean(axis=0)

    if verbose:
        print(f"[baseline] {len(Xn)} process-observations collected, "
              f"{Xn.shape[1]} features.")
        print(f"[baseline] trained Isolation Forest "
              f"({n_trees} trees, max_samples={max_samples}).")
        print(f"[baseline] 99th-percentile threshold = {threshold:.4f} "
              f"(scores range {base_scores.min():.4f}-{base_scores.max():.4f})")
    return model, lo, hi, threshold, baseline_mean


def top_contributing_features(vec_n, baseline_mean, k=3):
    """Cheap explainability: which normalised features deviate most from
    the baseline mean for this sample."""
    dev = np.abs(vec_n - baseline_mean)
    idx = np.argsort(dev)[::-1][:k]
    return [(FEATURE_NAMES[i], round(float(vec_n[i]), 3)) for i in idx]


def run_live(monitor, model, lo, hi, threshold, baseline_mean, duration, interval,
             log_path=None, spawn_demo=False, verbose=True):
    demo_proc = None
    demo_pid = None
    n_polls = max(1, int(duration / interval))
    spawn_at = n_polls // 3

    alerts = []
    log_f = open(log_path, 'w') if log_path else None

    if verbose:
        print(f"\n[live] monitoring for {duration:.0f}s, scoring every "
              f"{interval}s against the calibrated threshold...")

    try:
        for i in range(n_polls):
            if spawn_demo and demo_proc is None and i == spawn_at:
                script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'demo_workload.py')
                remaining = max(5, int((n_polls - i) * interval) - 2)
                demo_proc = subprocess.Popen(
                    [sys.executable, script, str(remaining)])
                demo_pid = demo_proc.pid
                if verbose:
                    print(f"[live] t={i*interval:.0f}s — spawned demo_workload.py "
                          f"(pid={demo_pid}) to validate detection")

            snap = monitor.poll()
            poll_alerts = []
            for pid, feat in snap.items():
                vec = np.array(featurevec(feat), dtype=np.float64)
                vec_n = normalize(vec.reshape(1, -1), lo, hi)[0]
                score = float(model.anomaly_scores(vec_n.reshape(1, -1))[0])
                flagged = score >= threshold
                if flagged:
                    why = top_contributing_features(vec_n, baseline_mean)
                    rec = {'t': round(i * interval, 1), 'pid': pid,
                           'name': feat['_name'], 'score': round(score, 4),
                           'threshold': round(threshold, 4), 'top_features': why}
                    poll_alerts.append(rec)
                    alerts.append(rec)
                if log_f:
                    log_f.write(json.dumps({
                        't': round(i * interval, 1), 'pid': pid,
                        'name': feat['_name'], 'score': round(score, 4),
                        'flagged': flagged,
                    }) + '\n')

            if verbose:
                for rec in poll_alerts:
                    print(f"  [ALERT] t={rec['t']:>5}s pid={rec['pid']:<7} "
                          f"name={rec['name']:<16} score={rec['score']:.4f} "
                          f"(threshold {rec['threshold']:.4f}) "
                          f"top features: {rec['top_features']}")

            time.sleep(interval)
    finally:
        if log_f:
            log_f.close()
        if demo_proc is not None and demo_proc.poll() is None:
            demo_proc.wait(timeout=10)

    if verbose:
        print(f"\n[live] done. {len(alerts)} alert(s) raised.")
        if spawn_demo:
            caught = any(a['pid'] == demo_pid for a in alerts)
            status = "DETECTED" if caught else "MISSED"
            print(f"[self-check] demo_workload.py (pid={demo_pid}) was {status}.")
    return alerts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline-duration', type=float, default=12.0)
    ap.add_argument('--live-duration', type=float, default=25.0)
    ap.add_argument('--interval', type=float, default=1.0)
    ap.add_argument('--syscall-window', type=int, default=8)
    ap.add_argument('--threshold-pct', type=float, default=99.0,
                     help='Percentile of normal scores used as anomaly threshold (default 99).')
    ap.add_argument('--spawn-demo', action='store_true', default=True,
                     help='Spawn demo_workload.py mid-run to validate detection (default on).')
    ap.add_argument('--no-spawn-demo', dest='spawn_demo', action='store_false')
    ap.add_argument('--log', type=str, default=None,
                     help='Optional path to write a JSON-lines per-poll log.')
    args = ap.parse_args()

    monitor = ProcMonitor(syscall_window=args.syscall_window)
    model, lo, hi, threshold, baseline_mean = run_baseline(
        monitor, args.baseline_duration, args.interval,
        threshold_pct=args.threshold_pct)
    run_live(monitor, model, lo, hi, threshold, baseline_mean,
              args.live_duration, args.interval,
              log_path=args.log, spawn_demo=args.spawn_demo)


if __name__ == '__main__':
    main()
