"""
MemoryWatch — synthetic /proc memory dataset generator
=======================================================
Generates a realistic CSV dataset that simulates what the live
memory monitor (memory_features.py) would collect from a real Linux VM
with diverse processes running, including injected attack behaviour.

Matches the 11-feature schema from FEATURE_NAMES in memory_features.py:
  vmrss_kb, vmrss_delta, fd_count, fd_proc_mem_open,
  maps_total_regions, maps_exec_region_count, maps_anonymous_exec,
  exec_region_growth_rate, syscall_entropy, io_read_bytes_delta,
  io_write_bytes_delta

Normal process types modelled (with realistic value distributions):
  systemd daemon, web browser (V8 JIT), text editor, database daemon,
  sshd/network service, python scripting process, shell session

Attack profiles (labelled separately):
  heap_spray    — many anonymous executable regions, very low syscall
                  entropy (repetitive pattern), growing RSS
  proc_scraper  — fd pointing into another process's /proc/[pid]/mem,
                  massive read I/O (scanning victim memory), low entropy
  mem_injection — sudden large vmrss_delta spike, rapid exec-region
                  growth, several anonymous executable mappings

Output: memory_dataset.csv
  columns: pid, name, poll_id, is_attack, attack_type, <11 features>

Usage: python3 generate_memory_dataset.py [--seed N] [--polls N]
"""

import argparse
import csv
import os
import math
import random

FEATURE_NAMES = [
    'vmrss_kb',
    'vmrss_delta',
    'fd_count',
    'fd_proc_mem_open',
    'maps_total_regions',
    'maps_exec_region_count',
    'maps_anonymous_exec',
    'exec_region_growth_rate',
    'syscall_entropy',
    'io_read_bytes_delta',
    'io_write_bytes_delta',
]

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'memory_dataset.csv')


def rng_normal(mu, sigma, lo=0, hi=None, rng=random):
    v = rng.gauss(mu, sigma)
    v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


# ──────────────────────────────────────────────────────────────────────
# Normal process profiles
# Each entry: (display_name, feature_samplers_dict)
# Values are (mu, sigma, lo, hi) for Gaussian sampling, or fixed.
# ──────────────────────────────────────────────────────────────────────

NORMAL_PROFILES = [
    ('systemd',     dict(vmrss=(8000,  2000,  2000, 20000),
                         fd=   (12,    4,     3,    30),
                         maps= (65,    10,    40,   100),
                         exec= (10,    2,     6,    18),
                         anon= (0,     0.3,   0,    1),
                         entr= (2.5,   0.3,   1.5,  3.5),
                         ior=  (1000,  800,   0,    5000),
                         iow=  (500,   400,   0,    3000))),
    ('chrome',      dict(vmrss=(350000, 100000, 80000, 900000),
                         fd=   (90,     30,     40,    200),
                         maps= (220,    40,     120,   350),
                         exec= (38,     6,      20,    60),
                         anon= (2,      0.7,    0,     4),   # V8 JIT
                         entr= (3.0,    0.3,    2.0,   3.8),
                         ior=  (80000,  40000,  5000,  300000),
                         iow=  (20000,  10000,  1000,  80000))),
    ('vim',         dict(vmrss=(40000,  15000,  8000,  120000),
                         fd=   (25,     8,      8,     60),
                         maps= (100,    20,     60,    160),
                         exec= (18,     4,      10,    30),
                         anon= (0,      0.4,    0,     1),
                         entr= (2.3,    0.3,    1.5,   3.2),
                         ior=  (3000,   2000,   0,     15000),
                         iow=  (1000,   800,    0,     6000))),
    ('postgres',    dict(vmrss=(180000, 60000,  50000, 500000),
                         fd=   (80,     25,     30,    200),
                         maps= (150,    25,     80,    220),
                         exec= (28,     5,      15,    45),
                         anon= (1,      0.5,    0,     2),
                         entr= (2.2,    0.4,    1.4,   3.2),
                         ior=  (200000, 100000, 5000,  800000),
                         iow=  (80000,  40000,  2000,  300000))),
    ('sshd',        dict(vmrss=(6000,   2000,   2000,  15000),
                         fd=   (10,     3,      4,     20),
                         maps= (55,     8,      35,    80),
                         exec= (9,      2,      5,     14),
                         anon= (0,      0.2,    0,     1),
                         entr= (2.6,    0.3,    1.8,   3.4),
                         ior=  (2000,   1500,   0,     8000),
                         iow=  (1000,   800,    0,     5000))),
    ('python3',     dict(vmrss=(90000,  40000,  20000, 300000),
                         fd=   (18,     6,      6,     40),
                         maps= (95,     18,     55,    150),
                         exec= (14,     3,      8,     22),
                         anon= (1,      0.6,    0,     3),
                         entr= (2.7,    0.4,    1.8,   3.5),
                         ior=  (10000,  8000,   0,     50000),
                         iow=  (3000,   2500,   0,     15000))),
    ('bash',        dict(vmrss=(5000,   1500,   2000,  12000),
                         fd=   (8,      3,      3,     20),
                         maps= (50,     8,      30,    75),
                         exec= (8,      2,      4,     14),
                         anon= (0,      0.2,    0,     1),
                         entr= (2.4,    0.4,    1.5,   3.3),
                         ior=  (500,    400,    0,     3000),
                         iow=  (200,    200,    0,     1500))),
]


def sample_normal(profile, rng, vmrss_prev=None):
    p = profile
    g = p['vmrss']
    vmrss = int(rng_normal(*g, rng=rng))
    vmrss_delta = int(rng_normal(0, 200, -2000, 2000, rng=rng)) if vmrss_prev is None \
        else int(vmrss - vmrss_prev)

    fd = int(rng_normal(*p['fd'], rng=rng))
    maps = int(rng_normal(*p['maps'], rng=rng))
    exec_ = int(rng_normal(*p['exec'], rng=rng))
    anon = max(0, int(round(rng_normal(*p['anon'], rng=rng))))
    entr = round(rng_normal(*p['entr'], rng=rng), 3)
    ior = int(max(0, rng_normal(*p['ior'], rng=rng)))
    iow = int(max(0, rng_normal(*p['iow'], rng=rng)))

    return dict(
        vmrss_kb=vmrss, vmrss_delta=vmrss_delta,
        fd_count=fd, fd_proc_mem_open=0,
        maps_total_regions=maps, maps_exec_region_count=exec_,
        maps_anonymous_exec=min(anon, exec_),
        exec_region_growth_rate=0,
        syscall_entropy=entr,
        io_read_bytes_delta=ior, io_write_bytes_delta=iow,
    )


# ──────────────────────────────────────────────────────────────────────
# Attack profiles
# ──────────────────────────────────────────────────────────────────────

def sample_heap_spray(rng, poll_idx):
    """Anonymous executable region count grows over time (exec_region_growth_rate > 0),
    syscall entropy is very low (tight repetitive loop), RSS grows."""
    anon = int(rng_normal(8 + poll_idx * 0.5, 1.5, 4, 24, rng=rng))
    exec_ = int(rng_normal(20 + anon * 0.3, 3, 15, 50, rng=rng))
    return dict(
        vmrss_kb=int(rng_normal(150000, 40000, 30000, 400000, rng=rng)),
        vmrss_delta=int(rng_normal(8000, 3000, 1000, 40000, rng=rng)),
        fd_count=int(rng_normal(10, 4, 3, 25, rng=rng)),
        fd_proc_mem_open=0,
        maps_total_regions=int(rng_normal(130 + anon, 20, 80, 250, rng=rng)),
        maps_exec_region_count=exec_,
        maps_anonymous_exec=anon,
        exec_region_growth_rate=int(rng_normal(3, 1, 1, 8, rng=rng)),
        syscall_entropy=round(rng_normal(0.2, 0.1, 0.0, 0.5, rng=rng), 3),
        io_read_bytes_delta=int(rng_normal(5000, 3000, 0, 20000, rng=rng)),
        io_write_bytes_delta=int(rng_normal(1000, 800, 0, 5000, rng=rng)),
    )


def sample_proc_scraper(rng, poll_idx):
    """Holds an fd into another process's /proc/mem; massive read I/O;
    low entropy (tight read loop)."""
    return dict(
        vmrss_kb=int(rng_normal(25000, 8000, 8000, 60000, rng=rng)),
        vmrss_delta=int(rng_normal(200, 300, -500, 2000, rng=rng)),
        fd_count=int(rng_normal(12, 4, 4, 30, rng=rng)),
        fd_proc_mem_open=1,                               # ← the key signal
        maps_total_regions=int(rng_normal(70, 12, 40, 110, rng=rng)),
        maps_exec_region_count=int(rng_normal(12, 3, 6, 20, rng=rng)),
        maps_anonymous_exec=int(rng_normal(1, 0.7, 0, 3, rng=rng)),
        exec_region_growth_rate=0,
        syscall_entropy=round(rng_normal(0.7, 0.2, 0.2, 1.2, rng=rng), 3),
        io_read_bytes_delta=int(rng_normal(1500000, 500000, 200000, 4000000, rng=rng)),
        io_write_bytes_delta=int(rng_normal(500, 400, 0, 3000, rng=rng)),
    )


def sample_mem_injection(rng, poll_idx):
    """Sudden large RSS spike + rapid exec-region growth + several
    anonymous executable mappings appearing mid-run."""
    anon = int(rng_normal(7 + poll_idx * 0.8, 2, 3, 18, rng=rng))
    exec_ = int(rng_normal(20, 5, 12, 35, rng=rng))
    return dict(
        vmrss_kb=int(rng_normal(120000, 40000, 30000, 350000, rng=rng)),
        vmrss_delta=int(rng_normal(40000, 15000, 5000, 120000, rng=rng)),
        fd_count=int(rng_normal(15, 5, 4, 35, rng=rng)),
        fd_proc_mem_open=0,
        maps_total_regions=int(rng_normal(140, 25, 80, 230, rng=rng)),
        maps_exec_region_count=exec_,
        maps_anonymous_exec=min(anon, exec_),
        exec_region_growth_rate=int(rng_normal(5, 2, 2, 12, rng=rng)),
        syscall_entropy=round(rng_normal(0.6, 0.2, 0.1, 1.1, rng=rng), 3),
        io_read_bytes_delta=int(rng_normal(40000, 20000, 2000, 150000, rng=rng)),
        io_write_bytes_delta=int(rng_normal(8000, 5000, 0, 40000, rng=rng)),
    )


# ──────────────────────────────────────────────────────────────────────
# Dataset assembly
# ──────────────────────────────────────────────────────────────────────

def generate(n_polls=20, seed=42):
    rng = random.Random(seed)
    rows = []

    # ── Normal processes ──────────────────────────────────────────────
    # Give each process type a stable PID and simulate n_polls observations
    # of it (with memory state carried forward between polls).
    next_pid = 1000
    for profile_name, profile in NORMAL_PROFILES:
        # 3–5 instances of each process type
        n_instances = rng.randint(3, 5)
        for _ in range(n_instances):
            pid = next_pid; next_pid += 1
            vmrss_prev = None
            for poll in range(n_polls):
                feat = sample_normal(profile, rng, vmrss_prev)
                vmrss_prev = feat['vmrss_kb']
                rows.append(dict(pid=pid, name=profile_name,
                                 poll_id=poll,
                                 is_attack=0, attack_type='normal',
                                 **feat))

    # ── Attack processes ──────────────────────────────────────────────
    # Fewer instances, start appearing partway through the poll window
    # to mirror a real attack (process starts, ramps up, persists).
    ATTACK_SAMPLERS = [
        ('heap_spray',    sample_heap_spray,    3),
        ('proc_scraper',  sample_proc_scraper,  2),
        ('mem_injection', sample_mem_injection, 2),
    ]
    attack_start_poll = n_polls // 3   # attacks begin 1/3 into the window

    for attack_type, sampler, n_instances in ATTACK_SAMPLERS:
        for _ in range(n_instances):
            pid = next_pid; next_pid += 1
            for poll in range(attack_start_poll, n_polls):
                local_idx = poll - attack_start_poll
                feat = sampler(rng, local_idx)
                rows.append(dict(pid=pid, name=f'[{attack_type}]',
                                 poll_id=poll,
                                 is_attack=1, attack_type=attack_type,
                                 **feat))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed',   type=int, default=42)
    ap.add_argument('--polls',  type=int, default=20,
                     help='Number of simulated polling intervals (default 20)')
    ap.add_argument('--out',    type=str, default=OUT_PATH)
    args = ap.parse_args()

    rows = generate(n_polls=args.polls, seed=args.seed)

    fieldnames = ['pid', 'name', 'poll_id', 'is_attack', 'attack_type'] + FEATURE_NAMES
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    normal  = sum(1 for r in rows if r['is_attack'] == 0)
    attacks = sum(1 for r in rows if r['is_attack'] == 1)
    print(f"Saved {len(rows)} rows ({normal} normal, {attacks} attack) → {args.out}")
    from collections import Counter
    cat_counts = Counter(r['attack_type'] for r in rows if r['is_attack'])
    for k, v in cat_counts.items():
        print(f"  {k}: {v} rows")


if __name__ == '__main__':
    main()
