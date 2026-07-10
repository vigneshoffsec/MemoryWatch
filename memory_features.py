"""
MemoryWatch — live /proc feature extraction
=============================================
Implements the data-collection + feature-extraction logic described in
Section 4.2 / 4.3 of MemoryWatch_Threat_Model_System_Design.md, using only
unprivileged /proc reads (no ptrace, no eBPF, no root) so it runs on a
normal Linux host or sandbox out of the box.

Feature schema (subset of the full design that is actually obtainable
without elevated privileges — see README for what was descoped and why):

  vmrss_kb                 current resident memory (status)
  vmrss_delta              change in RSS since the previous poll
  fd_count                 number of open file descriptors
  fd_proc_mem_open         1 if this process holds an fd into another
                           process's /proc/[pid]/mem (credential-scraping /
                           cross-process memory access indicator)
  maps_total_regions       number of memory mappings
  maps_exec_region_count   number of mappings with execute permission
  maps_anonymous_exec      anonymous (no backing file) + executable
                           mappings — classic shellcode/heap-spray signature
  exec_region_growth_rate  change in maps_exec_region_count since last poll
  syscall_entropy          Shannon entropy of sampled syscall IDs over a
                           sliding window (low entropy = repetitive,
                           heap-spray-like syscall pattern)
  io_read_bytes_delta      bytes read since last poll
  io_write_bytes_delta     bytes written since last poll

`syscall_entropy` is approximated by point-sampling /proc/[pid]/syscall on
every poll rather than full ptrace-based tracing (Section 4.2's
"`/proc/[pid]/syscall` ... for real-time syscall observation without full
tracing overhead"). `ptrace_attach_count` and `proc_vm_read_count` from the
original design table are NOT implemented here — both require a kernel
audit subsystem (auditd) or eBPF with root, which an unprivileged poller
cannot see. That gap is disclosed, not hidden.
"""

import os
import re
import math
from collections import deque, Counter

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

_MEM_FD_RE = re.compile(r'^/proc/(\d+)/mem$')


def list_pids():
    pids = []
    for entry in os.listdir('/proc'):
        if entry.isdigit():
            pids.append(int(entry))
    return pids


def read_status(pid):
    out = {'name': '?', 'vmrss_kb': 0, 'uid': None}
    with open(f'/proc/{pid}/status', 'r') as f:
        for line in f:
            if line.startswith('Name:'):
                out['name'] = line.split(':', 1)[1].strip()
            elif line.startswith('VmRSS:'):
                parts = line.split()
                out['vmrss_kb'] = int(parts[1]) if len(parts) > 1 else 0
            elif line.startswith('Uid:'):
                parts = line.split()
                if len(parts) > 1:
                    out['uid'] = int(parts[1])
    return out


def read_io(pid):
    out = {'rchar': 0, 'wchar': 0}
    with open(f'/proc/{pid}/io', 'r') as f:
        for line in f:
            k, _, v = line.partition(':')
            k = k.strip()
            if k in ('rchar', 'wchar'):
                out[k] = int(v.strip())
    return out


def read_maps_features(pid):
    """Returns (total_regions, exec_region_count, anonymous_exec_count)."""
    total = exec_count = anon_exec = 0
    with open(f'/proc/{pid}/maps', 'r') as f:
        for line in f:
            parts = line.split(None, 5)
            if len(parts) < 5:
                continue
            perms = parts[1]
            path = parts[5].strip() if len(parts) > 5 else ''
            total += 1
            if 'x' in perms:
                exec_count += 1
                if path == '' or path.startswith('[') is False and path == '':
                    anon_exec += 1
                elif path == '':
                    anon_exec += 1
    return total, exec_count, anon_exec


def read_fd_features(pid):
    """Returns (fd_count, fd_proc_mem_open: 0/1)."""
    fd_dir = f'/proc/{pid}/fd'
    count = 0
    flagged = 0
    for fd in os.listdir(fd_dir):
        count += 1
        try:
            target = os.readlink(os.path.join(fd_dir, fd))
        except OSError:
            continue
        m = _MEM_FD_RE.match(target)
        if m and int(m.group(1)) != pid:
            flagged = 1
    return count, flagged


def read_syscall_id(pid):
    """Point-in-time syscall sample. Returns None if not currently
    blocked in a syscall (kernel reports 'running') or unreadable."""
    with open(f'/proc/{pid}/syscall', 'r') as f:
        line = f.read().strip()
    if not line or line.startswith('running'):
        return None
    try:
        return int(line.split()[0])
    except (ValueError, IndexError):
        return None


def shannon_entropy(counts):
    counts = [c for c in counts if c > 0]
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        p = c / total
        h -= p * math.log2(p)
    return h


class ProcMonitor:
    """Stateful poller: tracks per-pid deltas and syscall-sample windows
    across successive polls of /proc."""

    def __init__(self, syscall_window=10):
        self.syscall_window = syscall_window
        self._hist = {}  # pid -> dict(vmrss, exec, rchar, wchar, sys=deque)

    def poll(self):
        """One full /proc sweep. Returns {pid: feature_dict}. Processes
        that vanish mid-read or aren't readable (permission denied,
        kernel threads with no maps, etc.) are silently skipped — this
        mirrors how a real lightweight poller behaves under normal
        operating conditions."""
        rows = {}
        live_pids = set()
        for pid in list_pids():
            live_pids.add(pid)
            try:
                rows[pid] = self._extract(pid)
            except (FileNotFoundError, ProcessLookupError, PermissionError,
                    OSError):
                continue
        # drop history for processes that have exited
        for pid in list(self._hist.keys()):
            if pid not in live_pids:
                del self._hist[pid]
        return rows

    def _extract(self, pid):
        status = read_status(pid)
        io = read_io(pid)
        total, exec_count, anon_exec = read_maps_features(pid)
        fd_count, fd_flag = read_fd_features(pid)
        sysid = read_syscall_id(pid)

        h = self._hist.get(pid)
        if h is None:
            h = {
                'vmrss': status['vmrss_kb'], 'exec': exec_count,
                'rchar': io['rchar'], 'wchar': io['wchar'],
                'sys': deque(maxlen=self.syscall_window),
            }
            self._hist[pid] = h

        vmrss_delta = status['vmrss_kb'] - h['vmrss']
        exec_growth = exec_count - h['exec']
        rchar_delta = max(io['rchar'] - h['rchar'], 0)
        wchar_delta = max(io['wchar'] - h['wchar'], 0)

        if sysid is not None:
            h['sys'].append(sysid)
        entropy = shannon_entropy(Counter(h['sys']).values())

        h['vmrss'], h['exec'] = status['vmrss_kb'], exec_count
        h['rchar'], h['wchar'] = io['rchar'], io['wchar']

        feat = {
            'vmrss_kb': status['vmrss_kb'],
            'vmrss_delta': vmrss_delta,
            'fd_count': fd_count,
            'fd_proc_mem_open': fd_flag,
            'maps_total_regions': total,
            'maps_exec_region_count': exec_count,
            'maps_anonymous_exec': anon_exec,
            'exec_region_growth_rate': exec_growth,
            'syscall_entropy': entropy,
            'io_read_bytes_delta': rchar_delta,
            'io_write_bytes_delta': wchar_delta,
        }
        feat['_name'] = status['name']
        feat['_pid'] = pid
        return feat


def featurevec(feat_dict):
    """Fixed-order numeric vector matching FEATURE_NAMES."""
    return [feat_dict[k] for k in FEATURE_NAMES]
