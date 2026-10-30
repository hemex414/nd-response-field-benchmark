# -*- coding: utf-8 -*-
"""pool.py -- case-parallel execution of the CPU planners (FMM/A*/Theta*/Lattice/RRT*/BIT*).

Each case is completely independent (its own field seed and rates), so a process pool
computes the same result as the single-case runner, one case per task.  The pool is
created with the "spawn" start method on Windows; workers import the package and rebuild
the field from ``(N, field_seed)``, so no large state is pickled per task.
"""
from __future__ import annotations

import multiprocessing as mp
import os


def _worker(args):
    method, n, field_seed, case = args
    from ..field import FieldND
    from . import plan_baseline
    field = FieldND(n, field_seed)
    return plan_baseline(method, field, case)


def plan_pool(method, cases, workers=16):
    """Run `method` on all cases with `workers` processes; returns records in case order."""
    tasks = [(method, int(c["N"]), int(c["field_seed"]), c) for c in cases]
    workers = max(1, int(workers))
    if workers == 1:
        return [_worker(t) for t in tasks]
    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    with ctx.Pool(processes=workers) as pool:
        out = pool.map(_worker, tasks, chunksize=1)
    return out
