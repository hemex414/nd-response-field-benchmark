# -*- coding: utf-8 -*-
"""baselines -- public N-D baseline planners (single case; batch wrappers in batch.py).

Every algorithm keeps the reference hyper-parameters verbatim (ALGORITHM_REGISTRY.md);
only the field/dimension interface is public and per-case ``rates`` are passed explicitly.
"""
from __future__ import annotations

import time

import numpy as np

from . import common
from .common import PARAMS, METHODS, PUB, WORKSPACE, GRID_RES, GridTooLarge

from . import fmm as _fmm
from . import astar as _astar
from . import theta_star as _theta_star
from . import lattice as _lattice
from . import rrt_star as _rrt_star
from . import bit_star as _bit_star
from . import ga as _ga
from . import gd as _gd
from . import sa as _sa
from . import chomp as _chomp
from . import random as _random

PLANNERS = {
    "fmm": _fmm.plan,
    "astar": _astar.plan,
    "theta_star": _theta_star.plan,
    "lattice": _lattice.plan,
    "rrt_star": _rrt_star.plan,
    "bit_star": _bit_star.plan,
    "ga": _ga.plan,
    "gd": _gd.plan,
    "sa": _sa.plan,
    "chomp": _chomp.plan,
    "random": _random.plan,
}
# algorithms that are accelerated by a case-parallel process pool
# case-parallel (process pool) planners; GD keeps the exact per-case walk and is
# parallelised at the case level (design decision: no vectorised GD walk)
POOL_METHODS = ("fmm", "astar", "theta_star", "lattice", "rrt_star", "bit_star", "gd")
# algorithms with a genuine tensor-batch implementation
TENSOR_BATCH_METHODS = ("ga", "sa", "chomp")
# any batched method (tensor batch or pooled)
BATCH_METHODS = POOL_METHODS + TENSOR_BATCH_METHODS
_IMPL = PLANNERS


def plan_baseline(method, field_nd, case, seed=0):
    t0 = time.perf_counter()
    m = method.strip().lower()
    if m not in _IMPL:
        raise KeyError("unknown baseline %r (known: %s)" % (method, ",".join(_IMPL)))
    try:
        rec = _IMPL[m](field_nd, case)
    except GridTooLarge as exc:
        rec = {"case_id": case["case_id"], "N": int(field_nd.N), "method": m,
               "grid_too_large": True, "status": "timeout", "notes": str(exc),
               "J": None, "feasible": None, "subset": case["subset"],
               "field_seed": case["field_seed"], "A": case["A"], "B": case["B"],
               "TR": case["TR"], "rates": case["rates"], "device": "cpu",
               "params": PARAMS[m]}
    rec["runtime_s"] = time.perf_counter() - t0
    return rec


def plan_single(method, case):
    """Single-case entry point used by run_single.py (field rebuilt from the case)."""
    from ..field import FieldND
    field = FieldND(int(case["N"]), int(case["field_seed"]))
    return plan_baseline(method, field, case)


def plan_batch(method, cases, workers=16, batch_size=16):
    """Batch entry point used by run_batch.py.

    * tensor-batch methods (GA/GD/SA/CHOMP): cases are processed ``batch_size`` at a time
      with a stacked :class:`~public_field.code.field.FieldBank`;
    * CPU planners (FMM/A*/Theta*/Lattice/RRT*/BIT*): case-parallel process pool;
    * ``workers=1`` / ``batch_size=1`` reproduce the single-case path exactly.
    """
    method = method.strip().lower()
    cases = list(cases)
    from ..field import FieldBank, FieldND
    if method in TENSOR_BATCH_METHODS:
        from . import batch as batch_mod
        fn = batch_mod.BATCH_PLANNERS[method]
        out = []
        bsz = max(1, int(batch_size))
        for i in range(0, len(cases), bsz):
            chunk = cases[i:i + bsz]
            bank = FieldBank([FieldND(int(c["N"]), int(c["field_seed"])) for c in chunk])
            t0 = time.perf_counter()
            recs = fn(bank, chunk)
            dt = time.perf_counter() - t0
            # runtime of a batched case: its share of the chunk wall-clock
            for r in recs:
                r["runtime_s"] = dt / max(len(chunk), 1)
                r["runtime_mode"] = "batch_share"
            out.extend(recs)
        return out
    if method in POOL_METHODS:
        from . import pool
        return pool.plan_pool(method, cases, workers=workers)
    if method in _IMPL:
        return [plan_single(method, c) for c in cases]
    raise KeyError("unknown baseline %r" % method)


__all__ = ["plan_baseline", "plan_single", "plan_batch", "PARAMS", "METHODS",
           "POOL_METHODS", "TENSOR_BATCH_METHODS", "BATCH_METHODS", "PrivField", "in_ellipsoid",
           "path_to_record", "PUB", "WORKSPACE", "GRID_RES", "GridTooLarge"]
