# -*- coding: utf-8 -*-
"""random -- random in-corridor trajectory baseline (public package).

Reference baseline kept for the public field: K = 8 segments (9 nodes), nodes sampled
uniformly **inside the time-indexed corridor** at the uniform-time reference points
``xbar(k/K)``, uniform segment durations ``T_b / K``.  No tuning: the per-case RNG seed is
the case id, so the baseline is reproducible and case-independent in expectation.

Feasibility is reported exactly as the shared evaluator sees it (endpoint / rate /
time-indexed corridor / T_b), without post-processing repair -- the random baseline is a
sanity/reference arm, not an optimiser.

``params``: K=8, rho=0.2, delta=0.02, uniform durations, seed = case_id.
"""
from __future__ import annotations

import numpy as np

from .. import evaluate as B
from .common import path_to_record, PUB, WORKSPACE, BETA  # noqa: F401  (interface parity)

K_RANDOM = 8
RHO = 0.2
DELTA = 0.02


def _corr_time(prob, nodes, durations):
    t = np.concatenate([[0.0], np.cumsum(np.asarray(durations, float))])
    tau = t / t[-1]
    xb = prob.a[None, :] + tau[:, None] * prob.dir[None, :]
    return float(np.max(np.clip(np.abs(np.asarray(nodes) - xb) / prob.bd[None, :] - 1.0,
                                0.0, None)))


def plan(field_nd, case):
    n = int(field_nd.N)
    a = np.asarray(case["A"], float)
    b = np.asarray(case["B"], float)
    rates = np.asarray(case["rates"], float)
    t_fast = float(np.max(np.abs(b - a) / rates))
    T_b = float(case["TR"]) * t_fast
    prob = B.Problem(field_nd, a, b, T_b, rates, K=K_RANDOM, rho=RHO, delta=DELTA)
    rng = np.random.default_rng(int(case.get("case_id", 0)))
    nodes = prob.base.copy()                      # xbar(k/K), uniform-time reference
    bd = prob.bd
    if K_RANDOM > 1:
        off = rng.uniform(-1.0, 1.0, size=(K_RANDOM - 1, n)) * bd[None, :]
        nodes[1:-1] = nodes[1:-1] + off
    # stay inside the box and the time-indexed corridor (alternating projection)
    for _ in range(8):
        nodes = np.clip(nodes, prob.lo[None, :], prob.hi[None, :])
        nodes = np.clip(nodes, prob.base - bd[None, :], prob.base + bd[None, :])
    nodes[0] = a
    nodes[-1] = b
    durations = np.full(K_RANDOM, prob.dt)
    ev = prob.eval(nodes, durations=durations)
    corr_time = _corr_time(prob, nodes, durations)
    corr_geom = float(np.max(prob.corr_viol(nodes)))
    ep = float(ev["endpoint_error"])
    rv = float(ev["rate_violation"])
    feas = (ep <= 1e-3 and rv <= 1e-2 and corr_time <= 1e-6
            and abs(ev["T"] - T_b) <= max(0.01 * T_b, prob.dt))
    return {
        "case_id": case["case_id"], "N": n, "subset": case["subset"],
        "field_seed": case["field_seed"], "A": case["A"], "B": case["B"],
        "TR": case["TR"], "rates": case["rates"],
        "device": "cpu", "method": "random", "K": K_RANDOM,
        "J": float(ev["J"]), "J_uniform": float(ev["J"]),
        "T": float(ev["T"]), "durations": [float(v) for v in durations],
        "time_alloc": "none (uniform durations)", "time_alloc_applied": False,
        "endpoint_error": ep, "rate_violation": rv,
        "corridor_violation": corr_time,
        "corridor_violation_time": corr_time,
        "corridor_violation_geom": corr_geom,
        "feasible": bool(feas),
        "repair_fac": 1.0,
        "points": np.asarray(nodes, float).round(6).tolist(),
        "status": "ok",
        "notes": "random in-corridor sampling, uniform durations, no tuning/repair",
        "params": {"K": K_RANDOM, "rho": RHO, "delta": DELTA,
                   "seed": "case_id", "durations": "uniform",
                   "sampling": "uniform inside time-indexed corridor"},
    }

