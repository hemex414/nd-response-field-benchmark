# -*- coding: utf-8 -*-
"""sa -- single-case public implementation (algorithm and parameters from the
reference default, unchanged; only the field/dimension interface is public).
"""
import heapq
import math

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree

from .. import evaluate as B
from .common import *          # noqa: F401,F403  (helpers, PARAMS, constants)


def plan(field_nd, case):
    p = PARAMS["sa"]
    pf = PrivField(field_nd)
    n = int(field_nd.N)
    a = np.asarray(case["A"], float) / PUB
    b = np.asarray(case["B"], float) / PUB
    rate = np.asarray(case["rates"], float) / PUB
    best_path, best_cost = None, np.inf
    for seed in p["SEEDS"]:
        rng = np.random.default_rng(seed)
        x = np.linspace(a, b, p["N_POINTS"])
        cur = float(np.mean(pf.interp(x)))
        bx, bc = x.copy(), cur
        temp = p["T_INIT"]
        for _ in range(p["N_ITER"]):
            idx = int(rng.integers(1, p["N_POINTS"] - 1))
            cand = x.copy()
            cand[idx] += rng.normal(0, p["PERTURB_SCALE"], size=n)
            cand = np.clip(cand, 0, WORKSPACE)
            if not in_ellipsoid(cand[idx], a, b, BETA):
                continue
            nc = float(np.mean(pf.interp(cand)))
            delta = nc - cur
            if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                x, cur = cand, nc
                if cur < bc:
                    bx, bc = x.copy(), cur
            temp *= p["COOLING"]
        bx[0] = a; bx[-1] = b
        if bc < best_cost:
            best_cost, best_path = bc, bx
    return path_to_record("sa", field_nd, case, best_path,
                          extra={"n_points": p["N_POINTS"], "iters": p["N_ITER"],
                                 "seeds": p["SEEDS"]})
