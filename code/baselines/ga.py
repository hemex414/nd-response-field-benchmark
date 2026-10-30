# -*- coding: utf-8 -*-
"""ga -- single-case public implementation (algorithm and parameters from the
reference default, unchanged; only the field/dimension interface is public).
"""
import heapq
import math

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree

from .. import evaluate as B
from .common import *          # noqa: F401,F403  (helpers, PARAMS, constants)


def _bspline_path(control, degree, n_eval):
    m = len(control)
    t = np.concatenate([np.zeros(degree), np.arange(1, m - degree + 1),
                        np.full(degree, m - degree + 1)])
    return BSpline(t, control, degree)(np.linspace(0, m - degree + 1, n_eval))


def plan(field_nd, case):
    p = PARAMS["ga"]
    pf = PrivField(field_nd)
    n = int(field_nd.N)
    a = np.asarray(case["A"], float) / PUB
    b = np.asarray(case["B"], float) / PUB
    rate = np.asarray(case["rates"], float) / PUB
    rng = np.random.default_rng(p["seed"])
    m = p["N_CONTROL"]
    base = np.linspace(a, b, m)
    sigma = p["INIT_SIGMA"]
    x = base + rng.normal(0, sigma * 0.3, size=base.shape)
    x[0] = a; x[-1] = b

    def fitness(ctrl):
        path = np.clip(_bspline_path(ctrl, p["DEGREE"], p["N_EVAL_PTS"]), 0, WORKSPACE)
        frac = 1.0 - float(in_ellipsoid(path, a, b, BETA).mean())
        return mean_field(pf, path, rate) + 20.0 * frac

    fx = fitness(x)
    for _ in range(p["N_GEN"]):
        cand = x + rng.normal(0, sigma, size=x.shape)
        cand[0] = a; cand[-1] = b
        fc = fitness(cand)
        if fc < fx:
            x, fx = cand, fc
            sigma *= p["SIGMA_UP"]
        else:
            sigma *= p["SIGMA_DOWN"]
        sigma = min(max(sigma, 0.5), 500.0)
    path = np.clip(_bspline_path(x, p["DEGREE"], p["N_EVAL_PTS"]), 0, WORKSPACE)
    path[0] = a; path[-1] = b
    return path_to_record("ga", field_nd, case, path,
                          extra={"gens": p["N_GEN"], "ctrl": p["N_CONTROL"],
                                 "sigma0": p["INIT_SIGMA"]})
