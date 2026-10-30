# -*- coding: utf-8 -*-
"""chomp -- single-case public implementation (algorithm and parameters from the
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
    p = PARAMS["chomp"]
    pf = PrivField(field_nd)
    n = int(field_nd.N)
    a = np.asarray(case["A"], float) / PUB
    b = np.asarray(case["B"], float) / PUB
    rate = np.asarray(case["rates"], float) / PUB
    eps = p["GRAD_EPS"]
    npts = p["N_POINTS"]

    def smooth_grad(x):
        N_ = x.shape[0]; g = np.zeros_like(x); d = np.zeros((N_, n))
        d[1:N_ - 1] = x[2:] - 2 * x[1:N_ - 1] + x[:N_ - 2]
        g[1:N_ - 1] = 2 * d[:N_ - 2] - 4 * d[1:N_ - 1] + 2 * d[2:]
        return g

    def vib_grad(x):
        interior = x[1:-1]
        M, _ = interior.shape
        pts = np.empty((M, n, 2, n))
        for d in range(n):
            off = np.zeros(n); off[d] = eps
            pts[:, d, 0, :] = interior + off
            pts[:, d, 1, :] = interior - off
        vals = pf.interp(np.clip(pts.reshape(-1, n), 0, WORKSPACE)).reshape(M, n, 2)
        return (vals[:, :, 0] - vals[:, :, 1]) / (2 * eps)

    best_path, best_obj = None, np.inf
    for w_s in p["W_SMOOTH_CAND"]:
        for w_v in p["W_VIB_CAND"]:
            x = np.linspace(a, b, npts)
            for _ in range(p["N_ITER"]):
                g = w_s * smooth_grad(x) + w_v * np.vstack(
                    [np.zeros(n), vib_grad(x), np.zeros(n)])
                xn = x.copy()
                xn[1:-1] = x[1:-1] - p["LR"] * g[1:-1]
                xn = np.clip(xn, 0, WORKSPACE)
                ok = in_ellipsoid(xn[1:-1], a, b, BETA)
                if not np.all(ok):
                    xn[1:-1] = x[1:-1]
                x = xn
            obj = mean_field(pf, x, rate)
            if obj < best_obj:
                best_obj, best_path = obj, x
    return path_to_record("chomp", field_nd, case, best_path,
                          extra={"n_points": npts, "iters": p["N_ITER"],
                                 "w_smooth": p["W_SMOOTH_CAND"],
                                 "w_vib": p["W_VIB_CAND"]})
