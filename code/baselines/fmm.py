# -*- coding: utf-8 -*-
"""fmm -- single-case public implementation (algorithm and parameters from the
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
    import skfmm
    p = PARAMS["fmm"]
    pf = PrivField(field_nd)
    n = int(field_nd.N)
    a = np.asarray(case["A"], float) / PUB
    b = np.asarray(case["B"], float) / PUB
    V, axes = make_grid(pf, n)
    res = GRID_RES
    dx = axes[1] - axes[0]
    mesh = np.stack(np.meshgrid(*([axes] * n), indexing="ij"), -1).reshape(-1, n)
    mask = in_ellipsoid(mesh, a, b, BETA).reshape((res,) * n)
    si = tuple(np.clip(np.rint(a / dx).astype(int), 0, res - 1))
    ei = tuple(np.clip(np.rint(b / dx).astype(int), 0, res - 1))
    cache = {}

    def trace(T):
        grads = np.gradient(T)
        grads = [grads] if isinstance(grads, np.ndarray) else list(grads)
        i = np.array(ei); seq = [i.copy()]; seen = set()
        for _ in range(5000):
            key = tuple(i.tolist())
            if key in seen:
                break
            seen.add(key)
            if np.max(np.abs(i - np.asarray(si))) <= 1:
                break
            g = np.array([grads[d][key] for d in range(n)])
            gn = float(np.linalg.norm(g))
            if not np.isfinite(gn) or gn < 1e-12:
                break
            nxt = np.clip(np.rint(i - g / gn * p["trace_step"]).astype(int), 0, res - 1)
            if np.all(nxt == i):
                break
            i = nxt; seq.append(i.copy())
        seq.append(np.asarray(si)); seq = seq[::-1]
        pts = np.array([np.asarray(k, float) * dx for k in seq])
        return pts

    def run_alpha(alpha):
        if alpha in cache:
            return cache[alpha]
        if si == ei:
            # start and end fall into the same grid cell -> eikonal source degenerates
            # (skfmm: "no zero contour"); the only grid path is the straight segment
            cache[alpha] = np.vstack([a, b])
            return cache[alpha]
        speed = np.exp(-alpha * V)
        eps_out = float(np.exp(-alpha * float(np.max(V)))) * 1e-2
        speed = np.where(mask, speed, eps_out)
        phi = np.ones_like(V); phi[si] = -1.0
        try:
            T = skfmm.travel_time(phi, speed=speed, dx=(dx,) * n)
        except ValueError:
            # degenerate configuration (no zero contour): straight-line fallback
            path = np.vstack([a, b])
            cache[alpha] = path
            return path
        T = np.asarray(np.ma.filled(T, np.nan), float)
        if not np.all(np.isfinite(T)):
            T = np.where(np.isfinite(T), T, np.nanmax(T[np.isfinite(T)])
                         if np.any(np.isfinite(T)) else 1.0)
        path = trace(T); path[0] = a; path[-1] = b
        cache[alpha] = path
        return path

    dens = 0.5 * 100.0          # reference DENSE_STEP*? -> keep ~0.5% of workspace
    best, best_j = None, np.inf
    jobs = list(p["COARSE_ALPHAS"])
    for alpha in jobs:
        path = densify(run_alpha(alpha), dens)
        rec = path_to_record("fmm", field_nd, case, path)
        if rec["J"] < best_j:
            best_j, best = rec["J"], (alpha, path)
    coarse_star = best[0]
    for alpha in (round(coarse_star - p["refine"], 2), round(coarse_star + p["refine"], 2)):
        if p["alpha_lo"] <= alpha <= p["alpha_hi"] and alpha not in cache:
            path = densify(run_alpha(alpha), dens)
            rec = path_to_record("fmm", field_nd, case, path)
            if rec["J"] < best_j:
                best_j, best = rec["J"], (alpha, path)
    return path_to_record("fmm", field_nd, case, best[1],
                          extra={"alpha": best[0], "grid_nodes": int(V.size)})
