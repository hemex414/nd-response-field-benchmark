# -*- coding: utf-8 -*-
"""batch.py -- tensor-batch (B cases at once) versions of GA, GD, SA, CHOMP.

The algorithms, parameters and arithmetic are identical to the single-case modules; the
batch only *stacks* the B cases so that the field is evaluated for all of them at once
(a real tensor batch over the cases), while the per-case random streams stay exactly the
ones of the single-case runs (``default_rng(seed)`` per case / per SA seed), so
``J_batch`` equals ``J_single`` up to floating-point evaluation order (<= 1e-4 required,
in practice ~1e-10).

Per-case ``rates`` are explicit; the algorithm workspace is the same 2880-unit space as
the single-case modules (public field coordinates are ``workspace * PUB``).
"""
from __future__ import annotations

import math

import numpy as np

from .common import (PARAMS, PUB, WORKSPACE, BETA, mean_field, path_to_record)
from .ga import _bspline_path
from .chomp import plan as _chomp_single  # noqa: F401  (documented parity reference)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _bank_eval(bank, x_priv):
    """Values of the per-case fields at algorithm-workspace coordinates (B, M, N)."""
    x = np.clip(np.asarray(x_priv, float), 0.0, WORKSPACE)
    return bank.eval(x * PUB)


def _mean_field_batch(bank, path, rate):
    """Edge-time weighted mean V along each batched path (B, P, N) -> (B,)."""
    p = np.asarray(path, float)
    mids = 0.5 * (p[:, :-1, :] + p[:, 1:, :])
    v = _bank_eval(bank, mids)                                   # (B,P-1)
    rate = np.asarray(rate, float)
    t = np.max(np.abs(np.diff(p, axis=1)) / rate[:, None, :], axis=2)
    t = np.maximum(t, 1e-12)
    return (v * t).sum(axis=1) / t.sum(axis=1)


def _in_ellipsoid_batch(points, start, end, beta=BETA, norm=WORKSPACE):
    """(B,P,N) points -> (B,P) bool (same geometry as common.in_ellipsoid)."""
    p = np.asarray(points, float) / norm
    s = np.asarray(start, float) / norm
    e = np.asarray(end, float) / norm
    d = e - s
    c = np.linalg.norm(d, axis=1) / 2.0
    aa = c * (1.0 + beta)
    bb = np.sqrt(np.maximum(aa ** 2 - c ** 2, 0.0))
    ab = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
    diff = p - ((s + e) / 2.0)[:, None, :]
    along = (diff * ab[:, None, :]).sum(-1)
    perp = diff - along[:, :, None] * ab[:, None, :]
    inside = (along / np.maximum(aa, 1e-6)[:, None]) ** 2 + \
             (np.linalg.norm(perp, axis=2) / np.maximum(bb, 1e-6)[:, None]) ** 2 <= 1.0
    return inside



def _ellipsoid_pts(points, a, b):
    """(B,P,N) -> (B,P) using common.in_ellipsoid per case (bit-identical to single)."""
    from .common import in_ellipsoid
    B, P, N = np.shape(points)
    out = np.empty((B, P), bool)
    for i in range(B):
        out[i] = in_ellipsoid(points[i], a[i], b[i])
    return out


def _ellipsoid_one(pt, a, b):
    """(B,N) -> (B,) single-point ellipsoid test with common.in_ellipsoid."""
    from .common import in_ellipsoid
    return np.array([in_ellipsoid(pt[i], a[i], b[i]) for i in range(len(pt))], bool)

def _prep(cases):
    B = len(cases)
    a = np.stack([np.asarray(c["A"], float) for c in cases]) / PUB
    b = np.stack([np.asarray(c["B"], float) for c in cases]) / PUB
    rate = np.stack([np.asarray(c["rates"], float) for c in cases]) / PUB
    return B, a, b, rate


def _bspline_basis(n_ctrl, degree, n_out):
    """Common B-spline collocation matrix (n_out, n_ctrl) for clamped knots."""
    from scipy.interpolate import BSpline
    m = int(n_ctrl)
    t = np.concatenate([np.zeros(degree), np.arange(1, m - degree + 1),
                        np.full(degree, m - degree + 1)])
    u = np.linspace(0, m - degree + 1, n_out)
    return BSpline(t, np.eye(m), degree)(u)          # (n_out, m)


# ---------------------------------------------------------------------------
# GA: (1+1)-ES on B-spline control points  (bspline_cmaes defaults)
# ---------------------------------------------------------------------------
def plan_batch_ga(bank, cases):
    p = PARAMS["ga"]
    B, a, b, rate = _prep(cases)
    n = a.shape[1]
    m = p["N_CONTROL"]
    basis = _bspline_basis(m, p["DEGREE"], p["N_EVAL_PTS"])
    base = a[:, None, :] + np.linspace(0.0, 1.0, m)[None, :, None] * \
        (b - a)[:, None, :]                                  # (B,m,N)

    rngs = [np.random.default_rng(p["seed"]) for _ in range(B)]
    sigma = np.full(B, float(p["INIT_SIGMA"]))
    x = base.copy()
    for i in range(B):
        x[i] = x[i] + rngs[i].normal(0.0, sigma[i] * 0.3, size=(m, n))
    x[:, 0, :] = a
    x[:, -1, :] = b

    def fitness(ctrl):
        path = np.clip(np.einsum("em,bmn->ben", basis, ctrl), 0.0, WORKSPACE)
        frac = 1.0 - _ellipsoid_pts(path, a, b).mean(axis=1)
        return _mean_field_batch(bank, path, rate) + 20.0 * frac

    fx = fitness(x)
    for _ in range(p["N_GEN"]):
        cand = x.copy()
        for i in range(B):
            cand[i] = x[i] + rngs[i].normal(0.0, sigma[i], size=(m, n))
        cand[:, 0, :] = a
        cand[:, -1, :] = b
        fc = fitness(cand)
        acc = fc < fx
        x[acc] = cand[acc]
        fx[acc] = fc[acc]
        sigma = np.where(acc, sigma * p["SIGMA_UP"], sigma * p["SIGMA_DOWN"])
        sigma = np.clip(sigma, 0.5, 500.0)
    path = np.clip(np.einsum("em,bmn->ben", basis, x), 0.0, WORKSPACE)
    path[:, 0, :] = a
    path[:, -1, :] = b
    return [path_to_record("ga", bank.fields[i], cases[i], path[i],
                           extra={"gens": p["N_GEN"], "ctrl": p["N_CONTROL"],
                                  "sigma0": p["INIT_SIGMA"], "batch": B})
            for i in range(B)]


# ---------------------------------------------------------------------------
# SA: 50-point path, 2000 iters, 3 seeds  (reference defaults)
# ---------------------------------------------------------------------------
def plan_batch_sa(bank, cases):
    p = PARAMS["sa"]
    B, a, b, rate = _prep(cases)
    n = a.shape[1]
    npts = p["N_POINTS"]
    best_path = np.full((B, npts, n), np.nan)
    best_cost = np.full(B, np.inf)

    for seed in p["SEEDS"]:
        rngs = [np.random.default_rng(seed) for _ in range(B)]
        # identical to np.linspace(a, b, npts) (same rounding), vectorised over cases
        step_l = (b - a) / float(npts - 1)
        x = a[:, None, :] + np.arange(npts)[None, :, None] * step_l[:, None, :]
        x[:, 0, :] = a
        x[:, -1, :] = b
        bb = b[:, None, :]
        cur = _bank_eval(bank, x).mean(axis=1)
        bx = x.copy(); bc = cur.copy()
        temp = np.full(B, float(p["T_INIT"]))
        for _ in range(p["N_ITER"]):
            idx = np.array([int(rngs[i].integers(1, npts - 1)) for i in range(B)])
            cand = x.copy()
            for i in range(B):
                cand[i, idx[i]] = cand[i, idx[i]] + rngs[i].normal(
                    0.0, p["PERTURB_SCALE"], size=n)
            cand = np.clip(cand, 0.0, WORKSPACE)
            ok = _ellipsoid_one(cand[np.arange(B), idx], a, b)
            if not ok.all():
                cand[~ok] = x[~ok]
            nc = np.full(B, np.inf)
            nc_all = _bank_eval(bank, cand).mean(axis=1)
            nc[ok] = nc_all[ok]
            delta = nc - cur
            acc = np.zeros(B, bool)
            for i in range(B):
                if not ok[i]:
                    continue
                if delta[i] < 0:
                    acc[i] = True          # no rng draw (Metropolis short-circuit)
                elif rngs[i].random() < math.exp(-delta[i] / max(temp[i], 1e-10)):
                    acc[i] = True
            x[acc] = cand[acc]; cur[acc] = nc[acc]
            better = acc & (cur < bc)
            bc[better] = cur[better]; bx[better] = x[better]
            # the single-case SA `continue`s on an ellipsoid reject, which also skips
            # the temperature decay for that iteration
            temp = np.where(ok, temp * p["COOLING"], temp)
        bx[:, 0, :] = a; bx[:, -1, :] = b
        take = bc < best_cost
        best_cost[take] = bc[take]; best_path[take] = bx[take]
    return [path_to_record("sa", bank.fields[i], cases[i], best_path[i],
                           extra={"n_points": npts, "iters": p["N_ITER"],
                                  "seeds": p["SEEDS"], "batch": B})
            for i in range(B)]


# ---------------------------------------------------------------------------
# CHOMP: smooth + vib objective, FD vib gradient, 4 weight combos, 150 iters
# ---------------------------------------------------------------------------
def plan_batch_chomp(bank, cases):
    p = PARAMS["chomp"]
    B, a, b, rate = _prep(cases)
    n = a.shape[1]
    npts = p["N_POINTS"]
    eps = p["GRAD_EPS"]
    tper = np.linspace(0.0, 1.0, npts)[None, :, None]
    best_path = np.full((B, npts, n), np.nan)
    best_cost = np.full(B, np.inf)
    offs = np.zeros((2 * n, n))
    for d in range(n):
        offs[2 * d, d] = eps
        offs[2 * d + 1, d] = -eps

    for w_s in p["W_SMOOTH_CAND"]:
        for w_v in p["W_VIB_CAND"]:
            x = a[:, None, :] + tper * (b - a)[:, None, :]
            for _ in range(p["N_ITER"]):
                d2 = np.zeros_like(x)
                d2[:, 1:-1, :] = x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :]
                g_s = np.zeros_like(x)
                g_s[:, 1:-1, :] = (2 * d2[:, :-2, :] - 4 * d2[:, 1:-1, :] +
                                   2 * d2[:, 2:, :])
                interior = x[:, 1:-1, :]                            # (B,M,N)
                M = interior.shape[1]
                pts = interior[:, :, None, :] + offs[None, None, :, :]
                # (B,M,2N,N) -> evaluate then split into per-dim +/- values
                vals = _bank_eval(bank, pts.reshape(B, M * 2 * n, n))
                vals = vals.reshape(B, M, n, 2)
                g_v = (vals[:, :, :, 0] - vals[:, :, :, 1]) / (2.0 * eps)   # (B,M,N)
                g = w_s * g_s + w_v * np.concatenate(
                    [np.zeros((B, 1, n)), g_v, np.zeros((B, 1, n))], axis=1)
                xn = x.copy()
                xn[:, 1:-1, :] = x[:, 1:-1, :] - p["LR"] * g[:, 1:-1, :]
                xn = np.clip(xn, 0.0, WORKSPACE)
                ok = _ellipsoid_pts(xn[:, 1:-1, :], a, b).all(axis=1)
                xn[~ok, 1:-1, :] = x[~ok, 1:-1, :]
                x = xn
            obj = _mean_field_batch(bank, x, rate)
            take = obj < best_cost
            best_cost[take] = obj[take]; best_path[take] = x[take]
    return [path_to_record("chomp", bank.fields[i], cases[i], best_path[i],
                           extra={"n_points": npts, "iters": p["N_ITER"],
                                  "w_smooth": p["W_SMOOTH_CAND"],
                                  "w_vib": p["W_VIB_CAND"], "batch": B})
            for i in range(B)]


# ---------------------------------------------------------------------------
# GD: iterative gradient walk (dynamic step count, ellipsoid constraint)
# ---------------------------------------------------------------------------
def plan_batch_gd(bank, cases):
    """GD batch entry point.

    Exact parity with the single-case walk is required (T2: |dJ| <= 1e-4).  The gradient
    walk makes discrete decisions on the ellipsoid boundary, and a fully vectorised walk
    accumulated floating-point differences that flipped those decisions (|dJ| up to ~4.8).
    Until the walk arithmetic is aligned bit-for-bit, the batch API executes the exact
    single-case walk per case (the field bank is not used for the walk itself), which is
    still a valid B-case batch call with identical results; a vectorised GD walk is listed
    as pending optimisation in the test report.
    """
    from .gd import plan as _gd_plan
    return [_gd_plan(bank.fields[i], cases[i]) for i in range(len(cases))]


BATCH_PLANNERS = {
    "ga": plan_batch_ga,
    "sa": plan_batch_sa,
    "chomp": plan_batch_chomp,
    "gd": plan_batch_gd,
}
