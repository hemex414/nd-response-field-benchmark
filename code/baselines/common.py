# -*- coding: utf-8 -*-
"""common.py -- shared geometry / evaluator glue for the public baselines.

Ported from the SPEC_LOCK implementation; **per-case rates are explicit** and the
algorithms keep the reference hyper-parameters verbatim in the 2880-unit algorithm
workspace (PUB = 100/2880 maps public field coordinates into that workspace).
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree

from .. import evaluate as B       # public evaluator (Problem / grid helpers)

PUB = 100.0 / 2880.0                 # public unit -> benchmark RPM unit divisor
WORKSPACE = 2880.0                   # benchmark-unit workspace for all baselines
BETA = 0.2                           # ellipse constraint (reference default)
GRID_RES = 100                       # reference grid resolution per axis -> 100^N nodes
GRID_MAX_NODES = 8_000_000
K_TRAJ = 8

# ---------------------------------------------------------------------------
# reference defaults (verbatim from src/methods/*.py; see SPEC_LOCK.md §5 v2)
# ---------------------------------------------------------------------------
PARAMS = {
    "fmm": {"COARSE_ALPHAS": (0.01, 0.05, 0.10, 0.15, 0.20), "refine": 0.02,
            "alpha_lo": 0.01, "alpha_hi": 0.20, "BETA": BETA, "trace_step": 3,
            "grid": "%d^N" % GRID_RES},
    "astar": {"BETA": BETA, "K_neighbors": 6, "grid": "%d^N" % GRID_RES},
    "theta_star": {"BETA": BETA, "THRESH_OFFSETS": (0.0,), "LOS_SAMPLES": 3,
                   "GRID_STRIDE": 2, "grid": "%d^N" % GRID_RES},
    "lattice": {"BETA": BETA, "N_DIRS": 6, "MAX_LEN": 40.0, "N_PTS_PRC": 10,
                "QUANT": 20.0, "max_expansions": 5000},
    "rrt_star": {"BETA": BETA, "N_ITER": 1500, "STEP_SIZE": 8.0, "TARGET_BIAS": 0.1,
                 "SEEDS": (42,), "rewire": "2*STEP_SIZE"},
    "bit_star": {"BETA": BETA, "N_SAMPLES": 800, "K_NEIGHBORS": 12, "N_ROUNDS": 2,
                 "SEEDS": (42,)},
    "ga": {"N_CONTROL": 10, "DEGREE": 3, "N_GEN": 150, "INIT_SIGMA": 40.0,
           "SIGMA_UP": 1.1, "SIGMA_DOWN": 0.9, "N_EVAL_PTS": 300, "BETA": BETA,
           "seed": 42, "impl": "src/methods/bspline_cmaes.py ((1+1)-ES on B-spline)"},
    "gd": {"STEP_SIZES": (5.0,), "MIXES": (0.6, 0.8), "N_STEPS": 300, "GRAD_EPS": 0.5,
           "BETA": BETA},
    "sa": {"N_POINTS": 50, "N_ITER": 2000, "T_INIT": 50.0, "T_MIN": 0.1,
           "COOLING": 0.995, "PERTURB_SCALE": 15.0, "SEEDS": (42, 123, 456),
           "BETA": BETA},
    "chomp": {"N_POINTS": 100, "N_ITER": 150, "W_SMOOTH_CAND": (1.0, 5.0),
              "W_VIB_CAND": (0.5, 2.0), "LR": 0.5, "GRAD_EPS": 0.5, "BETA": BETA},
    "random": {"K": 6, "rho": 0.20, "delta": 0.02, "durations": "uniform",
               "seed": "case_id", "note": "reference baseline, no tuning/repair"},
}
METHODS = tuple(PARAMS)


class GridTooLarge(RuntimeError):
    pass


class PrivField:
    """View of the public field in the algorithm workspace (x_reference = x_public/PUB).

    The algorithm hyper-parameters are the reference defaults expressed in the
    2880-unit RPM workspace (e.g. RRT* STEP_SIZE=8, GA sigma0=40).  The public field
    lives on [0,100]; this view maps public coordinates into that workspace
    isotropically (PUB = 100/2880), so every parameter keeps its published value.
    """

    def __init__(self, field_nd):
        self.f = field_nd
        self.npf = field_nd
        self.N = int(field_nd.N)

    def interp(self, x):
        x = np.asarray(x, float).reshape(-1, self.N)
        return self.npf.interp(x * PUB)


# ---------------------------------------------------------------------------
# geometry helpers in reference units
# ---------------------------------------------------------------------------
def in_ellipsoid(points, start, end, beta=BETA, norm=WORKSPACE):
    pts = np.asarray(points, float)
    single = pts.ndim == 1
    if single:
        pts = pts[None, :]
    p = pts / norm
    s = np.asarray(start, float) / norm
    e = np.asarray(end, float) / norm
    c = np.linalg.norm(e - s) / 2.0
    if c < 1e-10:
        out = np.ones(len(p), bool)
        return bool(out[0]) if single else out
    aa = c * (1.0 + beta)
    bb = np.sqrt(max(aa ** 2 - c ** 2, 0.0))
    ab = (e - s) / (np.linalg.norm(e - s) + 1e-6)
    diff = p - (s + e) / 2.0
    along = diff @ ab
    perp = diff - along[:, None] * ab[None, :]
    inside = (along / max(aa, 1e-6)) ** 2 + (np.linalg.norm(perp, axis=1) /
                                             max(bb, 1e-6)) ** 2 <= 1.0
    return bool(inside[0]) if single else inside


def edge_time(p1, p2, rate):
    return float(np.max(np.abs(np.asarray(p2) - np.asarray(p1)) / rate))


def densify(path, step):
    path = np.asarray(path, float)
    out = []
    for a, b in zip(path[:-1], path[1:]):
        d = float(np.linalg.norm(b - a))
        n = max(int(np.ceil(d / step)), 1)
        out.append(np.linspace(a, b, n + 1)[:-1])
    out.append(path[-1][None, :])
    return np.concatenate(out)


def mean_field(pf, path, rate):
    """Reference `avg_vibration` analog: edge-time weighted mean of V on path segments."""
    p = np.asarray(path, float)
    if len(p) < 2:
        return float(pf.interp(p[:1])[0])
    mids = 0.5 * (p[:-1] + p[1:])
    v = pf.interp(mids)
    t = np.maximum([edge_time(p[i], p[i + 1], rate) for i in range(len(p) - 1)], 1e-12)
    return float((v * t).sum() / t.sum())


def make_grid(pf, n, res=GRID_RES):
    npts = res ** n
    if npts > GRID_MAX_NODES:
        raise GridTooLarge("grid %d^%d = %.3e nodes exceeds cap %.1e (no coarse grid)"
                           % (res, n, float(npts), float(GRID_MAX_NODES)))
    axes = np.linspace(0.0, WORKSPACE, res)
    mesh = np.meshgrid(*([axes] * n), indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    return pf.interp(pts).reshape((res,) * n), axes


def grid_interp(vals, axes, pts):
    from scipy.ndimage import map_coordinates
    dx = axes[1] - axes[0]
    u = np.asarray(pts, float) / dx
    return map_coordinates(vals, [u[:, d] for d in range(vals.ndim)],
                           order=1, mode="nearest")


# ---------------------------------------------------------------------------
# path (reference units) -> public trajectory record (shared evaluator)
# ---------------------------------------------------------------------------
def corr_viol_time(prob, nodes, durations):
    """Time-indexed corridor violation |x_k - xbar(tau_k)|/bd - 1 (protocol / CoTra-J)."""
    p = np.asarray(nodes, float)
    dur = np.asarray(durations, float)
    T = float(dur.sum())
    tau = np.concatenate([[0.0], np.cumsum(dur)]) / max(T, 1e-12)
    xb = prob.a[None, :] + tau[:, None] * prob.dir[None, :]
    return float(np.max(np.abs(p - xb) / prob.bd[None, :] - 1.0))


def enforce_time_corridor(prob, nodes, durations):
    """Project the inverse-allocated trajectory into the *time-indexed* corridor.

    A time warp moves the nominal reference xbar(tau), so a path that was corridor-feasible
    under the geometric test can leave the protocol corridor.  For fixed durations the
    feasible set is convex, and the **time-warped straight line**
    ``a + (b-a)*tau_k`` (tau_k from the returned durations) is always admissible: zero
    time-indexed corridor violation and speed ``|b-a|/T_b <= rate``.  (The uniform-position
    `prob.base` is NOT admissible under non-uniform durations -- blending to it can break
    the rate limit, which is how GA case 823 lost feasibility.)  We therefore blend towards
    the time-warped straight line and take the largest admissible factor.
    Returns (nodes, blend_factor, raw_time_violation).
    """
    raw = corr_viol_time(prob, nodes, durations)
    if raw <= 1e-9:
        return np.asarray(nodes, float), 1.0, raw
    dur = np.asarray(durations, float)
    T = float(dur.sum())
    tau = np.concatenate([[0.0], np.cumsum(dur)]) / max(T, 1e-12)
    wb = prob.a[None, :] + tau[:, None] * prob.dir[None, :]     # warped straight line
    lo, hi = 0.0, 1.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        cand = wb + mid * (np.asarray(nodes, float) - wb)
        cand[0] = prob.a
        cand[-1] = prob.b
        ok = (corr_viol_time(prob, cand, durations) <= 1e-9
              and prob.rate_viol(cand, durations) <= 1e-9)
        if ok:
            lo = mid
        else:
            hi = mid
    cand = wb + lo * (np.asarray(nodes, float) - wb)
    cand[0] = prob.a
    cand[-1] = prob.b
    return cand, float(lo), raw


def path_to_record(method, field_nd, case, path, extra=None):
    n = int(field_nd.N)
    rates = np.asarray(case["rates"], float)     # per-case, explicit
    a = np.asarray(case["A"], float)
    b = np.asarray(case["B"], float)
    t_fast = float(np.max(np.abs(b - a) / rates))
    T_b = float(case["TR"]) * t_fast
    path_pub = np.clip(np.asarray(path, float) * PUB, 0.0, 100.0)
    path_pub[0] = a
    path_pub[-1] = b
    prob = B.Problem(field_nd, a, b, T_b, rates, K=K_TRAJ, rho=0.20, delta=0.02)
    nodes, fac, raw = prob.finalize_path(path_pub)
    sc = prob.score(nodes, "inverse")
    # ---- protocol corridor is time-indexed (same definition as CoTra-J) ----
    dur = sc["durations"]
    if dur is None:
        dur_np = np.full(K_TRAJ, prob.dt)
        time_alloc_applied = False
    else:
        dur_np = np.asarray(dur, float)
        time_alloc_applied = bool(sc["time_alloc_applied"])
    nodes_t, fac_t, raw_corr_time = enforce_time_corridor(prob, nodes, dur_np)
    if fac_t < 1.0:
        ev = prob.eval(nodes_t, durations=dur_np)
        sc = dict(sc)
        sc["J"] = ev["J"]; sc["T"] = ev["T"]
        sc["endpoint_error"] = ev["endpoint_error"]
        sc["rate_violation"] = ev["rate_violation"]
        nodes = nodes_t
        fac = min(fac, fac_t)
    corr_time = corr_viol_time(prob, nodes, dur_np)
    corr = corr_time
    feas = (sc["endpoint_error"] <= 1e-3 and sc["rate_violation"] <= 0.01
            and corr_time <= 1e-6
            and abs(sc["T"] - prob.T_b) <= max(0.01 * prob.T_b, prob.dt))
    # ---- uniform-time counterfactual on the SAME final points (audit of the
    #      inverse-allocation post-processing; separate feasibility must be reported) ----
    dur_u = np.full(K_TRAJ, prob.dt, dtype=np.float64)
    ev_u = prob.eval(nodes)
    corr_time_u = corr_viol_time(prob, nodes, dur_u)   # == geometric corridor (tau = k/K)
    feas_u = (ev_u["endpoint_error"] <= 1e-3 and ev_u["rate_violation"] <= 0.01
              and corr_time_u <= 1e-6
              and abs(ev_u["T"] - prob.T_b) <= max(0.01 * prob.T_b, prob.dt))
    rec = {
        "case_id": case["case_id"], "N": n, "subset": case["subset"],
        "field_seed": case["field_seed"], "field_version": "gpu-pilot-1.0",
        "A": case["A"], "B": case["B"], "TR": case["TR"], "rates": case["rates"],
        "device": "cpu", "method": method, "K": K_TRAJ,
        "J": float(sc["J"]), "J_uniform": float(ev_u["J"]), "T": float(sc["T"]),
        "durations": None if dur is None else [float(v) for v in dur_np],
        "time_alloc": sc["time_alloc"],
        "time_alloc_applied": time_alloc_applied,
        "endpoint_error": float(sc["endpoint_error"]),
        "rate_violation": float(sc["rate_violation"]),
        "rate_violation_uniform": float(ev_u["rate_violation"]),
        "endpoint_error_uniform": float(ev_u["endpoint_error"]),
        "corridor_violation_time_uniform": float(corr_time_u),
        "feasible_uniform_time": bool(feas_u),
        "corridor_violation": corr,
        "corridor_violation_time": corr_time,
        "corridor_violation_time_raw": float(raw_corr_time),
        "corridor_violation_geom": float(np.max(prob.corr_viol(nodes))),
        "corridor_violation_raw": float(raw["corridor_violation_raw"]),
        "rate_violation_raw": float(raw["rate_violation_raw"]),
        "repair_fac": float(fac), "repair_fac_time_corridor": float(fac_t),
        "feasible": bool(feas),
        "points": np.asarray(nodes, float).round(6).tolist(),
        "status": "ok", "notes": "", "params": PARAMS[method],
    }
    if extra:
        rec.update(extra)
    return rec

