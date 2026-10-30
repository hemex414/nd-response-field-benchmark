# -*- coding: utf-8 -*-
"""evaluate.py -- the single public evaluator: J, T, endpoint, rate, time-indexed corridor.

Ported from the experiment evaluator (identical formula), with **per-case rates passed
explicitly** into :class:`Problem` (there is no global RATE).  Used identically by the
single-case and batch runners.
"""
from __future__ import annotations

import numpy as np

RHO = 0.2
DELTA = 0.02
K_DEFAULT = 8
S_EVAL = 4


def _np(x):
    return np.asarray(x, dtype=np.float64)


def np_field_of(field):
    """Public fields expose ``interp`` directly; kept for interface parity."""
    return field


def field_eval(field, pts):
    """V(pts) via the numpy field -> np (M,)."""
    return np_field_of(field).interp(_np(pts))




def eval_np(npf, points, durations, a, b, rate, S=S_EVAL):
    """numpy port of ``v2_nd.evaluate_nd`` (batch), identical formula.

    points (B,K+1,N), durations (B,K), a/b/rate (B,N) -> dict(J, T, ep, rate_viol).
    """
    points = np.asarray(points, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    B, K1, N = points.shape
    K = K1 - 1
    T = durations.sum(axis=1)
    frac = (np.arange(S, dtype=np.float64) + 0.5) / S
    seg = points[:, 1:, :] - points[:, :-1, :]                       # (B,K,N)
    pos = points[:, :-1, :][:, :, None, :] + seg[:, :, None, :] * frac[None, None, :, None]
    F = npf.interp(pos)                                              # (B,K,S)
    w = durations / np.maximum(T, 1e-9)[:, None] / S                 # (B,K)
    J = (F * w[:, :, None]).sum(axis=(1, 2))
    speed = np.abs(seg) / np.maximum(durations, 1e-9)[:, :, None]    # (B,K,N)
    rv = np.clip(speed / rate[:, None, :] - 1.0, 0.0, None).max(axis=(1, 2))
    ep = np.maximum(np.abs(points[:, 0, :] - a).max(axis=1),
                    np.abs(points[:, -1, :] - b).max(axis=1))
    return {"J": J, "T": T, "ep": ep, "rate_viol": rv}

def _resample_polyline(path, m):
    """Resample a polyline to m points by arc-length (used to densify sparse paths)."""
    path = _np(path)
    if len(path) < 2:
        return np.repeat(path[:1], m, axis=0)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-12:
        return np.repeat(path[:1], m, axis=0)
    tgt = np.linspace(0.0, cum[-1], m)
    return np.stack([np.interp(tgt, cum, path[:, d]) for d in range(path.shape[1])], axis=1)

class Problem:
    """One case: start/end, time budget, rate limits, corridor, field."""

    def __init__(self, field, a, b, T_b, rates, K=K_DEFAULT, rho=RHO, delta=DELTA):
        self.field = field
        self.npf = field
        self.N = int(field.N)
        self.device = 'cpu'
        # per-case rates are passed explicitly (no global RATE anywhere)
        rate = _np(rates).reshape(-1)
        if rate.size == 1 and self.N > 1:
            rate = np.full(self.N, float(rate[0]))
        assert rate.size == self.N
        self.rate = rate
        self.rpm = _np(getattr(field, 'rpm_max', np.full(self.N, 100.0)))
        self.a = _np(a).reshape(-1)
        self.b = _np(b).reshape(-1)
        self.K = int(K)
        self.T_b = float(T_b)
        dt = self.T_b / self.K
        # T_b = 0 would be a degenerate zero-length problem; keep a tiny positive budget
        self.dt = float(dt) if dt > 0 else 1e-6
        self.rho = float(rho)
        self.delta = float(delta)
        self.dir = self.b - self.a
        self.dlen2 = float(self.dir @ self.dir)
        # fixed dimensions (a_i == b_i) must not move: zero corridor width.
        self.active = (np.abs(self.dir) > 1e-9).astype(np.float64)
        self.bd = self.rho * np.abs(self.dir) + self.delta * self.rpm * self.active
        self.bd = np.maximum(self.bd, 1e-9)
        self.lo = np.zeros(self.N)
        self.hi = self.rpm.copy()
        self.t_fast = float(np.max(np.abs(self.dir) / np.maximum(self.rate, 1e-9)))
        self.base = self.xbar(np.linspace(0.0, 1.0, self.K + 1))   # (K+1, N)

    # -- geometry ----------------------------------------------------------
    def xbar(self, t):
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        return self.a[None, :] + t[:, None] * self.dir[None, :]

    def line_t(self, x):
        """Projection of x onto the straight segment A->B (fraction in [0,1])."""
        x = _np(x)
        if self.dlen2 < 1e-18:
            return np.zeros(x.shape[:-1])
        return np.clip(((x - self.a) @ self.dir) / self.dlen2, 0.0, 1.0)

    def corr_viol(self, x):
        """Corridor violation per point (<=0 means inside)."""
        x = _np(x)
        t = self.line_t(x)
        xb = self.a[None, :] + t[..., None] * self.dir[None, :]
        return np.max(np.clip(np.abs(x - xb) / self.bd[None, :] - 1.0, 0.0, None), axis=-1)

    def rate_viol(self, x, durations=None):
        """Rate violation; durations (K,) may be non-uniform (default: uniform dt)."""
        x = _np(x)
        if durations is None:
            dur = np.full(x.shape[0] - 1, self.dt, dtype=np.float64)
        else:
            dur = np.asarray(durations, dtype=np.float64).reshape(-1)
        v = np.diff(x, axis=0) / np.maximum(dur, 1e-12)[:, None]
        return float(np.max(np.clip(np.abs(v) / self.rate[None, :] - 1.0, 0.0, None)))

    def clip_corridor(self, x):
        """Point-wise projection into the corridor tube (endpoints preserved)."""
        x = np.array(x, dtype=np.float64, copy=True)
        t = self.line_t(x)
        xb = self.a[None, :] + t[..., None] * self.dir[None, :]
        x = np.clip(x, xb - self.bd[None, :], xb + self.bd[None, :])
        x = np.clip(x, self.lo[None, :], self.hi[None, :])
        x[0] = self.a
        x[-1] = self.b
        return x

    def feasible(self, x, durations=None, rate_tol=1e-9, corr_tol=1e-9):
        x = _np(x)
        if x.shape != (self.K + 1, self.N):
            return False
        return (self.rate_viol(x, durations) <= rate_tol
                and float(np.max(self.corr_viol(x))) <= corr_tol)

    # -- evaluation --------------------------------------------------------
    def eval(self, nodes, durations=None, S=S_EVAL):
        """Evaluate a (K+1, N) trajectory with the numpy evaluator.

        ``durations`` (K,) may be non-uniform; ``J`` is the time-weighted mean of V
        along the trajectory, so non-uniform durations change J (dwell allocation).
        The formula is identical to ``v2_nd.evaluate_nd`` (S sub-samples per segment).
        """
        nodes = _np(nodes)
        if durations is None:
            dur_np = np.full(self.K, self.dt, dtype=np.float64)
        else:
            dur_np = np.asarray(durations, dtype=np.float64).reshape(-1)
        ev = eval_np(self.npf, nodes[None], dur_np[None], self.a[None], self.b[None],
                     self.rate[None], S=S)
        return {
            "J": float(ev["J"][0]),
            "T": float(ev["T"][0]),
            "endpoint_error": float(ev["ep"][0]),
            "rate_violation": float(ev["rate_viol"][0]),
        }

    def eval_batch(self, nodes, S=2):
        """Evaluate (P, K+1, N) trajectories -> (J, rate_viol) numpy arrays."""
        nodes = _np(nodes)
        P = nodes.shape[0]
        dur = np.full((P, self.K), self.dt, dtype=np.float64)
        ev = eval_np(self.npf, nodes, dur,
                     np.repeat(self.a[None], P, 0),
                     np.repeat(self.b[None], P, 0),
                     np.repeat(self.rate[None], P, 0), S=S)
        return np.asarray(ev["J"], dtype=np.float64), np.asarray(ev["rate_viol"], dtype=np.float64)

    def corr_viol_batch(self, nodes):
        t = self.line_t(nodes)
        xb = self.a[None, None, :] + t[..., None] * self.dir[None, None, :]
        return np.max(np.clip(np.abs(nodes - xb) / self.bd[None, None, :] - 1.0, 0.0, None), axis=-1)

    # -- candidate -> admissible trajectory --------------------------------
    def repair(self, x, durations=None):
        """Return (feasible nodes, blend factor, raw violations).

        The feasible set of fixed-duration trajectories (corridor box + rate box +
        fixed endpoints) is convex, so the straight-line trajectory is feasible and a
        blend ``base + f*(cand - base)`` is feasible for some ``f``; we take the largest
        such ``f`` (bisection).  A point-wise corridor projection is tried first.
        """
        x = np.array(x, dtype=np.float64, copy=True)
        x[0] = self.a
        x[-1] = self.b
        raw = dict(
            rate_violation_raw=self.rate_viol(x, durations),
            corridor_violation_raw=float(np.max(self.corr_viol(x))),
        )
        if self.feasible(x, durations):
            return x, 1.0, raw
        xp = self.clip_corridor(x)
        if self.feasible(xp, durations):
            return xp, 1.0, raw
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            cand = self.base + mid * (xp - self.base)
            if self.feasible(cand, durations):
                lo = mid
            else:
                hi = mid
        cand = self.base + lo * (xp - self.base)
        cand[0] = self.a
        cand[-1] = self.b
        return cand, float(lo), raw

    def retime(self, path):
        """Geometric path -> uniform-time (K+1, N) trajectory.

        Nodes are placed at equal fractions of the *minimum feasible duration* of the
        path, then the path is slowed down uniformly to the budget T_b.  If the path
        cannot be traversed within T_b it is blended towards the straight line first.
        """
        path = _np(path)
        if path.shape[0] < 2:
            return self.base.copy()
        path = np.array(path, copy=True)
        path[0] = self.a
        path[-1] = self.b
        straight = self.a[None, :] + np.linspace(0.0, 1.0, len(path))[:, None] * self.dir[None, :]

        def _cost(p):
            seg = np.diff(p, axis=0)
            st = np.max(np.abs(seg) / np.maximum(self.rate[None, :], 1e-12), axis=1)
            return st

        seg_t = _cost(path)
        S = float(seg_t.sum())
        if not np.isfinite(S) or S <= 1e-12:
            return self.base.copy()
        if S > self.T_b * (1.0 - 1e-9):
            chosen = straight.copy()
            for fac in np.linspace(1.0, 0.0, 21):
                cand = straight + fac * (path - straight)
                st = _cost(cand)
                if float(st.sum()) <= self.T_b * (1.0 - 1e-9):
                    chosen = cand
                    seg_t = st
                    S = float(st.sum())
                    break
            path = chosen
        tau = np.concatenate([[0.0], np.cumsum(seg_t)])
        tau[-1] = S
        tgt = np.linspace(0.0, S, self.K + 1)
        nodes = np.stack([np.interp(tgt, tau, path[:, d]) for d in range(self.N)], axis=1)
        nodes[0] = self.a
        nodes[-1] = self.b
        return nodes

    def finalize_path(self, path):
        nodes = self.retime(path)
        nodes, fac, raw = self.repair(nodes)
        return nodes, fac, raw

    def finalize_nodes(self, nodes):
        nodes, fac, raw = self.repair(nodes)
        return nodes, fac, raw

    # -- time allocation (the project's "inverse-field" post-processing) ----------
    def segment_min_durations(self, nodes):
        """Minimum feasible duration of each of the K uniform-node segments."""
        nodes = _np(nodes)
        seg = np.diff(nodes, axis=0)
        return np.max(np.abs(seg) / np.maximum(self.rate[None, :], 1e-12), axis=1)

    def segment_mean_V(self, nodes, n_inner=4):
        """Mean field value on each of the K segments (n_inner sub-samples each)."""
        nodes = _np(nodes)
        seg = np.diff(nodes, axis=0)
        frac = (np.arange(int(n_inner)) + 0.5) / float(n_inner)
        pts = (nodes[:-1][:, None, :] + frac[None, :, None] * seg[:, None, :])
        pts = pts.reshape(-1, self.N)
        v = field_eval(self.field, pts).reshape(-1, int(n_inner))
        return v.mean(axis=1)

    def allocate_inverse(self, nodes, power=1.0, n_inner=4):
        """Inverse-field time allocation over the K segments (geometric planners).

            t_min,k = max_i |dx_ki| / rate_i          (minimum feasible segment time)
            S       = T_b - sum_k t_min,k             (remaining budget)
            w_k     = 1 / mean(V on segment k)        (weight, ``power`` = 1 by default)
            t_k     = t_min,k + S * w_k / sum_j w_j

        ``sum(t) == T_b`` and ``t_k >= t_min,k``, so the trajectory stays admissible;
        only the dwell distribution changes (lower-field segments get more time).
        Returns ``None`` if the remaining budget is not positive.
        """
        nodes = np.array(nodes, dtype=np.float64, copy=True)
        t_min = self.segment_min_durations(nodes)
        S = float(self.T_b - float(t_min.sum()))
        if not np.isfinite(S) or S <= 1e-12:
            return None
        vals = self.segment_mean_V(nodes, n_inner=n_inner)
        w = 1.0 / np.maximum(vals, 1e-9) ** float(power)
        if not np.all(np.isfinite(w)) or float(w.sum()) <= 0.0:
            return None
        t = t_min + S * w / float(w.sum())
        # numerical guards: keep t_k >= t_min,k and sum(t) == T_b
        t = np.maximum(t, t_min)
        t[int(np.argmax(t))] += self.T_b - float(t.sum())
        t = np.maximum(t, t_min)
        t[int(np.argmax(t))] += self.T_b - float(t.sum())
        return t

    def score(self, nodes, time_alloc="none"):
        """Uniform-time (raw geometric) J plus, optionally, the inverse-allocated J."""
        out = dict(J=None, J_uniform=None, T=None, durations=None,
                   endpoint_error=None, rate_violation=None,
                   time_alloc=("inverse" if time_alloc == "inverse" else "none"),
                   time_alloc_applied=False)
        ev = self.eval(nodes)
        out.update(J=ev["J"], J_uniform=ev["J"], T=ev["T"],
                   endpoint_error=ev["endpoint_error"],
                   rate_violation=ev["rate_violation"])
        if time_alloc == "inverse":
            dur = self.allocate_inverse(nodes)
            if dur is not None:
                ev2 = self.eval(nodes, durations=dur)
                out.update(J=ev2["J"], T=ev2["T"], durations=dur.tolist(),
                           endpoint_error=ev2["endpoint_error"],
                           rate_violation=ev2["rate_violation"],
                           time_alloc_applied=True)
        return out


# ---------------------------------------------------------------------------
# materialised N-D grid + corridor mask
# ---------------------------------------------------------------------------

def _grid_strides(shape):
    strides = []
    acc = 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= s
    return list(reversed(strides))

def _reconstruct(parent, src, tgt, grid, prob):
    if tgt != src and parent[tgt] < 0:
        return np.vstack([prob.a, prob.b])
    seq = []
    u = tgt
    while u >= 0 and u != src:
        seq.append(u)
        u = parent[u]
    seq.append(src)
    seq.reverse()
    path = np.array([grid.coord(np.unravel_index(u, (grid.res,) * grid.N)) for u in seq])
    if len(path) < 2:
        return np.vstack([prob.a, prob.b])
    path[0] = prob.a
    path[-1] = prob.b
    return path


# ---------------------------------------------------------------------------
# method 3: Theta* (grid + line-of-sight shortcuts)
# ---------------------------------------------------------------------------
THETA_STRIDE = 2
THETA_LOS_SAMPLES = 3
