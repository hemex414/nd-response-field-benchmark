# -*- coding: utf-8 -*-
"""cotra_j.py -- standalone CoTra-J (joint path-time) planner for public fields.

This module is a **self-contained port** of the CoTra-J planner used in the ESWA
route-A study.  It has *no* imports from the private paper scripts and no global
rate/RPM state: every case carries its own ``rates`` row and its own box bound
``rpm_max``, and every search stage uses those per-row values.

Time allocation (rate-anchored wire)
------------------------------------
CoTra-J optimises the same 28 parameters as before -- the ``K-1 = 7`` interior
path offsets ``d`` and the ``K-1 = 7`` cumulative-warp coordinates ``delta``
(``K = 8``) -- but the durations are now anchored at the per-segment rate floor
instead of being a plain share of the budget::

    tmin_k(p) = max_i |p_{k+1,i} - p_{k,i}| / rates_i     (smallest legal time)
    dt_k      = tmin_k(p) + (T_b - sum_j tmin_j(p)) * w_k  ,  w = warp(delta)

so every sample whose geometry fits the budget (``sum tmin <= T_b``) is
rate-feasible *by construction*, and the warp only distributes the residual free
time.  Because the placement axis ``tau`` (where the nodes sit) and ``tmin(p)``
(which depends on that placement) are mutually dependent, the decode solves the
fixed point ``tau <- cumsum(dt(tau)) / T_b`` by iteration and then re-places the
nodes on the converged axis, so ``|d| <= bd`` still implies the corridor exactly.
Samples whose geometry does not fit the budget fall back to the plain
``dt = T_b * w`` wire (exactly consistent, rate-infeasible), which the CEM
objective then avoids through its rate penalty.  ``sum_k dt_k == T_b`` holds in
both branches.

Public API
----------
    cfg = CoTraJConfig()                       # paper defaults
    points, durations, info = plan_batch(field_bank, A, B, T_b, rates, cfg)

      field_bank : object with
                     .eval(points, case_idx) -> (M,) field values
                     .rpm_max  -> (N,) tensor/array, (B,N) tensor/array, or
                                  callable()/callable(case_idx) -> (B,N)/(N,)
      A, B       : (B,N) start / end states in the field's coordinate units
      T_b        : (B,)   time budget (sum of the returned durations == T_b)
      rates      : (B,N)  per-case, per-dimension speed limits
      returns points (B,K+1,N), durations (B,K), info dict
                   info = {J, T, rate_viol, corridor_viol, T_rel, ep,
                           feasible, hard_feasible, fallback}
      The evaluation of ``J`` always uses the returned ``durations`` (never the
      equal-duration allocation).

Design guarantees
-----------------
* K = 8, rho = 0.2, delta = 0.02, path-only CEM 1024/200 + 512/150, joint CEM
  512x250 + 512x150 by default, i.e. exactly the published CoTra-J search budget.
* The internal path-only warm start is part of the search only; it is not
  returned and not evaluated a second time.
* The returned durations are rate-anchored (see above): ``dt_k >= tmin_k`` for
  every segment whenever the budget allows it, and ``sum_k dt_k == T_b`` always.
  ``J`` is always evaluated with exactly those durations.
* ``field.eval`` is called in chunks (``cfg.chunk_points``), so an N-D field is
  never materialised as a big dense array.

Field protocol
--------------
Any object implementing ``eval(points, case_idx)`` can be passed.  A convenience
``AnalyticNDFieldBank`` is provided: it stores one per-case analytic evaluator
(e.g. a ``pilot_runner.FieldND`` instance or a plain callable) and evaluates only
the points belonging to that case, so memory stays O(number of points) instead of
O(8^N).  ``SharedFieldBank`` wraps a single field shared by all cases (used by the
equivalence test against the 3-D reference field).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import torch

__all__ = [
    "CoTraJConfig",
    "plan_batch",
    "eval_curve",
    "decode",
    "encode",
    "warp_durations",
    "segment_min_time",
    "bd_of",
    "AnalyticNDFieldBank",
    "SharedFieldBank",
]


# --------------------------------------------------------------------------- config
@dataclass
class CoTraJConfig:
    # --- geometry / search (paper CoTra-J defaults) ---------------------------
    K: int = 8
    rho: float = 0.2
    delta: float = 0.02
    # path-only warm start: stage 1 (position CEM) + stage 2 (corridor CEM)
    path_pop: int = 1024
    path_iters: int = 200
    stage2_pop: int = 512
    stage2_iters: int = 150
    # joint refinement: two CEM stages
    joint_pop: int = 512
    iters_a: int = 250
    iters_b: int = 150
    seed: int = 0
    # --- evaluator ------------------------------------------------------------
    s_samples: int = 8          # sub-samples per segment (same as the paper)
    chunk_points: int = 1 << 20  # max points per field.eval call
    # --- penalties / bounds ---------------------------------------------------
    pen_rate: float = 500.0
    pen_corr: float = 5000.0
    min_w: float = 1e-3
    tau_bound: float = 0.35
    elite_frac: float = 0.15
    # --- rate-anchored duration wire -----------------------------------------
    fp_iters: int = 30           # fixed-point sweeps for the placement axis tau
    fp_tol: float = 1e-10        # residual below which the fixed point stops
    anchor_safety: float = 0.9999  # final inward shrink of d (keeps the corridor)
    # --- feasibility tolerances (paper protocol) ------------------------------
    tol_t: float = 1e-2
    tol_rate: float = 1e-2
    tol_corr: float = 1e-6
    tol_box: float = 1e-2
    hard_rate: float = 1e-6
    ep_tol: float = 1e-3      # endpoint tolerance in coordinate units
    grid_step: float = 1.0    # coordinate units per "grid step" (only for info)

    def bd(self, a: torch.Tensor, b: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        """Corridor half-width (B,N); rpm may be (N,) or (B,N).

        Fixed dimensions (A_i == B_i) get zero width: they must not move.
        """
        active = ((b - a).abs() > 1e-9).to(dtype=a.dtype)
        return self.rho * (b - a).abs() + self.delta * rpm * active


# --------------------------------------------------------------------------- field protocol
class SharedFieldBank:
    """One field shared by all cases (e.g. the canonical 3-D field)."""

    def __init__(self, eval_fn: Callable[[torch.Tensor], torch.Tensor],
                 rpm_max: Union[torch.Tensor, Sequence[float], float],
                 dim: Optional[int] = None):
        self._fn = eval_fn
        self.rpm_max = torch.as_tensor(rpm_max, dtype=torch.float32)
        self.dim = dim if dim is not None else (self.rpm_max.numel()
                                                if self.rpm_max.ndim else None)

    def eval(self, points: torch.Tensor, case_idx: torch.Tensor) -> torch.Tensor:
        return self._fn(points)


class AnalyticNDFieldBank:
    """Per-case analytic field bank (no dense N-D array).

    Parameters
    ----------
    fields : sequence indexed by case index.  Each entry is either a callable
             ``f(points (M,N)) -> (M,)`` or an object with an ``eval(points)`` or
             ``interp(points)`` method.
    rpm_max : scalar, (N,), or (B,N) box bound of the coordinate domain.
    """

    def __init__(self, fields: Sequence, rpm_max: Union[torch.Tensor, Sequence[float], float]):
        self.fields = list(fields)
        self.rpm_max = torch.as_tensor(rpm_max, dtype=torch.float32)
        n = len(self.fields)
        if self.rpm_max.ndim == 0:
            raise ValueError("rpm_max must be (N,) or (B,N) for AnalyticNDFieldBank")
        self.dim = int(self.rpm_max.shape[-1]) if self.rpm_max.ndim == 2 else \
            int(self.rpm_max.numel())
        if self.rpm_max.ndim == 1:
            self.rpm_max = self.rpm_max.expand(n, -1).contiguous()

    def eval(self, points: torch.Tensor, case_idx: torch.Tensor) -> torch.Tensor:
        flat = points.reshape(-1, points.shape[-1])
        ci = case_idx.reshape(-1)
        out = flat.new_empty(flat.shape[0])
        for c in torch.unique(ci).tolist():
            m = ci == c
            f = self.fields[int(c)]
            fn = f if callable(f) else getattr(f, "eval", None) or getattr(f, "interp")
            out[m] = fn(flat[m])
        return out


# --------------------------------------------------------------------------- geometry
def bd_of(cfg: CoTraJConfig, a: torch.Tensor, b: torch.Tensor,
          rpm: torch.Tensor) -> torch.Tensor:
    rpm = rpm.to(device=a.device, dtype=a.dtype)
    return cfg.bd(a, b, rpm)


def warp_durations(cfg: CoTraJConfig, delta_f: torch.Tensor) -> torch.Tensor:
    """Cumulative progress warp (B,K-1) -> segment fractions (B,K), sum == 1."""
    B = delta_f.shape[0]
    K = cfg.K
    z = delta_f.new_zeros(B, 1)
    raw = 1.0 / K + torch.diff(torch.cat([z, delta_f, z], dim=1), dim=1)
    w = torch.clamp(raw, min=cfg.min_w)
    return w / w.sum(dim=1, keepdim=True)


def segment_min_time(points: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
    """Per-segment minimum traversal time ``tmin`` (B,K).

    ``tmin_k = max_i |p_{k+1,i} - p_{k,i}| / rates_i`` is the smallest time in
    which segment ``k`` can be traversed under the per-axis rate limits of that
    case.  ``sum_k tmin_k <= T_b`` is exactly the condition for the geometry to
    fit the time budget.
    """
    seg = points[:, 1:, :] - points[:, :-1, :]
    return (seg.abs() / rates.unsqueeze(1).clamp(min=1e-12)).amax(dim=2)


def _build_points(cfg: CoTraJConfig, d: Optional[torch.Tensor], tau: torch.Tensor,
                  a: torch.Tensor, b: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
    """``p_k = a + (b-a) tau_k + d_k``, box-clipped, endpoints pinned."""
    base = a.unsqueeze(1) + (b - a).unsqueeze(1) * tau.unsqueeze(-1)
    p = base.clone()
    if d is not None:
        p[:, 1:-1, :] = base[:, 1:-1, :] + d
    hi = rpm.unsqueeze(1) if rpm.dim() == 2 else rpm.view(1, 1, -1)
    p = torch.minimum(torch.maximum(p, torch.zeros_like(p)), hi)
    p[:, 0, :] = a
    p[:, -1, :] = b
    return p


def _anchor_durations(w: torch.Tensor, p: torch.Tensor, T_b: torch.Tensor,
                      rates: torch.Tensor):
    """Rate-anchored durations at a fixed geometry.

    ``dt = tmin(p) + clamp(T_b - sum tmin(p), min=0) * w`` together with the
    signed slack ``T_b - sum tmin(p)``; a non-negative slack means the geometry
    fits the budget and the durations are rate-feasible by construction.
    """
    tmin = segment_min_time(p, rates)
    slack = T_b.view(-1) - tmin.sum(dim=1)
    dt = tmin + slack.clamp(min=0.0).view(-1, 1) * w
    return dt, tmin, slack


def decode(cfg: CoTraJConfig, d: torch.Tensor, delta_f: torch.Tensor,
           a: torch.Tensor, b: torch.Tensor, T_b: torch.Tensor,
           rpm: torch.Tensor, rates: torch.Tensor):
    """(d, delta) -> points (B,K+1,N), durations (B,K), progress tau (B,K+1).

    Rate-anchored wire: the durations are the per-segment rate floor plus a
    share of the residual free time, and the placement axis is the fixed point
    of ``tau <- cumsum(dt(tau)) / T_b``.  ``rates`` is (B,N) and carries the
    per-case, per-dimension speed limits.  Samples whose geometry does not fit
    the budget (``sum tmin > T_b``) get the plain ``dt = T_b * w`` wire, which
    is exactly consistent but rate-infeasible; ``sum dt == T_b`` in both cases.
    """
    K = cfg.K
    w = warp_durations(cfg, delta_f)
    B = w.shape[0]
    Tb = T_b.view(-1, 1).clamp(min=1e-12)
    tau = torch.cat([w.new_zeros(B, 1), w.cumsum(dim=1)], dim=1)
    tau = tau / tau[:, -1:].clamp(min=1e-12)
    p = _build_points(cfg, d, tau, a, b, rpm)
    dt, tmin, slack = _anchor_durations(w, p, T_b, rates)
    ok = slack >= 0
    # fixed point of the placement axis (damped-free; 30 sweeps by default)
    for _ in range(int(cfg.fp_iters)):
        t = torch.cat([dt.new_zeros(B, 1), dt.cumsum(dim=1)], dim=1)
        tau_new = torch.where(ok.view(-1, 1), t / Tb, tau)
        resid = float((tau_new - tau).abs().max())
        tau = tau_new
        p = _build_points(cfg, d, tau, a, b, rpm)
        dt, tmin, slack = _anchor_durations(w, p, T_b, rates)
        if resid < cfg.fp_tol:
            break
    # finalise at the converged geometry; fall back to the plain wire if the
    # geometry no longer fits, then place the nodes on the axis it implies
    dt_f, _, slack_f = _anchor_durations(w, p, T_b, rates)
    okf = slack_f >= 0
    dt = torch.where(okf.view(-1, 1), dt_f, w * Tb)
    omega = dt / dt.sum(dim=1, keepdim=True).clamp(min=1e-12)
    tau = torch.cat([dt.new_zeros(B, 1), omega.cumsum(dim=1)], dim=1)
    bd = bd_of(cfg, a, b, rpm).unsqueeze(1)
    if d is not None:
        d = torch.clamp(d, -cfg.anchor_safety * bd, cfg.anchor_safety * bd)
    p = _build_points(cfg, d, tau, a, b, rpm)
    return p, dt, tau


def encode(cfg: CoTraJConfig, p: torch.Tensor, dur: torch.Tensor,
           a: torch.Tensor, b: torch.Tensor):
    """points/durations -> (d, delta): the search coordinates of a curve.

    The map is well defined for *any* durations that sum to ``T_b``; with the
    rate-anchored wire it is the natural inverse on the reachable set (the
    decode re-anchors the durations, so ``decode(encode(p, dur))`` reproduces
    ``p`` whenever ``dur`` was itself rate-anchored).
    """
    K = cfg.K
    B = p.shape[0]
    # normalise the durations *before* the cumsum, exactly like the reference
    # implementation (same arithmetic order -> bitwise identical rounding)
    w = dur / dur.sum(dim=1, keepdim=True).clamp(min=1e-12)
    tau = torch.cat([w.new_zeros(B, 1), w.cumsum(dim=1)], dim=1)
    tau = tau / tau[:, -1:].clamp(min=1e-12)
    base = a.unsqueeze(1) + (b - a).unsqueeze(1) * tau.unsqueeze(-1)
    d = p[:, 1:-1, :] - base[:, 1:-1, :]
    u = torch.arange(1, K, device=p.device, dtype=p.dtype) / K
    return d, tau[:, 1:-1] - u.unsqueeze(0)


# --------------------------------------------------------------------------- evaluator
def _batch_rpm(field_bank, B: int, device, dtype=torch.float32) -> torch.Tensor:
    r = getattr(field_bank, "rpm_max", None)
    if r is None:
        raise ValueError("field_bank must expose `rpm_max` (N,) or (B,N)")
    if callable(r):
        try:
            r = r(torch.arange(B, device=device))
        except TypeError:
            r = r()
    t = torch.as_tensor(r, dtype=dtype, device=device)
    if t.ndim == 0:
        raise ValueError("rpm_max must be (N,) or (B,N)")
    if t.ndim == 1:
        t = t.unsqueeze(0).expand(B, -1).contiguous()
    if t.shape[0] > B:
        t = t[:B]
    if t.shape[0] != B:
        raise ValueError("rpm_max has %d rows but batch size is %d" % (t.shape[0], B))
    return t


def _field_eval(field_bank, points: torch.Tensor, case_idx: torch.Tensor,
                cfg: CoTraJConfig) -> torch.Tensor:
    """Evaluate field at ``points`` (..., N); chunked to bound memory."""
    N = points.shape[-1]
    shape = points.shape[:-1]
    flat = points.reshape(-1, N)
    M = flat.shape[0]
    ci = case_idx.reshape(-1)
    out = flat.new_empty(M)
    step = max(int(cfg.chunk_points), 1)
    for i in range(0, M, step):
        out[i:i + step] = field_bank.eval(flat[i:i + step], ci[i:i + step])
    return out.reshape(shape)


def eval_curve(field_bank, points: torch.Tensor, durations: torch.Tensor,
               A: torch.Tensor, B: torch.Tensor, T_b: torch.Tensor,
               rates: torch.Tensor, rpm: torch.Tensor,
               cfg: Optional[CoTraJConfig] = None, case_idx: Optional[torch.Tensor] = None):
    """Shared evaluator.  ``J`` always uses the supplied per-segment durations."""
    cfg = cfg or CoTraJConfig()
    rpm = rpm.to(device=points.device, dtype=points.dtype)
    Bn, K1, N = points.shape
    K = K1 - 1
    S = cfg.s_samples
    T = durations.sum(dim=1)
    frac = (torch.arange(S, device=points.device, dtype=points.dtype) + 0.5) / S
    seg = points[:, 1:, :] - points[:, :-1, :]
    pos = points[:, :-1, :].unsqueeze(2) + seg.unsqueeze(2) * frac.view(1, 1, S, 1)
    if case_idx is None:
        case_idx = torch.arange(Bn, device=points.device).repeat_interleave(K * S)
    F = _field_eval(field_bank, pos, case_idx, cfg).reshape(Bn, K, S)
    w = durations / T.clamp(min=1e-9).unsqueeze(1) / S
    J = (F * w.unsqueeze(-1)).sum(dim=(1, 2))
    speed = seg.abs() / durations.clamp(min=1e-9).unsqueeze(-1)
    rate_viol = torch.clamp(speed / rates.view(Bn, 1, N) - 1.0, min=0.0).amax(dim=(1, 2))
    ep = torch.maximum((points[:, 0, :] - A).abs().amax(dim=1),
                       (points[:, -1, :] - B).abs().amax(dim=1))
    t = torch.cat([durations.new_zeros(Bn, 1), durations.cumsum(dim=1)], dim=1)
    tau = t / t[:, -1:].clamp(min=1e-12)
    xb = A.unsqueeze(1) + (B - A).unsqueeze(1) * tau.unsqueeze(-1)
    bdv = bd_of(cfg, A, B, rpm).clamp(min=1e-6).unsqueeze(1)
    cv = torch.clamp((points - xb).abs() / bdv - 1.0, min=0.0).amax(dim=(1, 2))
    T_rel = (T - T_b.view(-1)).abs() / T_b.view(-1).clamp(min=1e-9)
    rpm = rpm.to(device=points.device, dtype=points.dtype)
    hi = rpm.unsqueeze(1) if rpm.dim() == 2 else rpm.view(1, 1, -1)
    box = ((points < -cfg.tol_box) | (points > hi + cfg.tol_box)).any(dim=(1, 2)).float()
    common = ((cv <= cfg.tol_corr) & (ep <= cfg.ep_tol)
              & (T_rel <= cfg.tol_t) & (box == 0))
    feasible = common & (rate_viol <= cfg.tol_rate)
    hard = common & (rate_viol <= cfg.hard_rate)
    return {"J": J, "T": T, "rate_viol": rate_viol, "corridor_viol": cv,
            "T_rel": T_rel, "ep": ep, "ep_steps": ep / max(cfg.grid_step, 1e-12),
            "box_viol": box, "feasible": feasible, "hard_feasible": hard}


# --------------------------------------------------------------------------- searching
def _straight(cfg: CoTraJConfig, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    K = cfg.K
    u = torch.linspace(0.0, 1.0, K + 1, device=a.device).view(1, K + 1, 1)
    return a.unsqueeze(1) + (b - a).unsqueeze(1) * u


def _project(cfg: CoTraJConfig, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
             T_b: torch.Tensor, rates: torch.Tensor, rpm: torch.Tensor,
             iters: int = 25) -> torch.Tensor:
    """Alternating projection: corridor clip, then rate clip + reintegrate."""
    K = cfg.K
    Bn = x.shape[0]
    bdv = bd_of(cfg, A, B, rpm).unsqueeze(1)
    xb = _straight(cfg, A, B)
    dt = (T_b / K).view(-1, 1).unsqueeze(-1)
    for _ in range(iters):
        x = torch.max(torch.min(x, xb + bdv), xb - bdv)
        x[:, 0, :] = A
        x[:, -1, :] = B
        v = (x[:, 1:, :] - x[:, :-1, :]) / dt
        v = torch.max(torch.min(v, rates.view(Bn, 1, -1)), -rates.view(Bn, 1, -1))
        inc = v * dt
        x = torch.cat([A.unsqueeze(1), A.unsqueeze(1) + inc.cumsum(dim=1)], dim=1)
    return x


def _repair_curve(cfg: CoTraJConfig, points: torch.Tensor, A: torch.Tensor,
                  B: torch.Tensor, dur: torch.Tensor, rates: torch.Tensor,
                  rpm: torch.Tensor) -> torch.Tensor:
    """Scale interior deviations toward the straight line until rates hold."""
    Bn = points.shape[0]
    base = _straight(cfg, A, B)
    W = points[:, 1:-1, :] - base[:, 1:-1, :]
    best = base.clone()
    for fac in (0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0):
        p = base.clone()
        p[:, 1:-1, :] = base[:, 1:-1, :] + fac * W
        lo = torch.zeros_like(p)
        hi = rpm.unsqueeze(1) if rpm.dim() == 2 else rpm.view(1, 1, -1)
        p = torch.minimum(torch.maximum(p, lo), hi)
        seg = p[:, 1:, :] - p[:, :-1, :]
        speed = seg.abs() / dur.clamp(min=1e-9).unsqueeze(-1)
        ok = (torch.clamp(speed / rates.view(Bn, 1, -1) - 1.0, min=0.0)
              .amax(dim=(1, 2)) <= 1e-9)
        best = torch.where(ok.view(Bn, 1, 1), p, best)
    return best


def _cem_plan_stage1(field_bank, A, B, T_b, rates, rpm, cfg):
    """Path-only stage 1: CEM over interior positions, equal durations."""
    torch.manual_seed(cfg.seed)
    K, pop, iters = cfg.K, cfg.path_pop, cfg.path_iters
    Bn, N = A.shape
    dt = (T_b / K).view(Bn, 1)
    base = _straight(cfg, A, B)[:, 1:-1, :]
    span = (B - A).unsqueeze(1)
    mean = base.clone()
    std = (0.25 * span.abs() + 1e-2).expand(Bn, K - 1, N).clone()
    best_obj = torch.full((Bn,), 1e9, device=A.device)
    best_pts = _straight(cfg, A, B)
    dur = dt.expand(Bn, K).contiguous()
    A_rep = A.repeat_interleave(pop, dim=0)
    B_rep = B.repeat_interleave(pop, dim=0)
    T_rep = T_b.repeat_interleave(pop, dim=0)
    r_rep = rpm.repeat_interleave(pop, dim=0)
    ci_rep = torch.arange(Bn, device=A.device).repeat_interleave(pop)
    ci_pos = ci_rep.repeat_interleave(K * cfg.s_samples)
    for _ in range(iters):
        w = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
            Bn, pop, K - 1, N, device=A.device)
        p = torch.cat([A.unsqueeze(1).unsqueeze(2).expand(Bn, pop, 1, N), w,
                       B.unsqueeze(1).unsqueeze(2).expand(Bn, pop, 1, N)], dim=2)
        hi = rpm.view(Bn, 1, 1, N)
        p = torch.minimum(torch.maximum(p, torch.zeros_like(p)), hi)
        flat_p = p.reshape(Bn * pop, K + 1, N)
        flat_d = dur.unsqueeze(1).expand(Bn, pop, K).reshape(Bn * pop, K)
        ev = eval_curve(field_bank, flat_p, flat_d, A_rep, B_rep, T_rep,
                        rates.repeat_interleave(pop, dim=0), r_rep, cfg, ci_pos)
        Jv = ev["J"].view(Bn, pop)
        ov = ev["rate_viol"].view(Bn, pop)
        obj = Jv + cfg.pen_rate * ov
        k = max(2, int(cfg.elite_frac * pop))
        idx = obj.argsort(dim=1)[:, :k]
        idx4 = idx.view(Bn, k, 1, 1).expand(Bn, k, K - 1, N)
        elite_p = p[:, :, 1:-1, :].gather(1, idx4)
        mean = elite_p.mean(dim=1)
        std = elite_p.std(dim=1, unbiased=False) + 1e-2 * (span.abs() + 1.0)
        bobj = obj.min(dim=1).values
        better = bobj < best_obj
        bi = obj.argmin(dim=1).view(Bn, 1, 1, 1).expand(Bn, 1, K + 1, N)
        cand = p.gather(1, bi).squeeze(1)
        best_pts = torch.where(better.view(Bn, 1, 1), cand, best_pts)
        best_obj = torch.where(better, bobj, best_obj)
    return _repair_curve(cfg, best_pts, A, B, dur, rates, rpm)


def _corridor_uniform(cfg: CoTraJConfig, x: torch.Tensor, A: torch.Tensor,
                      B: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
    """Corridor violation at the *uniform* node times (analytic k/K grid).

    Used by the path-only stage, whose nodes are pinned to equal time slots; the
    analytic grid reproduces the reference implementation exactly (the joint
    stage uses :func:`eval_curve` with the warped node times instead).
    """
    K = x.shape[1] - 1
    u = torch.linspace(0.0, 1.0, K + 1, device=x.device).view(1, K + 1, 1)
    xb = A.unsqueeze(1) + (B - A).unsqueeze(1) * u
    bdv = bd_of(cfg, A, B, rpm).clamp(min=1e-6).unsqueeze(1)
    return torch.clamp((x - xb).abs() / bdv - 1.0, min=0.0).amax(dim=(1, 2))


def _cem_corridor_stage2(field_bank, A, B, T_b, rates, rpm, cfg, pts):
    """Path-only stage 2: corridor-penalised local CEM + projection + fallback."""
    K, pop, iters = cfg.K, cfg.stage2_pop, cfg.stage2_iters
    Bn, N = A.shape
    bdv = bd_of(cfg, A, B, rpm)
    xb = _straight(cfg, A, B)
    dur = (T_b / K).unsqueeze(1).expand(Bn, K).contiguous()
    mean = (pts[:, 1:-1, :] - xb[:, 1:-1, :]).clone()
    mean = torch.max(torch.min(mean, bdv.unsqueeze(1)), -bdv.unsqueeze(1))
    std = (0.25 * bdv).unsqueeze(1).expand(Bn, K - 1, N).clone()
    best_x = _project(cfg, pts, A, B, T_b, rates, rpm)
    A_rep = A.repeat_interleave(pop, dim=0)
    B_rep = B.repeat_interleave(pop, dim=0)
    T_rep = T_b.repeat_interleave(pop, dim=0)
    r_rep = rpm.repeat_interleave(pop, dim=0)
    ci_pos = torch.arange(Bn, device=A.device).repeat_interleave(pop).repeat_interleave(
        K * cfg.s_samples)
    ev = eval_curve(field_bank, best_x, dur, A, B, T_b, rates, rpm, cfg)
    best_obj = (ev["J"] + cfg.pen_rate * ev["rate_viol"]
                + cfg.pen_corr * _corridor_uniform(cfg, best_x, A, B, rpm))
    for _ in range(iters):
        d = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
            Bn, pop, K - 1, N, device=A.device)
        d = torch.max(torch.min(d, bdv.unsqueeze(1).unsqueeze(1)),
                      -bdv.unsqueeze(1).unsqueeze(1))
        x = xb.unsqueeze(1) + torch.cat(
            [xb.new_zeros(Bn, pop, 1, N), d, xb.new_zeros(Bn, pop, 1, N)], dim=2)
        x = x.reshape(Bn * pop, K + 1, N)
        hi = r_rep.view(Bn * pop, 1, N)
        x = torch.minimum(torch.maximum(x, torch.zeros_like(x)), hi)
        dur_e = dur.unsqueeze(1).expand(Bn, pop, K).reshape(Bn * pop, K)
        ev = eval_curve(field_bank, x, dur_e, A_rep, B_rep, T_rep,
                        rates.repeat_interleave(pop, dim=0), r_rep, cfg, ci_pos)
        obj = (ev["J"] + cfg.pen_rate * ev["rate_viol"]
               + cfg.pen_corr * _corridor_uniform(cfg, x, A_rep, B_rep,
                                                  r_rep)).view(Bn, pop)
        k = max(2, int(cfg.elite_frac * pop))
        idx = obj.argsort(dim=1)[:, :k]
        idx4 = idx.view(Bn, k, 1, 1).expand(Bn, k, K - 1, N)
        elite = d.view(Bn, pop, K - 1, N).gather(1, idx4)
        mean = elite.mean(dim=1)
        std = elite.std(dim=1, unbiased=False) + 1e-2 * bdv.unsqueeze(1)
        bobj = obj.min(dim=1).values
        better = bobj < best_obj
        bi = obj.argmin(dim=1).view(Bn, 1, 1, 1).expand(Bn, 1, K + 1, N)
        cand = x.view(Bn, pop, K + 1, N).gather(1, bi).squeeze(1)
        best_x = torch.where(better.view(Bn, 1, 1), cand, best_x)
        best_obj = torch.where(better, bobj, best_obj)
    best_x = _project(cfg, best_x, A, B, T_b, rates, rpm)
    ev = eval_curve(field_bank, best_x, dur, A, B, T_b, rates, rpm, cfg)
    bad = (torch.clamp(ev["rate_viol"] - 1e-6, min=0)
           + _corridor_uniform(cfg, best_x, A, B, rpm)) > 1e-6
    best_x = torch.where(bad.view(Bn, 1, 1), xb, best_x)
    return best_x


def _pathonly_warm_start(field_bank, A, B, T_b, rates, rpm, cfg):
    """Internal path-only solution (never returned / never evaluated separately)."""
    torch.manual_seed(cfg.seed)
    pts = _cem_plan_stage1(field_bank, A, B, T_b, rates, rpm, cfg)
    return _cem_corridor_stage2(field_bank, A, B, T_b, rates, rpm, cfg, pts)


def _cem_joint_stage(field_bank, A, B, T_b, rates, rpm, cfg,
                     mean_d, std_d, mean_t, std_t, pop, iters, seed):
    """One joint CEM stage over (d, delta)."""
    torch.manual_seed(seed)
    K, N = cfg.K, A.shape[1]
    Bn = A.shape[0]
    bdv = bd_of(cfg, A, B, rpm)
    a_rep = A.repeat_interleave(pop, dim=0)
    b_rep = B.repeat_interleave(pop, dim=0)
    T_rep = T_b.repeat_interleave(pop, dim=0)
    r_rep = rpm.repeat_interleave(pop, dim=0)
    rates_rep = rates.repeat_interleave(pop, dim=0)
    ci_rep = torch.arange(Bn, device=A.device).repeat_interleave(pop)
    ci_pos = ci_rep.repeat_interleave(K * cfg.s_samples)
    zeros_d = torch.zeros(Bn, K - 1, N, device=A.device)
    zeros_t = torch.zeros(Bn, K - 1, device=A.device)
    best_hard_p = best_hard_dur = None
    best_hard_J = torch.full((Bn,), float("nan"), device=A.device)
    best_hard_score = torch.full((Bn,), 1e10, device=A.device)
    best_pen_p = best_pen_dur = None
    best_pen = torch.full((Bn,), 1e10, device=A.device)
    for _ in range(iters):
        d = mean_d.unsqueeze(1) + std_d.unsqueeze(1) * torch.randn(
            Bn, pop, K - 1, N, device=A.device)
        lim = bdv.unsqueeze(1).unsqueeze(1)
        d = torch.minimum(torch.maximum(d, -lim), lim)
        dl = mean_t.unsqueeze(1) + std_t.unsqueeze(1) * torch.randn(
            Bn, pop, K - 1, device=A.device)
        dl = torch.clamp(dl, -cfg.tau_bound, cfg.tau_bound)
        p, dur, _ = decode(cfg, d.reshape(Bn * pop, K - 1, N),
                           dl.reshape(Bn * pop, K - 1),
                           a_rep, b_rep, T_rep, r_rep, rates_rep)
        ev = eval_curve(field_bank, p, dur, a_rep, b_rep, T_rep,
                        rates_rep, r_rep, cfg, ci_pos)
        Jv = ev["J"].view(Bn, pop)
        obj = (Jv + cfg.pen_rate * ev["rate_viol"].view(Bn, pop)
               + cfg.pen_corr * ev["corridor_viol"].view(Bn, pop))
        hard = ev["hard_feasible"].view(Bn, pop)
        score = torch.where(hard, Jv, torch.full_like(Jv, 1e9))
        cand_score = score.min(dim=1).values
        bi = score.argmin(dim=1)
        cand_p = p.view(Bn, pop, K + 1, N).gather(
            1, bi.view(Bn, 1, 1, 1).expand(Bn, 1, K + 1, N)).squeeze(1)
        cand_dur = dur.view(Bn, pop, K).gather(
            1, bi.view(Bn, 1, 1).expand(Bn, 1, K)).squeeze(1)
        cand_J = Jv.gather(1, bi.view(Bn, 1)).squeeze(1)
        better = cand_score < best_hard_score
        if best_hard_p is None:
            better = torch.ones_like(better)
            best_hard_p = cand_p.clone()
            best_hard_dur = cand_dur.clone()
        best_hard_p = torch.where(better.view(Bn, 1, 1), cand_p, best_hard_p)
        best_hard_dur = torch.where(better.view(Bn, 1), cand_dur, best_hard_dur)
        best_hard_J = torch.where(better, cand_J, best_hard_J)
        best_hard_score = torch.where(better, cand_score, best_hard_score)
        bobj = obj.min(dim=1).values
        bi2 = obj.argmin(dim=1)
        cand_p2 = p.view(Bn, pop, K + 1, N).gather(
            1, bi2.view(Bn, 1, 1, 1).expand(Bn, 1, K + 1, N)).squeeze(1)
        cand_dur2 = dur.view(Bn, pop, K).gather(
            1, bi2.view(Bn, 1, 1).expand(Bn, 1, K)).squeeze(1)
        better2 = bobj < best_pen
        if best_pen_p is None:
            better2 = torch.ones_like(better2)
            best_pen_p = cand_p2.clone()
            best_pen_dur = cand_dur2.clone()
        best_pen_p = torch.where(better2.view(Bn, 1, 1), cand_p2, best_pen_p)
        best_pen_dur = torch.where(better2.view(Bn, 1), cand_dur2, best_pen_dur)
        best_pen = torch.where(better2, bobj, best_pen)
        k = max(2, int(cfg.elite_frac * pop))
        idx = obj.argsort(dim=1)[:, :k]
        ed = d.view(Bn, pop, K - 1, N).gather(
            1, idx.view(Bn, k, 1, 1).expand(Bn, k, K - 1, N))
        mean_d = ed.mean(dim=1)
        std_d = ed.std(dim=1, unbiased=False) + 1e-2 * bdv.unsqueeze(1)
        et = dl.view(Bn, pop, K - 1).gather(1, idx.view(Bn, k, 1).expand(Bn, k, K - 1))
        mean_t = et.mean(dim=1)
        std_t = et.std(dim=1, unbiased=False) + 1e-2
    hard_found = best_hard_score < 1e8
    best_hard_J = torch.where(hard_found, best_hard_J,
                              torch.full_like(best_hard_J, float("nan")))
    return {"p_hard": best_hard_p, "dur_hard": best_hard_dur, "J_hard": best_hard_J,
            "hard_found": hard_found,
            "p_pen": best_pen_p, "dur_pen": best_pen_dur,
            "mean_d": mean_d, "std_d": std_d, "mean_t": mean_t, "std_t": std_t}


def _stage_best(res, Bn: int):
    has = res["hard_found"].view(Bn, 1, 1)
    p = torch.where(has, res["p_hard"], res["p_pen"])
    dur = torch.where(has.view(Bn, 1), res["dur_hard"], res["dur_pen"])
    return p, dur


# --------------------------------------------------------------------------- public API
def plan_batch(field_bank, A: torch.Tensor, B: torch.Tensor, T_b: torch.Tensor,
               rates: torch.Tensor, cfg: Optional[CoTraJConfig] = None):
    """Plan B cases jointly; returns (points, durations, info).

    All search variables/evaluations use the per-case ``rates`` rows and the
    per-case box bound from ``field_bank.rpm_max``.  The returned durations are
    rate-anchored, sum to ``T_b`` exactly and are what the reported ``J`` was
    evaluated with.  ``dt_k >= tmin_k`` (rate-feasible by construction) holds
    for every segment whose geometry fits the budget; see the module docstring.
    """
    cfg = cfg or CoTraJConfig()
    if A.dim() != 2 or B.dim() != 2 or rates.dim() != 2:
        raise ValueError("A, B and rates must be (B,N) tensors")
    Bn, N = A.shape
    if B.shape != A.shape or rates.shape != A.shape:
        raise ValueError("A, B, rates must share shape (B,N)")
    if T_b.dim() != 1 or T_b.shape[0] != Bn:
        raise ValueError("T_b must be (B,)")
    device = A.device
    rpm = _batch_rpm(field_bank, Bn, device)
    K = cfg.K
    with torch.no_grad():
        # 1) internal path-only warm start (search only)
        x0 = _pathonly_warm_start(field_bank, A, B, T_b, rates, rpm, cfg)
        dur0 = (T_b / K).unsqueeze(1).expand(Bn, K).contiguous()
        bdv = bd_of(cfg, A, B, rpm)
        d0, t0 = encode(cfg, x0, dur0, A, B)
        # 2) joint stage A (broad)
        resA = _cem_joint_stage(
            field_bank, A, B, T_b, rates, rpm, cfg,
            d0, (0.25 * bdv).unsqueeze(1).expand(Bn, K - 1, N).clone(),
            t0, torch.full((Bn, K - 1), 0.10, device=device),
            cfg.joint_pop, cfg.iters_a, cfg.seed + 101)
        pA, durA = _stage_best(resA, Bn)
        # 3) joint stage B (narrow)
        eA_d, eA_t = encode(cfg, pA, durA, A, B)
        resB = _cem_joint_stage(
            field_bank, A, B, T_b, rates, rpm, cfg,
            eA_d, (0.10 * bdv).unsqueeze(1).expand(Bn, K - 1, N).clone(),
            eA_t, torch.full((Bn, K - 1), 0.03, device=device),
            cfg.joint_pop, cfg.iters_b, cfg.seed + 202)
        pB, durB = _stage_best(resB, Bn)
        # 4) candidate pool: joint B, joint A, 3 interpolations, incumbent
        cands = [(pB, durB), (pA, durA)]
        labels = ["joint", "jointA"]
        for alpha in (0.75, 0.5, 0.25):
            cands.append((x0 + alpha * (pB - x0),
                          (1.0 - alpha) * dur0 + alpha * durB))
            labels.append("interp%.2f" % alpha)
        cands.append((x0, dur0))
        labels.append("pathonly")
        Nc = len(cands)
        P = torch.cat([c[0] for c in cands], dim=0)
        D = torch.cat([c[1] for c in cands], dim=0)
        A_rep = A.repeat(Nc, 1)
        B_rep = B.repeat(Nc, 1)
        T_rep = T_b.repeat(Nc)
        r_rep = rates.repeat(Nc, 1)
        rpm_rep = rpm.repeat(Nc, 1)
        ci_pos = torch.arange(Nc * Bn, device=device).repeat_interleave(K * cfg.s_samples)
        # case index inside the stacked candidate evaluation must map back to the
        # original case: candidate-major stacking -> case = row % Bn
        case_ids = (torch.arange(Nc * Bn, device=device) % Bn)
        ci_pos = case_ids.repeat_interleave(K * cfg.s_samples)
        ev = eval_curve(field_bank, P, D, A_rep, B_rep, T_rep,
                        r_rep, rpm_rep, cfg, ci_pos)
        Jc = ev["J"].view(Nc, Bn)
        hardc = ev["hard_feasible"].view(Nc, Bn)
        score = torch.where(hardc, Jc, torch.full_like(Jc, 1e9))
        kb = score.argmin(dim=0)
        ar = torch.arange(Bn, device=device)
        p_sel = P.view(Nc, Bn, K + 1, N)[kb, ar]
        dur_sel = D.view(Nc, Bn, K)[kb, ar]
        # 5) evaluate the selected curve once (with its own durations)
        ev_sel = eval_curve(field_bank, p_sel, dur_sel, A, B, T_b, rates, rpm, cfg)
    label = [labels[int(k)] for k in kb.cpu().numpy()]
    info = {
        "J": ev_sel["J"], "T": ev_sel["T"],
        "rate_viol": ev_sel["rate_viol"], "corridor_viol": ev_sel["corridor_viol"],
        "T_rel": ev_sel["T_rel"], "ep": ev_sel["ep"],
        "feasible": ev_sel["feasible"], "hard_feasible": ev_sel["hard_feasible"],
        "fallback": label,
        "config": {"K": K, "rho": cfg.rho, "delta": cfg.delta,
                   "path_pop": cfg.path_pop, "path_iters": cfg.path_iters,
                   "stage2_pop": cfg.stage2_pop, "stage2_iters": cfg.stage2_iters,
                   "joint_pop": cfg.joint_pop, "iters_a": cfg.iters_a,
                   "iters_b": cfg.iters_b, "seed": cfg.seed},
    }
    return p_sel, dur_sel, info
