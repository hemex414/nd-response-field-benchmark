# -*- coding: utf-8 -*-
"""Tests / benchmarks for the standalone ``cotra_j`` module.

This file is *test-only* and fully self-contained: it never imports anything
outside this directory (``cotra_j.py`` itself has no third-party imports beyond
``torch``).

What is checked
---------------
P1  Contract test with a per-case analytic field bank (N=4, 8 cases): output
    shapes, sum(durations)==T_b, J recomputed with the returned durations equals
    info['J'], and per-case rates are really used (same geometry, different
    rates -> different plans and feasibility).
P2  Single-process batch benchmark B=1/4/8/16 on the analytic N-D bank:
    wall time per case and peak GPU memory; B=4 is the measured reference point.

Run:
    python code/test_cotra_j.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cotra_j as CJ  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class _GaussField:
    """Small analytic per-case field (vectorised) for the contract/perf tests."""

    def __init__(self, n, seed, n_comp=64):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.n = n
        self.c = torch.rand(n_comp, n, device=DEV, generator=g) * 100.0
        self.s = 3.0 + torch.rand(n_comp, n, device=DEV, generator=g) * 9.0
        self.a = (torch.rand(n_comp, device=DEV, generator=g) * 2 - 1) * 4.0
        self.rpm_max = torch.full((n,), 100.0, device=DEV)

    def eval(self, points):
        u = points.reshape(-1, self.n)
        d = torch.zeros(u.shape[0], 1, device=u.device)
        for k in range(0, self.c.shape[0], 16):
            c = self.c[k:k + 16]
            s = self.s[k:k + 16]
            a = self.a[k:k + 16]
            d = d + (a.view(1, -1) * torch.exp(
                -0.5 * (((u[:, None, :] - c[None]) / s[None]) ** 2).sum(-1))).sum(1, keepdim=True)
        return 100.0 + d.squeeze(1)


# --------------------------------------------------------------------------- P1
def test_contract():
    print("\n=== P1  contract test with per-case analytic N-D field ===")
    N, B = 4, 8
    fields = [_GaussField(N, seed=100 + i) for i in range(B)]
    bank = CJ.AnalyticNDFieldBank(fields, rpm_max=torch.full((N,), 100.0))
    g = torch.Generator(device=DEV).manual_seed(7)
    A = torch.rand(B, N, device=DEV, generator=g) * 100.0
    Bv = torch.rand(B, N, device=DEV, generator=g) * 100.0
    rates = torch.full((B, N), 8.0, device=DEV)
    T_b = 2.5 * ((Bv - A).abs() / rates).amax(dim=1)
    cfg = CJ.CoTraJConfig(K=6)   # public-field convention is K=6 (any K supported)
    p, d, info = CJ.plan_batch(bank, A, Bv, T_b, rates, cfg)
    checks = {
        "points_shape": tuple(p.shape) == (B, cfg.K + 1, N),
        "durations_shape": tuple(d.shape) == (B, cfg.K),
        "sum_durations_Tb": float(((d.sum(1) - T_b).abs() / T_b).max()) <= 1e-6,
        "J_finite": bool(torch.isfinite(info["J"]).all()),
        "feasible": bool(info["feasible"].all()),
    }
    # J must be reproducible from the returned durations (not equal durations)
    rpm = bank.rpm_max
    ev = CJ.eval_curve(bank, p, d, A, Bv, T_b, rates, rpm, cfg)
    checks["J_uses_returned_durations"] = float((ev["J"] - info["J"]).abs().max()) <= 1e-6
    # per-case rates are really used: same geometry, one case gets much tighter rates
    A2 = torch.cat([A[:1], A[:1]], 0)
    B2 = torch.cat([Bv[:1], Bv[:1]], 0)
    rates2 = torch.stack([rates[0], rates[0] * 0.35], 0)
    bank2 = CJ.AnalyticNDFieldBank([fields[0], fields[0]],
                                  rpm_max=torch.full((N,), 100.0))
    T2 = 2.5 * ((B2 - A2).abs() / rates2).amax(dim=1)
    p2, d2, info2 = CJ.plan_batch(bank2, A2, B2, T2, rates2, cfg)
    checks["rates_per_case_used"] = float((p2[0] - p2[1]).abs().max()) > 1e-3
    for k, v in checks.items():
        print("  %-32s %s" % (k, "PASS" if v else "FAIL"))
    return checks


# --------------------------------------------------------------------------- P2
def _time_batch(bank, A, Bv, T_b, rates, cfg):
    """One timed plan_batch call; returns (seconds, peak GPU MB)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    CJ.plan_batch(bank, A, Bv, T_b, rates, cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20 \
        if torch.cuda.is_available() else float("nan")
    return dt, peak


def test_perf():
    print("\n=== P2  batch benchmark (single process, analytic N-D bank) ===")
    N, Bmax = 4, 16
    fields = [_GaussField(N, seed=200 + i) for i in range(Bmax)]
    bank = CJ.AnalyticNDFieldBank(fields, rpm_max=torch.full((N,), 100.0))
    g = torch.Generator(device=DEV).manual_seed(11)
    A = torch.rand(Bmax, N, device=DEV, generator=g) * 100.0
    Bv = torch.rand(Bmax, N, device=DEV, generator=g) * 100.0
    rates = torch.full((Bmax, N), 8.0, device=DEV)
    T_b = 2.5 * ((Bv - A).abs() / rates).amax(dim=1)
    cfg = CJ.CoTraJConfig()
    rows = []
    for B in (1, 4, 8, 16):
        a, b = A[:B].contiguous(), Bv[:B].contiguous()
        t, r = T_b[:B].contiguous(), rates[:B].contiguous()
        CJ.plan_batch(bank, a, b, t, r, cfg)              # warm-up
        dt, peak = _time_batch(bank, a, b, t, r, cfg)
        rows.append({"B": B, "seconds": dt, "s_per_case": dt / B,
                     "peak_gpu_mb": peak})
        print("  B=%-2d  %7.2f s  %6.3f s/case  peakGPU=%8.1f MB (%.0f MiB)"
              % (B, dt, dt / B, peak, peak))
    # speed-up: sequential B x 1-case time / batched time
    t1 = rows[0]["seconds"]
    for r in rows:
        r["speedup"] = (r["B"] * t1) / r["seconds"] if r["seconds"] > 0 else float("nan")
        print("  B=%-2d  speed-up vs sequential B=1: %.2fx" % (r["B"], r["speedup"]))
    return rows


# --------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "test_cotra_j_report.md"))
    args = ap.parse_args()
    print("device:", DEV, "| torch", torch.__version__)
    report = ["# cotra_j standalone module - test report", "",
              "device: `%s`, torch `%s`" % (DEV, torch.__version__), ""]
    ok = True
    c = test_contract()
    ok = ok and all(c.values())
    report += ["## P1 - API contract (per-case analytic bank)", "",
               "| check | result |", "|---|---|"]
    for k, v in c.items():
        report.append("| %s | %s |" % (k, "PASS" if v else "FAIL"))
    report.append("")
    pr = test_perf()
    report += ["## P2 - batch benchmark (single process, analytic N-D bank)", "",
               "| B | wall time (s) | s/case | peak GPU (MB) | speed-up vs sequential B=1 |",
               "|---|---|---|---|---|"]
    for r in pr:
        report.append("| %d | %.2f | %.3f | %.1f | %.2fx |"
                      % (r["B"], r["seconds"], r["s_per_case"], r["peak_gpu_mb"],
                         r["speedup"]))
    report += ["", "Note: CEM is stochastic; timings are single-run single-process and "
               "include the internal path-only warm start (no separate evaluation).", ""]
    report += ["## Verdict", "", "ALL PASS" if ok else "SOME CHECKS FAILED", ""]
    open(args.out, "w", encoding="utf8").write("\n".join(report))
    print("\nreport written to", args.out)
    print("VERDICT:", "ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
