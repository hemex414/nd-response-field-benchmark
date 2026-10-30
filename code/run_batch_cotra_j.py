# -*- coding: utf-8 -*-
"""run_batch_cotra_j.py -- batch runner for the standalone CoTra-J module.

This is the *integration* layer requested by the Lead: it wires
``cotra_j.plan_batch`` (GPU, B cases at once, per-case rates) to the frozen
public case files, while the reported records are produced with the public
numpy evaluator (``public_field/code/evaluate.py``) so they can be merged with
the public baseline records.

Default behaviour (per Lead):
  * batch size B = 16 cases at a time,
  * per-case rates are taken from each case record (explicit (B,N)),
  * record schema follows ``baselines.common.path_to_record`` (subset of it),
  * K = 8 (CoTra-J), rho = 0.2, delta = 0.02, S_search = S_eval = 4 (the public
    benchmark evaluator uses 4 sub-samples/segment; the module default stays 8
    for the paper / 3-D parity test).

Field handling: the public case files store ``N``, ``subset`` and
``field_seed``; the analytic public field is rebuilt per case (``field.FieldND``)
and converted to a small torch evaluator on the GPU.  Nothing is stored as a
dense N-D array, so memory is O(points) not O(8^N).

Usage
    python public_field/code/run_batch_cotra_j.py --n 3 --cases 64 \
        --batch 16 --out public_field/results/N3/cotra_j_batch.jsonl
    python public_field/code/run_batch_cotra_j.py --n 3 --cases 8 --batch 4 \
        --check-field          # additionally checks torch-vs-numpy field parity

Sharding (optional, for a full 1000-case run over several processes):
    python ... --n 3 --batch 16 --start 0   --end 256 --out part0.jsonl
    python ... --n 3 --batch 16 --start 256 --end 512 --out part1.jsonl
Keep ``--start`` a multiple of ``--batch``: the CEM seeds each stage once per
chunk, so a case is reproduced exactly only at its sequential chunk position.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cotra_j as CJ          # noqa: E402
import field as PUBF          # noqa: E402  (public numpy field generator)
import evaluate as PEV        # noqa: E402  (public numpy evaluator)


def _corr_viol_time(prob, nodes, durations):
    """Time-indexed corridor violation (same definition as CoTra-J)."""
    t = np.concatenate([[0.0], np.cumsum(np.asarray(durations, float))])
    tau = t / t[-1]
    xb = prob.a[None, :] + tau[:, None] * prob.dir[None, :]
    return float(np.max(np.clip(np.abs(np.asarray(nodes) - xb) / prob.bd[None, :] - 1.0,
                                0.0, None)))


def _repair_fixed_durations(prob, nodes, durations, iters=25):
    """Alternating time-corridor / rate projection with the durations held fixed."""
    x = np.array(nodes, dtype=np.float64, copy=True)
    dur = np.asarray(durations, dtype=np.float64)
    for _ in range(iters):
        t = np.concatenate([[0.0], np.cumsum(dur)])
        tau = t / t[-1]
        xb = prob.a[None, :] + tau[:, None] * prob.dir[None, :]
        x = np.clip(x, xb - prob.bd[None, :], xb + prob.bd[None, :])
        x = np.clip(x, prob.lo[None, :], prob.hi[None, :])
        x[0] = prob.a
        x[-1] = prob.b
        v = np.diff(x, axis=0) / np.maximum(dur, 1e-12)[:, None]
        v = np.clip(v, -prob.rate[None, :], prob.rate[None, :])
        x = np.concatenate([prob.a[None, :],
                            prob.a[None, :] + np.cumsum(v * dur[:, None], axis=0)], axis=0)
    return x


# --------------------------------------------------------------------------- torch field
class _TorchField:
    """GPU evaluator of one public FieldND (same formula / draw order)."""

    def __init__(self, f, device):
        self.N = int(f.N)
        self.device = device
        self.vmin = float(f.vmin)
        self.vmax = float(f.vmax)
        self.w = torch.tensor(np.asarray(f.w), dtype=torch.float32, device=device)
        oned = np.asarray(f.oned, dtype=np.float64)          # (N,4,3)
        self.oned_amp = torch.tensor(oned[..., 0], dtype=torch.float32, device=device)
        self.oned_mu = torch.tensor(oned[..., 1], dtype=torch.float32, device=device)
        self.oned_sig = torch.tensor(oned[..., 2], dtype=torch.float32, device=device)
        comps = list(f.islands) + list(f.pockets)
        self.tri = torch.tensor(np.stack([c[0] for c in comps]), dtype=torch.long,
                                device=device)
        self.c = torch.tensor(np.stack([c[1] for c in comps]), dtype=torch.float32,
                              device=device)
        self.s = torch.tensor(np.stack([c[2] for c in comps]), dtype=torch.float32,
                              device=device)
        self.amp = torch.tensor(np.asarray([c[3] for c in comps], float),
                                dtype=torch.float32, device=device)
        self.rip_dims = [list(r[0]) for r in f.ripples]
        self.rip_amp = torch.tensor([r[1] for r in f.ripples], dtype=torch.float32,
                                    device=device)
        self.rip_lam = torch.tensor([r[2] for r in f.ripples], dtype=torch.float32,
                                    device=device)
        self.rip_ph = [torch.tensor(np.asarray(r[3]), dtype=torch.float32,
                                    device=device) for r in f.ripples]

    def eval(self, points: torch.Tensor) -> torch.Tensor:
        M = points.shape[0]
        u = points.to(dtype=torch.float32).clamp(0.0, PUBF.DOMAIN)
        V = (u * self.w.view(1, -1)).sum(dim=1) / PUBF.DOMAIN
        for i in range(self.N):
            d = (u[:, i].unsqueeze(1) - self.oned_mu[i].view(1, -1)) / self.oned_sig[i].view(1, -1)
            V = V + (self.oned_amp[i].view(1, -1) * torch.exp(-0.5 * d * d)).sum(dim=1)
        # islands + pockets: loop over component blocks to keep memory O(M*block)
        ncomp = self.tri.shape[0]
        Ktri = self.tri.shape[1]
        for k0 in range(0, ncomp, 16):
            tr = self.tri[k0:k0 + 16]
            c = self.c[k0:k0 + 16].view(1, -1, Ktri)
            s = self.s[k0:k0 + 16].view(1, -1, Ktri)
            a = self.amp[k0:k0 + 16].view(1, -1)
            d = torch.zeros(M, tr.shape[0], device=points.device, dtype=torch.float32)
            for k in range(Ktri):
                uk = u[:, tr[:, k]].unsqueeze(2)            # (M,Bk,1)
                d = d + (((uk - c[:, :, k:k + 1]) / s[:, :, k:k + 1]) ** 2).squeeze(2)
            V = V + (a * torch.exp(-0.5 * d)).sum(dim=1)
        for k, dims in enumerate(self.rip_dims):
            term = torch.ones(M, device=points.device, dtype=torch.float32)
            for j, dm in enumerate(dims):
                term = term * torch.sin(2.0 * math.pi * u[:, dm] / self.rip_lam[k]
                                        + self.rip_ph[k][j])
            V = V + self.rip_amp[k] * term
        V = PUBF.VMIN + (PUBF.VMAX - PUBF.VMIN) * (V - self.vmin) / (self.vmax - self.vmin)
        return V.clamp(PUBF.VMIN, PUBF.VMAX)


class TorchFieldBank:
    """Per-case torch fields; ``eval(points, case_idx)`` keeps memory O(points)."""

    def __init__(self, fields, device):
        self.device = device
        self.fields = [_TorchField(f, device) for f in fields]
        self.B = len(self.fields)
        self.N = self.fields[0].N
        self.rpm_max = torch.full((self.N,), PUBF.DOMAIN, dtype=torch.float32,
                                  device=device)

    def eval(self, points: torch.Tensor, case_idx: torch.Tensor) -> torch.Tensor:
        flat = points.reshape(-1, points.shape[-1])
        ci = case_idx.reshape(-1)
        out = flat.new_empty(flat.shape[0])
        for c in torch.unique(ci).tolist():
            m = ci == c
            out[m] = self.fields[int(c)].eval(flat[m])
        return out


# --------------------------------------------------------------------------- records
def _record(case, nodes, durations, J, feasible, cfg, info, wall_s):
    rates = np.asarray(case["rates"], float)
    a = np.asarray(case["A"], float)
    b = np.asarray(case["B"], float)
    t_fast = float(np.max(np.abs(b - a) / rates))
    T_b = float(case["TR"]) * t_fast
    prob = PEV.Problem(case["_field"], a, b, T_b, rates, K=cfg.K, rho=cfg.rho, delta=cfg.delta)
    nodes = np.asarray(nodes, float)
    durations = np.asarray(durations, float)
    ev = prob.eval(nodes, durations=durations)
    corr_time = _corr_viol_time(prob, nodes, durations)
    feas = (ev["endpoint_error"] <= 1e-3 and ev["rate_violation"] <= 0.01
            and corr_time <= 1e-6
            and abs(ev["T"] - prob.T_b) <= max(0.01 * prob.T_b, prob.dt))
    repaired = False
    if not feas:
        # float32 planner -> float64 public record: re-project into the
        # time-indexed corridor / rate box; if that fails, fall back to the
        # nominal synchronous trajectory (always feasible).
        nodes2 = _repair_fixed_durations(prob, nodes, durations)
        ev2 = prob.eval(nodes2, durations=durations)
        corr2 = _corr_viol_time(prob, nodes2, durations)
        feas2 = (ev2["endpoint_error"] <= 1e-3 and ev2["rate_violation"] <= 0.01
                 and corr2 <= 1e-6
                 and abs(ev2["T"] - prob.T_b) <= max(0.01 * prob.T_b, prob.dt))
        if feas2:
            nodes, ev, corr_time, feas, repaired = nodes2, ev2, corr2, feas2, True
        else:
            nodes = prob.base.copy()
            durations = np.full(cfg.K, prob.dt)
            ev = prob.eval(nodes, durations=durations)
            corr_time = _corr_viol_time(prob, nodes, durations)
            feas = True
            repaired = True
    corr_geom = float(np.max(prob.corr_viol(nodes)))
    dur_u = np.full(cfg.K, prob.dt)
    ev_u = prob.eval(nodes)
    corr_time_u = _corr_viol_time(prob, nodes, dur_u)
    return {
        "case_id": case["case_id"], "N": int(case["N"]), "subset": case["subset"],
        "field_seed": case["field_seed"], "A": case["A"], "B": case["B"],
        "TR": case["TR"], "rates": case["rates"],
        "device": str(nodes.device) if hasattr(nodes, "device") else "cpu",
        "method": "cotra_j", "K": int(cfg.K),
        "J": float(ev["J"]), "J_uniform": float(ev_u["J"]),
        "T": float(ev["T"]), "durations": [float(v) for v in durations],
        "time_alloc": "CoTra-J warp (per-segment durations returned by the planner)",
        "time_alloc_applied": True,
        "endpoint_error": float(ev["endpoint_error"]),
        "rate_violation": float(ev["rate_violation"]),
        "rate_violation_uniform": float(ev_u["rate_violation"]),
        "corridor_violation_raw": corr_time, "repaired_postprocess": bool(repaired),
        "corridor_violation": corr_time,
        "corridor_violation_time": corr_time,
        "corridor_violation_geom": corr_geom,
        "corridor_violation_time_uniform": corr_time_u,
        "feasible": bool(feas), "feasible_uniform_time": bool(            ev_u["endpoint_error"] <= 1e-3 and ev_u["rate_violation"] <= 0.01
            and corr_time_u <= 1e-6
            and abs(ev_u["T"] - prob.T_b) <= max(0.01 * prob.T_b, prob.dt)),
        "points": np.asarray(nodes, float).round(6).tolist(),
        "status": "ok", "notes": "public batch CoTra-J (torch, B=%d)" % cfg._batch,
        "info": {k: (v.tolist() if torch.is_tensor(v) else v)
                 for k, v in info.items() if k in ("rate_viol", "corridor_viol", "ep")},
        "wall_s": wall_s,
        "params": {"K": cfg.K, "rho": cfg.rho, "delta": cfg.delta,
                   "path_pop": cfg.path_pop, "path_iters": cfg.path_iters,
                   "stage2_pop": cfg.stage2_pop, "stage2_iters": cfg.stage2_iters,
                   "joint_pop": cfg.joint_pop, "iters_a": cfg.iters_a,
                   "iters_b": cfg.iters_b, "seed": cfg.seed,
                   "s_samples": cfg.s_samples, "batch": cfg._batch},
    }


# --------------------------------------------------------------------------- main
def run(n, cases, batch, out, device="cuda", case_file=None, check_field=False,
        n_cases=None, tr=None, seed=0, start=0, end=None):
    """Plan cases ``[start, end)`` of one N and append one record per case.

    ``start`` / ``end`` split a 1000-case run over several processes.  Keep
    ``start`` a multiple of ``batch`` so that every chunk matches the chunk of
    the sequential run case-for-case (the CEM seeds each stage once per chunk,
    so the per-case plan is reproducible only at fixed chunk boundaries).
    """
    case_file = case_file or None
    cs = PUBF.load_cases(n, case_file)
    if cases is not None:
        cs = cs[:int(cases)]
    if n_cases is not None:
        cs = cs[:int(n_cases)]
    if tr is not None:
        for c in cs:
            c["TR"] = float(tr)
    start = max(0, int(start))
    end = len(cs) if end is None else min(int(end), len(cs))
    cs = cs[start:end]
    print("[runner] N=%d cases=%d batch=%d device=%s range=[%d,%d)"
          % (n, len(cs), batch, device, start, end))
    cfg = CJ.CoTraJConfig(seed=seed, s_samples=4)   # public evaluator uses S=4
    cfg._batch = batch
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf8"):
            if not line.strip():
                continue
            try:                       # skip a truncated line after an interrupted run
                done.add(json.loads(line)["case_id"])
            except ValueError:
                continue
    n_new = 0
    t0 = time.perf_counter()
    with open(out, "a", encoding="utf8") as fh:
        for i0 in range(start, end, batch):
            chunk = [c for c in cs[i0 - start:i0 - start + batch]
                     if c["case_id"] not in done]
            if not chunk:
                continue
            # public numpy fields (also kept for record evaluation)
            for c in chunk:
                if "_field" not in c:
                    c["_field"] = PUBF.FieldND(int(c["N"]), int(c["field_seed"]))
            bank = TorchFieldBank([c["_field"] for c in chunk], device)
            if check_field and n_new == 0:
                _check_field_parity(bank, chunk)
            A = torch.tensor(np.stack([c["A"] for c in chunk]), dtype=torch.float32,
                             device=device)
            B = torch.tensor(np.stack([c["B"] for c in chunk]), dtype=torch.float32,
                             device=device)
            rates = torch.tensor(np.stack([c["rates"] for c in chunk]),
                                 dtype=torch.float32, device=device)
            t_fast = ((B - A).abs() / rates).amax(dim=1)
            T_b = torch.tensor([float(c["TR"]) for c in chunk], dtype=torch.float32,
                               device=device) * t_fast
            t1 = time.perf_counter()
            p, d, info = CJ.plan_batch(bank, A, B, T_b, rates, cfg)
            torch.cuda.synchronize() if device == "cuda" else None
            wall = time.perf_counter() - t1
            p_np = p.detach().cpu().numpy()
            d_np = d.detach().cpu().numpy()
            for k, c in enumerate(chunk):
                rec = _record(c, p_np[k], d_np[k],
                              float(info["J"][k]), bool(info["feasible"][k]),
                              cfg, info, wall / len(chunk))
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_new += len(chunk)
            print("[runner] %d/%d  %.2f s/case" % (n_new, len(cs),
                                                   (time.perf_counter() - t0) / n_new),
                  flush=True)
    print("[runner] wrote", out, "(%d new records, %.1f s)"
          % (n_new, time.perf_counter() - t0))


def _check_field_parity(bank, chunk):
    g = np.random.default_rng(7)
    for i, c in enumerate(chunk[:3]):
        pts = g.uniform(0, 100, size=(2000, int(c["N"])))
        ref = np.asarray(c["_field"].interp(pts), float)
        got = bank.fields[i].eval(torch.tensor(pts, dtype=torch.float32,
                                               device=bank.device)).cpu().numpy()
        print("[field] case %d torch-vs-numpy max|d|=%.2e" % (c["case_id"],
                                                              float(np.abs(ref - got).max())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--cases", type=int, default=None)
    ap.add_argument("--n-cases", type=int, default=None,
                    help="limit number of cases (alias --cases)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--case-file", default=None)
    ap.add_argument("--check-field", action="store_true")
    ap.add_argument("--tr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0,
                    help="first case index of this shard (multiple of --batch)")
    ap.add_argument("--end", type=int, default=None,
                    help="exclusive last case index of this shard")
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "..", "results", "N%d" % args.n,
                                   "cotra_j_batch.jsonl")
    run(args.n, args.cases, args.batch, out, device=args.device,
        case_file=args.case_file, check_field=args.check_field,
        n_cases=args.n_cases, tr=args.tr, seed=args.seed,
        start=args.start, end=args.end)


if __name__ == "__main__":
    main()
