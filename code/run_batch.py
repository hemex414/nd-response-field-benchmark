#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_batch.py -- batch baseline runner for the public field (resume + merge).

Usage
-----
python code/run_batch.py --N 3 --methods fmm astar rrt_star ga sa --out results/E3 --workers 12
python code/run_batch.py --cases cases/N3/cases.jsonl --methods ... --resume
python code/run_batch.py --N 3 --merge-only --out results/E3

Result layout (PUBLIC_PACKAGE_SPEC.md §4):
    <out>/run_manifest.json
    <out>/methods/<method>_N<n>.json
    <out>/per_case_N<n>.jsonl
    <out>/summary.json
    <out>/timing.csv
    <out>/timeout.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from code import field as F                    # noqa: E402
from code.baselines import (plan_batch, plan_single, PARAMS, METHODS,  # noqa: E402
                                         POOL_METHODS, TENSOR_BATCH_METHODS,
                                         BATCH_METHODS)


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _append_jsonl(path, recs):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
        fh.flush()


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=float)
    os.replace(tmp, path)


IDLE_REF_PATH = os.path.join(HERE, "idle_reference.json")


def _idle_ref(method, n):
    """Idle single-case seconds/case for (method, N), or None if not tabulated."""
    try:
        with open(IDLE_REF_PATH, encoding="utf-8") as fh:
            ref = json.load(fh)
        return float(ref.get(method, {}).get(str(n)))
    except Exception:                                  # noqa: BLE001
        return None


def load_cases(args):
    if args.cases:
        return F.load_cases(args.N, args.cases)
    return F.load_cases(args.N)


def merge(out, n, methods):
    per_method = {}
    timing_rows = []
    timeout_rows = []
    cases_by_id = {}
    for m in methods:
        jl = os.path.join(out, "methods", "%s_N%d.jsonl" % (m, n))
        if os.path.exists(jl):
            # jsonl is the resumable source of truth; rebuild the per-method JSON from it
            recs = _read_jsonl(jl)
            d = {"method": m, "N": n, "n_cases": len(recs),
                 "config_hash": json.dumps(PARAMS[m], sort_keys=True, default=str),
                 "code_version": "public-baselines-1.0.0", "cases": recs}
            _atomic_json(os.path.join(out, "methods", "%s_N%d.json" % (m, n)), d)
        else:
            path = os.path.join(out, "methods", "%s_N%d.json" % (m, n))
            if not os.path.exists(path):
                continue
            d = json.load(open(path, encoding="utf-8"))
        per_method[m] = d
        # E5 layout: also expose a root-level per-method JSON next to summary.json
        _atomic_json(os.path.join(out, "%s.json" % m), d)
        for c in d["cases"]:
            cases_by_id.setdefault(c["case_id"], {})[m] = c
            timing_rows.append({"method": m, "N": n, "case_id": c["case_id"],
                                "runtime_s": c.get("runtime_s"),
                                "device": c.get("device", "cpu")})
            if c.get("status", "ok") != "ok":
                timeout_rows.append({"method": m, "N": n, "case_id": c["case_id"],
                                     "runtime_s": c.get("runtime_s"),
                                     "device": c.get("device", "cpu"),
                                     "status": c.get("status"),
                                     "reason": c.get("notes", "")})
    # per_case jsonl
    with open(os.path.join(out, "per_case_N%d.jsonl" % n), "w", encoding="utf-8") as fh:
        for cid in sorted(cases_by_id):
            row = {"case_id": cid, "methods": cases_by_id[cid]}
            fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
    summary = {"N": n, "methods": sorted(per_method), "n_ok": {}, "mean_J": {},
               "median_J": {}, "feasible_rate": {}, "mean_runtime_s": {},
               "rank": {}, "notes": ""}
    for m, d in per_method.items():
        ok = [c for c in d["cases"] if c.get("status") == "ok"]
        js = [c["J"] for c in ok if c.get("J") is not None]
        summary["n_ok"][m] = len(ok)
        summary["mean_J"][m] = float(np.mean(js)) if js else None
        summary["median_J"][m] = float(np.median(js)) if js else None
        rts = [c.get("runtime_s") for c in ok if c.get("runtime_s") is not None]
        summary["mean_runtime_s"][m] = float(np.mean(rts)) if rts else None
        summary["feasible_rate"][m] = (float(np.mean([bool(c.get("feasible")) for c in ok]))
                                       if ok else None)
    ranked = [m for m in per_method if summary["mean_J"].get(m) is not None]
    ranked.sort(key=lambda m: summary["mean_J"][m])
    summary["rank"] = {m: i + 1 for i, m in enumerate(ranked)}
    _atomic_json(os.path.join(out, "summary.json"), summary)
    if timing_rows:
        with open(os.path.join(out, "timing.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(timing_rows[0].keys()))
            w.writeheader(); w.writerows(timing_rows)
    if timeout_rows:
        with open(os.path.join(out, "timeout.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(timeout_rows[0].keys()))
            w.writeheader(); w.writerows(timeout_rows)
    elif os.path.exists(os.path.join(out, "timeout.csv")):
        os.remove(os.path.join(out, "timeout.csv"))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--cases", default=None, help="cases json/jsonl (default: frozen cases)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only first K cases (tests)")
    ap.add_argument("--max-case-hours", type=float, default=6.0,
                    help="pilot-projected wall-clock budget per 1000 cases; above -> too_slow")
    ap.add_argument("--force-slow", action="store_true", help="disable the too_slow guard")
    args = ap.parse_args()
    out = args.out or os.path.join(PROJ, "results",
                                   "E_public_N%d" % args.N)
    os.makedirs(os.path.join(out, "methods"), exist_ok=True)

    if args.merge_only:
        s = merge(out, args.N, args.methods)
        print(json.dumps({"N": args.N, "n_ok": s["n_ok"]}, indent=2))
        return

    cases = load_cases(args)
    if args.limit:
        cases = cases[:args.limit]
    t_start = time.time()
    for m in args.methods:
        jpath = os.path.join(out, "methods", "%s_N%d.jsonl" % (m, args.N))
        done = {r["case_id"] for r in _read_jsonl(jpath)}
        todo = [c for c in cases if c["case_id"] not in done]
        print("[N%d] %-11s todo=%d/%d" % (args.N, m, len(todo), len(cases)), flush=True)
        if not todo:
            continue
        # pilot: estimate the 1000-case wall-clock from a single-case run (project rule:
        # a cell projected above the budget is marked too_slow, no partial J is kept)
        if not args.force_slow and len(todo) > 8:
            t_p = time.time()
            pilot = plan_single(m, todo[0])
            dt_p = time.time() - t_p
            # too_slow criterion uses the *idle* single-case reference when tabulated,
            # so concurrent sweeps cannot inflate the projection (project rule 2026-10-26)
            ref = _idle_ref(m, args.N)
            basis_s = ref if ref is not None else dt_p
            proj_h = basis_s * 1000.0 / 3600.0
            print("[N%d] %-11s pilot %.2fs (idle ref %s) -> projected %.1f h"
                  % (args.N, m, dt_p, ("%.2fs" % ref) if ref is not None else "none",
                     proj_h), flush=True)
            if pilot.get("status") == "timeout" and pilot.get("grid_too_large"):
                pass                              # handled below (records carry the flag)
            elif proj_h > args.max_case_hours:
                note = ("%s %.3f s/case -> projected %.1f h for 1000 cases > %.0f h "
                        "budget; too_slow (no partial J)" % (
                            "idle-ref" if ref is not None else "pilot", basis_s,
                            proj_h, args.max_case_hours))
                ts = os.path.join(out, "too_slow_N%d.csv" % args.N)
                with open(ts, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["method", "N", "pilot_case_id", "pilot_s", "projected_h",
                                "status", "reason"])
                    w.writerow([m, args.N, todo[0]["case_id"], round(dt_p, 3),
                                round(proj_h, 2), "too_slow", note])
                print("[N%d] %-11s TOO_SLOW %s" % (args.N, m, note), flush=True)
                continue
        t0 = time.time()
        recs = plan_batch(m, todo, workers=args.workers, batch_size=args.batch_size)
        _append_jsonl(jpath, recs)
        # per-method json (last write wins on resume)
        allrecs = _read_jsonl(jpath)
        _atomic_json(os.path.join(out, "methods", "%s_N%d.json" % (m, args.N)),
                     {"method": m, "N": args.N, "n_cases": len(allrecs),
                      "config_hash": json.dumps(PARAMS[m], sort_keys=True, default=str),
                      "code_version": "public-baselines-1.0.0",
                      "cases": allrecs})
        print("[N%d] %-11s done in %.2fs (mean %.3fs/case)"
              % (args.N, m, time.time() - t0,
                 (time.time() - t0) / max(len(recs), 1)), flush=True)
    manifest = {
        "run_id": "public_N%d_%s" % (args.N, time.strftime("%Y%m%d_%H%M%S")),
        "field_version": "public-field-1.0.0",
        "N": args.N, "n_cases": len(cases),
        "methods": args.methods,
        "workers": args.workers, "batch_size": args.batch_size,
        "device": "cpu", "python": platform.python_version(),
        "cpu_count": os.cpu_count(), "platform": platform.platform(),
        "wall_s": time.time() - t_start,
        "method_kind": {m: ("tensor_batch" if m in TENSOR_BATCH_METHODS else
                            "process_pool" if m in POOL_METHODS else "sequential")
                        for m in args.methods},
    }
    _atomic_json(os.path.join(out, "run_manifest.json"), manifest)
    summary = merge(out, args.N, args.methods)
    print(json.dumps({"N": args.N, "n_ok": summary["n_ok"],
                      "mean_J": summary["mean_J"]}, indent=2))


if __name__ == "__main__":
    main()
