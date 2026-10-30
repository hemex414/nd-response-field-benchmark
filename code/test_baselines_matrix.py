#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_baselines_matrix.py -- T1..T6 acceptance matrix for the public baselines.

Per the the project's instruction the matrix is first run on **N=2/3 x 16 cases** (not 1000).

T1 single-case smoke        every algorithm, N=2 and N=3, case 0: runs, feasible, complete fields
T2 batch == single          GA/GD/SA/CHOMP tensor batch (B=16) and the 6 CPU planners with a
                            12-worker process pool vs. the single-case runs: |dJ| <= 1e-4,
                            T/endpoints/rate/feasibility identical
T3 small batch (N=3)        same comparison at N=3, 16 cases
T4 resources                wall time single vs batch, workers, peak RSS, CPU count
T5 1000 cases               NOT RUN here (explicit instruction: do not launch 1000 yet)
T6 reproducibility          rerun the batch and compare J (<= 1e-6)

Writes public_field/results/TEST_MATRIX.md and test_matrix.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from code import field as F                                  # noqa: E402
from code.baselines import (plan_single, plan_batch, PARAMS,
                                         METHODS, POOL_METHODS,
                                         TENSOR_BATCH_METHODS, BATCH_METHODS)

OUT = os.path.join(PROJ, "results")
TOR = 1e-4


def _rss_mb():
    """Current process RSS in MB (psutil on Windows/Linux; 0 if unavailable)."""
    if psutil is None:
        return 0.0
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:                                  # noqa: BLE001
        return 0.0


def _run_singles(method, cases):
    t0 = time.perf_counter()
    recs = [plan_single(method, c) for c in cases]
    return recs, time.perf_counter() - t0


def _run_batch(method, cases, workers, batch_size):
    t0 = time.perf_counter()
    recs = plan_batch(method, cases, workers=workers, batch_size=batch_size)
    return recs, time.perf_counter() - t0


def compare(method, singles, batch, tol=TOR):
    dJ, dT, same_feas, bad = 0.0, 0.0, True, []
    for s, b in zip(singles, batch):
        if s.get("status") != b.get("status"):
            bad.append((s.get("case_id"), "status", s.get("status"), b.get("status")))
            continue
        if s.get("status") == "ok":
            dJ = max(dJ, abs(float(s["J"]) - float(b["J"])))
            dT = max(dT, abs(float(s["T"]) - float(b["T"])))
            same_feas &= (bool(s.get("feasible")) == bool(b.get("feasible")))
        else:
            bad.append((s.get("case_id"), "non-ok", s.get("status"), b.get("status")))
    ok = (dJ <= tol) and (dT <= 1e-6) and same_feas and not bad
    return ok, {"max_abs_dJ": dJ, "max_abs_dT": dT, "same_feasible": same_feas,
                "mismatches": bad[:5]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", default="2,3")
    ap.add_argument("--cases", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]
    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "ns": ns,
              "n_cases": args.cases, "workers": args.workers,
              "batch_size": args.batch_size, "tests": {}, "resources": {}}
    os.makedirs(OUT, exist_ok=True)

    # ---------------- T1: single-case smoke ----------------
    t1 = {"rows": [], "passed": True}
    for n in ns:
        cases = F.load_cases(n)
        for m in args.methods:
            try:
                r, dt = _run_singles(m, [cases[0]])
                rec = r[0]
                ok = (rec.get("status") == "ok" and rec.get("feasible")
                      and rec.get("points") is not None and rec.get("durations") is not None
                      and np.isfinite(rec.get("J", float("nan"))))
                t1["rows"].append({"N": n, "method": m, "ok": bool(ok),
                                   "J": rec.get("J"), "feasible": rec.get("feasible"),
                                   "runtime_s": dt, "status": rec.get("status"),
                                   "error": None})
                t1["passed"] &= bool(ok)
            except Exception as exc:                       # noqa: BLE001
                t1["rows"].append({"N": n, "method": m, "ok": False, "error": repr(exc)})
                t1["passed"] = False
    report["tests"]["T1_single_smoke"] = t1
    print("T1 single smoke:", "PASS" if t1["passed"] else "FAIL", flush=True)

    # ---------------- T2/T3: batch/parallel == single ----------------
    for tag, n in (("T2_batch_vs_single_N2", ns[0]), ("T3_batch_vs_single_N3", ns[-1])):
        cases = F.load_cases(n)[:args.cases]
        rows = []
        passed = True
        for m in args.methods:
            try:
                singles, t_single = _run_singles(m, cases)
                batch, t_batch = _run_batch(m, cases, args.workers, args.batch_size)
                ok, stats = compare(m, singles, batch)
                kind = "tensor_batch" if m in TENSOR_BATCH_METHODS else (
                    "process_pool" if m in POOL_METHODS else "sequential")
                rows.append({"method": m, "kind": kind, "ok": bool(ok), **stats,
                             "t_single_s": t_single, "t_batch_s": t_batch})
                passed &= bool(ok)
            except Exception as exc:                       # noqa: BLE001
                rows.append({"method": m, "ok": False, "error": repr(exc),
                             "trace": traceback.format_exc(limit=2)})
                passed = False
        report["tests"][tag] = {"N": n, "passed": passed, "rows": rows}
        print(tag, "PASS" if passed else "FAIL", flush=True)

    # ---------------- T4: resources ----------------
    t4 = {"workers": args.workers, "batch_size": args.batch_size,
          "cpu_count": os.cpu_count(), "rows": []}
    for m in args.methods:
        n2 = ns[0]
        cases = F.load_cases(n2)[:args.cases]
        rss0 = _rss_mb()
        _, t1 = _run_singles(m, cases)
        _, t2 = _run_batch(m, cases, args.workers, args.batch_size)
        t4["rows"].append({"method": m, "T_single16": t1, "T_batch16": t2,
                           "speedup": (t1 / t2) if t2 > 0 else None,
                           "rss_mb": _rss_mb() - rss0})
    t4["peak_rss_mb"] = _rss_mb()
    report["resources"] = t4
    print("T4 resources: peak RSS %.0f MB, %d CPUs" % (t4["peak_rss_mb"], os.cpu_count()),
          flush=True)

    # ---------------- T5: 1000 cases (not run) ----------------
    report["tests"]["T5_1000_cases"] = {
        "passed": None, "status": "not_run",
        "note": "per instruction: run N=2/3 x16 acceptance first; 1000-case sweep follows "
                "only after this matrix is green"}

    # ---------------- T6: reproducibility ----------------
    cases = F.load_cases(ns[0])[:args.cases]
    t6 = {"passed": True, "rows": []}
    for m in args.methods:
        a, _ = _run_batch(m, cases, args.workers, args.batch_size)
        b, _ = _run_batch(m, cases, args.workers, args.batch_size)
        d = max(abs(float(x["J"]) - float(y["J"])) for x, y in zip(a, b)
                if x.get("status") == "ok" and y.get("status") == "ok")
        t6["rows"].append({"method": m, "max_abs_dJ_rerun": d, "ok": d <= 1e-6})
        t6["passed"] &= d <= 1e-6
    report["tests"]["T6_reproducibility"] = t6
    print("T6 reproducibility:", "PASS" if t6["passed"] else "FAIL", flush=True)

    report["all_passed"] = all(v.get("passed") for v in report["tests"].values()
                               if isinstance(v, dict) and v.get("passed") is not None)
    with open(os.path.join(OUT, "test_matrix.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=float)
    md = ["# Public baselines test matrix (T1-T6)", "",
          "generated: %s | N=%s | %d cases | workers=%d | batch=%d"
          % (report["generated_at"], ns, args.cases, args.workers, args.batch_size), ""]
    for name, t in report["tests"].items():
        if "rows" in t and name.startswith(("T1", "T2", "T3")):
            md += ["## %s -- %s" % (name, "PASS" if t.get("passed") else "FAIL"), "",
                   "| method | kind | max|dJ| | max|dT| | feasible same | ok |",
                   "|---|---|---|---|---|---|"]
            for r in t["rows"]:
                md.append("| %s | %s | %.3e | %.3e | %s | %s |" % (
                    r.get("method"), r.get("kind", "-"),
                    r.get("max_abs_dJ", float("nan")), r.get("max_abs_dT", float("nan")),
                    r.get("same_feasible", "-"), "PASS" if r.get("ok") else "FAIL"))
            md.append("")
        else:
            md.append("## %s: %s" % (name, t.get("passed") if "passed" in t else t.get("status")))
    md += ["## resources", "", "| method | T single 16 [s] | T batch 16 [s] | speedup |",
           "|---|---|---|---|"]
    for r in report["resources"]["rows"]:
        md.append("| %s | %.2f | %.2f | %.2fx |" % (r["method"], r["T_single16"],
                                                    r["T_batch16"], r["speedup"] or 0))
    md += ["", "peak RSS: %.0f MB, CPUs: %s" % (report["resources"]["peak_rss_mb"],
                                                os.cpu_count())]
    with open(os.path.join(OUT, "TEST_MATRIX.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("wrote", os.path.join(OUT, "TEST_MATRIX.md"), flush=True)
    print("ALL_PASSED =", report["all_passed"], flush=True)


if __name__ == "__main__":
    main()
