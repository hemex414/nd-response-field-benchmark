#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""merge_e5_cotra.py -- merge E5 baselines with the public CoTra-J batch results.

Reads
  public_field/results/E5_batch/N<n>/methods/<method>_N<n>.jsonl   (baselines)
  public_field/results/E5_batch/N<n>/too_slow_N<n>.csv             (eliminated cells)
  public_field/results/E4_batch/N<n>/cotra_j_batch.jsonl           (public CoTra-J)
Writes
  public_field/results/E5_batch/N<n>/merged_summary.json
  public_field/results/FINAL_BY_N.md
and prints the per-N table (n, n_grid_too_large, mean J, feasible, paired vs cotra_j).

Cells eliminated as `too_slow` appear explicitly (status column) so a final per-N table
is complete even when a method did not run to 1000 cases.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
E5 = os.path.join(RES, "E5_batch")
E4 = os.path.join(RES, "E4_batch")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def method_records(d, n):
    """Read the resumable per-method jsonl files (fallback: per-method json)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "methods", "*_N%d.jsonl" % n))):
        m = os.path.basename(path)[:-len("_N%d.jsonl" % n)]
        out[m] = load_jsonl(path)
    for path in sorted(glob.glob(os.path.join(d, "methods", "*_N%d.json" % n))):
        m = os.path.basename(path)[:-len("_N%d.json" % n)]
        if m in out:
            continue
        try:
            out[m] = json.load(open(path, encoding="utf-8")).get("cases", [])
        except Exception:                            # noqa: BLE001
            pass
    return out


def stats(records):
    ok = [r for r in records if r.get("status", "ok") == "ok" and r.get("J") is not None]
    js = [float(r["J"]) for r in ok]
    rts = [float(r["runtime_s"]) for r in ok if r.get("runtime_s") is not None]
    n_grid = sum(1 for r in records
                 if r.get("status") == "timeout" and r.get("grid_too_large"))
    return {
        "n": len(ok),
        "n_records": len(records),
        "n_grid_too_large": n_grid,
        "mean_J": statistics.fmean(js) if js else None,
        "median_J": statistics.median(js) if js else None,
        "feasible_rate": (sum(1 for r in ok if r.get("feasible")) / len(ok)) if ok else None,
        "mean_runtime_s": statistics.fmean(rts) if rts else None,
        "status": "ok" if ok and not n_grid else
                  ("grid_too_large" if n_grid and not ok else
                   ("partial" if ok else "no_data")),
    }


def main():
    rows = []
    report = {}
    for n in range(1, 9):
        d5 = os.path.join(E5, "N%d" % n)
        if not os.path.isdir(d5):
            continue
        per_method = {}
        recs = method_records(d5, n)
        for m, rs in recs.items():
            per_method[m] = stats(rs)
            rows.append({"N": n, "method": m, "kind": "baseline", **per_method[m]})
        # too_slow cells (no partial J by protocol)
        for r in load_csv(os.path.join(d5, "too_slow_N%d.csv" % n)):
            m = r.get("method", "?")
            per_method[m] = {"n": 0, "n_records": 0, "n_grid_too_large": 0,
                             "mean_J": None, "median_J": None,
                             "feasible_rate": None, "mean_runtime_s": None,
                             "status": "too_slow",
                             "reason": r.get("reason", "")}
            rows.append({"N": n, "method": m, "kind": "baseline", **per_method[m]})
        # public CoTra-J
        cotra = load_jsonl(os.path.join(E4, "N%d" % n, "cotra_j_batch.jsonl"))
        if cotra:
            per_method["cotra_j"] = stats(cotra)
            per_method["cotra_j"]["status"] = "ok"
            rows.append({"N": n, "method": "cotra_j", "kind": "cotra_j",
                         **per_method["cotra_j"]})
            cotra_j = {r["case_id"]: float(r["J"]) for r in cotra
                       if r.get("status", "ok") == "ok" and r.get("J") is not None}
            for m, rs in recs.items():
                base = {r["case_id"]: float(r["J"]) for r in rs
                        if r.get("status", "ok") == "ok" and r.get("J") is not None}
                common = sorted(set(base) & set(cotra_j))
                if not common:
                    continue
                ds = [base[c] - cotra_j[c] for c in common]
                per_method[m]["n_paired_with_cotra"] = len(common)
                per_method[m]["mean_dJ_vs_cotra"] = statistics.fmean(ds)
        report["N%d" % n] = per_method
        with open(os.path.join(d5, "merged_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(per_method, fh, indent=2, ensure_ascii=False, default=float)
    with open(os.path.join(RES, "merged_rows.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=float)

    md = ["# Public-field batch results by N (baselines E5 + public CoTra-J E4)", "",
          "`too_slow` = projected 1000-case wall-clock > 6 h (no partial J); "
          "`grid_too_large` = N>=4 100^N grid above the memory cap.", ""]
    md.append("| N | method | kind | status | n_ok | n_grid_too_large | mean J | median J | "
              "feasible | mean rt [s] | n_paired_cotra | mean dJ vs cotra |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x["N"], -1 if x["mean_J"] is None else 0,
                                          -(x["mean_J"] or 0))):
        mark = report.get("N%d" % r["N"], {}).get(r["method"], {})
        md.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["N"], r["method"], r["kind"], r.get("status", "-"), r["n"],
            r.get("n_grid_too_large", 0),
            "%.3f" % r["mean_J"] if r["mean_J"] is not None else "-",
            "%.3f" % r["median_J"] if r["median_J"] is not None else "-",
            "%.3f" % r["feasible_rate"] if r["feasible_rate"] is not None else "-",
            "%.3f" % r["mean_runtime_s"] if r["mean_runtime_s"] is not None else "-",
            mark.get("n_paired_with_cotra", "-"),
            "%.3f" % mark["mean_dJ_vs_cotra"]
            if mark.get("mean_dJ_vs_cotra") is not None else "-"))
    with open(os.path.join(RES, "FINAL_BY_N.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("wrote", os.path.join(RES, "FINAL_BY_N.md"))
    print("\n".join(md[-16:]))


if __name__ == "__main__":
    main()
