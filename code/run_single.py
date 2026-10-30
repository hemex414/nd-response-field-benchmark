#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_single.py -- single-case baseline planner for the public field.

Example
-------
python code/run_single.py --method fmm --N 3 --subset 0,3,5 \
    --A 10,80,20 --B 90,15,70 --TR 2.5 --rates 4,11,7 \
    --field-seed 202600017 --out results/single_fmm_N3.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from code.baselines import plan_single, PARAMS, METHODS  # noqa: E402


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--subset", default="")
    ap.add_argument("--A", required=True)
    ap.add_argument("--B", required=True)
    ap.add_argument("--TR", type=float, default=2.5)
    ap.add_argument("--rates", required=True)
    ap.add_argument("--field-seed", type=int, required=True)
    ap.add_argument("--field-version", default="public-field-1.0.0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A = _floats(args.A)
    B = _floats(args.B)
    rates = _floats(args.rates)
    assert len(A) == len(B) == len(rates) == args.N, "A/B/rates must have length N"
    subset = _ints(args.subset) if args.subset else list(range(args.N))
    case = {
        "case_id": 0, "N": args.N, "subset": subset,
        "field_seed": args.field_seed, "field_version": args.field_version,
        "A": A, "B": B, "TR": args.TR, "rates": rates,
    }
    rec = plan_single(args.method, case)
    rec["params"] = PARAMS[args.method]
    rec["config_hash"] = hashlib.sha256(
        json.dumps(PARAMS[args.method], sort_keys=True, default=str).encode()).hexdigest()[:16]
    rec["code_version"] = "public-baselines-1.0.0"
    out = args.out or os.path.join(
        PROJ, "results", "single_%s_N%d.json" % (args.method, args.N))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, ensure_ascii=False, default=float)
    print("%s N=%d J=%.4f feasible=%s runtime=%.2fs -> %s"
          % (args.method, args.N, rec.get("J", float("nan")), rec.get("feasible"),
             rec.get("runtime_s", float("nan")), out))


if __name__ == "__main__":
    main()
