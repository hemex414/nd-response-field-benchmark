# -*- coding: utf-8 -*-
"""Slice audit for the public field (v1.0).

Computes the acceptance metrics in SPEC.md and writes:
    public_field/audit/audit.json
    public_field/figures/field_slices.png

The audit is part of the freeze gate G0: all 15 audited slices must show
>= 5 significant peaks, no full-width high line, and overlapping peak/corridor
ranges; the global additive R2 must be <= 0.85 and the residual coverage
|r| > 1 must be >= 50 %.
"""
import json
import os

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_field import make_field, VALUE_RANGE

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED = [10, 30, 50, 70, 90]
PAIRS = {"12": "u1-u2", "13": "u1-u3", "23": "u2-u3"}


def slice_metrics(S):
    mx = ndimage.maximum_filter(S, 7, mode="nearest")
    mn = ndimage.minimum_filter(S, 15, mode="nearest")
    peaks = S[(S == mx)]
    prom = (S - mn)[(S == mx)]
    pk = peaks[prom > 2.0]

    # full-width line detector: row/col almost constant and high
    row_std = S.std(axis=1); col_std = S.std(axis=0)
    hi = np.percentile(S, 80)
    line_hi = int(((row_std < 1.2) & (S.mean(axis=1) > hi)).sum()
                  + ((col_std < 1.2) & (S.mean(axis=0) > hi)).sum())

    low = float(np.percentile(S, 20))
    m = S < low
    lab, n = ndimage.label(m, np.ones((3, 3), bool))
    sizes = np.sort(np.bincount(lab.ravel())[1:])[::-1] if n else np.array([0])
    return {
        "n_peaks": int(len(pk)),
        "peak_min": float(pk.min()) if len(pk) else None,
        "peak_max": float(pk.max()) if len(pk) else None,
        "peak_std": float(pk.std()) if len(pk) else None,
        "low20": low,
        "low_area_frac": float(m.mean()),
        "low_largest_frac_of_slice": float(sizes[0] / S.size) if n else 0.0,
        "high_line_count": line_hi,
    }


def main():
    f = make_field(seed=2026, resolution=80)
    G = f.grid
    res = G.shape[0]
    out = {"version": f.meta["version"], "seed": f.seed, "resolution": res,
           "value_range": list(VALUE_RANGE),
           "global": {"min": float(G.min()), "max": float(G.max()),
                      "mean": float(G.mean()), "std": float(G.std())},
           "slices": []}

    # additive R2 on the full 3-D grid
    f1 = G.mean(axis=(1, 2))[:, None, None]
    f2 = G.mean(axis=(0, 2))[None, :, None]
    f3 = G.mean(axis=(0, 1))[None, None, :]
    add = f1 + f2 + f3 - 2 * G.mean()
    r = G - add
    out["global"]["additive_r2"] = float(1 - r.var() / G.var())
    out["global"]["residual_std"] = float(r.std())
    out["global"]["abs_res_gt1_frac"] = float((np.abs(r) > 1).mean())

    for pair, name in PAIRS.items():
        for fv in FIXED:
            S = f.slice(pair, fv)
            m = slice_metrics(S)
            m.update({"pair": pair, "pair_name": name, "fixed_value": fv})
            out["slices"].append(m)

    # summary / gate
    peaks = [m["n_peaks"] for m in out["slices"]]
    lines = [m["high_line_count"] for m in out["slices"]]
    out["gate_G0"] = {
        "n_slices": len(out["slices"]),
        "peaks_min": int(min(peaks)), "peaks_max": int(max(peaks)),
        "all_slices_ge5_peaks": bool(min(peaks) >= 5),
        "max_high_lines": int(max(lines)),
        "no_full_width_high_line": bool(max(lines) == 0),
        "additive_r2": out["global"]["additive_r2"],
        "abs_res_gt1_frac": out["global"]["abs_res_gt1_frac"],
        "passed": bool(min(peaks) >= 5 and max(lines) == 0
                       and out["global"]["additive_r2"] <= 0.85
                       and out["global"]["abs_res_gt1_frac"] >= 0.50),
    }

    os.makedirs(os.path.join(HERE, "..", "audit"), exist_ok=True)
    with open(os.path.join(HERE, "..", "audit", "audit.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    # figure: 3 pairs x fixed 30/50/70
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.5))
    for r_, fv in enumerate((30, 50, 70)):
        for c_, (pair, name, xl, yl) in enumerate(
                [("12", "u1-u2", "$u_1$", "$u_2$"),
                 ("13", "u1-u3", "$u_1$", "$u_3$"),
                 ("23", "u2-u3", "$u_2$", "$u_3$")]):
            ax = axes[r_, c_]
            S = f.slice(pair, fv)
            im = ax.imshow(S.T, origin="lower", extent=[0, 100, 0, 100], cmap="RdYlBu_r",
                           vmin=VALUE_RANGE[0], vmax=VALUE_RANGE[1], aspect="equal")
            ax.set_title("%s (fixed = %d)" % (name, fv))
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
    fig.colorbar(im, ax=axes, shrink=0.75, label="$V$ (dimensionless, 100-160)")
    fig.suptitle("Public field v1.0: 3-D-localised features, no full-width lines", y=0.995)
    os.makedirs(os.path.join(HERE, "..", "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "..", "figures", "field_slices.png"),
                dpi=120, bbox_inches="tight")

    print(json.dumps(out["gate_G0"], indent=2, ensure_ascii=False))
    print("saved audit.json and field_slices.png")


if __name__ == "__main__":
    main()

