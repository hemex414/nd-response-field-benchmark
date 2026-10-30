# -*- coding: utf-8 -*-
"""Public dimensionless response-field generator (v1.0, canonical 3-D instance).

Construction: trend + per-axis 1-D profiles + 3-D anisotropic two-lobe islands
+ 3-D negative pockets + 3-D winding corridor tubes (negative Gaussian chains)
+ small 3-D product ripple.  Everything is localised in all three dimensions, so
2-D slices show finite islands / bands / pockets and never full-width lines.

Domain:   u_i in [0, 100], i = 1..3
Values:   V in [100, 160]  (dimensionless; positive affine map of a [0,1] field)
Seed:     2026 (canonical); any integer reproduces a different field.
Version:  1.0.0

Usage:
    python make_field.py --out ../data/canonical_3d_seed2026.npz --resolution 80
    from make_field import make_field
    f = make_field(seed=2026, resolution=80)
    V = f(points)          # points: (M,3) in [0,100] -> (M,) in [100,160]
    ax, G = f.grid         # 80^3 array
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

VERSION = "1.0.0"
CANONICAL_SEED = 2026
RESOLUTION = 80
VALUE_RANGE = (100.0, 160.0)
TREND = (6.0, 3.0, 5.0)


# --------------------------------------------------------------------------
# parameter sampling (order of RNG draws is part of the reproducibility contract)
# --------------------------------------------------------------------------
def build_params(seed: int = CANONICAL_SEED) -> dict:
    rng = np.random.default_rng(seed)
    p = {"seed": int(seed), "version": VERSION, "trend": list(TREND),
         "oned": [], "islands": [], "pockets": [], "corridors": [], "ripples": []}

    # per-axis 1-D profiles: 4 Gaussian bumps per axis
    for _ in range(3):
        comps = []
        for _ in range(4):
            amp = float(rng.uniform(0.5, 1.6))
            mu = float(rng.uniform(5.0, 95.0))
            sig = float(rng.uniform(8.0, 18.0))
            comps.append((amp, mu, sig))
        p["oned"].append(comps)

    # 3-D islands: 48 anisotropic two-lobe Gaussians, different heights
    for _ in range(48):
        c = rng.uniform(8.0, 92.0, 3)
        s = rng.uniform(4.5, 11.0, 3) * rng.uniform(0.55, 1.0, 3)
        A = float(rng.uniform(2.0, 6.0))
        c2 = c + rng.uniform(-1.2, 1.2, 3) * s
        p["islands"].append({"c": c.tolist(), "s": s.tolist(), "A": A, "c2": c2.tolist()})

    # 3-D negative pockets
    for _ in range(14):
        c = rng.uniform(8.0, 92.0, 3)
        s = rng.uniform(5.0, 12.0, 3)
        A = -float(rng.uniform(1.0, 2.5))
        p["pockets"].append({"c": c.tolist(), "s": s.tolist(), "A": A})

    # 3-D winding corridors: 5 negative Gaussian chains along 3-D sine curves
    for _ in range(5):
        x0 = float(rng.uniform(30.0, 70.0)); y0 = float(rng.uniform(30.0, 70.0)); z0 = float(rng.uniform(30.0, 70.0))
        ax_ = float(rng.uniform(12.0, 28.0)); ay_ = float(rng.uniform(12.0, 28.0)); az_ = float(rng.uniform(12.0, 28.0))
        lx = float(rng.uniform(40.0, 80.0)); ly = float(rng.uniform(40.0, 80.0)); lz = float(rng.uniform(40.0, 80.0))
        px = float(rng.uniform(0.0, 6.3)); py = float(rng.uniform(0.0, 6.3)); pz = float(rng.uniform(0.0, 6.3))
        B = float(rng.uniform(2.0, 3.8))
        p["corridors"].append({"x0": x0, "y0": y0, "z0": z0, "ax": ax_, "ay": ay_, "az": az_,
                               "lx": lx, "ly": ly, "lz": lz, "px": px, "py": py, "pz": pz, "B": B})

    # 3-D product ripple
    for _ in range(4):
        a = float(rng.uniform(0.05, 0.12))
        l = float(rng.uniform(25.0, 60.0))
        p["ripples"].append({"a": a, "l": l,
                             "p1": float(rng.uniform(0.0, 6.3)),
                             "p2": float(rng.uniform(0.0, 6.3)),
                             "p3": float(rng.uniform(0.0, 6.3))})
    return p


# --------------------------------------------------------------------------
# field evaluation
# --------------------------------------------------------------------------
def _raw(V, x, y, z, p):
    V += x * TREND[0] / 100.0 + y * TREND[1] / 100.0 + z * TREND[2] / 100.0
    for i, comps in enumerate(p["oned"]):
        u = (x, y, z)[i]
        for amp, mu, sig in comps:
            V += amp * np.exp(-0.5 * ((u - mu) / sig) ** 2)
    for isl in p["islands"]:
        c, s, A, c2 = isl["c"], isl["s"], isl["A"], isl["c2"]
        V += A * np.exp(-0.5 * (((x - c[0]) / s[0]) ** 2 + ((y - c[1]) / s[1]) ** 2 + ((z - c[2]) / s[2]) ** 2))
        s2 = [0.55 * v for v in s]
        V += 0.45 * A * np.exp(-0.5 * (((x - c2[0]) / s2[0]) ** 2 + ((y - c2[1]) / s2[1]) ** 2 + ((z - c2[2]) / s2[2]) ** 2))
    for pk in p["pockets"]:
        c, s, A = pk["c"], pk["s"], pk["A"]
        V += A * np.exp(-0.5 * (((x - c[0]) / s[0]) ** 2 + ((y - c[1]) / s[1]) ** 2 + ((z - c[2]) / s[2]) ** 2))
    return V


def build_raw_grid(p: dict, resolution: int = RESOLUTION):
    ax = np.linspace(0.0, 100.0, resolution)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    V = np.zeros_like(X)
    V = _raw(V, X, Y, Z, p)
    # corridors: negative Gaussian chains along 3-D sine curves
    for c in p["corridors"]:
        for t in np.linspace(0.0, 1.0, 30):
            cx = c["x0"] + c["ax"] * np.sin(2 * np.pi * 100.0 * t / c["lx"] + c["px"])
            cy = c["y0"] + c["ay"] * np.sin(2 * np.pi * 100.0 * t / c["ly"] + c["py"])
            cz = c["z0"] + c["az"] * np.sin(2 * np.pi * 100.0 * t / c["lz"] + c["pz"])
            V += -c["B"] * np.exp(-0.5 * (((X - cx) / 2.8) ** 2 + ((Y - cy) / 2.8) ** 2 + ((Z - cz) / 2.8) ** 2))
    for r in p["ripples"]:
        V += r["a"] * np.sin(2 * np.pi * X / r["l"] + r["p1"]) * \
             np.sin(2 * np.pi * Y / (r["l"] * 1.3) + r["p2"]) * \
             np.sin(2 * np.pi * Z / (r["l"] * 0.8) + r["p3"])
    return ax, V


class Field:
    """Canonical 3-D public field: callable at arbitrary points + stored grid."""

    def __init__(self, seed: int = CANONICAL_SEED, resolution: int = RESOLUTION):
        self.seed = int(seed)
        self.resolution = int(resolution)
        self.params = build_params(self.seed)
        ax, raw = build_raw_grid(self.params, self.resolution)
        self.vmin = float(raw.min()); self.vmax = float(raw.max())
        self.axes = ax
        self.grid = 100.0 + 60.0 * (raw - self.vmin) / (self.vmax - self.vmin)
        self.meta = {"version": VERSION, "seed": self.seed, "resolution": self.resolution,
                     "value_range": list(VALUE_RANGE), "trend": list(TREND),
                     "n_islands": len(self.params["islands"]),
                     "n_pockets": len(self.params["pockets"]),
                     "n_corridors": len(self.params["corridors"]),
                     "vmin_raw": self.vmin, "vmax_raw": self.vmax}

    def __call__(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        V = np.zeros(pts.shape[0])
        V = _raw(V, pts[:, 0], pts[:, 1], pts[:, 2], self.params)
        for c in self.params["corridors"]:
            for t in np.linspace(0.0, 1.0, 30):
                cx = c["x0"] + c["ax"] * np.sin(2 * np.pi * 100.0 * t / c["lx"] + c["px"])
                cy = c["y0"] + c["ay"] * np.sin(2 * np.pi * 100.0 * t / c["ly"] + c["py"])
                cz = c["z0"] + c["az"] * np.sin(2 * np.pi * 100.0 * t / c["lz"] + c["pz"])
                V += -c["B"] * np.exp(-0.5 * (((pts[:, 0] - cx) / 2.8) ** 2
                                             + ((pts[:, 1] - cy) / 2.8) ** 2
                                             + ((pts[:, 2] - cz) / 2.8) ** 2))
        for r in self.params["ripples"]:
            V += r["a"] * np.sin(2 * np.pi * pts[:, 0] / r["l"] + r["p1"]) * \
                 np.sin(2 * np.pi * pts[:, 1] / (r["l"] * 1.3) + r["p2"]) * \
                 np.sin(2 * np.pi * pts[:, 2] / (r["l"] * 0.8) + r["p3"])
        V = 100.0 + 60.0 * (V - self.vmin) / (self.vmax - self.vmin)
        return np.clip(V, *VALUE_RANGE)

    def slice(self, pair="12", fixed_value=50.0):
        """2-D slice of the canonical grid. pair in {'12','13','23'}."""
        k = int(np.argmin(np.abs(self.axes - fixed_value)))
        if pair == "12":
            return self.grid[:, :, k]
        if pair == "13":
            return self.grid[:, k, :]
        if pair == "23":
            return self.grid[k, :, :]
        raise ValueError(pair)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path,
                            axes=self.axes.astype(np.float32),
                            V=self.grid.astype(np.float32),
                            meta=json.dumps(self.meta, ensure_ascii=False))
        return path


def make_field(seed: int = CANONICAL_SEED, resolution: int = RESOLUTION) -> Field:
    return Field(seed=seed, resolution=resolution)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data",
                                                  "canonical_3d_seed2026.npz"))
    ap.add_argument("--resolution", type=int, default=RESOLUTION)
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED)
    args = ap.parse_args()
    f = make_field(seed=args.seed, resolution=args.resolution)
    path = f.save(args.out)
    print("saved", os.path.abspath(path))
    print("grid", f.grid.shape, "range %.2f..%.2f" % (f.grid.min(), f.grid.max()))


if __name__ == "__main__":
    main()
