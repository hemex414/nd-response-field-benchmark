# -*- coding: utf-8 -*-
"""field.py -- self-contained public N-D response field (numpy).

The field family is the public N-D generalisation used by the public benchmark:
trend + 1-D Gaussian profiles + random anisotropic 3-D islands/pockets on
dimension triples + 3-D product ripples, normalised to V in [100, 160] on
``u_i in [0, 100]``.  It has **no** dependency on the external code; the
draw order of ``numpy.random.default_rng(seed)`` is part of the reproducibility
contract (case files store the seed).

Case metadata (important)
-------------------------
A case JSON stores ``N``, ``subset``, ``field_seed``, ``field_version``,
``A``, ``B``, ``rates`` and ``TR``.  The virtual field is a deterministic
function of ``(N, field_seed)`` only.  ``subset`` does **not** participate in
field generation: it records which indices of the original 8-state-dimension
candidate set this N-D case uses, so that ``A``, ``B`` and ``rates`` can be
mapped back to the corresponding state dimensions.  Two cases with the same
``(N, field_seed)`` therefore share the same field even if their ``subset``
labels differ; use different ``field_seed`` values when independent fields are
required.

Classes
-------
FieldND      one N-D field instance (``.eval(points)`` / ``.interp(points)``)
FieldBank    a stack of same-N fields, evaluated as a tensor batch
load_cases   read a frozen case jsonl / json case file
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

DOMAIN = 100.0
VMIN, VMAX = 100.0, 160.0


class FieldND:
    """One N-dimensional public field (analytic, dimensionless, u in [0,100]^N)."""

    def __init__(self, n, seed):
        self.N = int(n)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.w = rng.uniform(2.0, 6.0, self.N)
        self.oned = []          # per dim: 4 x (amp, mu, sigma)
        for _ in range(self.N):
            comps = [(rng.uniform(0.6, 1.5), rng.uniform(5, 95), rng.uniform(8, 18))
                     for _ in range(4)]
            self.oned.append(comps)
        self.islands, self.pockets = [], []          # (tri, c, s, amp)
        self._struct = "3d"
        if self.N >= 3:
            n_tri = min(12, max(4, self.N * 2))
            self.triples = [tuple(rng.choice(self.N, size=3, replace=False).tolist())
                            for _ in range(n_tri)]
            for tri in self.triples:
                for _ in range(6):
                    c = rng.uniform(8, 92, 3)
                    s = rng.uniform(4, 10, 3) * rng.uniform(0.6, 1.0, 3)
                    self.islands.append((tri, c, s, float(rng.uniform(2.0, 5.5))))
                for _ in range(2):
                    c = rng.uniform(8, 92, 3)
                    s = rng.uniform(5, 12, 3)
                    self.pockets.append((tri, c, s, -float(rng.uniform(1.0, 2.2))))
        else:
            self._struct = "nd"
            self.triples = []
            for _ in range(8 if self.N == 2 else 6):
                c = rng.uniform(8, 92, self.N)
                s = rng.uniform(6, 16, self.N)
                self.islands.append((tuple(range(self.N)), c, s,
                                     float(rng.uniform(2.0, 5.0))))
        self.ripples = []
        for _ in range(4):
            dims = (rng.choice(self.N, size=min(3, self.N), replace=False).tolist()
                    if self.N >= 2 else [0])
            a = float(rng.uniform(0.05, 0.15))
            lam = float(rng.uniform(25, 60))
            ph = rng.uniform(0, 6.3, len(dims))
            self.ripples.append((dims, a, lam, ph))
        # normalisation from one fixed sample draw (part of the draw-order contract)
        pts = rng.uniform(0.0, DOMAIN, size=(20000, self.N))
        raw = self._raw(pts)
        self.vmin, self.vmax = float(raw.min()), float(raw.max())
        self.rpm_max = np.full(self.N, DOMAIN)
        self.device = "cpu"

    # -- analytic evaluation ------------------------------------------------
    def _raw(self, u):
        u = np.asarray(u, dtype=np.float64).reshape(-1, self.N)
        V = (u * self.w[None, :]).sum(axis=1) / 100.0
        for i, comps in enumerate(self.oned):
            for amp, mu, sig in comps:
                d = (u[:, i] - mu) / sig
                V = V + amp * np.exp(-0.5 * d * d)
        for tri, c, s, amp in self.islands + self.pockets:
            tri = list(tri)
            d = np.zeros(len(u))
            for k, dm in enumerate(tri):
                d = d + ((u[:, dm] - c[k]) / s[k]) ** 2
            V = V + amp * np.exp(-0.5 * d)
        for dims, a, lam, ph in self.ripples:
            term = np.ones(len(u))
            for k, dm in enumerate(dims):
                term = term * np.sin(2.0 * math.pi * u[:, dm] / lam + ph[k])
            V = V + a * term
        return V

    def interp(self, x):
        """Field values at points x (..., N); clamps to the box and to [100,160]."""
        x = np.asarray(x, dtype=np.float64)
        shape = x.shape[:-1]
        u = x.reshape(-1, self.N)
        raw = self._raw(np.clip(u, 0.0, DOMAIN))
        V = VMIN + (VMAX - VMIN) * (raw - self.vmin) / (self.vmax - self.vmin)
        return np.clip(V, VMIN, VMAX).reshape(shape)

    # alias used by some planner code paths
    eval = interp


class FieldBank:
    """Tensor batch of same-N fields (one per case); evaluates (B, M, N) points."""

    def __init__(self, fields):
        self.fields = list(fields)
        self.B = len(self.fields)
        assert self.B > 0
        self.N = self.fields[0].N
        assert all(f.N == self.N for f in self.fields)
        self.vmin = np.array([f.vmin for f in self.fields], float)
        self.vmax = np.array([f.vmax for f in self.fields], float)
        # oned: (B,N,4)
        self.oned_amp = np.stack([[[c[0] for c in comps] for comps in f.oned]
                                  for f in self.fields])
        self.oned_mu = np.stack([[[c[1] for c in comps] for comps in f.oned]
                                 for f in self.fields])
        self.oned_sig = np.stack([[[c[2] for c in comps] for comps in f.oned]
                                  for f in self.fields])
        # islands+pockets: counts are identical for identical N
        feats = [[f.islands + f.pockets for f in self.fields]][0]
        assert all(len(x) == len(feats[0]) for x in feats), "same-N field structure mismatch"
        self.tri = np.stack([[np.asarray(t[0], int) for t in x] for x in feats])   # (B,F,3)
        self.c = np.stack([[np.asarray(t[1], float) for t in x] for x in feats])   # (B,F,3)
        self.sig = np.stack([[np.asarray(t[2], float) for t in x] for x in feats])
        self.amp = np.stack([[float(t[3]) for t in x] for x in feats])             # (B,F)
        # ripples: same number, dims count identical for same N
        self.rip = [[(np.asarray(dims, int), float(a), float(l), np.asarray(ph, float))
                     for dims, a, l, ph in f.ripples] for f in self.fields]

    def eval(self, points):
        """points (B, M, N) -> (B, M), with the same accumulation order as FieldND._raw
        (so a batched evaluation is bit-identical to the single-case one)."""
        pts = np.asarray(points, float)
        B, M, N = pts.shape
        assert B == self.B and N == self.N
        w = np.stack([f.w for f in self.fields])                        # (B,N)
        V = (pts * w[:, None, :]).sum(-1) / 100.0                        # (B,M)
        for i in range(N):
            for c in range(self.oned_amp.shape[2]):
                d = (pts[:, :, i] - self.oned_mu[:, i, c][:, None]) / \
                    self.oned_sig[:, i, c][:, None]
                V = V + self.oned_amp[:, i, c][:, None] * np.exp(-0.5 * d * d)
        F = self.tri.shape[1]
        for f in range(F):
            tri = self.tri[:, f, :]                                     # (B,3)
            G = np.take_along_axis(pts, tri[:, None, :], axis=-1)        # (B,M,3)
            dd = (G - self.c[:, f, None, :]) / self.sig[:, f, None, :]
            V = V + self.amp[:, f][:, None] * np.exp(-0.5 * (dd * dd).sum(-1))
        nrip = len(self.rip[0])
        for k in range(nrip):
            a_all = np.array([r[k][1] for r in self.rip])
            l_all = np.array([r[k][2] for r in self.rip])
            ph_all = [r[k][3] for r in self.rip]
            dims_k = [r[k][0] for r in self.rip]
            tern = np.ones((B, M))
            for q in range(max(len(dd) for dd in dims_k)):
                vals = np.ones((B, M))
                for b in range(B):
                    if q < len(dims_k[b]):
                        vals[b] = np.sin(2.0 * math.pi * pts[b, :, dims_k[b][q]] /
                                         l_all[b] + ph_all[b][q])
                tern = tern * vals
            V = V + a_all[:, None] * tern
        V = VMIN + (VMAX - VMIN) * (V - self.vmin[:, None]) / (self.vmax[:, None] - self.vmin[:, None])
        return np.clip(V, VMIN, VMAX)

    def eval_single(self, points, case_idx):
        return self.fields[case_idx].interp(points)


def make_field(n, seed):
    return FieldND(n, seed)


def load_cases(n, path=None):
    """Load frozen cases from a public case file.

    Each case contains ``case_id``, ``N``, ``subset``, ``field_seed``,
    ``field_version``, ``A``, ``B``, ``rates`` and ``TR``.  ``subset`` is
    metadata for mapping the N coordinates back to the 8-state-dimension
    candidate set; the field itself is generated by ``FieldND(N, field_seed)``.
    """
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(os.path.dirname(here), "cases", "N%d" % n, "cases.jsonl")
    with open(path, encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in fh if line.strip()]
        return json.load(fh)["cases"]
