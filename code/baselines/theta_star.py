# -*- coding: utf-8 -*-
"""theta_star -- single-case public implementation (algorithm and parameters from the
reference default, unchanged; only the field/dimension interface is public).
"""
import heapq
import math

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree

from .. import evaluate as B
from .common import *          # noqa: F401,F403  (helpers, PARAMS, constants)


def plan(field_nd, case):
    p = PARAMS["theta_star"]
    pf = PrivField(field_nd)
    n = int(field_nd.N)
    a = np.asarray(case["A"], float) / PUB
    b = np.asarray(case["B"], float) / PUB
    V, axes = make_grid(pf, n)
    res = GRID_RES; dx = axes[1] - axes[0]
    stride = p["GRID_STRIDE"]
    idx1d = list(range(0, res, stride))
    if idx1d[-1] != res - 1:
        idx1d.append(res - 1)
    shape = tuple(len(idx1d) for _ in range(n))
    coords = [axes[idx1d] for _ in range(n)]
    mesh = np.meshgrid(*coords, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], 1)
    mask = in_ellipsoid(pts, a, b, BETA).reshape(shape)
    vsub = grid_interp(V, axes, pts).reshape(shape)
    rate = np.asarray(case["rates"], float) / PUB

    def flat_idx(pt):
        ii = [int(np.argmin(np.abs(coords[d] - pt[d]))) for d in range(n)]
        return int(np.ravel_multi_index(tuple(ii), shape))

    def to_xyz(u):
        idx = np.unravel_index(u, shape)
        return np.array([coords[d][idx[d]] for d in range(n)])

    thresh = mean_field(pf, np.linspace(a, b, 200), rate)
    src = flat_idx(a); tgt = flat_idx(b)
    g = {src: 0.0}; parent = {}; closed = set(); heap = [(0.0, src)]; pops = 0
    while heap:
        _, u = heapq.heappop(heap)
        pops += 1
        if u == tgt:
            break
        if u in closed:
            continue
        closed.add(u)
        ui = np.unravel_index(u, shape)
        cu = to_xyz(u); pu = parent.get(u, u); c_pu = to_xyz(pu)
        for d in range(n):
            for sign in (1, -1):
                ni = list(ui); ni[d] += sign
                if ni[d] < 0 or ni[d] >= shape[d]:
                    continue
                ni = tuple(ni)
                if not mask[ni]:
                    continue
                v = int(np.ravel_multi_index(ni, shape))
                if v in closed:
                    continue
                cv = to_xyz(v)
                seg = np.linspace(c_pu, cv, p["LOS_SAMPLES"])
                if bool(np.all(pf.interp(seg) < thresh)):
                    mid = 0.5 * (c_pu + cv)
                    cand = g.get(pu, 0.0) + float(pf.interp(mid[None])[0]) * float(
                        np.linalg.norm(cv - c_pu))
                    np_ = pu
                else:
                    cand = g[u] + float(vsub[ni]) * float(np.linalg.norm(cv - cu))
                    np_ = u
                if cand < g.get(v, np.inf):
                    g[v] = cand; parent[v] = np_
                    heapq.heappush(heap, (cand, v))
    if tgt not in parent and tgt != src:
        path = np.vstack([a, b])
    else:
        seq = []; u = tgt
        while True:
            seq.append(u)
            if u == src:
                break
            u = parent.get(u, src)
        seq = seq[::-1]
        path = np.array([to_xyz(u) for u in seq])
        path[0] = a; path[-1] = b
    return path_to_record("theta_star", field_nd, case, path,
                          extra={"grid_nodes": int(np.prod(shape)), "pops": pops})
