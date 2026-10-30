# -*- coding: utf-8 -*-
"""astar -- single-case public implementation (algorithm and parameters from the
reference default, unchanged; only the field/dimension interface is public).
"""
import heapq
import math

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree

from .. import evaluate as B
from .common import *          # noqa: F401,F403  (helpers, PARAMS, constants)


class _GridProxy:
    def __init__(self, axes, res, n):
        self.axes = axes; self.res = res; self.N = int(n)
        self.dx = axes[1] - axes[0]

    def coord(self, idx):
        return np.asarray(idx, float) * self.dx


class _CaseProxy:
    def __init__(self, a, b):
        self.a = a; self.b = b


def plan(field_nd, case):
    p = PARAMS["astar"]
    pf = PrivField(field_nd)
    n = int(field_nd.N)
    a = np.asarray(case["A"], float) / PUB
    b = np.asarray(case["B"], float) / PUB
    V, axes = make_grid(pf, n)
    res = GRID_RES; dx = axes[1] - axes[0]
    shape = (res,) * n
    strides = B._grid_strides(shape)
    mesh = np.stack(np.meshgrid(*([axes] * n), indexing="ij"), -1).reshape(-1, n)
    mask = in_ellipsoid(mesh, a, b, BETA).reshape(-1)
    src = int(np.ravel_multi_index(tuple(np.clip(np.rint(a / dx).astype(int), 0, res - 1)), shape))
    tgt = int(np.ravel_multi_index(tuple(np.clip(np.rint(b / dx).astype(int), 0, res - 1)), shape))
    nn = int(V.size)
    ar = np.arange(nn, dtype=np.int64)
    pos = [(ar // strides[d]) % res for d in range(n)]
    vflat = V.reshape(-1)
    si = tuple(int(x) for x in np.rint(a / dx).astype(int))

    def h(u):                     # distance formula: normalized distance to START
        s = 0.0
        for d in range(n):
            s += ((axes[pos[d][u]] - axes[si[d]]) / WORKSPACE) ** 2
        return math.sqrt(s)

    g = np.full(nn, np.inf); parent = np.full(nn, -1, np.int64)
    closed = np.zeros(nn, bool); g[src] = 0.0
    heap = [(h(src), src)]
    while heap:
        _, u = heapq.heappop(heap)
        if u == tgt:
            break
        if closed[u]:
            continue
        closed[u] = True
        cu = g[u]
        for d in range(n):
            s = strides[d]; pd = pos[d][u]
            for sign in (1, -1):
                if (sign > 0 and pd >= res - 1) or (sign < 0 and pd <= 0):
                    continue
                v = u + sign * s
                if not mask[v] or closed[v]:
                    continue
                ng = cu + float(vflat[v]) * dx
                if ng < g[v]:
                    g[v] = ng; parent[v] = u
                    heapq.heappush(heap, (ng + h(v), v))
    path = B._reconstruct(parent, src, tgt, _GridProxy(axes, res, n), _CaseProxy(a, b))
    return path_to_record("astar", field_nd, case, path, extra={"grid_nodes": nn})
