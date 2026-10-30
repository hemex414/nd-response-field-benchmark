# -*- coding: utf-8 -*-
"""lattice -- single-case public implementation (algorithm and parameters from the
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
    p = PARAMS["lattice"]
    pf = PrivField(field_nd)
    n = int(field_nd.N)
    a = np.asarray(case["A"], float) / PUB
    b = np.asarray(case["B"], float) / PUB
    rate = np.asarray(case["rates"], float) / PUB
    max_len, quant = p["MAX_LEN"], p["QUANT"]
    tt = np.linspace(0.0, 1.0, p["N_PTS_PRC"])
    prims = []
    for d in range(n):
        for sign in (1.0, -1.0):
            e = np.zeros(n); e[d] = sign
            prims.append(np.array([np.zeros(n), e * max_len * 0.5, e * max_len]))

    def bezier(t, ctrl):
        return (1 - t) ** 2 * ctrl[0] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * ctrl[2]

    def qkey(pos, d):
        return tuple(np.rint(pos / quant).astype(int).tolist()) + (int(d),)

    start = qkey(a, 0)
    g = {start: 0.0}; parent = {}; heap = [(0.0, start)]; closed = set(); exp = 0
    while heap and exp < p["max_expansions"]:
        _, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u); exp += 1
        pos = np.asarray(u[:-1], float) * quant
        for pi, ctrl in enumerate(prims):
            pts = np.array([bezier(t, ctrl) for t in tt]) + pos
            if np.any(pts < 0) or np.any(pts > WORKSPACE):
                continue
            if not in_ellipsoid(pts, a, b, BETA).all():
                continue
            vib = float(np.mean(pf.interp(pts)))
            tc = float(max_len / np.max(rate)) if np.max(rate) > 0 else 0.0
            cost = g[u] + vib * tc
            v = qkey(pts[-1], pi)
            if cost < g.get(v, np.inf):
                g[v] = cost; parent[v] = u
                heapq.heappush(heap, (cost, v))
    best_key, best_d = None, np.inf
    for k in g:
        pos = np.asarray(k[:-1], float) * quant
        d = float(np.linalg.norm(pos - b))
        if d < best_d and d <= max_len * 1.01:
            best_d, best_key = d, k
    if best_key is None:
        path = np.vstack([a, b])
    else:
        seq = []; k = best_key
        while k in parent:
            seq.append(np.asarray(k[:-1], float) * quant)
            k = parent[k]
        seq.append(a.copy()); seq = seq[::-1]
        path = np.vstack([np.array(seq), b])
    return path_to_record("lattice", field_nd, case, path, extra={"expansions": exp})
