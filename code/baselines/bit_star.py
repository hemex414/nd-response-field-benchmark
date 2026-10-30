# -*- coding: utf-8 -*-
"""bit_star -- single-case public implementation (algorithm and parameters from the
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
    p = PARAMS["bit_star"]
    pf = PrivField(field_nd)
    n = int(field_nd.N)
    a = np.asarray(case["A"], float) / PUB
    b = np.asarray(case["B"], float) / PUB
    rate = np.asarray(case["rates"], float) / PUB
    best_path, best_obj = np.vstack([a, b]), np.inf
    for seed in p["SEEDS"]:
        rng = np.random.default_rng(seed)
        box = None
        for rnd in range(p["N_ROUNDS"]):
            lo = np.zeros(n) if box is None else np.maximum(0, box[0])
            hi = np.full(n, WORKSPACE) if box is None else np.minimum(WORKSPACE, box[1])
            pts = []
            while len(pts) < p["N_SAMPLES"]:
                s = rng.uniform(lo, hi, size=(max(p["N_SAMPLES"] * 4, 64), n))
                pts.append(s[in_ellipsoid(s, a, b, BETA)])
            samples = np.concatenate(pts)[:p["N_SAMPLES"]]
            nodes = np.vstack([a[None], samples, b[None]])
            M = len(nodes)
            tree = cKDTree(nodes)
            _, nbr = tree.query(nodes, k=min(p["K_NEIGHBORS"] + 1, M))
            adj = [[] for _ in range(M)]
            for i in range(M):
                for j in nbr[i]:
                    j = int(j)
                    if j <= i:
                        continue
                    mid = 0.5 * (nodes[i] + nodes[j])
                    w = float(pf.interp(mid[None])[0]) * float(np.linalg.norm(nodes[j] - nodes[i]))
                    adj[i].append((j, w)); adj[j].append((i, w))
            g = {0: 0.0}; parent = {}; closed = set(); heap = [(0.0, 0)]
            while heap:
                _, u = heapq.heappop(heap)
                if u == M - 1:
                    break
                if u in closed:
                    continue
                closed.add(u)
                for v, w in adj[u]:
                    if v in closed:
                        continue
                    ng = g[u] + w
                    if ng < g.get(v, np.inf):
                        g[v] = ng; parent[v] = u
                        heapq.heappush(heap, (ng, v))
            if M - 1 not in parent:
                continue
            seq = [M - 1]; u = M - 1
            while u != 0:
                u = parent[u]; seq.append(u)
            seq = seq[::-1]
            path = nodes[seq]
            obj = mean_field(pf, path, rate)
            if obj < best_obj:
                best_obj, best_path = obj, path
                lo = np.maximum(0, np.min(path, 0) - (0.15 * (np.max(path, 0) - np.min(path, 0)) + 5))
                hi = np.minimum(WORKSPACE, np.max(path, 0) + (0.15 * (np.max(path, 0) - np.min(path, 0)) + 5))
                box = (lo, hi)
    return path_to_record("bit_star", field_nd, case, best_path,
                          extra={"samples": p["N_SAMPLES"], "k": p["K_NEIGHBORS"],
                                 "rounds": p["N_ROUNDS"]})
