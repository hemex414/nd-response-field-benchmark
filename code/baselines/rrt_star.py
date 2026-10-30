# -*- coding: utf-8 -*-
"""rrt_star -- single-case public implementation (algorithm and parameters from the
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
    p = PARAMS["rrt_star"]
    pf = PrivField(field_nd)
    n = int(field_nd.N)
    a = np.asarray(case["A"], float) / PUB
    b = np.asarray(case["B"], float) / PUB
    rate = np.asarray(case["rates"], float) / PUB
    best_path, best_obj = np.vstack([a, b]), np.inf
    step = p["STEP_SIZE"]
    for seed in p["SEEDS"]:
        rng = np.random.default_rng(seed)
        nodes = [a.copy()]; costs = [0.0]; parent = [-1]
        for _ in range(p["N_ITER"]):
            if rng.random() < p["TARGET_BIAS"]:
                sample = b.copy()
            else:
                sample = rng.uniform(0, WORKSPACE, size=n)
                if not in_ellipsoid(sample, a, b, BETA):
                    sample = a + rng.random() * (b - a)
            arr = np.asarray(nodes)
            ni = int(np.argmin(np.linalg.norm(arr - sample, axis=1)))
            direction = sample - nodes[ni]
            dist = float(np.linalg.norm(direction))
            if dist < 1e-9:
                continue
            new = nodes[ni] + direction / dist * min(step, dist)
            new = np.clip(new, 0, WORKSPACE)
            if not in_ellipsoid(new, a, b, BETA):
                continue
            vib = float(pf.interp(new[None])[0])
            new_cost = costs[ni] + vib * edge_time(nodes[ni], new, rate)
            near = np.where(np.linalg.norm(arr - new, axis=1) < 2 * step)[0]
            bp, bc = ni, new_cost
            for j in near:
                alt = costs[int(j)] + vib * edge_time(nodes[int(j)], new, rate)
                if alt < bc:
                    bc, bp = alt, int(j)
            nodes.append(new); costs.append(bc); parent.append(bp)
        arr = np.asarray(nodes)
        bi = int(np.argmin(np.linalg.norm(arr - b, axis=1)))
        seq = []; u = bi
        while u != -1:
            seq.append(u); u = parent[u]
        seq = seq[::-1]
        path = np.vstack([arr[seq], b])
        obj = mean_field(pf, path, rate)
        if obj < best_obj:
            best_obj, best_path = obj, path
    return path_to_record("rrt_star", field_nd, case, best_path,
                          extra={"tree_samples": p["N_ITER"], "seeds": p["SEEDS"]})
