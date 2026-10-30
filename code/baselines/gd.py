# -*- coding: utf-8 -*-
"""gd -- single-case public implementation (algorithm and parameters from the
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
    p = PARAMS["gd"]
    pf = PrivField(field_nd)
    n = int(field_nd.N)
    a = np.asarray(case["A"], float) / PUB
    b = np.asarray(case["B"], float) / PUB
    rate = np.asarray(case["rates"], float) / PUB
    eps = p["GRAD_EPS"]

    def grad(pos):
        g = np.zeros(n)
        for d in range(n):
            off = np.zeros(n); off[d] = eps
            pp = np.clip(pos + off, 0, WORKSPACE); pm = np.clip(pos - off, 0, WORKSPACE)
            g[d] = (float(pf.interp(pp[None])[0]) - float(pf.interp(pm[None])[0])) / (2 * eps)
        return g

    best_path, best_obj = np.vstack([a, b]), np.inf
    for ss in p["STEP_SIZES"]:
        for mix in p["MIXES"]:
            pos = a.copy(); path = [pos.copy()]
            dist = float(np.linalg.norm(b - a))
            n_steps = max(int(dist / ss * 2.5) + 100, p["N_STEPS"])
            for _ in range(n_steps):
                if np.linalg.norm(pos - b) < ss:
                    break
                gr = grad(pos)
                te = (b - pos) / (np.linalg.norm(b - pos) + 1e-6)
                aw = -gr / (np.linalg.norm(gr) + 1e-6)
                mv = mix * te + (1.0 - mix) * aw
                mv = mv / (np.linalg.norm(mv) + 1e-6) * ss
                new = np.clip(pos + mv, 0, WORKSPACE)
                if in_ellipsoid(new, a, b, BETA):
                    pos = new; path.append(pos.copy())
            cur = path[-1]
            d = float(np.linalg.norm(cur - b))
            if d > ss:
                nf_ = max(int(np.ceil(d / ss)), 1)
                for i in range(1, nf_ + 1):
                    path.append(cur + (i / nf_) * (b - cur))
            elif d > 1e-9:
                path.append(b.copy())
            path = np.array(path)
            obj = mean_field(pf, path, rate)
            if obj < best_obj:
                best_obj, best_path = obj, path
    return path_to_record("gd", field_nd, case, best_path,
                          extra={"step": p["STEP_SIZES"][0], "mixes": p["MIXES"]})
