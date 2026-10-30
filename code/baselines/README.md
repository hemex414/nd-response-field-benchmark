# Public baselines -- independent implementation for the public N-D field

Self-contained (no imports from the external paper scripts). One module per algorithm (10 optimisers + the `random` reference baseline);
shared evaluator and field are in `public_field/code/`.

## Interface

```python
from public_field.code import field as F
from public_field.code.baselines import plan_single, plan_batch

case = F.load_cases(3)[0]                     # {"case_id","N","subset","field_seed",
                                              #  "A","B","TR","rates"}
rec  = plan_single("fmm", case)               # single case, any subset/N
recs = plan_batch("sa", cases, batch_size=16) # tensor batch (GA/SA/CHOMP)
recs = plan_batch("rrt_star", cases, workers=12)  # process pool (incl. GD)
```

`rec` fields: `case_id, N, subset, field_seed, A, B, TR, rates, method, K,
J, J_uniform, T, durations, time_alloc, time_alloc_applied, endpoint_error,
rate_violation, corridor_violation_time, corridor_violation_geom, feasible,
repair_fac, points (K+1,N), params, status, runtime_s`. Grid methods at N>=4 return
`status="timeout"`, `grid_too_large=True` ("no coarse grid").

## Algorithms and parameters (verbatim reference defaults)

| file | algorithm | key parameters |
|---|---|---|
| fmm.py | FMM (skfmm travel time + trace) | alphas 0.01/0.05/0.10/0.15/0.20 (+-0.02 refine), 100^N grid, trace_step 3 cells |
| astar.py | A* | 100^N grid, 6-neighbour, ellipsoid beta=0.2 |
| theta_star.py | Theta* | LOS_SAMPLES 3, GRID_STRIDE 2, 100^N |
| lattice.py | Lattice | N_DIRS 6, MAX_LEN 40, QUANT 20, N_PTS_PRC 10, max_expansions 5000 |
| rrt_star.py | RRT* | N_ITER 1500, STEP_SIZE 8, TARGET_BIAS 0.1, 2*step rewire |
| bit_star.py | BIT* | 800 samples, 12 neighbours, 2 rounds |
| ga.py | B-spline (1+1)-ES (bspline_cmaes) | 10 control points, deg 3, 150 gens, sigma0=40, x1.1/x0.9, 300 eval pts |
| gd.py | gradient walk | step 5.0, mixes {0.6,0.8}, 300 steps, FD eps 0.5 |
| sa.py | simulated annealing | 50 points, 2000 iters, T0=50, cooling 0.995, perturb 15, seeds (42,123,456) |
| chomp.py | CHOMP (smooth + vibration, restored original) | 100 points, 150 iters, weights {1,5} x {0.5,2}, LR 0.5, FD eps 0.5 |
| random.py | random reference baseline (kept in the public package) | K=6, nodes sampled uniformly inside the time-indexed corridor, uniform durations, seed=case_id, no tuning/repair |

Algorithm workspace: the reference defaults are expressed on the 2880-unit RPM axis; the
public field lives on `[0,100]`, so coordinates are mapped isotropically
(`PUB = 100/2880`) and **every parameter keeps its published value**. The public
evaluator (`code/evaluate.py`) is the only evaluator used for J / feasibility:
`J` uses the returned per-segment durations, feasibility uses the **time-indexed
corridor** (numeric-boundary cases 1e-6 < corr <= 1e-5 are reported separately), and
the path-level arms get the inverse-field time allocation post-processing
(`J_alloc` main, `J_uniform` counterfactual).

## Batch modes

* `tensor_batch` (GA, SA, CHOMP): cases stacked in a `FieldBank`; batched field evaluation
  is **bit-identical** to the single-case field (`FieldND` accumulation order preserved),
  J difference vs single-case <= 1e-7.
* `process_pool` (FMM, A*, Theta*, Lattice, RRT*, BIT*, GD): `multiprocessing` pool,
  one case per task; results are bit-identical to single-case (GD keeps the exact
  per-case walk; no vectorised walk by decision of the experiment owner).
* GD/GA/SA/CHOMP random streams: per-case `default_rng(seed)` exactly as in the
  single-case implementations.

## Runners

* `run_single.py` -- single case, arbitrary subset/N.
* `run_batch.py` -- batch with resume (`methods/<m>_N<n>.jsonl`), `--merge-only`,
  `--workers`, `--batch-size`, `--max-case-hours 6` (pilot projection > budget ->
  `too_slow`, no partial J; N>=4 grids -> `grid_too_large`).
* `test_baselines_matrix.py` -- T1..T6 acceptance matrix (first run on N=2/3 x 16 cases).
* `merge_e5_cotra.py` -- merges E5 baselines with the public CoTra-J batch results
  (same case_id) into `results/FINAL_BY_N.md`.
