# Algorithm Registry

All methods use the same `evaluate.py`, the same time-indexed corridor and the same per-case rates. Parameters below are frozen for the public benchmark.

| Method | Implementation | Key parameters | Status | N |
|---|---|---|---|---|
| **CoTra-J** | `code/cotra_j.py` | K=8; ρ=0.2; δ=0.02; path CEM 1024×200 + 512×150; joint CEM 512×250+150; batch=16 | Implemented; P1/P2/P3 passed | 1–8 |
| FMM | `code/baselines/fmm.py` | 100 grid nodes per axis for N≤3; inverse-field time allocation; time-indexed corridor | Implemented | 1–3 |
| A* | `code/baselines/astar.py` | 100 grid nodes per axis for N≤3; time-indexed corridor | Implemented | 1–3 |
| Theta* | `code/baselines/theta_star.py` | 100 grid nodes per axis for N≤3; time-indexed corridor | Implemented | 1–3 |
| Lattice | `code/baselines/lattice.py` | resolution 100 for N≤3; graph search; time-indexed corridor | Implemented | 1–3 |
| RRT* | `code/baselines/rrt_star.py` | 100-grid sampling; 2000 iterations; K=8 | Implemented | 1–8 |
| BIT* | `code/baselines/bit_star.py` | batch-informed tree; K=8 | Implemented | 1–8 |
| GA | `code/baselines/ga.py` | population 512; 150 generations; K=8 | Implemented | 1–8 |
| SA | `code/baselines/sa.py` | 5000 iterations; K=8 | Implemented | 1–8 |
| GD | `code/baselines/gd.py` | finite-difference gradient descent; 200 steps; K=8 | Implemented | 1–7 (N=8 too slow) |
| CHOMP | `code/baselines/chomp.py` | 1000 iterations; K=8 | Implemented | 1–8 |
| random | `code/baselines/random.py` | K=8; uniform random nodes inside the time-indexed corridor; uniform durations; no repair | Implemented | 1–8 |

## Notes

- All trajectory methods use K=8 segments (9 nodes) for a fair comparison.
- Per-dimension rate limits are passed into `evaluate.Problem` for every case; there is no global rate.
- Fixed dimensions ($A_i=B_i$) have zero corridor width and are locked at $A_i$ by the evaluator.
- Grid planners (FMM, A*, Theta*, Lattice) are marked `grid_too_large` for N≥4 because $100^N$ exceeds the memory limit.
- Lattice is marked `too_slow` for N≥4; GD is marked `too_slow` at N=8.
- `cotra_j.py` never imports external scripts; it is a standalone implementation.

## Consistency checks

```bash
python code/test_cotra_j.py
python code/test_baselines_matrix.py
```

`test_cotra_j.py` runs the API-contract and batch-benchmark tests against a self-contained analytic field bank; it imports nothing outside `code/`.
