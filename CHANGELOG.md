# Changelog

## 2026-10-29

- CoTra-J now uses the rate-anchored duration wire:
  `dt_k = tmin_k(p) + (T_b - sum_j tmin_j(p)) * w_k` with a fixed-point solve on the
  placement axis `tau = cumsum(dt) / T_b`. Every accepted sample with
  `sum tmin <= T_b` is rate-feasible by construction.
- All CoTra-J results regenerated with this wire: `results/E4_batch/`,
  `results/FINAL_BY_N.md`, `COMPARISON_TABLE.md` and `TIMING_TABLE.md`.
- Search budget unchanged: K=8, rho=0.2, delta=0.02, path-only 1024/200 + 512/150,
  joint 512x250 + 512x150.
