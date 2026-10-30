# cotra_j standalone module - test report

device: `cuda`, torch `2.6.0+cu126`

## P1 - API contract (per-case analytic bank)

| check | result |
|---|---|
| points_shape | PASS |
| durations_shape | PASS |
| sum_durations_Tb | PASS |
| J_finite | PASS |
| feasible | PASS |
| J_uses_returned_durations | PASS |
| rates_per_case_used | PASS |

## P2 - batch benchmark (single process, analytic N-D bank)

| B | wall time (s) | s/case | peak GPU (MB) | speed-up vs sequential B=1 |
|---|---|---|---|---|
| 1 | 21.42 | 21.423 | 35.8 | 1.00x |
| 4 | 25.90 | 6.475 | 43.7 | 3.31x |
| 8 | 33.37 | 4.171 | 54.4 | 5.14x |
| 16 | 42.84 | 2.678 | 75.5 | 8.00x |

Note: CEM is stochastic; timings are single-run single-process and include the internal path-only warm start (no separate evaluation).

## Verdict

ALL PASS
