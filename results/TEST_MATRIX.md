# Public baselines test matrix (T1-T6)

generated: 2026-10-26 18:34:14 | N=[2, 3] | 16 cases | workers=12 | batch=16

## T1_single_smoke -- PASS

| method | kind | max|dJ| | max|dT| | feasible same | ok |
|---|---|---|---|---|---|
| fmm | - | nan | nan | - | PASS |
| astar | - | nan | nan | - | PASS |
| theta_star | - | nan | nan | - | PASS |
| lattice | - | nan | nan | - | PASS |
| rrt_star | - | nan | nan | - | PASS |
| bit_star | - | nan | nan | - | PASS |
| ga | - | nan | nan | - | PASS |
| gd | - | nan | nan | - | PASS |
| sa | - | nan | nan | - | PASS |
| chomp | - | nan | nan | - | PASS |
| fmm | - | nan | nan | - | PASS |
| astar | - | nan | nan | - | PASS |
| theta_star | - | nan | nan | - | PASS |
| lattice | - | nan | nan | - | PASS |
| rrt_star | - | nan | nan | - | PASS |
| bit_star | - | nan | nan | - | PASS |
| ga | - | nan | nan | - | PASS |
| gd | - | nan | nan | - | PASS |
| sa | - | nan | nan | - | PASS |
| chomp | - | nan | nan | - | PASS |

## T2_batch_vs_single_N2 -- PASS

| method | kind | max|dJ| | max|dT| | feasible same | ok |
|---|---|---|---|---|---|
| fmm | process_pool | 0.000e+00 | 0.000e+00 | True | PASS |
| astar | process_pool | 0.000e+00 | 0.000e+00 | True | PASS |
| theta_star | process_pool | 0.000e+00 | 0.000e+00 | True | PASS |
| lattice | process_pool | 0.000e+00 | 0.000e+00 | True | PASS |
| rrt_star | process_pool | 0.000e+00 | 0.000e+00 | True | PASS |
| bit_star | process_pool | 0.000e+00 | 0.000e+00 | True | PASS |
| ga | tensor_batch | 2.842e-14 | 3.553e-15 | True | PASS |
| gd | tensor_batch | 0.000e+00 | 0.000e+00 | True | PASS |
| sa | tensor_batch | 0.000e+00 | 0.000e+00 | True | PASS |
| chomp | tensor_batch | 9.739e-08 | 3.553e-15 | True | PASS |

## T3_batch_vs_single_N3 -- PASS

| method | kind | max|dJ| | max|dT| | feasible same | ok |
|---|---|---|---|---|---|
| fmm | process_pool | 0.000e+00 | 0.000e+00 | True | PASS |
| astar | process_pool | 0.000e+00 | 0.000e+00 | True | PASS |
| theta_star | process_pool | 0.000e+00 | 0.000e+00 | True | PASS |
| lattice | process_pool | 0.000e+00 | 0.000e+00 | True | PASS |
| rrt_star | process_pool | 0.000e+00 | 0.000e+00 | True | PASS |
| bit_star | process_pool | 0.000e+00 | 0.000e+00 | True | PASS |
| ga | tensor_batch | 2.842e-14 | 0.000e+00 | True | PASS |
| gd | tensor_batch | 0.000e+00 | 0.000e+00 | True | PASS |
| sa | tensor_batch | 0.000e+00 | 0.000e+00 | True | PASS |
| chomp | tensor_batch | 8.454e-08 | 3.553e-15 | True | PASS |

## T5_1000_cases: None
## T6_reproducibility: True
## resources

| method | T single 16 [s] | T batch 16 [s] | speedup |
|---|---|---|---|
| fmm | 0.76 | 0.80 | 0.95x |
| astar | 0.24 | 0.71 | 0.33x |
| theta_star | 3.10 | 1.53 | 2.02x |
| lattice | 36.27 | 8.22 | 4.41x |
| rrt_star | 9.45 | 3.42 | 2.76x |
| bit_star | 23.89 | 5.59 | 4.28x |
| ga | 2.88 | 0.59 | 4.91x |
| gd | 5.14 | 5.37 | 0.96x |
| sa | 14.79 | 7.73 | 1.91x |
| chomp | 2.58 | 2.03 | 1.27x |

peak RSS: 66 MB, CPUs: 20
