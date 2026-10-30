# Public N-D Benchmark: All-Method Comparison (Mean)

> N=1..8, 1000 cases per N, TR=2.5, K=8, S=4. Only means are reported. J is the dimensionless time-weighted vibration level; feasibility is the fraction of cases satisfying all four constraints; times are single-process absolute wall-clock seconds.
> `—` means the method is unavailable at that N: grid methods at N≥4 are `grid_too_large`; Lattice at N≥4 and GD at N=8 are `too_slow`.

## 1. Mean J (lower is better)

| Method | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---|---|---|---|---|---|---|---|
| CoTra-J | 130.993 | 127.143 | 123.936 | 125.070 | 124.040 | 123.783 | 125.153 | 126.223 |
| FMM | 137.534 | 131.071 | 126.932 | — | — | — | — | — |
| A* | 137.560 | 131.530 | 127.246 | — | — | — | — | — |
| Theta* | 137.560 | 132.869 | 128.801 | — | — | — | — | — |
| Lattice | 137.548 | 132.438 | 129.321 | — | — | — | — | — |
| RRT* | 137.560 | 133.737 | 129.354 | 131.366 | 130.564 | 130.934 | 132.955 | 133.799 |
| BIT* | 137.560 | 132.700 | 128.356 | 130.227 | 129.424 | 129.730 | 131.656 | 132.626 |
| GA | 134.788 | 130.844 | 127.329 | 129.049 | 128.388 | 128.736 | 130.615 | 131.714 |
| SA | 137.793 | 133.473 | 129.115 | 131.107 | 130.298 | 130.662 | 132.660 | 133.576 |
| GD | 137.560 | 131.763 | 127.500 | 129.088 | 128.148 | 128.173 | 130.027 | — |
| CHOMP | 136.503 | 132.835 | 128.855 | 130.770 | 130.025 | 130.420 | 132.328 | 133.174 |
| random | 137.641 | 133.701 | 129.366 | 131.389 | 130.568 | 130.913 | 132.905 | 133.714 |

## 2. Feasibility rate (higher is better)

| Method | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---|---|---|---|---|---|---|---|
| CoTra-J | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| FMM | 1.000 | 1.000 | 1.000 | — | — | — | — | — |
| A* | 1.000 | 1.000 | 1.000 | — | — | — | — | — |
| Theta* | 1.000 | 1.000 | 1.000 | — | — | — | — | — |
| Lattice | 1.000 | 1.000 | 1.000 | — | — | — | — | — |
| RRT* | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| BIT* | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| GA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| GD | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — |
| CHOMP | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| random | 0.105 | 0.108 | 0.123 | 0.108 | 0.110 | 0.129 | 0.102 | 0.114 |

## 3. Single-case time / s (lower is better)

| Method | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---|---|---|---|---|---|---|---|
| CoTra-J | 6.8754 | 7.2620 | 8.1860 | 9.0672 | 9.6686 | 9.9291 | 10.0609 | 10.4010 |
| FMM | 0.0066 | 0.0379 | 6.6209 | — | — | — | — | — |
| A* | 0.0009 | 0.0086 | 2.3877 | — | — | — | — | — |
| Theta* | 0.0035 | 0.1345 | 15.7509 | — | — | — | — | — |
| Lattice | 0.0284 | 1.3944 | 14.1253 | 24.7504 | 36.8583 | 52.7125 | 62.5620 | 65.6852 |
| RRT* | 0.4816 | 0.4801 | 0.9036 | 1.0968 | 1.2614 | 1.4657 | 1.4981 | 1.5318 |
| BIT* | 0.7282 | 1.2765 | 4.2809 | 5.5290 | 6.6467 | 8.1040 | 8.2319 | 7.8139 |
| GA | 0.1411 | 0.1558 | 0.2233 | 0.2541 | 0.2705 | 0.3093 | 0.3143 | 0.3077 |
| SA | 0.5440 | 0.8215 | 2.7984 | 3.6696 | 4.3416 | 5.3314 | 5.2416 | 5.2892 |
| GD | 0.0642 | 0.2547 | 1.9388 | 5.1122 | 7.8918 | 11.8362 | 13.0942 | 22.6825 |
| CHOMP | 0.0729 | 0.1349 | 0.5093 | 0.7173 | 0.9677 | 1.2508 | 1.3599 | 1.5028 |
| random | 0.0003 | 0.0003 | 0.0007 | 0.0008 | 0.0010 | 0.0011 | 0.0011 | 0.0011 |

## 4. batch=16 per-case time / s (lower is better)

| Method | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---|---|---|---|---|---|---|---|
| CoTra-J | 1.6003 | 1.5529 | 2.4401 | 2.7630 | 3.0269 | 3.6740 | 3.3536 | 3.4386 |
| FMM | — | — | — | — | — | — | — | — |
| A* | — | — | — | — | — | — | — | — |
| Theta* | — | — | — | — | — | — | — | — |
| Lattice | — | — | — | — | — | — | — | — |
| RRT* | — | — | — | — | — | — | — | — |
| BIT* | — | — | — | — | — | — | — | — |
| GA | 0.0131 | 0.0314 | 0.1027 | 0.1319 | 0.1580 | 0.1931 | 0.1946 | 0.1927 |
| SA | 0.3135 | 0.4616 | 1.2271 | 1.5101 | 1.8519 | 2.8930 | 2.8363 | 2.8822 |
| GD | 0.0622 | 0.2587 | 1.9418 | 5.1081 | 7.9929 | 11.8648 | 12.9883 | 22.8625 |
| CHOMP | 0.0409 | 0.1129 | 0.6495 | 1.1063 | 1.7093 | 2.4094 | 2.7573 | 3.1327 |
| random | — | — | — | — | — | — | — | — |

## 5. Notes

- CoTra-J has the lowest mean J and 100% feasibility at every N; pairwise mean differences are listed in `results/FINAL_BY_N.md`.
- Only means are listed. Per-case distributions, paired differences and confidence intervals can be computed from `results/E4_batch` and `results/E5_batch`.
- random is an untuned reference baseline (uniform durations, no repair); it is not the strongest competitor.
- Times are absolute costs only; no speed-up ratio is computed. `batch=16` is marked — for methods without a batch mode.
- GD at N=8 and Lattice at N≥4 are `too_slow` for the 1000-case benchmark; their single-case times are still listed in Table 3 for reference.
