# Timing table (mean absolute wall-clock time)

Protocol: N=1..8, 1000 cases per N, TR=2.5, K=8, S=4. Times are single-process absolute wall-clock seconds. Only the mean is reported; no speed-up ratios are computed. `—` means the method is not available at that N (grid methods: `grid_too_large`; Lattice N≥4 and GD N=8: `too_slow`).

## 1. Single-case time / s

| method | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---|---|---|---|---|---|---|---|
| CoTra-J | 6.875 | 7.262 | 8.186 | 9.067 | 9.669 | 9.929 | 10.061 | 10.401 |
| FMM | 0.00660 | 0.0379 | 6.621 | — | — | — | — | — |
| A* | 0.00090 | 0.00860 | 2.388 | — | — | — | — | — |
| Theta* | 0.00350 | 0.1345 | 15.751 | — | — | — | — | — |
| Lattice | 0.0284 | 1.394 | 14.125 | 24.750 | 36.858 | 52.712 | 62.562 | 65.685 |
| RRT* | 0.4816 | 0.4801 | 0.9036 | 1.097 | 1.261 | 1.466 | 1.498 | 1.532 |
| BIT* | 0.7282 | 1.276 | 4.281 | 5.529 | 6.647 | 8.104 | 8.232 | 7.814 |
| GA | 0.1411 | 0.1558 | 0.2233 | 0.2541 | 0.2705 | 0.3093 | 0.3143 | 0.3077 |
| SA | 0.5440 | 0.8215 | 2.798 | 3.670 | 4.342 | 5.331 | 5.242 | 5.289 |
| GD | 0.0642 | 0.2547 | 1.939 | 5.112 | 7.892 | 11.836 | 13.094 | 22.683 |
| CHOMP | 0.0729 | 0.1349 | 0.5093 | 0.7173 | 0.9677 | 1.251 | 1.360 | 1.503 |
| random | 0.00028 | 0.00032 | 0.00068 | 0.00082 | 0.00099 | 0.00110 | 0.00109 | 0.00115 |

## 2. batch=16 per-case time / s

| method | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---|---|---|---|---|---|---|---|
| CoTra-J | 1.600 | 1.553 | 2.440 | 2.763 | 3.027 | 3.674 | 3.354 | 3.439 |
| FMM | — | — | — | — | — | — | — | — |
| A* | — | — | — | — | — | — | — | — |
| Theta* | — | — | — | — | — | — | — | — |
| Lattice | — | — | — | — | — | — | — | — |
| RRT* | — | — | — | — | — | — | — | — |
| BIT* | — | — | — | — | — | — | — | — |
| GA | 0.0131 | 0.0314 | 0.1027 | 0.1319 | 0.1580 | 0.1931 | 0.1946 | 0.1927 |
| SA | 0.3135 | 0.4616 | 1.227 | 1.510 | 1.852 | 2.893 | 2.836 | 2.882 |
| GD | 0.0622 | 0.2587 | 1.942 | 5.108 | 7.993 | 11.865 | 12.988 | 22.863 |
| CHOMP | 0.0409 | 0.1129 | 0.6495 | 1.106 | 1.709 | 2.409 | 2.757 | 3.133 |
| random | — | — | — | — | — | — | — | — |

## 3. Notes

- `batch=16` is the per-case time when 16 cases are planned together; it is not a single-case latency.
- Grid planners (FMM, A*, Theta*, Lattice) do not support batch=16 and are marked `—` in Table 2.
- random is a reference baseline implemented on CPU; it has no batch mode.
- Raw per-method measurements are in `results/timing_single.csv` (columns: method, N, n, batch, workers, mean_s, median_s, min_s, max_s, device, feasible, status, note).
