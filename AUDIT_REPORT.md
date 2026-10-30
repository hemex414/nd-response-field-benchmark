# Public Benchmark Audit Report

Scope: `public_field/cases`, `results/E4_batch`, `results/E5_batch`, `timing_single.csv`, and the shared evaluator. This report documents the released state; it is not a development log.

## 1. Data scale

| N | Cases | Subset length | Rates in [2,15] | Fixed-dim cases | SHA256 prefix |
|---|---|---|---|---|---|
| 1 | 1000 | pass | pass | 0 | `582ce376add0` |
| 2 | 1000 | pass | pass | 0 | `ae5882fe78a9` |
| 3 | 1000 | pass | pass | 0 | `55b5503ebcea` |
| 4 | 1000 | pass | pass | 0 | `8f7b8c855c9a` |
| 5 | 1000 | pass | pass | 0 | `30bee9a54ef4` |
| 6 | 1000 | pass | pass | 0 | `25212daaf191` |
| 7 | 1000 | pass | pass | 0 | `ddc98667dccc` |
| 8 | 1000 | pass | pass | 0 | `c702bd298ebe` |

- 8000 cases in total; each case is generated directly as an N-D field, not sliced from an 8-D field.
- No public benchmark case has a fixed dimension; fixed-dimension behavior is covered separately by the evaluator contract.

## 2. Result coverage

| N | CoTra-J n | CoTra-J mean J | Best baseline | Baseline J | ΔJ | Methods reported |
|---|---|---|---|---|---|---|
| 1 | 1000 | 130.993 | GA | 134.788 | 3.795 | 12 |
| 2 | 1000 | 127.143 | GA | 130.844 | 3.701 | 12 |
| 3 | 1000 | 123.936 | FMM | 126.932 | 2.996 | 12 |
| 4 | 1000 | 125.070 | GA | 129.049 | 3.979 | 12 |
| 5 | 1000 | 124.040 | GD | 128.148 | 4.108 | 12 |
| 6 | 1000 | 123.783 | GD | 128.173 | 4.390 | 12 |
| 7 | 1000 | 125.153 | GD | 130.027 | 4.874 | 12 |
| 8 | 1000 | 126.223 | GA | 131.714 | 5.491 | 11 |

- Every N is an independent 1000-case paired comparison; absolute J is not pooled across N.
- Baseline set: 10 optimizers + random = 11 baselines, plus CoTra-J = 12 methods.
- Grid planners are `grid_too_large` for N≥4; Lattice is `too_slow` for N≥4; GD is `too_slow` at N=8.

## 3. Timing matrix

- 136 rows: 121 `ok`, 15 `grid_too_large`.
- 12 methods, N=1..8.
- Single-process measurements; batch=16 is the per-case time for a batch of 16.
- No speed-up ratio is computed or reported.
- Raw data: `results/timing_single.csv`; formatted tables: `TIMING_TABLE.md`, `COMPARISON_TABLE.md`.

## 4. Consistency checks

| Check | Result | Note |
|---|---|---|
| Self-contained cases | PASS | `cases/N*/cases.jsonl` + `cases/MANIFEST.json`; `field.load_cases` reads the public directory first |
| Subset length | PASS | 8×1000 cases; `len(subset)=N` |
| Rate range | PASS | All rates in [2,15] |
| Fixed dimensions | PASS | Zero corridor width for `A_i=B_i`; output clipped to `A_i` |
| Result count | PASS | 1000 CoTra-J cases per N; 1000 valid baseline records per N when the method is available |
| Cross-N pooling | PASS | Only within-N paired comparisons are reported |
| Method count | PASS | 10 optimizers + random + CoTra-J |
| Timing labels | PASS | batch=16 panels contain only methods with batch=16 data |
| Random baseline | PASS | K=8, uniform durations, no repair; low feasibility is expected for a random path |

## 5. Reproduce the audit

```bash
python code/audit_field.py
python code/test_cotra_j.py
python code/test_baselines_matrix.py
```

## 6. Summary

The public benchmark is self-contained at the case, evaluator and result levels. All released 1000-case comparisons use the same cases, the same time-indexed corridor and the same evaluator. The formatted mean tables are in `COMPARISON_TABLE.md` and `TIMING_TABLE.md`.
