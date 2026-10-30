# Public Release Manifest

A self-contained, reproducible N-dimensional response-field benchmark. Any reader can run the single-case and 1000-case batch experiments without the external paper directory.

Authoritative references: [AUDIT_REPORT.md](AUDIT_REPORT.md), [cases/MANIFEST.json](cases/MANIFEST.json), [CHECKSUMS.txt](CHECKSUMS.txt).

## 1. Inventory

```
public_field/
├─ README.md
├─ SPEC.md
├─ PUBLIC_PACKAGE_SPEC.md
├─ ALGORITHM_REGISTRY.md
├─ AUDIT_REPORT.md
├─ COMPARISON_TABLE.md / .csv
├─ TIMING_TABLE.md
├─ CHECKSUMS.txt
├─ LICENSE / LICENSE-DATA.md
├─ requirements.txt
├─ cases/
│  ├─ MANIFEST.json
│  └─ N1..N8/cases.jsonl
├─ code/
│  ├─ field.py
│  ├─ evaluate.py
│  ├─ cotra_j.py
│  ├─ run_batch_cotra_j.py
│  ├─ run_batch.py
│  ├─ run_single.py
│  ├─ merge_e5_cotra.py
│  ├─ baselines/
│  └─ test_*.py
├─ data/canonical_3d_seed2026.npz
├─ figures/
└─ results/
   ├─ README.md
   ├─ FINAL_BY_N.md
   ├─ timing_single.csv
   ├─ per_case/N1..N8.csv.gz
   └─ E4_batch/ E5_batch/
```

## 2. Data

- 8 dimensions, 1000 cases each, 8000 cases total;
- each case is generated from `(N, field_seed)`; `subset` selects state-dimension indices, and the case is not an 8-D slice;
- rate limits are random integers in 2--15; TR=2.5;
- no fixed-dimension case exists in the public benchmark.

## 3. Methods

- **CoTra-J**: K=8, ρ=0.2, δ=0.02; path CEM 1024×200 + 512×150; joint CEM 512×250+150; per-case rates; batch=16;
- **10 optimization baselines**: FMM, A*, Theta*, Lattice, RRT*, BIT*, GA, SA, GD, CHOMP;
- **random**: K=8, uniform random nodes in the time-indexed corridor, uniform durations, no tuning;
- all methods use the same `evaluate.py`, the same time-indexed corridor and the same rates.

## 4. Results

Detailed all-method mean tables: [COMPARISON_TABLE.md](COMPARISON_TABLE.md) and [TIMING_TABLE.md](TIMING_TABLE.md).

| N | CoTra-J | Best baseline | Baseline J | ΔJ | CoTra feasible |
|---|---|---|---|---|---|
| 1 | 131.100 | GA | 134.788 | 3.688 | 1.000 |
| 2 | 127.320 | GA | 130.844 | 3.524 | 1.000 |
| 3 | 124.120 | FMM | 126.932 | 2.812 | 1.000 |
| 4 | 125.325 | GA | 129.049 | 3.724 | 1.000 |
| 5 | 124.330 | GD | 128.148 | 3.818 | 1.000 |
| 6 | 124.113 | GD | 128.173 | 4.060 | 1.000 |
| 7 | 125.549 | GD | 130.027 | 4.478 | 1.000 |
| 8 | 126.619 | GA | 131.714 | 5.095 | 1.000 |

- Each N is an independent 1000-case paired comparison; absolute J is not pooled across N.
- Grid planners are `grid_too_large` for N≥4; Lattice is `too_slow` for N≥4; GD is `too_slow` at N=8.
- random is an untuned reference baseline, not the strongest competitor.

## 5. Quick commands

Single baseline:

```bash
python code/run_single.py --method fmm --N 2 --A 10,20 --B 80,90 \
  --rates 5,7 --field-seed 202600001 --out results/single_fmm_N2.json
```

Batch:

```bash
python code/run_batch.py --N 3 --methods fmm astar rrt_star ga sa \
  --out results/check_N3
python code/run_batch_cotra_j.py --n 3 --batch 16 \
  --out results/check_N3/cotra_j_batch.jsonl
python code/merge_e5_cotra.py
```

## 6. Notes

- The public evaluator uses `S_EVAL=4`; all methods are compared under this protocol.
- Fixed dimensions are locked by the evaluator; the public benchmark has no fixed-dimension case.
- random is a reference lower bound, not a tuned competitor.
- The field is a controlled synthetic benchmark; conclusions are limited to dimension scalability and reproducibility.

## 7. Integrity

- `cases/MANIFEST.json`: SHA256, subset/rate checks and fixed-dimension counts for all 8 case files;
- `CHECKSUMS.txt`: SHA256 for cases, summary tables and timing;
- `AUDIT_REPORT.md`: data, result, timing and logic checks.
