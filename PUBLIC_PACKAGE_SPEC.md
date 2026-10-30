# Public Package: Data and File Format Specification

> Goal: anyone can reproduce, with the same code:
> 1) single-case planning for an arbitrary dimension subset;
> 2) the N=1..8 batch benchmark with 1000 cases per N;
> 3) the same evaluator, the same cases, and the same algorithm parameters.
>
> This package does not import external paper scripts. Algorithm parameters are listed in `ALGORITHM_REGISTRY.md`.

## 1. Layout

```
public_field/
├─ README.md
├─ SPEC.md                      # field definition, parameters, normalization
├─ PUBLIC_PACKAGE_SPEC.md       # this file
├─ ALGORITHM_REGISTRY.md
├─ AUDIT_REPORT.md
├─ COMPARISON_TABLE.md / .csv
├─ TIMING_TABLE.md
├─ CHECKSUMS.txt
├─ cases/
│  ├─ MANIFEST.json             # N, case counts, seed, hashes
│  └─ N1..N8/cases.jsonl        # one JSON case per line
├─ code/
│  ├─ field.py                  # field class and case loader
│  ├─ evaluate.py               # J, constraints, feasibility (single evaluator)
│  ├─ cotra_j.py                # standalone CoTra-J (batch, per-case rates)
│  ├─ baselines/                # one file per algorithm, shared interface
│  ├─ run_single.py             # single-case baseline runner
│  ├─ run_batch.py              # batch baseline runner
│  ├─ run_batch_cotra_j.py      # batch CoTra-J runner
│  └─ merge_e5_cotra.py
├─ data/canonical_3d_seed2026.npz
├─ figures/
└─ results/
   ├─ README.md
   ├─ FINAL_BY_N.md
   ├─ timing_single.csv
   ├─ E4_batch/N1..N8/
   └─ E5_batch/N1..N8/
```

## 2. Case format (`cases/N<n>/cases.jsonl`)

Each line is one self-contained JSON case:

```json
{
  "case_id": 0,
  "N": 3,
  "subset": [0, 3, 5],
  "field_seed": 202600017,
  "field_version": "public-field-1.0.0",
  "A": [10.0, 80.0, 20.0],
  "B": [90.0, 15.0, 70.0],
  "TR": 2.5,
  "rates": [4, 11, 7]
}
```

- `subset`: N indices selected from the 8 candidate dimensions; the case is **not** a slice of an 8-D field;
- the field is determined by `(N, field_seed, field_version)`; `subset` selects the state-dimension indices used by A/B/rates;
- `A`/`B` use the field coordinate system; `rates` are per-dimension maximum rates;
- cases are frozen after generation; hashes are in `MANIFEST.json`.

## 3. Field representation

- The package does not store a `1000 × 100^N` dense array;
- it stores generation parameters and seeds, and reconstructs fields deterministically in `field.py`;
- optional precomputed `.npz` caches may be provided for N≤3 as accelerators only;
- the single-case API accepts any subset, not only the frozen benchmark cases.

## 4. Result format

### `methods/<method>_N<n>.json`

```json
{
  "method": "cotra_j",
  "N": 3,
  "n_cases": 1000,
  "config_hash": "...",
  "code_version": "...",
  "cases": [
    {
      "case_id": 0,
      "J": 124.14,
      "T": 12.3,
      "feasible": true,
      "endpoint_error": 0.0,
      "rate_violation": 0.0,
      "corridor_violation": 0.0,
      "durations": [1.2, 1.1],
      "points": [[0.0, 0.0, 0.0]],
      "status": "ok",
      "runtime_s": 1.69
    }
  ]
}
```

### `per_case_N<n>.jsonl`

One `(case, method)` result per line; used for resume, audit and paired comparison. Fields are the same as above plus `method`.

### `summary.json`

```json
{
  "N": 3,
  "methods": ["cotra_j", "fmm"],
  "n_ok": {"cotra_j": 1000},
  "mean_J": {"cotra_j": 124.14},
  "median_J": {},
  "feasible_rate": {"cotra_j": 1.0},
  "mean_runtime_s": {},
  "rank": {},
  "notes": ""
}
```

### `run_manifest.json`

- `run_id`, timestamp, code version/commit;
- `field_version`, case manifest hash;
- `config_hash`;
- Python/torch/CUDA versions, GPU/CPU model;
- status of each method (`ok`, `timeout`, `too_slow`, `grid_too_large`).

## 5. Command line

Single case (baseline; CoTra-J uses the Python API in `README.md`):

```bash
python code/run_single.py --method fmm --N 3 --subset 0,3,5 \
  --A 10,80,20 --B 90,15,70 --TR 2.5 --rates 4,11,7 \
  --field-seed 202600017 --out results/single_fmm_N3.json
```

Batch:

```bash
python code/run_batch.py --N 3 --methods fmm astar rrt_star ga sa \
  --out results/check_N3
```

- `run_batch.py` supports `--resume` (skip cases already in `per_case_N*.jsonl`);
- `--merge-only` rebuilds method JSON and summary from JSONL;
- single-case and batch runs use the **same** `evaluate.py`.

## 6. Reproducibility requirements

- every result carries `field_version`, `case_id`, `config_hash`, and `code_version`;
- algorithm parameters come from `ALGORITHM_REGISTRY.md`; do not silently change defaults;
- implementations inconsistent with the registry are marked `invalid_simplified` and excluded from released results;
- single-case runs may use any subset; batch runs must use the frozen cases in `cases/`.

## 7. Independence

- The public package and the reference study share algorithm semantics and parameters, but not code paths;
- no external script is imported by the public package;
- public results are generated independently and are not copied from external results;
- parameter-level consistency is tracked in `ALGORITHM_REGISTRY.md`.
