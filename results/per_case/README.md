# Per-case results (1000 cases per N)

This directory contains compact per-case records for all released 1000-case runs. Files are gzip-compressed CSV so that all 8000 cases × available methods fit in a small Git repository.

## Files

| File | Rows | Contents |
|---|---|---|
| `N1.csv.gz` | 12000 | 12 methods × 1000 cases |
| `N2.csv.gz` | 12000 | 12 methods × 1000 cases |
| `N3.csv.gz` | 12000 | 12 methods × 1000 cases |
| `N4.csv.gz` | 11000 | 11 methods × 1000 cases (Lattice unavailable) |
| `N5.csv.gz` | 11000 | 11 methods × 1000 cases |
| `N6.csv.gz` | 11000 | 11 methods × 1000 cases |
| `N7.csv.gz` | 11000 | 11 methods × 1000 cases |
| `N8.csv.gz` | 10000 | 10 methods × 1000 cases (Lattice and GD unavailable) |
| `MANIFEST.json` | — | row counts, method lists, bytes and SHA256 |

Grid planners (FMM, A*, Theta*) appear with `status=timeout` for N≥4; this preserves one row per requested case. `J` and feasibility fields are empty for unavailable cases. The benchmark-level summary excludes those cells as `grid_too_large`.

## Columns

```
N, method, case_id, J, J_uniform, feasible, status, T,
runtime_s, endpoint_error, rate_violation, corridor_violation_time, durations
```

- `J`: time-weighted vibration level (dimensionless);
- `feasible`: endpoint / rate / time-indexed corridor / duration all satisfied;
- `status`: `ok` for evaluated cases; `timeout` / `grid_too_large` for unavailable grid cells;
- `durations`: 8 segment durations, semicolon-separated;
- `points`: omitted to keep the repository small; trajectories can be regenerated from the case files and code.

## Read

```python
import pandas as pd
df = pd.read_csv("results/per_case/N3.csv.gz", compression="gzip")
print(df.groupby("method")[["J", "feasible"]].mean())
```

A single case:

```python
row = df[(df.method == "cotra_j") & (df.case_id == 0)].iloc[0]
print(row.J, row.feasible, row.durations)
```

## Regenerate

The per-case files are derived from:

- `results/E4_batch/N<n>/cotra_j_batch.jsonl`;
- `results/E5_batch/N<n>/<method>.json`.

After rerunning the benchmark, the same compact table can be rebuilt. The per-case CSV is a release artifact; the canonical full records remain in the raw JSONL/JSON outputs described in `results/README.md`.
