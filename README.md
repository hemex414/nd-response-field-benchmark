# Public N-D Response-Field Benchmark

A synthetic, dimensionless, reproducible N-dimensional response-field benchmark for comparing path–time trajectory optimization algorithms in multi-device state transitions. The benchmark covers $N=1,\dots,8$, with 1000 randomized cases per dimension (8000 cases in total), one reference method (CoTra-J), 10 optimization baselines, and a random baseline.

> This repository contains only the public benchmark. It does not contain measured data, surrogate models, or restricted information. Coordinates and field values are dimensionless and carry no physical unit.

## 1. Contents

```
public_field/
├─ README.md                       # this file
├─ SPEC.md                         # frozen field specification
├─ params.json                     # frozen parameters
├─ PUBLIC_PACKAGE_SPEC.md          # case / result / CLI formats
├─ ALGORITHM_REGISTRY.md           # algorithm and parameter registry
├─ AUDIT_REPORT.md                 # data, result and logic audit
├─ PUBLIC_RELEASE_MANIFEST.md      # release inventory
├─ CITATION.cff                    # citation metadata
├─ COMPARISON_TABLE.md / .csv      # all-method × N mean comparison
├─ TIMING_TABLE.md                 # single-case and batch=16 timing
├─ CHECKSUMS.txt                   # SHA256 checksums
├─ cases/
│  ├─ MANIFEST.json
│  └─ N1..N8/cases.jsonl           # 8×1000 self-contained cases
├─ code/
│  ├─ field.py                     # public N-D field + case loader
│  ├─ evaluate.py                  # shared evaluator
│  ├─ cotra_j.py                   # standalone CoTra-J
│  ├─ run_batch_cotra_j.py
│  ├─ run_batch.py                 # baseline batch runner
│  ├─ run_single.py                # baseline single-case runner
│  ├─ merge_e5_cotra.py
│  ├─ baselines/                   # 10 optimizers + random
│  └─ test_*.py
├─ data/canonical_3d_seed2026.npz  # illustrative 3-D field
├─ figures/                        # field and result figures
└─ results/
   ├─ README.md
   ├─ FINAL_BY_N.md
   ├─ timing_single.csv
   ├─ E4_batch/N1..N8/             # CoTra-J per-case results
   └─ E5_batch/N1..N8/             # baseline per-case results
```

## 2. Benchmark definition

- Domain: $\mathbf{u}\in[0,100]^N$;
- Value range: $V\in[100,160]$;
- Each case is generated directly from the triple `(N, subset, field_seed)` as an $N$-dimensional field; it is not a slice of an 8-D field;
- The field is a sum of a trend, one-dimensional ripples, localized high-vibration islands / low-vibration pockets, curved low-vibration corridors, and multiplicative fine ripples;
- Per-dimension rate limits are random integers in 2–15; the budget ratio is $T_R=2.5$.

See [SPEC.md](SPEC.md) for the field definition, [PUBLIC_PACKAGE_SPEC.md](PUBLIC_PACKAGE_SPEC.md) for case/result schemas, and [AUDIT_REPORT.md](AUDIT_REPORT.md) for validation.

## 3. Results

Full 1000-case results (independent paired cases per $N$; absolute $J$ is not pooled across $N$). The complete all-method × all-$N$ mean table is in [COMPARISON_TABLE.md](COMPARISON_TABLE.md) and [TIMING_TABLE.md](TIMING_TABLE.md):

| N | CoTra-J | Best baseline | Baseline J | ΔJ | CoTra feasible |
|---|---|---|---|---|---|
| 1 | 130.993 | GA | 134.788 | 3.795 | 1.000 |
| 2 | 127.143 | GA | 130.844 | 3.701 | 1.000 |
| 3 | 123.936 | FMM | 126.932 | 2.996 | 1.000 |
| 4 | 125.070 | GA | 129.049 | 3.979 | 1.000 |
| 5 | 124.040 | GD | 128.148 | 4.108 | 1.000 |
| 6 | 123.783 | GD | 128.173 | 4.390 | 1.000 |
| 7 | 125.153 | GD | 130.027 | 4.874 | 1.000 |
| 8 | 126.223 | GA | 131.714 | 5.491 | 1.000 |

- CoTra-J has the lowest mean $J$ at every $N$ and 100% feasibility;
- Grid planners are marked `grid_too_large` for $N\ge4$; Lattice is marked `too_slow` for $N\ge4$; GD is `too_slow` at $N=8$;
- random is an untuned reference baseline and is not the strongest competitor.

Per-case files are under [results/](results/); the summary is [FINAL_BY_N.md](results/FINAL_BY_N.md). Raw per-case JSONL files are large and are not tracked by default; regenerate them with the commands in Section 5, or use the release assets.

## 4. Installation

```bash
# Python >= 3.9
pip install -r requirements.txt
```

GPU is optional. CoTra-J runs on CPU; batch=16 is faster on a GPU.

## 5. Quick start

### 5.1 Single case: baseline

```bash
python code/run_single.py --method fmm --N 3 --subset 0,3,5 \
  --A 10,80,20 --B 90,15,70 --TR 2.5 --rates 4,11,7 \
  --field-seed 202600017 --out results/single_fmm_N3.json
```

### 5.2 Single case: CoTra-J (Python API)

```python
import torch
from public_field.code import field as F
from public_field.code import cotra_j as CJ

n = 3
f = F.FieldND(n, seed=202600017)
bank = CJ.AnalyticNDFieldBank(
    [lambda pts: torch.as_tensor(f.eval(pts.detach().cpu().numpy()),
                                 device=pts.device, dtype=torch.float32)],
    rpm_max=[100.0] * n,
)
A = torch.tensor([[10.0, 80.0, 20.0]])
B = torch.tensor([[90.0, 15.0, 70.0]])
rates = torch.tensor([[4.0, 11.0, 7.0]])
TR = 2.5
T_b = torch.tensor([TR * ((B - A).abs() / rates).max().item()])
cfg = CJ.CoTraJConfig(K=8, s_samples=4)
points, durations, info = CJ.plan_batch(bank, A, B, T_b, rates, cfg)
print(points.shape, durations, info["J"], info["feasible"])
```

### 5.3 Batch: 1000 cases

```bash
# CoTra-J: N=1..8, batch=16
for n in 1 2 3 4 5 6 7 8; do
  python code/run_batch_cotra_j.py --n $n --batch 16 \
    --out results/E4_batch/N$n/cotra_j_batch.jsonl
done

# 11 baselines
for n in 1 2 3 4 5 6 7 8; do
  python code/run_batch.py --N $n \
    --methods fmm astar theta_star lattice rrt_star bit_star ga gd sa chomp random \
    --out results/E5_batch/N$n --workers 12 --batch-size 16
done

python code/merge_e5_cotra.py
```

For a quick check on a single $N$:

```bash
python code/run_batch.py --N 3 --methods fmm astar ga random --limit 8 \
  --out results/_check_N3
```

## 6. Result format

- `results/E4_batch/N<n>/cotra_j_batch.jsonl`: one JSON record per case, including `J`, `feasible`, `durations`, `points`, `runtime_s`;
- `results/E5_batch/N<n>/per_case_N<n>.jsonl`: merged per-case records for 11 baselines;
- `results/E5_batch/N<n>/<method>.json`: full records for one method;
- `results/timing_single.csv`: absolute single-case and batch=16 times;
- `results/FINAL_BY_N.md`: per-$N$ summary.

See [results/README.md](results/README.md) for field-level details.

## 7. Tests

```bash
python code/test_cotra_j.py
python code/test_baselines_matrix.py
```

Field audit and integrity:

```bash
python code/audit_field.py
```

## 8. Citation and license

- Code: MIT (see [LICENSE](LICENSE));
- Data and results: CC-BY-4.0 (see [LICENSE-DATA.md](LICENSE-DATA.md));
- Citation metadata: see [CITATION.cff](CITATION.cff). A DOI will be added when the archived release is registered.

If you use this benchmark, cite the corresponding paper and record `public-field-1.0.0`, `seed=42`, and the case-file hashes in [cases/MANIFEST.json](cases/MANIFEST.json).
