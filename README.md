# TailMender

Reference implementation of **TailMender**, a batch tabular imbalanced-regression method that trains a bulk teacher on uniformly weighted data and a residual-zero tail-student ensemble that corrects rare targets while anchoring predictions in the dense region. Final predictions combine the teacher mean with an aggregated student residual selected by held-out calibration over rare-tail and bulk accuracy.

## Requirements

- Python 3.10+
- See `requirements.txt` for Python packages (NumPy, pandas, scikit-learn, XGBoost, PyYAML, and supporting libraries).

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start (smoke test)

From the repository root (`2_Code/`), run a small sanity check on one benchmark:

```powershell
python scripts\run_radical_smoke.py --profile smoke --datasets abalone --seeds 42 --n-estimators 12 --max-train 300 --max-test 120 --output-dir results\smoke
```

On Linux or macOS, use forward slashes:

```bash
python scripts/run_radical_smoke.py --profile smoke --datasets abalone --seeds 42 --n-estimators 12 --max-train 300 --max-test 120 --output-dir results/smoke
```

The script writes metrics to the output directory (CSV and JSON summaries). Those files are local artifacts and are not part of the source tree.

## Main experiment script

`scripts/run_radical_smoke.py` evaluates TailMender against published imbalanced-regression baselines on the UAI9 benchmark suite.

**All nine datasets (default):**

```bash
python scripts/run_radical_smoke.py --profile full --seeds 42,43,44 --n-estimators 80 --output-dir results/uai9_full
```

**Subset of datasets:**

```bash
python scripts/run_radical_smoke.py --profile full --datasets abalone,california_housing,insurance --seeds 42 --output-dir results/subset
```

**Methods included:** `No-Rebalance`, `StaticTailWeight`, `InverseDensityWeight`, `LDSWeight`, `RandomTailReplay`, `SMOTER-Batch`, `SMOGN-Batch`, `REBAGG-Resample`, and `TailMender`.

Evaluation protocol: chronological 70/30 train/test split, features and targets standardized, rare region defined as the bottom 10% and top 10% of training targets (fixed thresholds), gradient-boosted tree regressors (XGBoost when available).

## UAI9 datasets

The nine benchmarks follow the UAI TabOversample regression roster. Data are fetched at runtime from public sources (OpenML, scikit-learn bundled datasets, and other open repositories) via `scripts/run_router_pilot.py`; no proprietary files are bundled in this repository.

| Dataset | Rows (approx.) | Features (approx.) |
|---------|----------------:|-------------------:|
| abalone | 4,177 | 8 |
| cpu_activity | 8,192 | 21 |
| california_housing | 20,640 | 8 |
| house_16h | 22,784 | 16 |
| insurance | 1,338 | 6 |
| ames_housing | 2,930 | 80 |
| ibm_employee | 1,470 | 34 |
| melbourne_housing | 27,247 | 18 |
| bike_sharing | 17,379 | 12 |

An internet connection is required on first download for datasets that are not cached locally by scikit-learn or OpenML.

## Reproducing results

1. Install dependencies with `pip install -r requirements.txt`.
2. Run `scripts/run_radical_smoke.py` with `--profile full`, three seeds (`--seeds 42,43,44`), and default dataset list. Full runs are CPU-intensive; allow several hours on a multi-core machine.
3. Inspect `radical_full_results.csv` and `radical_full_summary.json` in your chosen `--output-dir`. Rank summaries aggregate per-dataset rare-region and overall metrics across methods.

Hyperparameters (ensemble grid, calibration objective, small-sample fallback) are defined in `scripts/run_radical_smoke.py`.

## Repository layout

```
scripts/
  run_radical_smoke.py   # TailMender + baselines on UAI9
  run_router_pilot.py    # dataset loaders and shared utilities
src/
  evaluation/metrics.py  # rare-region metrics
requirements.txt
```

## Contact

Nathaniel Kang — School of Computer Science and Engineering, Kyungpook National University — natekang@knu.ac.kr
