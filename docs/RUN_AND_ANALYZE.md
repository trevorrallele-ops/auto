# How to run the project and analyze results

This document gives a clear, ordered set of commands to run the codebase from raw daily CSV data to trade-ready outputs and analyses, followed by a practical guide for reading and interpreting the results.

Prerequisites
- Python 3.10+ (the dev container uses Ubuntu + Python 3.x)
- Create and activate a virtual environment, then install required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# additional packages used by some scripts (if not in requirements)
pip install shap mplfinance plotly
```

Quick file list (most important scripts)
- `features.py` — feature engineering and `prepare_features(path, target_horizon)`
- `models.py` — build classifier pipelines
- `backtest.py` — simple & advanced backtests
- `run_experiments.py` — baseline end-to-end run (train simple models, basic backtest)
- `run_tuned_experiments.py` — randomized tuning (TimeSeriesSplit)
- `viz_results.py` — generate figures & summary CSVs (has `--full-retrain` option)
- `trade_ready_signals.py` — create per-model, per-horizon probabilistic signals and contributions
- `make_analysis_summary.py` — merge signals with backtest metrics and produce text report
- `export_broker_orders.py` — export broker-ready CSV (`figures/broker_orders.csv`)
- `plot_candles_with_positions.py` — create yearly candlestick PNGs with trade markers
- `analyses.py` — permutation tests, walk-forward CV, vol-sizing backtests and ensemble evaluation
- `diagnose_mlp.py` — MLP-specific diagnostics

1. Minimal end-to-end run (fast smoke test)

1. Prepare your CSV: place `GLD_daily.csv` (or `IAU_daily.csv`) in the repository root. Ensure header includes Date, Close (or Adj Close), Open, High, Low, Volume.
2. Run the baseline experiment (quick):

```bash
python run_experiments.py GLD_daily.csv
```

This trains the light models, runs simple backtests, and writes `results_summary.csv` / `results_summary.json` and some plots to `figures/`.

2. Recommended full-run (train all models, produce trade-ready signals)

```bash
# train and produce visualizations (this may retrain heavy models and take several minutes)
python viz_results.py GLD_daily.csv --full-retrain

# then produce trade-ready signals (1..5 horizons) with SHAP where available
python trade_ready_signals.py GLD_daily.csv --horizons 1 2 3 4 5 --train-if-missing --perm-repeats 5

# create merged analysis & textual report for next 5 days
python make_analysis_summary.py --days 5

# export broker-ready CSV (one row per model signal)
python export_broker_orders.py --symbol GLD --input figures/trade_ready_signals.csv --out figures/broker_orders.csv

# plot candles with overlaid signals (saves candlesticks per year under figures/)
python plot_candles_with_positions.py
```

Notes:
- Add `--no-shap` to `trade_ready_signals.py` to skip SHAP (faster); SHAP is already installed if you followed prerequisites.
- Use `--train-if-missing` to force training models for the current dataset if model artifacts are absent.
- Change `--perm-repeats` to control permutation-importance accuracy vs runtime.

3. Deeper validation and analyses

- Run nested / aggregated analyses:

```bash
python analyses.py GLD_daily.csv
```

This script will run ensemble evaluation, MLP permutation tests, nested walk-forward CV (over models) and vol-sizing backtests; it writes many artifacts under `figures/` (files described below).

- Run MLP specific diagnostics:

```bash
python diagnose_mlp.py GLD_daily.csv
```

4. Where outputs are and what they mean

- `figures/results_summary.csv` — per-model backtest & classification metrics (accuracy, f1, auc, total_return, ann_ret, ann_vol, sharpe). Use it to quickly compare historical simulated performance.
- `figures/trade_ready_signals.csv` — core signal outputs; columns:
  - `as_of_date` — the date of the last observed bar used for prediction
  - `target_date` — the calendar business date the model predicts for (YYYY-MM-DD)
  - `horizon_days` — lookahead days
  - `model` — model name
  - `prob` — predicted probability (0..1) of the target being 'up'
  - `val_auc` — validation AUC on held-out split
  - `action` — derived label (BUY/SELL/HOLD) using thresholds
  - `suggested_size` — numeric size (shares) computed either fixed or `vol` (ATR) mode
  - `copy_trade` — one-line copy-paste instruction
  - `top_feature_contribs` — top features contributing to the prediction (SHAP or permutation)
- `figures/trade_ready_contribs.json` — full per-model contribution dictionaries
- `figures/analysis_summary.csv` — merged table combining `results_summary.csv` metrics with the signals (handy for filtering signals by model historical performance)
- `figures/copy_trade_report.txt` — human-readable, grouped report of upcoming orders
- `figures/broker_orders.csv` — broker-ready orders (symbol, side, qty, order_type, target_date, model)
- `figures/candles_{year}.png` — yearly candlestick PNGs with BUY/SELL markers plotted at target dates

5. How to read the results — a short primer

Start here: `figures/trade_ready_signals.csv` and `figures/analysis_summary.csv`.

- Probability (`prob`): model's confidence in an up move. Treat it as signal strength, not a guarantee.
- Validation AUC (`val_auc`): prefer models with val_auc > 0.52–0.55; lower values indicate weak discrimination.
- Backtest metrics (`results_summary.csv`): look at `sharpe`, `ann_ret`, `ann_vol` to evaluate risk-adjusted historic performance.
- Feature contributions: SHAP (preferred for tree models) tells you which features moved the prediction for that sample. Permutation importances give a global sense of feature importance.

Simple decision rules (examples)
- Conservative: trade only when ensemble `avg_prob` ≥ 0.58 or ≤ 0.42 AND at least two trusted models agree AND val_auc ≥ 0.52 for those models.
- Aggressive: single-model strong signal with prob ≥ 0.65 (use reduced size and strict stop).

Sanity checklist before placing orders
1. Confirm `target_date` is an actual trading day (no holidays).
2. Visual check on `figures/candles_{year}.png` for the target date.
3. Confirm `suggested_size` aligns with your capital and risk rules.
4. If unsure, skip or reduce size.

6. Useful commands & snippets

- Quick peek at top signals:
```bash
head -n 20 figures/trade_ready_signals.csv | column -t -s,
```

- Compute ensemble average per target date (python snippet):
```python
import pandas as pd
df = pd.read_csv('figures/trade_ready_signals.csv', parse_dates=['target_date'])
ens = df.groupby('target_date')['prob'].mean()
print(ens)
```

- Export broker-ready CSV (already provided):
```bash
python export_broker_orders.py --symbol GLD --input figures/trade_ready_signals.csv --out figures/broker_orders.csv
```

7. Troubleshooting

- Missing columns on feature generation: run the feature check:
```bash
python - <<'PY'
from features import prepare_features
X,y,df = prepare_features('GLD_daily.csv', target_horizon=1)
print(X.columns[:40])
PY
```
- If SHAP is slow: use `--no-shap` or restrict SHAP to tree-based models (script already does this).
- If plots fail: ensure `figures/trade_ready_signals.csv` has `target_date` values that appear in your price index.

8. Next improvements you can request
- Broker CSV with stop and limit price columns (I can compute suggested stop = price ± n*ATR).
- Consensus aggregation instead of one row per model (majority vote / avg prob) for a single broker order per date.
- Interactive Plotly dashboard for filtering models, thresholds, and date ranges.
- Nested hyperparameter tuning per fold (robust but compute intensive).

If you'd like, I can add one of these next steps and re-run the pipeline for your chosen dataset (GLD or IAU). 

---
Saved: `docs/RUN_AND_ANALYZE.md`
