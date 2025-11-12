# Analysis updates — changes made

This report summarizes the changes made on or before 2025-11-12 to support deeper validation and analysis of the GLD ML experiments.

Files added/modified
- `analyses.py` (added): comprehensive analysis script that performs:
  - Permutation test (default 2000 iterations) for MLP strategy
  - Walk-forward CV using sklearn TimeSeriesSplit and `sklearn.base.clone` to avoid joblib deep-copy issues; saves per-fold P&L to `figures/wfcv_fold_{i}.csv`
  - Profit contribution analysis (top contributors CSV)
  - Calibration analysis (bucketed calibration plot)
  - Volatility-based sizing backtest for MLP signal and save perf/CSV
  - Ensemble evaluation (average probability) and creation of `figures/analysis_summary.csv`

- `viz_results.py` (updated): added `--full-retrain` flag and logic to save trained models under `models/`.
- `diagnose_mlp.py` (added): diagnostic plots for the MLP model saved under `figures/mlp/`.
- `GLD_ml_analysis.ipynb` (updated): cells added to display figures, saved models, quick inference, and (later) interactive visualizations.

How to reproduce

1. Full retrain and produce models and figures:

```bash
python viz_results.py GLD_daily.csv --full-retrain
```

2. Run the comprehensive analyses (ensemble, permutation test, walk-forward CV, vol-sizing):

```bash
python analyses.py GLD_daily.csv
```

Outputs
- `figures/` contains multiple analysis artifacts, including:
  - `figures/analysis_ensemble_model_probs.csv`
  - `figures/analysis_ensemble_perf.json`
  - `figures/analysis_ensemble_calib.png`
  - `figures/analysis_ensemble_top_contributors.csv`
  - `figures/analysis_mlp_permutation.png`
  - `figures/wfcv_fold_*.csv` (per-fold P&L for walk-forward CV)
  - `figures/analysis_summary.csv` (summary including ensemble row)

Notes and next steps
- Walk-forward CV now uses `sklearn.clone` and saves fold-level P&L. If you want nested CV or longer training/holding windows, we can parameterize the splitter.
- Permutation test uses 2000 iterations by default; increase to 10k if you need finer p-values (will take longer).
- I can add an interactive Plotly cell to the notebook to let you filter top contributors by date range and inspect trades.

If you'd like, I can now: (a) run a longer permutation test, (b) add interactive notebook cells (Plotly widgets), or (c) implement volatility-based position sizing variants.
