# GLD ML Experiments

Small experiment scaffold to train a battery of ML models on GLD historical data and run a simple signal backtest.

Files added:

- `features.py` — loads `GLD_daily.csv`, computes technical indicators and prepares X/y.
- `models.py` — builds and trains a set of classifiers (logistic, rf, svm, knn, mlp, xgb, lgb if installed).
- `backtest.py` — simple position-based backtest and Sharpe calculation.
- `run_experiments.py` — orchestrates the pipeline and saves results to `results_summary.json`.
- `requirements.txt` — suggested Python packages.

How to run (in the workspace root):

```bash
# create a venv and install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run experiments
python run_experiments.py GLD_daily.csv
```

Notes:
- The scripts guard heavy imports where possible; if `xgboost` or `lightgbm` are not installed, those models are skipped.
- The backtest is intentionally simple — it's for quick comparison of signals, not a production trading engine.

Next steps you might want:
- Add walk-forward cross-validation and hyperparameter tuning.
- Expand targets (multi-day return/regression) and include transaction slippage/position sizing.
- Add an interactive notebook for plotting signals and equity curves (I can add one if you want).
# IAU workflow and docs

If you have IAU (or another ETF) daily data and want to run the same pipeline, see `docs/IAU_workflow.md` for a step-by-step guide: preparing the CSV, retraining models, generating trade-ready signals, reading outputs, and exporting broker-ready orders.

You can also run the tools in this repo to produce `figures/trade_ready_signals.csv`, `figures/analysis_summary.csv`, and `figures/broker_orders.csv` for quick manual copy-trading.
# auto