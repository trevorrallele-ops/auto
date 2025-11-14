# IAU: Run data through the ML pipeline and read results

This document shows the minimal, clear steps to run your IAU daily data through the trained/ trainable models, generate trade-ready signals, and read/interpret the outputs.

Files produced by the pipeline
- `figures/trade_ready_signals.csv` — one row per (model, horizon) with: as_of_date, target_date (YYYY-MM-DD), horizon_days, model, prob, val_auc, action, suggested_size, copy_trade, top_feature_contribs
- `figures/trade_ready_contribs.json` — per-model contribution dicts (SHAP for tree models, permutation fallback for others)
- `figures/analysis_summary.csv` — merged per-model backtest metrics and the trade-ready signals
- `figures/copy_trade_report.txt` — plain-text, grouped copy-trade instructions for the next N days
- `models/<model>_h{h}.joblib` — saved model artifacts (if you retrain)
- `figures/candles_{year}.png` — yearly candlestick charts with BUY/SELL markers

Quick checklist (fast path)
1. Add your IAU daily CSV to the repo root and name it `IAU_daily.csv`.
2. Run the trade-ready generator (recommended: retrain models on IAU):

```bash
python3 trade_ready_signals.py IAU_daily.csv --horizons 1 2 3 4 5 --train-if-missing --perm-repeats 5 --size-mode vol --equity 100000 --target-risk 0.005
```

- `--horizons` controls how many business-day lookaheads to produce (here 1..5).
- `--train-if-missing` will train and save model artifacts for IAU if none exist.
- `--perm-repeats` controls how expensive permutation importances are (higher = more accurate but slower).
- `--size-mode vol` uses ATR-based sizing (recommended) with `--equity` and `--target-risk` to compute suggested size.

3. Create merged analysis and a plain-text copy-trade report:

```bash
python3 make_analysis_summary.py --days 5
```

4. Generate candles with trades:

```bash
python3 plot_candles_with_positions.py
```

(Or patch `plot_candles_with_positions.py` to accept `--csv IAU_daily.csv` if you want plots for IAU specifically.)

How to sanity-check the IAU run (before trading)
- Check `figures/results_summary.csv` and `figures/analysis_summary.csv` for each model's historical backtest metrics (total_return, ann_ret, ann_vol, sharpe). Prefer models with stable positive sharpe and reasonable ann_vol.
- Inspect `figures/trade_ready_signals.csv`:
  - `prob` — predicted up-probability (0..1). Use thresholds (example: >0.55 BUY, <0.45 SELL).
  - `val_auc` — model discrimination on held-out data. Prefer >0.52–0.55.
  - `suggested_size` — shares suggested (if `size-mode vol` uses ATR; otherwise fixed `--size`). Verify it meets your risk rules.
  - `top_feature_contribs` — quick per-row explanation; when SHAP is available for a model you’ll have per-sample contributions.
- View `figures/copy_trade_report.txt` — a human-readable set of one-line instructions grouped by target date for the next days.
- Open the generated candle images `figures/candles_{year}.png` and visually confirm BUY/SELL points are sensible relative to price action.

Interpretation guide — where the insights are
- Signal strength (prob): gives confidence directionally. Combine with model reliability (val_auc, backtest sharpe) for action.
- Model agreement: prefer consensus across top-performing models. Example rule: average of probs across chosen subset (mlp, svm, xgb, lgb); only trade when average passes threshold and at least two models agree.
- Feature contributions (SHAP/permutation): look for consistent contributors across models. If many models point to the same features (e.g., `macd_hist`, `atr_14`) this is a stronger sign.
- Volatility sizing: suggested_size is based on ATR; use it as a baseline and adjust if you have constraints (min/max shares, capital per trade). If you want risk not shares, convert suggested_size * ATR ≈ dollar-risk per trade.

Conservative decision rule (manual copy-trade)
- Conditions to place a trade:
  - Ensemble avg_prob ≥ 0.58 (BUY) or ≤ 0.42 (SELL)
  - At least two trusted models also have prob beyond individual thresholds (≥0.55 or ≤0.45)
  - Those trusted models have val_auc ≥ 0.52
  - No major news or market holidays on target_date
  - Visual confirmation on candle chart (not trading into obvious inverted signal)
- Place an order with stop ≈ 1×ATR (or 1.5×ATR for more cushion) and size = suggested_size (or scaled down by your risk appetite)

Aggressive (warning: more risk)
- Act when a single strong model shows prob ≥ 0.65 or ≤ 0.35, but reduce size (e.g., 50% of suggested_size) and use strict stop.

Troubleshooting & tips
- If `trade_ready_signals.py` errors on missing columns, run `python3 -c "from features import prepare_features; X,y,df = prepare_features('IAU_daily.csv') ; print(X.columns[:20])"` to verify features are created.
- SHAP can be slow on non-tree models and is skipped for SVM/MLP; the script falls back to permutation importance.
- If charts fail to show markers, ensure `figures/trade_ready_signals.csv` has correct `target_date` values that exist in your price index.

Quick commands recap
- Sanity-check features:
```bash
python3 - <<'PY'
from features import prepare_features
X,y,df = prepare_features('IAU_daily.csv', target_horizon=1)
print('rows,cols:', X.shape)
print('sample columns:', X.columns[:20])
PY
```

- Run full signals (recommended):
```bash
python3 trade_ready_signals.py IAU_daily.csv --horizons 1 2 3 4 5 --train-if-missing --perm-repeats 5 --size-mode vol --equity 100000 --target-risk 0.005
```

- Make merged summary & text report:
```bash
python3 make_analysis_summary.py --days 5
```

- View plots in the notebook (interactive): open `/figures/read_figures.ipynb` and run the last plotting cells.

Where to change behavior
- To change thresholds, update `--buy-threshold` and `--sell-threshold` on `trade_ready_signals.py`.
- To change sizing rule (e.g., stop multiplier), edit the sizing code in `trade_ready_signals.py` (search for `size_mode` and ATR usage) or ask me to adjust.

Next steps I can do for you (pick any)
- Create a broker-ready CSV with columns: symbol, side, qty, target_date, order_type.
- Add an interactive Plotly dashboard for filtering models and dates inside the notebook.
- Implement confidence-weighted sizing (scale sizes by |prob-0.5|).

If you'd like, I can add a short README entry to the project root linking to this doc. If you'd like any customization (different thresholds, stop multiple, or broker CSV), tell me which and I will implement it.
