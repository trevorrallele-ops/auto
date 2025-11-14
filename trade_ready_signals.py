#!/usr/bin/env python3
"""
Produce trade-ready probability signals per model and horizon and compute feature contributions.

Saves CSV to `figures/trade_ready_signals.csv` and per-model contribution JSONs under `figures/`.

Usage:
  python trade_ready_signals.py GLD_daily.csv --horizons 1 2 3 --no-shap

If SHAP is available the script will compute per-sample SHAP values; otherwise it falls back to
permutation importances (global) as a proxy.
"""
import argparse
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pandas.tseries.offsets import BDay

from features import prepare_features
from models import build_classifiers


def ensure_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def get_target_column(df):
    # find a column that looks like the target (starts with 'target' or is 'y')
    for c in df.columns:
        if c.lower().startswith("target") or c.lower() == "y":
            return c
    raise ValueError("No target column found in features dataframe")


def safe_predict_proba(model, X):
    # Return probability for class 1 for rows in X
    try:
        proba = model.predict_proba(X)
        # in case binary, take column for positive class
        if proba.shape[1] == 2:
            return proba[:, 1]
        # else if multi, assume positive is last
        return proba[:, -1]
    except Exception:
        # fallback: use decision_function and sigmoid
        try:
            scores = model.decision_function(X)
            # convert to prob via logistic sigmoid
            return 1.0 / (1.0 + np.exp(-scores))
        except Exception:
            # last resort: predict -> 0/1
            preds = model.predict(X)
            return preds.astype(float)


def compute_permutation_importance(model, X, y, n_repeats=20, random_state=42):
    from sklearn.inspection import permutation_importance

    try:
        r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=1)
        importances = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
        return importances
    except Exception as e:
        warnings.warn(f"Permutation importance failed: {e}")
        return pd.Series([], dtype=float)


def try_shap_values(model, X_train, X_row):
    # Return a Series mapping feature->shap value for the single row X_row
    try:
        import shap

        # Try to extract estimator if pipeline
        estimator = getattr(model, "named_steps", {}).get(list(model.named_steps.keys())[-1], model)

        # Only compute SHAP for tree-based estimators (fast). For others, skip and fall back.
        est_name = estimator.__class__.__name__.lower()
        tree_like_indicators = ("tree", "xgb", "lightgbm", "lgb", "randomforest", "hist")
        if any(tok in est_name for tok in tree_like_indicators):
            expl = shap.TreeExplainer(estimator)
        else:
            # skip expensive kernel/linear explainers for non-tree models (e.g., SVM)
            raise RuntimeError(f"SHAP skipped for non-tree estimator: {est_name}")

        vals = expl.shap_values(X_row)
        # shap returns list per class for some explainers, handle that
        if isinstance(vals, list):
            # pick last class contributions if multi
            shap_vals = np.array(vals[-1])
        else:
            shap_vals = np.array(vals)

        if shap_vals.ndim == 2:
            # (n_rows, n_features) -> take first row
            shap_row = shap_vals[0]
        else:
            shap_row = shap_vals

        return pd.Series(shap_row, index=X_row.columns)
    except Exception as e:
        warnings.warn(f"SHAP failed or not available: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV file with historical data (daily).")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1], help="Target horizons in days (1,2,...)")
    parser.add_argument("--buy-threshold", type=float, default=0.55, help="Probability threshold above which to suggest BUY")
    parser.add_argument("--sell-threshold", type=float, default=0.45, help="Probability threshold below which to suggest SELL")
    parser.add_argument("--size", type=int, default=100, help="Suggested number of shares for copy-trade helper")
    parser.add_argument("--size-mode", choices=["fixed","vol"], default="fixed", help="Sizing mode: fixed shares or volatility-based (uses ATR)")
    parser.add_argument("--equity", type=float, default=100000.0, help="Account equity used for vol-based sizing")
    parser.add_argument("--target-risk", type=float, default=0.005, help="Fractional risk per trade for vol-based sizing (e.g., 0.005 = 0.5%)")
    parser.add_argument("--perm-repeats", type=int, default=5, help="Number of repeats for permutation importance (smaller is faster)")
    parser.add_argument("--models", nargs="*", default=None, help="Which models to include (default: all from build_classifiers)")
    parser.add_argument("--symbol", type=str, default=None, help="Ticker/symbol to include in broker export (optional)")
    parser.add_argument("--stop-mult", type=float, default=1.5, help="Multiplier for ATR to compute suggested stop price")
    parser.add_argument("--limit-mult", type=float, default=2.0, help="Multiplier for ATR to compute suggested limit/target price")
    parser.add_argument("--consensus", action="store_true", help="Also write a consensus broker_orders_consensus.csv with one row per target_date")
    parser.add_argument("--no-shap", dest="shap", action="store_false", help="Disable SHAP and use permutation fallback")
    parser.add_argument("--train-if-missing", action="store_true", help="Train models if saved ones are not found (default: False)")
    parser.add_argument("--out", default="figures/trade_ready_signals.csv", help="Output CSV path")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    classifiers_template = build_classifiers(random_state=42)
    model_names = args.models if args.models else list(classifiers_template.keys())

    rows = []
    contrib_store = {}

    for h in args.horizons:
        print(f"Processing horizon={h}")
        # prepare_features returns (X, y, df)
        X_all, y_all, df_all = prepare_features(args.csv, target_horizon=h)
        # align and drop NA
        X_all = X_all.dropna()
        y_all = y_all.loc[X_all.index]

        if X_all.shape[0] < 50:
            warnings.warn(f"Not enough rows after feature preparation for horizon {h}, skipping")
            continue

        feature_cols = list(X_all.columns)

        # time-based split: first 80% train, last 20% test
        split_idx = int(len(X_all) * 0.8)
        X_train = X_all.iloc[:split_idx].copy()
        y_train = y_all.iloc[:split_idx].copy()
        X_test = X_all.iloc[split_idx:].copy()
        y_test = y_all.iloc[split_idx:].copy()

        # latest sample to produce signal for next-day/horizon
        X_row = X_test.iloc[[-1]]
        signal_date = X_test.index[-1] if isinstance(X_test.index, pd.DatetimeIndex) else X_test.index[-1]

        for name in model_names:
            if name not in classifiers_template:
                warnings.warn(f"Model {name} not found in available classifiers; skipping")
                continue

            clf_template = classifiers_template[name]
            model_file = Path(f"models/{name}_h{h}.joblib")
            model = None
            if model_file.exists():
                try:
                    model = joblib.load(model_file)
                except Exception as e:
                    warnings.warn(f"Failed to load {model_file}: {e}")

            if model is None:
                if args.train_if_missing:
                    print(f"Training model {name} for horizon {h} (will save to {model_file})")
                    model = clf_template
                    model.fit(X_train, y_train)
                    ensure_dir(model_file)
                    joblib.dump(model, model_file)
                else:
                    print(f"Model file {model_file} missing; skipping {name} for horizon {h} (use --train-if-missing to train)")
                    continue

            # predict probabilities and compute validation AUC where possible
            prob_row = float(safe_predict_proba(model, X_row)[0])
            try:
                val_proba = safe_predict_proba(model, X_test)
                val_auc = float(roc_auc_score(y_test, val_proba))
            except Exception:
                val_auc = float('nan')

            # contributions: try SHAP first
            contrib = None
            if args.shap:
                contrib = try_shap_values(model, X_train, X_row)

            if contrib is None:
                # fallback to permutation importance on test set (global)
                imp = compute_permutation_importance(model, X_test, y_test, n_repeats=args.perm_repeats)
                if not imp.empty:
                    # normalize
                    imp = imp.reindex(feature_cols).fillna(0.0)
                    contrib = imp / (imp.abs().sum() + 1e-12)
                else:
                    contrib = pd.Series(0.0, index=feature_cols)

            # store contributions as comma-separated top features
            top_feats = ", ".join([f"{f}:{v:.4f}" for f, v in contrib.sort_values(key=lambda s: s.abs(), ascending=False).head(10).items()])

            # compute the actual target calendar/business date for this horizon
            try:
                as_of_dt = pd.to_datetime(signal_date)
                # remove tz if present and get date
                try:
                    as_of_dt = as_of_dt.tz_convert(None)
                except Exception:
                    try:
                        as_of_dt = as_of_dt.tz_localize(None)
                    except Exception:
                        pass
                as_of_date = as_of_dt.date()
                target_date = (pd.to_datetime(as_of_date) + BDay(h)).date()
            except Exception:
                # fallback: use string offset if index isn't datetime
                as_of_date = str(signal_date)
                target_date = as_of_date

            # simple copy-trade helper based on probability thresholds
            prob = prob_row
            if prob >= args.buy_threshold:
                action = "BUY"
            elif prob <= args.sell_threshold:
                action = "SELL"
            else:
                action = "HOLD"

            # suggested size: fixed or volatility-based using ATR
            suggested_size = args.size
            try:
                # df_all contains ATR as 'atr_14' from prepare_features
                atr_val = None
                # X_row index corresponds to df_all index
                row_idx = X_row.index[0]
                if 'atr_14' in df_all.columns:
                    atr_val = float(df_all.loc[row_idx, 'atr_14'])
                price_val = float(X_row[ X_row.columns[0] ]) if X_row.shape[1] > 0 else None
                # better get price column from df_all
                price_col = 'Adj Close' if 'Adj Close' in df_all.columns else 'Close'
                price_val = float(df_all.loc[row_idx, price_col])
                if args.size_mode == 'vol' and atr_val is not None and atr_val > 0:
                    # per-share dollar risk approximated by ATR
                    risk_per_trade = args.equity * args.target_risk
                    suggested_size = int(max(1, (risk_per_trade) / (atr_val)))
            except Exception:
                suggested_size = args.size

            if action == 'BUY' or action == 'SELL':
                copy_trade = f"{action} {suggested_size} @market (prob={prob:.3f})"
            else:
                copy_trade = f"HOLD (prob={prob:.3f})"

            # compute suggested stop/limit prices using ATR multipliers
            stop_price = None
            limit_price = None
            try:
                if atr_val is not None and price_val is not None:
                    if action == 'BUY':
                        stop_price = float(max(0.0, price_val - args.stop_mult * atr_val))
                        limit_price = float(price_val + args.limit_mult * atr_val)
                    elif action == 'SELL':
                        # for sell (short) stop is above entry, limit is below
                        stop_price = float(price_val + args.stop_mult * atr_val)
                        limit_price = float(max(0.0, price_val - args.limit_mult * atr_val))
            except Exception:
                stop_price = None
                limit_price = None

            rows.append({
                "as_of_date": as_of_date.isoformat() if hasattr(as_of_date, 'isoformat') else str(as_of_date),
                "target_date": target_date.strftime("%Y-%m-%d") if hasattr(target_date, 'strftime') else str(target_date),
                "horizon_days": h,
                "model": name,
                "prob": prob_row,
                "val_auc": val_auc,
                "action": action,
                "stop_price": stop_price,
                "limit_price": limit_price,
                "suggested_size": suggested_size,
                "copy_trade": copy_trade,
                "top_feature_contribs": top_feats,
            })

            contrib_store[f"{name}_h{h}"] = contrib.to_dict()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    # save contributions as JSON
    contrib_file = out_path.parent / "trade_ready_contribs.json"
    pd.Series(contrib_store).to_json(contrib_file)

    # optionally write a single consensus broker-orders CSV per target date
    if args.consensus:
        try:
            cons_rows = []
            # group by target_date and aggregate
            g = out_df.groupby('target_date')
            for tgt, grp in g:
                avg_prob = float(grp['prob'].mean())
                models_included = int(len(grp))
                # decide consensus action by average probability
                if avg_prob >= args.buy_threshold:
                    side = 'BUY'
                elif avg_prob <= args.sell_threshold:
                    side = 'SELL'
                else:
                    side = 'HOLD'

                # suggested size: mean of sizes for non-HOLD rows, fallback to median overall
                sizes = grp.loc[grp['action'] != 'HOLD', 'suggested_size']
                if len(sizes) > 0:
                    qty = int(max(1, int(sizes.mean())))
                else:
                    qty = int(max(1, int(grp['suggested_size'].median())))

                # pick stop/limit by averaging non-null values
                stop_vals = grp['stop_price'].dropna()
                limit_vals = grp['limit_price'].dropna()
                stop_price_cons = float(stop_vals.mean()) if len(stop_vals) > 0 else None
                limit_price_cons = float(limit_vals.mean()) if len(limit_vals) > 0 else None

                cons_rows.append({
                    'symbol': args.symbol if args.symbol else '',
                    'target_date': tgt,
                    'side': side,
                    'qty': qty,
                    'order_type': 'market' if side != 'HOLD' else 'none',
                    'stop_price': stop_price_cons,
                    'limit_price': limit_price_cons,
                    'models_included': models_included,
                    'avg_prob': avg_prob,
                })

            cons_df = pd.DataFrame(cons_rows)
            cons_file = out_path.parent / 'broker_orders_consensus.csv'
            cons_df.to_csv(cons_file, index=False)
            print('Wrote consensus broker orders:', cons_file)
        except Exception as e:
            warnings.warn(f'Failed to write consensus broker orders: {e}')

    print("Wrote:", out_path)
    print("Contributions:", contrib_file)
    print(out_df)


if __name__ == '__main__':
    main()
