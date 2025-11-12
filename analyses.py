"""Comprehensive analyses for GLD ML experiments.

Functions included:
 - permutation_test: permutation p-value for observed strategy return
 - walk_forward_cv: time-series CV aggregated P&L for specified estimator
 - profit_contribution: show top N dates/trades contributing to P&L
 - calibration_analysis: bucket predicted probabilities and compute observed win-rate
 - vol_sizing_backtest: volatility-based sizing backtest with simple transaction costs
 - ensemble_eval: average-proba ensemble over saved models

Saves outputs under `figures/analysis_*` and CSV summaries under `figures/`.
"""
from __future__ import annotations

import os
import glob
import json
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.base import clone

from features import prepare_features
from backtest import backtest_advanced


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def backtest_position(prices: pd.Series, position: pd.Series, transaction_cost: float = 0.0005, slippage: float = 0.0005):
    """Backtest given daily position weights (can be fractional). Returns df and perf."""
    prices = prices.sort_index()
    position = position.reindex(prices.index).fillna(0).astype(float)
    ret = prices.pct_change().fillna(0)
    pnl = position.shift(0).fillna(0) * ret
    turnover = (position - position.shift(1)).abs().fillna(position.abs())
    pnl = pnl - turnover * (transaction_cost + slippage)
    cum = (1 + pnl).cumprod()
    total_return = cum.iloc[-1] - 1
    ann_ret = (1 + total_return) ** (252 / len(pnl)) - 1 if len(pnl) > 0 else np.nan
    ann_vol = pnl.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    df = pd.DataFrame({"price": prices, "position": position, "pnl": pnl, "cum": cum})
    perf = {"total_return": float(total_return), "ann_ret": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe)}
    return df, perf


def permutation_test(price: pd.Series, signal: pd.Series, n_iter: int = 2000, seed: int = 42):
    """Shuffle the signal and compute distribution of total returns; return p-value."""
    rng = np.random.default_rng(seed)
    observed_df, observed_perf = backtest_position(price, signal)
    obs = observed_perf["total_return"]
    sims = []
    arr = signal.values
    for i in range(n_iter):
        perm = rng.permutation(arr)
        _, perf = backtest_position(price, pd.Series(perm, index=signal.index))
        sims.append(perf["total_return"])
    sims = np.array(sims)
    pval = (np.sum(sims >= obs) + 1) / (len(sims) + 1)
    return obs, sims, pval


def walk_forward_cv(csv: str = "GLD_daily.csv", estimator_name: str = "mlp", n_splits: int = 5):
    """Run TimeSeriesSplit walk-forward CV using saved model type name; retrains model on each fold and collects P&L."""
    X, y, df = prepare_features(csv)
    tss = pd.Series(index=X.index)
    from sklearn.model_selection import TimeSeriesSplit
    from models import build_classifiers

    clf_map = build_classifiers()
    if estimator_name not in clf_map:
        raise ValueError(f"Estimator {estimator_name} not found")
    clf_template = clf_map[estimator_name]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    per_fold = []
    fold_idx = 0
    ensure_dir('figures/wfcv')
    for train_idx, test_idx in tscv.split(X):
        fold_idx += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        mdl = clone(clf_template)
        mdl.fit(X_train, y_train)
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(X_test)[:, 1]
        else:
            proba = mdl.predict(X_test)
        sig = pd.Series((proba > 0.5).astype(int), index=X_test.index)
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        price = df[price_col].loc[X_test.index]
        bt_df, perf = backtest_position(price, sig)
        # save fold-level pnl series
        bt_df.to_csv(f'figures/wfcv_fold_{fold_idx}.csv')
        per_fold.append({"fold": fold_idx, **perf, "pnl_file": f'figures/wfcv_fold_{fold_idx}.csv'})
    return per_fold


def profit_contribution(bt_df: pd.DataFrame, top_n: int = 20):
    """Return top N dates by pnl contribution and cumulative share."""
    daily = bt_df["pnl"].dropna()
    top = daily.sort_values(ascending=False).head(top_n)
    return top, top.sum(), daily.sum()


def calibration_analysis(y_true: pd.Series, proba: np.ndarray, bins: int = 10, out_prefix: str = "figures/analysis_calib"):
    ensure_dir(os.path.dirname(out_prefix) or "figures")

    df = pd.DataFrame({"y": y_true, "p": proba}, index=y_true.index)
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates='drop')
    agg = df.groupby("bin").agg(mean_p=("p", "mean"), obs_rate=("y", "mean"), count=("y", "size"))
    agg = agg.reset_index()
    plt.figure(figsize=(6, 6))
    plt.plot(agg["mean_p"], agg["obs_rate"], marker="o")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration plot")
    plt.tight_layout()
    plt.savefig(out_prefix + ".png")
    plt.close()
    return agg


def vol_sizing_signal_backtest(price: pd.Series, signal: pd.Series, window: int = 20, target_annual_vol: float = 0.08):
    """Compute position sizes based on rolling realized vol and backtest the scaled positions."""
    ret = price.pct_change().fillna(0)
    realized_vol = ret.rolling(window).std() * np.sqrt(252)
    size = target_annual_vol / (realized_vol + 1e-9)
    size = size.clip(0, 2.0)  # cap leverage
    # position is signal (0/1) times size shifted to use previous vol
    position = signal * size
    df, perf = backtest_position(price, position)
    return df, perf


def ensemble_eval(csv: str = "GLD_daily.csv"):
    X, y, df = prepare_features(csv)
    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    price = df[price_col].loc[X_test.index]

    model_files = glob.glob("models/*.joblib")
    probs = {}
    for mf in model_files:
        name = os.path.splitext(os.path.basename(mf))[0]
        mdl = joblib.load(mf)
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(X_test)[:, 1]
            probs[name] = proba
    if not probs:
        raise RuntimeError("No probabilistic models found to ensemble")
    proba_df = pd.DataFrame(probs, index=X_test.index)
    # simple average ensemble
    avg_proba = proba_df.mean(axis=1)
    signal = (avg_proba > 0.5).astype(int)
    bt_df, perf = backtest_position(price, pd.Series(signal, index=X_test.index))
    # calibration for ensemble
    calib = calibration_analysis(y_test, avg_proba, bins=10, out_prefix="figures/analysis_ensemble_calib")
    ensure_dir("figures")
    proba_df.to_csv("figures/analysis_ensemble_model_probs.csv")
    with open("figures/analysis_ensemble_perf.json", "w") as f:
        json.dump(perf, f, indent=2)
    # also append ensemble metrics to a summary CSV if results_summary exists
    # also append ensemble metrics to a summary CSV (merge with results_summary if present)
    try:
        if os.path.exists('figures/results_summary.csv'):
            base = pd.read_csv('figures/results_summary.csv', index_col=0)
        else:
            base = pd.DataFrame()
        ens_row = pd.Series(perf, name='ensemble')
        base = base.copy()
        # ensure consistent columns
        for c in ens_row.index:
            if c not in base.columns:
                base[c] = pd.NA
        # if base has extra columns, keep them but fill missing with NA for ensemble
        base.loc['ensemble'] = [ens_row.get(c, pd.NA) for c in base.columns]
        base.to_csv('figures/analysis_summary.csv')
    except Exception as e:
        print('Failed to update analysis_summary.csv:', e)
    return perf, bt_df, calib


def run_all(csv: str = "GLD_daily.csv"):
    ensure_dir("figures")
    print("Running ensemble evaluation...")
    try:
        perf, bt_df, calib = ensemble_eval(csv)
        print("Ensemble perf:", perf)
    except Exception as e:
        print("Ensemble failed:", e)

    # diagnose MLP signal permutation test
    print("Running MLP permutation test...")
    mdl_path = "models/mlp.joblib"
    if os.path.exists(mdl_path):
        print("Loading MLP")
        mdl = joblib.load(mdl_path)
        X, y, df = prepare_features(csv)
        split = int(len(X) * 0.8)
        X_test = X.iloc[split:]
        y_test = y.iloc[split:]
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        price = df[price_col].loc[X_test.index]
        proba = mdl.predict_proba(X_test)[:, 1] if hasattr(mdl, "predict_proba") else None
        signal = pd.Series((proba > 0.5).astype(int) if proba is not None else mdl.predict(X_test), index=X_test.index)
        obs, sims, pval = permutation_test(price, signal, n_iter=2000)
        print(f"MLP observed total_return={obs:.4f}, permutation p-value={pval:.4f}")
        # save sim histogram
        plt.figure(figsize=(6, 3))
        plt.hist(sims, bins=50)
        plt.axvline(obs, color='red', lw=2, label='observed')
        plt.legend()
        plt.title('Permutation distribution of total_return (MLP)')
        plt.tight_layout()
        plt.savefig('figures/analysis_mlp_permutation.png')
        plt.close()

    # nested walk-forward CV aggregated across multiple models
    print("Running nested walk-forward CV across all classifiers...")
    try:
        from models import build_classifiers
        clf_map = build_classifiers()
        all_wfcv = []
        for name in clf_map.keys():
            try:
                print(f"WFCV for {name}...")
                per_fold = walk_forward_cv(csv, estimator_name=name, n_splits=5)
                for r in per_fold:
                    r['model'] = name
                all_wfcv.extend(per_fold)
            except Exception as e:
                print(f"WFCV failed for {name}: {e}")
        if all_wfcv:
            df_wfcv = pd.DataFrame(all_wfcv)
            df_wfcv.to_csv('figures/analysis_all_models_walkforward.csv', index=False)
            print('Saved aggregated walk-forward CV to figures/analysis_all_models_walkforward.csv')
    except Exception as e:
        print('Nested walk-forward CV failed:', e)

    # profit contribution for ensemble
    try:
        ensemble_perf_file = 'figures/analysis_ensemble_perf.json'
        if os.path.exists(ensemble_perf_file):
            print('Analyzing ensemble profit contributors...')
            # reload ensemble backtest to compute daily pnl
            _, bt_df, _ = ensemble_eval(csv)
            top, top_sum, total = profit_contribution(bt_df, top_n=20)
            top.to_csv('figures/analysis_ensemble_top_contributors.csv')
            print('Top contributors saved to figures/analysis_ensemble_top_contributors.csv')
    except Exception as e:
        print('Profit contribution analysis failed:', e)

    # volatility sizing backtest using MLP signal
    try:
        if os.path.exists(mdl_path):
            print('Vol-sizing backtest (MLP)')
            df_vol, perf_vol = vol_sizing_signal_backtest(price, signal, window=20, target_annual_vol=0.08)
            df_vol.to_csv('figures/analysis_mlp_volsizing.csv')
            with open('figures/analysis_mlp_volsizing_perf.json', 'w') as f:
                json.dump(perf_vol, f, indent=2)
            print('Saved vol sizing results')
    except Exception as e:
        print('Vol sizing backtest failed:', e)


if __name__ == '__main__':
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else 'GLD_daily.csv'
    run_all(csv)
