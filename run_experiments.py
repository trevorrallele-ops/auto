"""Run experiments across multiple ML models on GLD data.

Usage: python run_experiments.py [path_to_csv]

This script:
 - loads and prepares features
 - splits data by time (train up to 80%, test last 20%)
 - trains a set of classifiers
 - evaluates classification metrics and runs a very simple backtest
 - saves results to `results_summary.csv`
"""
from __future__ import annotations

import os
import json
from pprint import pprint

import pandas as pd

from features import prepare_features
from models import train_and_evaluate
from backtest import backtest_signals


def time_train_test_split(X: pd.DataFrame, y: pd.Series, train_frac: float = 0.8):
    n = len(X)
    split = int(n * train_frac)
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    return X_train, X_test, y_train, y_test


def run(path: str = "GLD_daily.csv"):
    print("Loading and preparing features from:", path)
    X, y, df = prepare_features(path)
    print("Data prepared, rows:", X.shape)

    X_train, X_test, y_train, y_test = time_train_test_split(X, y)
    print("Train/Test sizes:", X_train.shape[0], X_test.shape[0])

    # Train models
    trained, results = train_and_evaluate(X_train, y_train, X_test, y_test)

    # Backtest each model's predictions
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    price = df[price_col].loc[X_test.index]
    backtest_results = {}
    for name, mdl in trained.items():
        try:
            if hasattr(mdl, "predict_proba"):
                proba = mdl.predict_proba(X_test)[:, 1]
                # simple rule: go long when prob>0.5
                signal = pd.Series((proba > 0.5).astype(int), index=X_test.index)
            else:
                preds = mdl.predict(X_test)
                signal = pd.Series(preds.astype(int), index=X_test.index)

            bt_df, perf = backtest_signals(price, signal, transaction_cost=0.0005)
            backtest_results[name] = perf
        except Exception as e:
            print(f"Backtest failed for {name}: {e}")

    # Combine and save
    summary = {}
    for k, v in results.items():
        summary[k] = {**v, **backtest_results.get(k, {})}

    out_path = "results_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved summary to:", out_path)
    pprint(summary)


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "GLD_daily.csv"
    if not os.path.exists(csv):
        print(f"CSV not found: {csv}. Please place GLD_daily.csv in the working folder.")
    else:
        run(csv)
