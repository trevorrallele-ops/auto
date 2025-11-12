"""Run tuned experiments with TimeSeriesSplit and RandomizedSearchCV.

This script performs light-weight randomized hyperparameter search on a few models
using TimeSeriesSplit. It saves best models to `models/` and writes a JSON summary.

Usage: python run_tuned_experiments.py [path_to_csv]
"""
from __future__ import annotations

import os
import json
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
import joblib

from features import prepare_features
from backtest import backtest_advanced


def small_param_grids():
    grids = {}
    grids["rf"] = {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [3, 5, None]}
    grids["logistic"] = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    return grids


def build_pipelines():
    pipes = {}
    pipes["rf"] = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=42))])
    pipes["logistic"] = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
    return pipes


def run(path: str = "GLD_daily.csv", n_iter: int = 8):
    print("Preparing features from:", path)
    X, y, df = prepare_features(path)

    # use first 80% for training CV, last 20% as holdout test
    split = int(len(X) * 0.8)
    X_train = X.iloc[:split]
    y_train = y.iloc[:split]
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    tscv = TimeSeriesSplit(n_splits=3)

    pipes = build_pipelines()
    grids = small_param_grids()

    ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
    ensure_dir("models")

    summary = {}

    for name, pipe in pipes.items():
        param_grid = grids.get(name, {})
        if param_grid:
            print(f"Tuning {name} with RandomizedSearchCV (n_iter={n_iter})")
            search = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=n_iter, cv=tscv, scoring=make_scorer(f1_score), random_state=42, n_jobs=1)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            print(f"Best params for {name}:", search.best_params_)
        else:
            print(f"Fitting {name} without tuning")
            pipe.fit(X_train, y_train)
            best = pipe

        # save model
        model_path = f"models/{name}.joblib"
        joblib.dump(best, model_path)

        # evaluate on holdout
        preds = best.predict(X_test)
        try:
            proba = best.predict_proba(X_test)[:, 1]
        except Exception:
            proba = None

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        price = df[price_col].loc[X_test.index]

        if proba is not None:
            signal = (proba > 0.5).astype(int)
            prob_series = pd.Series(proba, index=X_test.index)
        else:
            signal = preds.astype(int)
            prob_series = None

        signal_series = pd.Series(signal, index=X_test.index)
        bt_df, perf = backtest_advanced(price, signal_series, prob=prob_series, transaction_cost=0.0005, slippage=0.0005, leverage=1.0, long_short=False)

        # get serializable params
        try:
            params = getattr(best, "get_params", lambda: {})()
            params = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in params.items()}
        except Exception:
            params = {}
        summary[name] = {"params": params, "perf": perf}

    out = "models/tuning_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved models to models/ and summary to {out}")
    pprint(summary)


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "GLD_daily.csv"
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if not os.path.exists(csv):
        print(f"CSV not found: {csv}. Please place GLD_daily.csv in the working folder.")
    else:
        run(csv, n_iter=n_iter)
