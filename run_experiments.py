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
from typing import Tuple

import pandas as pd

from features import prepare_features
from models import train_and_evaluate
from backtest import backtest_signals
from config import TRAIN_TEST_SPLIT, FIGURES_DIR, RESULTS_FILE, PROB_THRESHOLD
from utils import ensure_dir, validate_file_path, logger


def time_train_test_split(X: pd.DataFrame, y: pd.Series, train_frac: float = TRAIN_TEST_SPLIT) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data by time with validation."""
    if len(X) == 0:
        raise ValueError("Input data is empty")
    if train_frac <= 0 or train_frac >= 1:
        raise ValueError("train_frac must be between 0 and 1")
    n = len(X)
    split = int(n * train_frac)
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    return X_train, X_test, y_train, y_test


def run(path: str = "GLD_daily.csv") -> None:
    """Run complete experiment pipeline with validation."""
    try:
        validate_file_path(path)
        logger.info(f"Loading and preparing features from: {path}")
        X, y, df = prepare_features(path)
        logger.info(f"Data prepared, rows: {X.shape}")
    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        raise

    try:
        X_train, X_test, y_train, y_test = time_train_test_split(X, y)
        logger.info(f"Train/Test sizes: {X_train.shape[0]}, {X_test.shape[0]}")
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        raise

    # Train models
    try:
        trained, results = train_and_evaluate(X_train, y_train, X_test, y_test)
    except Exception as e:
        logger.error(f"Failed to train models: {e}")
        raise

    # Backtest each model's predictions
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    price = df[price_col].loc[X_test.index]
    backtest_results = {}
    for name, mdl in trained.items():
        try:
            if hasattr(mdl, "predict_proba"):
                proba = mdl.predict_proba(X_test)[:, 1]
                # simple rule: go long when prob > threshold
                signal = pd.Series((proba > PROB_THRESHOLD).astype(int), index=X_test.index)
            else:
                preds = mdl.predict(X_test)
                signal = pd.Series(preds.astype(int), index=X_test.index)

            bt_df, perf = backtest_signals(price, signal)
            backtest_results[name] = perf
        except Exception as e:
            logger.warning(f"Backtest failed for {name}: {e}")

    # Combine and save
    summary = {}
    for k, v in results.items():
        summary[k] = {**v, **backtest_results.get(k, {})}

    ensure_dir(FIGURES_DIR)
    out_path = os.path.join(FIGURES_DIR, RESULTS_FILE)
    try:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to: {out_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "GLD_daily.csv"
    try:
        run(csv)
    except FileNotFoundError:
        logger.error(f"CSV not found: {csv}. Please place GLD_daily.csv in the working folder.")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
