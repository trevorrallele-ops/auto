"""Visualize experiment results and equity curves.

This script retrains the same models on the same time split, regenerates signals,
computes equity curves, and creates plots saved under `figures/`.

Usage: python viz_results.py [path_to_csv]
"""
from __future__ import annotations

import os
import json
from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from features import prepare_features
from models import train_and_evaluate
from backtest import backtest_signals, backtest_advanced
import joblib
import glob


def time_train_test_split_indices(n: int, train_frac: float = 0.8):
    split = int(n * train_frac)
    return split


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_metrics_table(summary: dict, out_path: str):
    # convert to DataFrame
    df = pd.DataFrame(summary).T
    # select columns of interest (some may be missing)
    cols = [c for c in ["accuracy", "f1", "precision", "recall", "auc", "total_return", "ann_ret", "ann_vol", "sharpe"] if c in df.columns]
    df = df[cols]
    plt.figure(figsize=(10, max(2, 0.5 * len(df))))
    sns.heatmap(df.astype(float), annot=True, fmt=".3f", cmap="vlag", cbar_kws={"label": "value"})
    plt.title("Model metrics & backtest performance")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_equity_curves(equities: dict, out_path: str):
    plt.figure(figsize=(12, 6))
    for name, ser in equities.items():
        ser.plot(label=name, lw=1)
    plt.legend()
    plt.title("Equity curves (test period)")
    plt.ylabel("Cumulative return")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run(path: str = "GLD_daily.csv", full_retrain: bool = False):
    print("Preparing features from:", path)
    X, y, df = prepare_features(path)

    n = len(X)
    split = time_train_test_split_indices(n)
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print("Train/Test sizes:", X_train.shape[0], X_test.shape[0])

    # Ensure models dir exists
    os.makedirs("models", exist_ok=True)

    # Prefer loading saved models from models/ if present
    trained = {}
    results = {}
    model_files = glob.glob("models/*.joblib")
    if model_files and not full_retrain:
        print("Found saved models in models/ — loading them for visualization")
        for mf in model_files:
            name = os.path.splitext(os.path.basename(mf))[0]
            try:
                mdl = joblib.load(mf)
                trained[name] = mdl
            except Exception as e:
                print(f"Failed to load {mf}: {e}")
        # Train only the missing models (avoid retraining heavy models like SVM/MLP unless requested)
        from models import build_classifiers
        candidate = build_classifiers()
        missing = [n for n in candidate.keys() if n not in trained]
        # skip heavy models by default to keep this step fast
        skip_heavy = {"svm", "mlp", "xgb", "lgb"}
        to_train = [n for n in missing if n not in skip_heavy]
        if missing and not to_train:
            print(f"Found missing models {missing} but skipping heavy ones {skip_heavy}. Run full retrain if you want them.")
        results = {}
        for name in to_train:
            mdl = candidate[name]
            try:
                print(f"Training missing model: {name}")
                mdl.fit(X_train, y_train)
                trained[name] = mdl
                # evaluate quickly
                res = {}
                try:
                    from models import evaluate_model
                    res = evaluate_model(mdl, X_test, y_test)
                except Exception:
                    res = {}
                results[name] = res
            except Exception as e:
                print(f"Failed training {name}: {e}")
        # Note: results currently contains metrics for models we trained here; saved-model metrics may be missing
    else:
        # Full retrain requested or no saved models: train all models and save them
        print("No saved models found or full retrain requested — training all models now")
        trained_all, results_all = train_and_evaluate(X_train, y_train, X_test, y_test)
        # save each trained model to models/
        for name, mdl in trained_all.items():
            try:
                joblib.dump(mdl, f"models/{name}.joblib")
                print(f"Saved model: models/{name}.joblib")
            except Exception as e:
                print(f"Failed to save model {name}: {e}")
        trained.update(trained_all)
        results.update(results_all)

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    price = df[price_col].loc[X_test.index]

    ensure_dir("figures")

    equities = {}
    combined_summary = {}
    for name, mdl in trained.items():
        try:
            if hasattr(mdl, "predict_proba"):
                proba = mdl.predict_proba(X_test)[:, 1]
                signal = pd.Series((proba > 0.5).astype(int), index=X_test.index)
            else:
                preds = mdl.predict(X_test)
                signal = pd.Series(preds.astype(int), index=X_test.index)

            # Use advanced backtest for better realism if possible
            bt_df, perf = backtest_advanced(price, signal, prob=pd.Series(proba, index=X_test.index) if 'proba' in locals() else None,
                                            transaction_cost=0.0005, slippage=0.0005, leverage=1.0, long_short=False)
            equities[name] = bt_df["cum"]
            # merge metrics
            metrics = results.get(name, {})
            combined_summary[name] = {**metrics, **perf}
            acc = metrics.get('accuracy', float('nan'))
            sharpe = perf.get('sharpe', float('nan'))
            tot = perf.get('total_return', float('nan'))
            print(f"{name}: acc={acc:.3f} sharpe={sharpe:.3f} total_return={tot:.3f}")
        except Exception as e:
            print(f"Failed plotting/backtest for {name}: {e}")

    # save combined summary
    summary_df = pd.DataFrame(combined_summary).T
    summary_df.to_csv("figures/results_summary.csv")

    # plots
    plot_metrics_table(combined_summary, "figures/metrics_heatmap.png")
    plot_equity_curves(equities, "figures/equity_curves.png")

    # Save JSON too
    with open("figures/results_summary.json", "w") as f:
        json.dump(combined_summary, f, indent=2)

    print("Saved figures to figures/ and summary to figures/results_summary.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize GLD ML results and equity curves")
    parser.add_argument("csv", nargs="?", default="GLD_daily.csv", help="path to GLD CSV")
    parser.add_argument("--full-retrain", action="store_true", help="retrain and save all models (can be slow)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}. Please place GLD_daily.csv in the working folder.")
    else:
        run(args.csv, full_retrain=args.full_retrain)
