"""Diagnostics for the saved MLP model.

Produces plots and metrics under `figures/` with prefix `mlp_`.
"""
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report

from features import prepare_features
from backtest import backtest_advanced


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def max_drawdown(series: pd.Series) -> float:
    cum = series.cummax()
    drawdown = (series - cum) / cum
    return drawdown.min()


def run(csv: str = "GLD_daily.csv"):
    ensure_dir("figures")
    model_path = "models/mlp.joblib"
    if not os.path.exists(model_path):
        print("MLP model not found at models/mlp.joblib. Run full retrain first: python viz_results.py --full-retrain")
        return

    print("Loading features and model")
    X, y, df = prepare_features(csv)
    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    price = df[price_col].loc[X_test.index]

    mdl = joblib.load(model_path)
    if hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(X_test)[:, 1]
    else:
        proba = None
    preds = mdl.predict(X_test)

    # classification report
    print("Classification report for MLP on test set:")
    print(classification_report(y_test, preds))

    ensure_dir("figures/mlp")

    # confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("MLP Confusion Matrix")
    plt.tight_layout()
    plt.savefig("figures/mlp/confusion_matrix.png")
    plt.close()

    # ROC and PR curves if proba
    if proba is not None:
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve (MLP)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/mlp/roc.png")
        plt.close()

        prec, rec, _ = precision_recall_curve(y_test, proba)
        plt.figure(figsize=(5, 4))
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (MLP)")
        plt.tight_layout()
        plt.savefig("figures/mlp/pr_curve.png")
        plt.close()

        # proba histogram
        plt.figure(figsize=(5, 3))
        plt.hist(proba, bins=30)
        plt.title("MLP predicted probability distribution")
        plt.tight_layout()
        plt.savefig("figures/mlp/proba_hist.png")
        plt.close()

    # Backtest
    signal = pd.Series((proba > 0.5).astype(int) if proba is not None else preds.astype(int), index=X_test.index)
    bt_df, perf = backtest_advanced(price, signal, prob=pd.Series(proba, index=X_test.index) if proba is not None else None,
                                    transaction_cost=0.0005, slippage=0.0005, leverage=1.0, long_short=False)

    print("Backtest perf:", perf)

    # plot equity and rolling stats
    plt.figure(figsize=(10, 4))
    bt_df['cum'].plot()
    plt.title('MLP equity curve (test)')
    plt.ylabel('Cumulative return')
    plt.tight_layout()
    plt.savefig('figures/mlp/equity.png')
    plt.close()

    # rolling Sharpe (30-day)
    roll_sharpe = bt_df['strat_return'].rolling(30).mean() / bt_df['strat_return'].rolling(30).std()
    plt.figure(figsize=(10, 3))
    roll_sharpe.plot()
    plt.title('MLP rolling Sharpe (30-day)')
    plt.tight_layout()
    plt.savefig('figures/mlp/rolling_sharpe.png')
    plt.close()

    # drawdown
    dd = (bt_df['cum'] / bt_df['cum'].cummax() - 1)
    plt.figure(figsize=(10, 3))
    dd.plot()
    plt.title('MLP drawdown')
    plt.tight_layout()
    plt.savefig('figures/mlp/drawdown.png')
    plt.close()

    print('Saved MLP diagnostics to figures/mlp/')


if __name__ == '__main__':
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else 'GLD_daily.csv'
    run(csv)
