"""Model training utilities: train a battery of ML models and return metrics.

The module keeps imports lightweight at module import time and guards heavy libs.
"""
from __future__ import annotations

import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")


def _safe_import_xgboost():
    try:
        import xgboost as xgb

        return xgb
    except Exception:
        return None


def _safe_import_lightgbm():
    try:
        import lightgbm as lgb

        return lgb
    except Exception:
        return None


def build_classifiers(random_state: int = 42) -> Dict[str, Any]:
    models = {}
    models["logistic"] = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    models["rf"] = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state))])
    models["svm"] = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True))])
    models["knn"] = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
    models["mlp"] = Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200))])

    xgb = _safe_import_xgboost()
    if xgb is not None:
        from xgboost import XGBClassifier

        models["xgb"] = Pipeline([("scaler", StandardScaler()), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0))])

    lgb = _safe_import_lightgbm()
    if lgb is not None:
        from lightgbm import LGBMClassifier

        models["lgb"] = Pipeline([("scaler", StandardScaler()), ("clf", LGBMClassifier())])

    return models


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    out = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)[:, 1]
            out["auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            out["auc"] = np.nan
    else:
        out["auc"] = np.nan
    return out


def train_and_evaluate(X_train, y_train, X_test, y_test, random_state: int = 42):
    models = build_classifiers(random_state=random_state)
    results = {}
    trained = {}
    for name, clf in models.items():
        try:
            clf.fit(X_train, y_train)
            metrics = evaluate_model(clf, X_test, y_test)
            results[name] = metrics
            trained[name] = clf
            print(f"Trained {name}: acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} auc={metrics.get('auc')}")
        except Exception as e:
            print(f"Failed {name}: {e}")
    return trained, results


if __name__ == "__main__":
    print("This module provides training helpers. Run run_experiments.py to execute an experiment.")
