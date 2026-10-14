"""Model building + training (Phase 5) and evaluation (Phase 6).

Three models are trained and compared:
  1. Logistic Regression  - simple, fast baseline          (supervised)
  2. Random Forest        - main model, multi-class        (supervised)
  3. Isolation Forest     - learns ONLY normal behaviour and flags outliers
                            (unsupervised -> can catch brand-new, unseen attacks)
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import (CATEGORICAL, ISO_MODEL_PATH, META_PATH, MODELS_DIR, NUMERIC,
                     OUTPUTS_DIR, RANDOM_STATE, RF_MODEL_PATH, TARGET)
from .evaluate import binary_metrics, multiclass_metrics, report_text
from .features import build_preprocessor


def train_all(df: pd.DataFrame, test_size=0.2):
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    features = NUMERIC + CATEGORICAL
    X, y = df[features], df[TARGET]
    idx_train, idx_test = train_test_split(df.index, test_size=test_size,
                                           stratify=y, random_state=RANDOM_STATE)
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    print(f"[train] train rows: {len(X_train)} | test rows: {len(X_test)}")

    # ---- 1. Logistic Regression baseline -------------------------------------------------
    lr = Pipeline([("prep", build_preprocessor()),
                   ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # ---- 2. Random Forest (main model) ---------------------------------------------------
    rf = Pipeline([("prep", build_preprocessor()),
                   ("model", RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                                    class_weight="balanced",
                                                    n_jobs=-1, random_state=RANDOM_STATE))])
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_conf = rf.predict_proba(X_test).max(axis=1)

    # ---- 3. Isolation Forest (trained on NORMAL traffic only) ----------------------------
    normal_train = X_train[y_train == "Normal"]
    iso = Pipeline([("prep", build_preprocessor()),
                    ("model", IsolationForest(n_estimators=200, contamination=0.05,
                                              random_state=RANDOM_STATE))])
    iso.fit(normal_train)
    iso_score = -iso.score_samples(X_test)                    # higher = more anomalous
    iso_flag = (iso.predict(X_test) == -1)
    # Alert threshold for "unknown" anomalies: 99th percentile of normal training scores
    normal_scores = -iso.score_samples(normal_train)
    iso_threshold = float(np.percentile(normal_scores, 99))

    # ---- Evaluation ----------------------------------------------------------------------
    true_attack = (y_test != "Normal").astype(int)
    results = {
        "LogisticRegression": binary_metrics(true_attack, (lr_pred != "Normal").astype(int)),
        "RandomForest": binary_metrics(true_attack, (rf_pred != "Normal").astype(int)),
        "IsolationForest": binary_metrics(true_attack, iso_flag.astype(int)),
    }
    multi = {"LogisticRegression": multiclass_metrics(y_test, lr_pred),
             "RandomForest": multiclass_metrics(y_test, rf_pred)}

    print("\n=== Attack-vs-Normal metrics (test set) ===")
    print(pd.DataFrame(results).T.round(4).to_string())
    print("\n=== Multi-class report: Random Forest ===")
    print(report_text(y_test, rf_pred))

    # ---- Save everything -----------------------------------------------------------------
    joblib.dump(rf, RF_MODEL_PATH, compress=3)
    joblib.dump(iso, ISO_MODEL_PATH, compress=3)
    META_PATH.write_text(json.dumps({"iso_alert_threshold": iso_threshold}, indent=2))
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(
        {"binary": results, "multiclass": multi, "train_rows": len(X_train),
         "test_rows": len(X_test)}, indent=2))
    (OUTPUTS_DIR / "classification_report.txt").write_text(report_text(y_test, rf_pred))

    preds = df.loc[idx_test, ["timestamp", "src_ip", "dst_ip"] + features + [TARGET]].copy()
    preds["rf_pred"], preds["rf_confidence"] = rf_pred, rf_conf
    preds["lr_pred"], preds["iso_score"], preds["iso_flag"] = lr_pred, iso_score, iso_flag
    preds.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)
    print(f"\n[train] saved models -> {MODELS_DIR}")
    return results
