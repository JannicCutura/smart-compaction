#!/usr/bin/env python3
"""
Train models to predict compaction necessity from Iceberg table metadata.

Models trained:
    Classification (needs_compaction 0/1):
        - XGBoost classifier
        - Logistic Regression (baseline)

    Regression (file_reduction_ratio 0–1):
        - XGBoost regressor
        - Linear Regression (baseline)

All models use only observable metadata features (file counts, sizes,
partition stats, derived ratios) – *not* the generation parameters
(num_rows, etc.) which would not be available in production.

Outputs (all under data/):
    model_clf.json         – XGBoost classifier (JSON format)
    model_reg.json         – XGBoost regressor (JSON format)
    train_metrics.json     – evaluation metrics for all models

Usage:
    python code/train.py
    python code/train.py --dataset data/dataset.csv --seed 42 --test-size 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

LOG = logging.getLogger("train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)-5s  %(message)s",
)

# ── Defaults ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "dataset.csv"
DEFAULT_OUT_DIR = ROOT / "data"

# Columns to DROP from feature matrix X.
# Generation parameters are not observable in production.
DROP_COLUMNS = {
    "config_id",
    # Generation parameters
    "num_rows",
    "num_columns",
    "num_partitions_cfg",
    "num_writers",
    "num_write_batches",
    "file_size_target_kb",
    "partition_skew",
    # Post-compaction columns (labels / leakage)
    "after_file_count",
    "after_total_size_bytes",
    "after_avg_file_size_bytes",
    "rewritten_data_files_count",
    "added_data_files_count",
    "compaction_duration_s",
    # Label columns
    "needs_compaction",
    "file_reduction_ratio",
    "size_change_ratio",
    "files_rewritten_ratio",
}

# Observable metadata features (kept for model input).
# Defined explicitly for clarity and column ordering.
FEATURE_COLUMNS = [
    "file_count",
    "total_size_bytes",
    "avg_file_size_bytes",
    "min_file_size_bytes",
    "max_file_size_bytes",
    "stddev_file_size_bytes",
    "total_records",
    "avg_records_per_file",
    "num_partitions_actual",
    "avg_files_per_partition",
    "max_files_per_partition",
    "min_files_per_partition",
    "stddev_files_per_partition",
    "num_snapshots",
    "small_file_ratio",
    "file_size_cv",
    "files_per_partition_cv",
]


# ── Training ────────────────────────────────────────────────────────────────


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
) -> tuple[XGBClassifier, dict]:
    """Train binary classifier and return model + metrics."""
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=seed,
        eval_metric="logloss",
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    # 5-fold stratified cross-validation on training data only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(
        clf,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
    )
    metrics["cv_f1_mean"] = float(cv_scores.mean())
    metrics["cv_f1_std"] = float(cv_scores.std())

    LOG.info("Classifier test metrics:")
    for k, v in metrics.items():
        LOG.info("  %-18s %.4f", k, v)

    LOG.info(
        "\n%s",
        classification_report(y_test, y_pred, target_names=["no_compact", "compact"]),
    )

    return clf, metrics


def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
) -> tuple[Pipeline, dict]:
    """Train logistic regression baseline and return pipeline + metrics."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
        )),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
    )
    metrics["cv_f1_mean"] = float(cv_scores.mean())
    metrics["cv_f1_std"] = float(cv_scores.std())

    LOG.info("Logistic Regression test metrics:")
    for k, v in metrics.items():
        LOG.info("  %-18s %.4f", k, v)

    LOG.info(
        "\n%s",
        classification_report(y_test, y_pred, target_names=["no_compact", "compact"]),
    )

    return pipe, metrics


def train_linear_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict]:
    """Train linear regression baseline and return pipeline + metrics."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression()),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    LOG.info("Linear Regression test metrics:")
    for k, v in metrics.items():
        LOG.info("  %-18s %.4f", k, v)

    return pipe, metrics


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
) -> tuple[XGBRegressor, dict]:
    """Train regressor for file_reduction_ratio and return model + metrics."""
    reg = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=seed,
        eval_metric="rmse",
    )

    reg.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = reg.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    LOG.info("Regressor test metrics:")
    for k, v in metrics.items():
        LOG.info("  %-18s %.4f", k, v)

    return reg, metrics


def get_feature_importance(
    model: XGBClassifier | XGBRegressor,
    feature_names: list[str],
) -> list[dict]:
    """Return sorted feature importances."""
    imp = model.feature_importances_
    pairs = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": float(i)} for f, i in pairs]


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train compaction prediction model")
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to dataset.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for model and metrics output",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--test-size", type=float, default=0.2, help="Test split fraction"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    df = pd.read_csv(args.dataset)
    LOG.info("Dataset: %d rows × %d cols", len(df), len(df.columns))

    # Validate expected feature columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        LOG.error("Missing feature columns: %s", missing)
        sys.exit(1)

    X = df[FEATURE_COLUMNS].copy()
    y_clf = df["needs_compaction"]
    y_reg = df["file_reduction_ratio"]

    LOG.info("Feature matrix: %d rows × %d features", *X.shape)
    LOG.info("Class balance: %s", y_clf.value_counts().to_dict())

    # Train/test split (stratified on classification label)
    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = (
        train_test_split(
            X,
            y_clf,
            y_reg,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y_clf,
        )
    )
    LOG.info("Train: %d, Test: %d", len(X_train), len(X_test))

    # ── XGBoost Classifier ───────────────────────────────────────────────
    LOG.info("--- XGBoost Classifier ---")
    clf, clf_metrics = train_classifier(
        X_train, y_clf_train, X_test, y_clf_test, args.seed
    )
    clf_importance = get_feature_importance(clf, FEATURE_COLUMNS)

    # ── Logistic Regression Baseline ────────────────────────────────────
    LOG.info("--- Logistic Regression Baseline ---")
    logit, logit_metrics = train_logistic(
        X_train, y_clf_train, X_test, y_clf_test, args.seed
    )

    # ── XGBoost Regressor ───────────────────────────────────────────────
    LOG.info("--- XGBoost Regressor ---")
    reg, reg_metrics = train_regressor(
        X_train, y_reg_train, X_test, y_reg_test, args.seed
    )
    reg_importance = get_feature_importance(reg, FEATURE_COLUMNS)

    # ── Linear Regression Baseline ──────────────────────────────────────
    LOG.info("--- Linear Regression Baseline ---")
    linreg, linreg_metrics = train_linear_regressor(
        X_train, y_reg_train, X_test, y_reg_test
    )

    # ── Save outputs ────────────────────────────────────────────────────
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    clf_path = out_dir / "model_clf.json"
    clf.save_model(str(clf_path))
    LOG.info("Saved XGBoost classifier → %s", clf_path)

    reg_path = out_dir / "model_reg.json"
    reg.save_model(str(reg_path))
    LOG.info("Saved XGBoost regressor → %s", reg_path)

    metrics_out = {
        "seed": args.seed,
        "test_size": args.test_size,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": FEATURE_COLUMNS,
        "classifier": {
            "xgboost": {
                "metrics": clf_metrics,
                "feature_importance": clf_importance,
            },
            "logistic_regression": {
                "metrics": logit_metrics,
            },
        },
        "regressor": {
            "xgboost": {
                "metrics": reg_metrics,
                "feature_importance": reg_importance,
            },
            "linear_regression": {
                "metrics": linreg_metrics,
            },
        },
    }
    metrics_path = out_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    LOG.info("Saved metrics → %s", metrics_path)


if __name__ == "__main__":
    main()
