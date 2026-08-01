#!/usr/bin/env python3
"""
Extended analyses for referee revision: feature ablation, hyperparameter
sensitivity, bootstrap confidence intervals, correlation matrix, and
class-prevalence robustness.

Reads:
    data/dataset.csv  -- full labelled dataset

Outputs (all under data/):
    ablation_metrics.json  -- all results from this script

Usage:
    python code/ablation.py
    python code/ablation.py --dataset data/dataset.csv --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

LOG = logging.getLogger("ablation")

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)-5s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "ablation.log"),
    ],
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "dataset.csv"
DEFAULT_OUT_DIR = ROOT / "data"

# Must match train.py exactly.
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


# ── Helpers ─────────────────────────────────────────────────────────────────


def _clf_metrics(y_true, y_pred, y_prob=None) -> dict:
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return m


def _reg_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _make_xgb_clf(seed: int, **overrides) -> XGBClassifier:
    params = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=seed,
        eval_metric="logloss",
    )
    params.update(overrides)
    return XGBClassifier(**params)


def _make_xgb_reg(seed: int, **overrides) -> XGBRegressor:
    params = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=seed,
        eval_metric="rmse",
    )
    params.update(overrides)
    return XGBRegressor(**params)


# ── 1. Feature ablation ────────────────────────────────────────────────────


def run_feature_ablation(
    df: pd.DataFrame, seed: int, test_size: float
) -> dict:
    """Train with progressive feature subsets and leave-one-out."""
    LOG.info("=== Feature Ablation ===")

    y_clf = df["needs_compaction"]
    y_reg = df["file_reduction_ratio"]

    # Load the existing feature importance order from a fresh XGB fit
    X_full = df[FEATURE_COLUMNS]
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X_full, y_clf, y_reg, test_size=test_size, random_state=seed, stratify=y_clf
    )

    clf_full = _make_xgb_clf(seed)
    clf_full.fit(X_tr, yc_tr, eval_set=[(X_te, yc_te)], verbose=False)
    importance_order = np.argsort(clf_full.feature_importances_)[::-1]
    ordered_features = [FEATURE_COLUMNS[i] for i in importance_order]

    results = {"importance_order": ordered_features, "progressive": [], "leave_one_out": []}

    # Progressive: top-1, top-2, top-3, top-5, top-10, all-17
    for k in [1, 2, 3, 5, 10, len(FEATURE_COLUMNS)]:
        feats = ordered_features[:k]
        X_sub = df[feats]
        X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
            X_sub, y_clf, y_reg, test_size=test_size, random_state=seed, stratify=y_clf
        )

        clf = _make_xgb_clf(seed)
        clf.fit(X_tr, yc_tr, eval_set=[(X_te, yc_te)], verbose=False)
        yc_pred = clf.predict(X_te)
        yc_prob = clf.predict_proba(X_te)[:, 1]

        reg = _make_xgb_reg(seed)
        reg.fit(X_tr, yr_tr, eval_set=[(X_te, yr_te)], verbose=False)
        yr_pred = reg.predict(X_te)

        entry = {
            "k": k,
            "features": feats,
            "clf": _clf_metrics(yc_te, yc_pred, yc_prob),
            "reg": _reg_metrics(yr_te, yr_pred),
        }
        results["progressive"].append(entry)
        LOG.info(
            "  top-%d: clf_f1=%.4f, reg_r2=%.4f",
            k, entry["clf"]["f1"], entry["reg"]["r2"],
        )

    # Leave-one-out: drop each feature individually
    for feat in FEATURE_COLUMNS:
        feats = [f for f in FEATURE_COLUMNS if f != feat]
        X_sub = df[feats]
        X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
            X_sub, y_clf, y_reg, test_size=test_size, random_state=seed, stratify=y_clf
        )

        clf = _make_xgb_clf(seed)
        clf.fit(X_tr, yc_tr, eval_set=[(X_te, yc_te)], verbose=False)
        yc_pred = clf.predict(X_te)
        yc_prob = clf.predict_proba(X_te)[:, 1]

        reg = _make_xgb_reg(seed)
        reg.fit(X_tr, yr_tr, eval_set=[(X_te, yr_te)], verbose=False)
        yr_pred = reg.predict(X_te)

        entry = {
            "dropped_feature": feat,
            "clf": _clf_metrics(yc_te, yc_pred, yc_prob),
            "reg": _reg_metrics(yr_te, yr_pred),
        }
        results["leave_one_out"].append(entry)

    return results


# ── 2. Hyperparameter sensitivity ──────────────────────────────────────────


def run_hyperparam_sensitivity(
    df: pd.DataFrame, seed: int, test_size: float
) -> dict:
    """Sweep XGBoost hyperparams + compare with simpler model families."""
    LOG.info("=== Hyperparameter Sensitivity ===")

    X = df[FEATURE_COLUMNS]
    y_clf = df["needs_compaction"]
    y_reg = df["file_reduction_ratio"]

    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_clf, y_reg, test_size=test_size, random_state=seed, stratify=y_clf
    )

    results = {"xgb_sweep": [], "model_families": []}

    # XGBoost sweep: max_depth x n_estimators
    for depth in [2, 4, 6, 8]:
        for n_est in [10, 50, 200, 500]:
            clf = _make_xgb_clf(seed, max_depth=depth, n_estimators=n_est)
            clf.fit(X_tr, yc_tr, eval_set=[(X_te, yc_te)], verbose=False)
            yc_pred = clf.predict(X_te)
            yc_prob = clf.predict_proba(X_te)[:, 1]

            reg = _make_xgb_reg(seed, max_depth=depth, n_estimators=n_est)
            reg.fit(X_tr, yr_tr, eval_set=[(X_te, yr_te)], verbose=False)
            yr_pred = reg.predict(X_te)

            entry = {
                "max_depth": depth,
                "n_estimators": n_est,
                "clf": _clf_metrics(yc_te, yc_pred, yc_prob),
                "reg": _reg_metrics(yr_te, yr_pred),
            }
            results["xgb_sweep"].append(entry)
            LOG.info(
                "  depth=%d, n_est=%d: clf_f1=%.4f, reg_r2=%.4f",
                depth, n_est, entry["clf"]["f1"], entry["reg"]["r2"],
            )

    # Alternative model families
    alt_models = {
        "decision_tree": {
            "clf": DecisionTreeClassifier(random_state=seed, max_depth=6),
            "reg": DecisionTreeRegressor(random_state=seed, max_depth=6),
        },
        "random_forest": {
            "clf": RandomForestClassifier(
                n_estimators=200, max_depth=6, random_state=seed, n_jobs=-1
            ),
            "reg": RandomForestRegressor(
                n_estimators=200, max_depth=6, random_state=seed, n_jobs=-1
            ),
        },
        "logistic_regression": {
            "clf": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
            ]),
            "reg": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]),
        },
    }

    for name, models in alt_models.items():
        c = models["clf"]
        c.fit(X_tr, yc_tr)
        yc_pred = c.predict(X_te)
        yc_prob = c.predict_proba(X_te)[:, 1] if hasattr(c, "predict_proba") else None

        r = models["reg"]
        r.fit(X_tr, yr_tr)
        yr_pred = r.predict(X_te)

        entry = {
            "model": name,
            "clf": _clf_metrics(yc_te, yc_pred, yc_prob),
            "reg": _reg_metrics(yr_te, yr_pred),
        }
        results["model_families"].append(entry)
        LOG.info(
            "  %s: clf_f1=%.4f, reg_r2=%.4f",
            name, entry["clf"]["f1"], entry["reg"]["r2"],
        )

    return results


# ── 3. Bootstrap confidence intervals ──────────────────────────────────────


def run_bootstrap_ci(
    df: pd.DataFrame, seed: int, test_size: float, n_repeats: int = 30
) -> dict:
    """Repeated stratified splits to compute 95% CIs on all metrics."""
    LOG.info("=== Bootstrap Confidence Intervals (%d repeats) ===", n_repeats)

    X = df[FEATURE_COLUMNS]
    y_clf = df["needs_compaction"]
    y_reg = df["file_reduction_ratio"]

    clf_runs = []
    reg_runs = []

    for i in range(n_repeats):
        s = seed + i
        X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
            X, y_clf, y_reg, test_size=test_size, random_state=s, stratify=y_clf
        )

        clf = _make_xgb_clf(s)
        clf.fit(X_tr, yc_tr, eval_set=[(X_te, yc_te)], verbose=False)
        yc_pred = clf.predict(X_te)
        yc_prob = clf.predict_proba(X_te)[:, 1]
        clf_runs.append(_clf_metrics(yc_te, yc_pred, yc_prob))

        reg = _make_xgb_reg(s)
        reg.fit(X_tr, yr_tr, eval_set=[(X_te, yr_te)], verbose=False)
        yr_pred = reg.predict(X_te)
        reg_runs.append(_reg_metrics(yr_te, yr_pred))

        if (i + 1) % 10 == 0:
            LOG.info("  Completed %d / %d repeats", i + 1, n_repeats)

    # Aggregate
    def _summarise(runs: list[dict]) -> dict:
        keys = runs[0].keys()
        summary = {}
        for k in keys:
            vals = [r[k] for r in runs]
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "ci_lo": float(np.percentile(vals, 2.5)),
                "ci_hi": float(np.percentile(vals, 97.5)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        return summary

    return {
        "n_repeats": n_repeats,
        "classifier": _summarise(clf_runs),
        "regressor": _summarise(reg_runs),
        "raw_clf": clf_runs,
        "raw_reg": reg_runs,
    }


# ── 4. Correlation matrix ──────────────────────────────────────────────────


def run_correlation_analysis(df: pd.DataFrame) -> dict:
    """Spearman and Pearson correlation among features and with labels."""
    LOG.info("=== Correlation Analysis ===")

    X = df[FEATURE_COLUMNS]

    spearman = X.corr(method="spearman")
    pearson = X.corr(method="pearson")

    # Feature-label correlations
    label_corr = {}
    for label in ["needs_compaction", "file_reduction_ratio"]:
        y = df[label]
        label_corr[label] = {
            "spearman": {f: float(X[f].corr(y, method="spearman")) for f in FEATURE_COLUMNS},
            "pearson": {f: float(X[f].corr(y)) for f in FEATURE_COLUMNS},
        }

    # Top-5 features inter-correlation (Spearman)
    # Use importance order from XGBoost if available; fall back to variance
    top5 = FEATURE_COLUMNS[:5]  # will be overridden by caller if needed

    return {
        "spearman": {r: {c: float(spearman.loc[r, c]) for c in FEATURE_COLUMNS} for r in FEATURE_COLUMNS},
        "pearson": {r: {c: float(pearson.loc[r, c]) for c in FEATURE_COLUMNS} for r in FEATURE_COLUMNS},
        "feature_label_correlations": label_corr,
    }


# ── 5. Class prevalence robustness ─────────────────────────────────────────


def run_prevalence_robustness(
    df: pd.DataFrame, seed: int, test_size: float
) -> dict:
    """Evaluate at different positive-class prevalence via downsampling."""
    LOG.info("=== Class Prevalence Robustness ===")

    y_clf = df["needs_compaction"]
    natural_prevalence = float(y_clf.mean())
    LOG.info("  Natural prevalence: %.3f", natural_prevalence)

    results = {"natural_prevalence": natural_prevalence, "levels": []}

    for target_prev in [natural_prevalence, 0.70, 0.50, 0.30]:
        rng = np.random.RandomState(seed)

        pos = df[df["needs_compaction"] == 1]
        neg = df[df["needs_compaction"] == 0]
        n_pos = len(pos)
        n_neg = len(neg)

        if target_prev >= natural_prevalence:
            # Use full dataset at natural prevalence
            df_sub = df.copy()
        elif target_prev <= 0:
            continue
        else:
            # Downsample positive class to achieve target prevalence
            # target = n_pos_new / (n_pos_new + n_neg)
            # n_pos_new = target * n_neg / (1 - target)
            n_pos_new = int(target_prev * n_neg / (1 - target_prev))
            if n_pos_new > n_pos:
                # Can't upsample, downsample negative instead
                n_neg_new = int(n_pos * (1 - target_prev) / target_prev)
                neg_sub = neg.sample(n=n_neg_new, random_state=seed)
                df_sub = pd.concat([pos, neg_sub])
            else:
                pos_sub = pos.sample(n=n_pos_new, random_state=seed)
                df_sub = pd.concat([pos_sub, neg])

        X_sub = df_sub[FEATURE_COLUMNS]
        y_clf_sub = df_sub["needs_compaction"]
        y_reg_sub = df_sub["file_reduction_ratio"]

        actual_prev = float(y_clf_sub.mean())
        LOG.info("  Target=%.2f, Actual=%.3f, N=%d", target_prev, actual_prev, len(df_sub))

        X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
            X_sub, y_clf_sub, y_reg_sub,
            test_size=test_size, random_state=seed, stratify=y_clf_sub,
        )

        # XGBoost
        clf = _make_xgb_clf(seed)
        clf.fit(X_tr, yc_tr, eval_set=[(X_te, yc_te)], verbose=False)
        yc_pred = clf.predict(X_te)
        yc_prob = clf.predict_proba(X_te)[:, 1]

        reg = _make_xgb_reg(seed)
        reg.fit(X_tr, yr_tr, eval_set=[(X_te, yr_te)], verbose=False)
        yr_pred = reg.predict(X_te)

        # Threshold sweep for this prevalence level
        threshold_results = []
        for k in range(1, 21):
            t_pred = (df_sub["max_files_per_partition"] > k).astype(int)
            t_acc = float(accuracy_score(y_clf_sub, t_pred))
            t_f1 = float(f1_score(y_clf_sub, t_pred, zero_division=0))
            threshold_results.append({"k": k, "accuracy": t_acc, "f1": t_f1})

        entry = {
            "target_prevalence": target_prev,
            "actual_prevalence": actual_prev,
            "n_samples": len(df_sub),
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "clf": _clf_metrics(yc_te, yc_pred, yc_prob),
            "reg": _reg_metrics(yr_te, yr_pred),
            "threshold_sweep": threshold_results,
        }
        results["levels"].append(entry)

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended analyses for referee revision")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--bootstrap-repeats", type=int, default=30)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    df = pd.read_csv(args.dataset)
    LOG.info("Dataset: %d rows x %d cols", len(df), len(df.columns))

    results = {}

    results["feature_ablation"] = run_feature_ablation(df, args.seed, args.test_size)
    results["hyperparam_sensitivity"] = run_hyperparam_sensitivity(
        df, args.seed, args.test_size
    )
    results["bootstrap_ci"] = run_bootstrap_ci(
        df, args.seed, args.test_size, n_repeats=args.bootstrap_repeats
    )
    results["correlation"] = run_correlation_analysis(df)
    results["prevalence_robustness"] = run_prevalence_robustness(
        df, args.seed, args.test_size
    )

    out_path = args.out_dir / "ablation_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    LOG.info("Saved all results -> %s", out_path)


if __name__ == "__main__":
    main()
