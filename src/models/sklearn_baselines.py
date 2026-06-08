from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from ..config import ProjectConfig
from ..utils import get_logger, save_json
from .model_utils import build_tabular_features, classification_metrics, estimate_parameter_count, regression_metrics


def train_baselines(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: ProjectConfig) -> dict[str, Any]:
    logger = get_logger(config, "train_baselines")
    X_train, X_val, X_test, columns = build_tabular_features(train_df, val_df, test_df, True, True, True, True)
    models = {
        "Logistic Regression": {
            "threat": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "severity": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "label": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "risk": RandomForestRegressor(n_estimators=150, random_state=config.seed, n_jobs=-1),
        },
        "Random Forest": {
            "threat": RandomForestClassifier(n_estimators=200, random_state=config.seed, n_jobs=-1, class_weight="balanced"),
            "severity": RandomForestClassifier(n_estimators=200, random_state=config.seed, n_jobs=-1, class_weight="balanced"),
            "label": RandomForestClassifier(n_estimators=200, random_state=config.seed, n_jobs=-1, class_weight="balanced"),
            "risk": RandomForestRegressor(n_estimators=200, random_state=config.seed, n_jobs=-1),
        },
        "Gradient Boosting": {
            "threat": GradientBoostingClassifier(random_state=config.seed),
            "severity": GradientBoostingClassifier(random_state=config.seed),
            "label": GradientBoostingClassifier(random_state=config.seed),
            "risk": GradientBoostingRegressor(random_state=config.seed),
        },
    }

    results: dict[str, Any] = {}
    for model_name, group in models.items():
        start = time.perf_counter()
        group["threat"].fit(X_train, train_df["threat_type"])
        group["severity"].fit(X_train, train_df["severity"])
        group["label"].fit(X_train, train_df["label"])
        group["risk"].fit(X_train, train_df["risk_score"])
        train_seconds = time.perf_counter() - start

        infer_start = time.perf_counter()
        threat_pred = group["threat"].predict(X_test)
        severity_pred = group["severity"].predict(X_test)
        label_pred = group["label"].predict(X_test)
        risk_pred = np.clip(group["risk"].predict(X_test), 0.0, 100.0)
        infer_seconds = time.perf_counter() - infer_start

        metrics = {
            "threat_type": classification_metrics(test_df["threat_type"].to_numpy(), threat_pred),
            "severity": classification_metrics(test_df["severity"].to_numpy(), severity_pred),
            "label": classification_metrics(test_df["label"].to_numpy(), label_pred),
            "risk_score": regression_metrics(test_df["risk_score"].to_numpy(), risk_pred),
            "efficiency": {
                "train_time_seconds": float(train_seconds),
                "average_latency_seconds_per_sample": float(infer_seconds / max(len(test_df), 1)),
                "feature_count": len(columns),
                "parameter_count_estimate": int(
                    estimate_parameter_count(group["threat"])
                    + estimate_parameter_count(group["severity"])
                    + estimate_parameter_count(group["label"])
                    + estimate_parameter_count(group["risk"])
                ),
            },
            "notes": "Tabular baseline using DERG-derived numeric features only.",
        }
        results[model_name] = metrics
        logger.info("Finished baseline %s", model_name)

    save_json(config.metrics_dir / "baseline_metrics.json", results)
    return results
