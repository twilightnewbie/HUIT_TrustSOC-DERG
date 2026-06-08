from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from ..calibration_metrics import calibration_summary
from ..config import OpenSOCBaseline, ProjectConfig
from ..trust_calibration import TrustOutputs, build_meta_features, decide_actions, fit_trust_calibrator
from ..utils import save_json


NUMERIC_DERG_COLUMNS = [
    "num_derg_nodes",
    "num_derg_edges",
    "graph_density",
    "avg_reliability",
    "max_reliability",
    "min_reliability",
    "reliability_std",
    "contradiction_score",
    "cti_match_score",
    "mitre_risk_score",
    "entity_risk_score",
    "evidence_diversity",
    "evidence_consistency",
    "adversarial_noise_score",
    "graph_centrality_score",
    "high_risk_node_ratio",
    "conflicting_evidence_ratio",
    "reliability_score",
]

TABULAR_COLUMNS = NUMERIC_DERG_COLUMNS + [
    "risk_score",
    "cti_match_count",
    "mitre_count",
    "evidence_node_count",
]


def safe_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=float)
    return frame[column].fillna(default).astype(float).to_numpy()


def build_sparse_feature_dict(row: pd.Series, include_cti: bool = True, include_mitre: bool = True, include_adversarial: bool = True) -> dict[str, Any]:
    feature_dict: dict[str, Any] = {
        f"source:{Path(str(row.get('source_file', 'unknown'))).name.lower()}": 1,
        f"case_type:{row.get('case_type', 'unknown')}": 1,
        "contradiction_score": float(row.get("contradiction_score", 0.0)),
        "evidence_diversity": float(row.get("evidence_diversity", 0.0)),
        "evidence_consistency": float(row.get("evidence_consistency", 1.0)),
        "reliability_score": float(row.get("reliability_score", 0.7)),
    }
    if include_cti:
        feature_dict["cti_match_count"] = float(row.get("cti_match_count", 0.0))
        feature_dict["cti_match_score"] = float(row.get("cti_match_score", 0.0))
    if include_mitre:
        feature_dict["mitre_count"] = float(row.get("mitre_count", 0.0))
        feature_dict["mitre_risk_score"] = float(row.get("mitre_risk_score", 0.0))
    if include_adversarial:
        feature_dict["adversarial_noise_score"] = float(row.get("adversarial_noise_score", 0.0))
        feature_dict[f"adversarial_type:{row.get('adversarial_type', 'normal_case')}"] = 1
    return feature_dict


@dataclass
class FeatureBundle:
    train_matrix: csr_matrix
    val_matrix: csr_matrix
    test_matrix: csr_matrix
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    dict_vectorizer: DictVectorizer
    numeric_columns: list[str]


def build_text_graph_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig,
    include_text: bool = True,
    include_cti: bool = True,
    include_mitre: bool = True,
    include_derg: bool = True,
    include_adversarial: bool = True,
) -> FeatureBundle:
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1 if len(train_df) <= config.small_dataset_threshold else 2,
        max_features=config.text_max_word_features,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1 if len(train_df) <= config.small_dataset_threshold else 2,
        max_features=config.text_max_char_features,
        sublinear_tf=True,
    )
    dict_vectorizer = DictVectorizer()

    if include_text:
        train_parts = [
            word_vectorizer.fit_transform(train_df["event_text"].fillna("")),
            char_vectorizer.fit_transform(train_df["event_text"].fillna("")),
        ]
        val_parts = [
            word_vectorizer.transform(val_df["event_text"].fillna("")),
            char_vectorizer.transform(val_df["event_text"].fillna("")),
        ]
        test_parts = [
            word_vectorizer.transform(test_df["event_text"].fillna("")),
            char_vectorizer.transform(test_df["event_text"].fillna("")),
        ]
    else:
        train_parts = []
        val_parts = []
        test_parts = []

    train_dict = [build_sparse_feature_dict(row, include_cti, include_mitre, include_adversarial) for _, row in train_df.iterrows()]
    val_dict = [build_sparse_feature_dict(row, include_cti, include_mitre, include_adversarial) for _, row in val_df.iterrows()]
    test_dict = [build_sparse_feature_dict(row, include_cti, include_mitre, include_adversarial) for _, row in test_df.iterrows()]

    train_parts.append(dict_vectorizer.fit_transform(train_dict))
    val_parts.append(dict_vectorizer.transform(val_dict))
    test_parts.append(dict_vectorizer.transform(test_dict))

    numeric_columns = NUMERIC_DERG_COLUMNS if include_derg else []
    if numeric_columns:
        train_parts.append(csr_matrix(train_df[numeric_columns].fillna(0.0).to_numpy(dtype=float)))
        val_parts.append(csr_matrix(val_df[numeric_columns].fillna(0.0).to_numpy(dtype=float)))
        test_parts.append(csr_matrix(test_df[numeric_columns].fillna(0.0).to_numpy(dtype=float)))

        # GNN integration (Phase 3 contribution)
        try:
            from .derg_gnn import compute_derg_embeddings
            from ..derg_builder import build_derg

            def reconstruct_graphs_from_df(df: pd.DataFrame) -> list:
                graphs = []
                for _, row in df.iterrows():
                    case_row = row.to_dict()
                    for field in ("raw_fields", "evidence_items", "mitre_list", "cti_matches"):
                        json_col = f"{field}_json"
                        if json_col in case_row and isinstance(case_row[json_col], str):
                            try:
                                case_row[field] = json.loads(case_row[json_col])
                            except Exception:
                                case_row[field] = []
                        elif field not in case_row or case_row[field] is None:
                            case_row[field] = []
                    graph = build_derg(case_row)
                    graphs.append(graph)
                return graphs

            train_graphs = reconstruct_graphs_from_df(train_df)
            val_graphs = reconstruct_graphs_from_df(val_df)
            test_graphs = reconstruct_graphs_from_df(test_df)

            train_gnn = compute_derg_embeddings(train_graphs, seed=config.seed)
            val_gnn = compute_derg_embeddings(val_graphs, seed=config.seed)
            test_gnn = compute_derg_embeddings(test_graphs, seed=config.seed)

            train_parts.append(csr_matrix(train_gnn))
            val_parts.append(csr_matrix(val_gnn))
            test_parts.append(csr_matrix(test_gnn))
        except ImportError:
            pass

    train_matrix = hstack(train_parts).tocsr()
    val_matrix = hstack(val_parts).tocsr()
    test_matrix = hstack(test_parts).tocsr()

    return FeatureBundle(train_matrix, val_matrix, test_matrix, word_vectorizer, char_vectorizer, dict_vectorizer, numeric_columns)


def build_tabular_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_cti: bool = True,
    include_mitre: bool = True,
    include_derg: bool = True,
    include_adversarial: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    columns = ["reliability_score", "contradiction_score", "evidence_consistency", "evidence_diversity"]
    if include_cti:
        columns.extend(["cti_match_count", "cti_match_score"])
    if include_mitre:
        columns.extend(["mitre_count", "mitre_risk_score"])
    if include_derg:
        columns.extend(NUMERIC_DERG_COLUMNS)
    if include_adversarial:
        columns.append("adversarial_noise_score")
    columns = list(dict.fromkeys(columns))
    return (
        train_df[columns].fillna(0.0).to_numpy(dtype=float),
        val_df[columns].fillna(0.0).to_numpy(dtype=float),
        test_df[columns].fillna(0.0).to_numpy(dtype=float),
        columns,
    )


def classifier_probability(model, matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(matrix)
    else:
        scores = model.decision_function(matrix)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores = scores - scores.max(axis=1, keepdims=True)
        exps = np.exp(np.clip(scores, -40.0, 40.0))
        probabilities = exps / exps.sum(axis=1, keepdims=True)

    ordered = np.sort(probabilities, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2] if ordered.shape[1] > 1 else confidence
    uncertainty = 1.0 - confidence
    return probabilities, confidence, margin


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) if mask.any() else 0.0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape,
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def estimate_parameter_count(model) -> int:
    total = 0
    for attr in ("coef_", "feature_importances_", "estimators_"):
        if hasattr(model, attr):
            value = getattr(model, attr)
            if hasattr(value, "shape"):
                total += int(np.prod(value.shape))
            elif isinstance(value, list):
                total += len(value)
    return total


def train_trust_layer(
    threat_model,
    severity_model,
    label_model,
    risk_model,
    train_meta_df: pd.DataFrame,
    val_meta_df: pd.DataFrame,
    test_meta_df: pd.DataFrame,
    train_matrix,
    val_matrix,
    test_matrix,
    train_pred_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    val_pred_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    test_pred_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> TrustOutputs:
    train_threat_probs, train_threat_conf, _ = classifier_probability(threat_model, train_matrix)
    train_severity_probs, train_severity_conf, _ = classifier_probability(severity_model, train_matrix)
    train_label_probs, train_label_conf, _ = classifier_probability(label_model, train_matrix)
    val_threat_probs, val_threat_conf, _ = classifier_probability(threat_model, val_matrix)
    val_severity_probs, val_severity_conf, _ = classifier_probability(severity_model, val_matrix)
    val_label_probs, val_label_conf, _ = classifier_probability(label_model, val_matrix)
    test_threat_probs, test_threat_conf, _ = classifier_probability(threat_model, test_matrix)
    test_severity_probs, test_severity_conf, _ = classifier_probability(severity_model, test_matrix)
    test_label_probs, test_label_conf, _ = classifier_probability(label_model, test_matrix)

    train_confidence = np.mean([train_threat_conf, train_severity_conf, train_label_conf], axis=0)
    val_confidence = np.mean([val_threat_conf, val_severity_conf, val_label_conf], axis=0)
    test_confidence = np.mean([test_threat_conf, test_severity_conf, test_label_conf], axis=0)

    train_uncertainty = 1.0 - train_confidence
    val_uncertainty = 1.0 - val_confidence
    test_uncertainty = 1.0 - test_confidence

    train_reliability = safe_series(train_meta_df, "avg_reliability")
    val_reliability = safe_series(val_meta_df, "avg_reliability")
    test_reliability = safe_series(test_meta_df, "avg_reliability")

    train_contradiction = safe_series(train_meta_df, "contradiction_score")
    val_contradiction = safe_series(val_meta_df, "contradiction_score")
    test_contradiction = safe_series(test_meta_df, "contradiction_score")

    train_noise = safe_series(train_meta_df, "adversarial_noise_score")
    val_noise = safe_series(val_meta_df, "adversarial_noise_score")
    test_noise = safe_series(test_meta_df, "adversarial_noise_score")

    train_risk = np.clip(train_pred_tuple[3] / 100.0, 0.0, 1.0)
    val_risk = np.clip(val_pred_tuple[3] / 100.0, 0.0, 1.0)
    test_risk = np.clip(test_pred_tuple[3] / 100.0, 0.0, 1.0)

    train_cti = safe_series(train_meta_df, "cti_match_score")
    val_cti = safe_series(val_meta_df, "cti_match_score")
    test_cti = safe_series(test_meta_df, "cti_match_score")

    train_consistency = safe_series(train_meta_df, "evidence_consistency", 1.0)
    val_consistency = safe_series(val_meta_df, "evidence_consistency", 1.0)
    test_consistency = safe_series(test_meta_df, "evidence_consistency", 1.0)

    train_correct = (
        (train_pred_tuple[0] == train_meta_df["threat_type"].to_numpy())
        & (train_pred_tuple[1] == train_meta_df["severity"].to_numpy())
        & (train_pred_tuple[2] == train_meta_df["label"].to_numpy())
    )
    val_correct = (
        (val_pred_tuple[0] == val_meta_df["threat_type"].to_numpy())
        & (val_pred_tuple[1] == val_meta_df["severity"].to_numpy())
        & (val_pred_tuple[2] == val_meta_df["label"].to_numpy())
    )
    test_correct = (
        (test_pred_tuple[0] == test_meta_df["threat_type"].to_numpy())
        & (test_pred_tuple[1] == test_meta_df["severity"].to_numpy())
        & (test_pred_tuple[2] == test_meta_df["label"].to_numpy())
    )

    train_meta = build_meta_features(
        train_confidence, train_uncertainty, train_reliability, train_contradiction, train_noise, train_risk, train_cti, train_consistency
    )
    val_meta = build_meta_features(
        val_confidence, val_uncertainty, val_reliability, val_contradiction, val_noise, val_risk, val_cti, val_consistency
    )
    test_meta = build_meta_features(
        test_confidence, test_uncertainty, test_reliability, test_contradiction, test_noise, test_risk, test_cti, test_consistency
    )

    split_point = max(16, len(val_meta_df) // 2)
    split_point = min(split_point, len(val_meta_df) - 1)
    calibrator, threshold, _ = fit_trust_calibrator(
        val_meta[:split_point],
        val_correct[:split_point],
        val_meta[split_point:],
        val_correct[split_point:],
        val_meta_df["adversarial_type"].ne("normal_case").to_numpy()[split_point:],
        val_consistency[split_point:],
    )
    trust_score = calibrator.predict_proba(test_meta)[:, 1]
    actions = decide_actions(trust_score, test_uncertainty, test_reliability, test_contradiction, test_noise, test_risk)
    metrics = calibration_summary(
        trust_score,
        test_correct.astype(bool),
        threshold,
        test_meta_df["adversarial_type"].ne("normal_case").to_numpy(),
        test_consistency,
    )
    return TrustOutputs(
        trust_score=trust_score,
        uncertainty_score=test_uncertainty,
        reliability_score=test_reliability,
        expected_action=actions,
        threshold=threshold,
        calibration_metrics=metrics,
    )


def save_model_bundle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_model_bundle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def comparison_row_from_metrics(
    model_name: str,
    metrics: dict[str, Any],
    latency_seconds_per_sample: float,
    train_time_minutes: float,
    parameter_count: int,
    notes: str,
) -> dict[str, Any]:
    return {
        "Model": model_name,
        "Accuracy": metrics["threat_type"]["accuracy"],
        "Macro F1": metrics["threat_type"]["f1_macro"],
        "Weighted F1": metrics["threat_type"]["f1_weighted"],
        "MAE": metrics["risk_score"]["mae"],
        "RMSE": metrics["risk_score"]["rmse"],
        "MAPE": metrics["risk_score"]["mape"],
        "R2": metrics["risk_score"]["r2"],
        "ECE": metrics["calibration"]["ece"],
        "Brier": metrics["calibration"]["brier"],
        "Robust-F1": metrics["robustness"].get("adversarial_case_f1", None),
        "Refusal Acc": metrics["calibration"].get("refusal_correctness", None),
        "Latency": latency_seconds_per_sample,
        "Train Time": train_time_minutes,
        "Params": parameter_count,
        "Notes": notes,
    }


def opensoc_reference_row(baseline: OpenSOCBaseline) -> dict[str, Any]:
    return {
        "Model": "OpenSOC-AI",
        "Accuracy": baseline.threat_accuracy,
        "Macro F1": getattr(baseline, "threat_macro_f1", None),
        "Weighted F1": baseline.threat_weighted_f1,
        "MAE": baseline.mae,
        "RMSE": baseline.rmse,
        "MAPE": baseline.mape,
        "R2": baseline.r2,
        "ECE": None,
        "Brier": None,
        "Robust-F1": None,
        "Refusal Acc": None,
        "Latency": baseline.latency_seconds_per_sample,
        "Train Time": baseline.train_time_minutes,
        "Params": baseline.parameter_count,
        "Notes": baseline.notes,
    }
