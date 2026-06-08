from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from .config import ProjectConfig
from .derg_builder import graph_for_case
from .models.model_utils import load_model_bundle

sns.set_theme(style="whitegrid")


def save_placeholder_figure(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_confusion(metrics: dict[str, Any], path: Path, normalized: bool = False) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    labels = metrics["labels"]
    if normalized and matrix.sum(axis=1).all():
        matrix = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1.0, None)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f" if normalized else ".0f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Threat Confusion Matrix" + (" (Normalized)" if normalized else ""))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_distribution(values: pd.Series, path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(values, kde=True, ax=ax, color="#3366cc")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_calibration(predictions: pd.DataFrame, path_curve: Path, path_reliability: Path) -> None:
    trust = predictions["trust_score"].astype(float).to_numpy()
    correct = predictions["joint_correct"].astype(int).to_numpy()
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(trust, bins) - 1
    centers = []
    accs = []
    confs = []
    counts = []
    for idx in range(len(bins) - 1):
        mask = bin_ids == idx
        if not mask.any():
            continue
        centers.append((bins[idx] + bins[idx + 1]) / 2.0)
        accs.append(correct[mask].mean())
        confs.append(trust[mask].mean())
        counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.plot(confs, accs, marker="o", color="#d62728")
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean Predicted Trust")
    ax.set_ylabel("Empirical Correctness")
    fig.tight_layout()
    fig.savefig(path_curve, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(centers, counts, width=0.08, alpha=0.35, color="#1f77b4", label="Samples")
    ax.plot(centers, confs, marker="o", label="Confidence", color="#ff7f0e")
    ax.plot(centers, accs, marker="s", label="Accuracy", color="#2ca02c")
    ax.set_title("Reliability Diagram")
    ax.set_xlabel("Trust Bin")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_reliability, dpi=300)
    plt.close(fig)


def plot_error_distribution(predictions: pd.DataFrame, path_hist: Path, path_residual: Path) -> None:
    residual = predictions["risk_pred"].astype(float) - predictions["risk_true"].astype(float)
    plot_distribution(residual, path_hist, "Regression Residual Distribution", "Predicted Risk - True Risk")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predictions["risk_true"], residual, alpha=0.6, color="#1f77b4")
    ax.axhline(0.0, linestyle="--", color="gray")
    ax.set_title("Residual Plot")
    ax.set_xlabel("True Risk Score")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(path_residual, dpi=300)
    plt.close(fig)


def plot_predicted_vs_actual_trust(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predictions["trust_score"], predictions["joint_correct"].astype(int), alpha=0.5, color="#9467bd")
    ax.set_title("Predicted Trust vs Actual Correctness")
    ax.set_xlabel("Predicted Trust Score")
    ax.set_ylabel("Correctness (0/1)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_true_vs_predicted_risk(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    true_risk = predictions["risk_true"].astype(float)
    pred_risk = predictions["risk_pred"].astype(float)
    ax.scatter(true_risk, pred_risk, alpha=0.55, color="#1f77b4")
    low = min(true_risk.min(), pred_risk.min())
    high = max(true_risk.max(), pred_risk.max())
    ax.plot([low, high], [low, high], linestyle="--", color="gray")
    ax.set_title("Predicted vs True Risk")
    ax.set_xlabel("True Risk Score")
    ax.set_ylabel("Predicted Risk Score")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_roc_pr(predictions: pd.DataFrame, roc_path: Path, pr_path: Path) -> None:
    y_true = predictions["joint_correct"].astype(int).to_numpy()
    y_score = predictions["trust_score"].astype(float).to_numpy()
    if len(np.unique(y_true)) < 2:
        save_placeholder_figure(roc_path, "ROC Curve", "ROC curve skipped because correctness has a single class.")
        save_placeholder_figure(pr_path, "Precision Recall Curve", "PR curve skipped because correctness has a single class.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#1f77b4", label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve for Trust Correctness")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color="#d62728", label=f"AUC={pr_auc:.3f}")
    ax.set_title("Precision Recall Curve for Trust Correctness")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pr_path, dpi=300)
    plt.close(fig)


def plot_training_history(history: pd.DataFrame, x_col: str, y_cols: list[str], title: str, ylabel: str, path: Path) -> None:
    available = [col for col in y_cols if col in history.columns]
    if history.empty or not available:
        save_placeholder_figure(path, title, f"History columns missing for {', '.join(y_cols)}.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("deep", len(available))
    for color, col in zip(palette, available, strict=False):
        ax.plot(history[x_col], history[col], marker="o", label=col, color=color)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_history_comparison(history_frames: dict[str, pd.DataFrame], metric_col: str, title: str, ylabel: str, path: Path) -> None:
    valid_frames = {name: frame for name, frame in history_frames.items() if not frame.empty and metric_col in frame.columns}
    if not valid_frames:
        save_placeholder_figure(path, title, f"Missing history for {metric_col}.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("deep", len(valid_frames))
    for color, (name, frame) in zip(palette, valid_frames.items(), strict=False):
        ax.plot(frame["epoch"], frame[metric_col], marker="o", label=name, color=color)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_feature_importance(bundle_path: Path, out_path: Path) -> None:
    if not bundle_path.exists():
        save_placeholder_figure(out_path, "Feature Importance", f"Bundle not found: {bundle_path}")
        return

    bundle = load_model_bundle(bundle_path)
    if "threat_model" not in bundle or not hasattr(bundle["threat_model"], "coef_"):
        save_placeholder_figure(out_path, "Feature Importance", "Selected model does not expose linear coefficients.")
        return

    word_names = bundle["feature_bundle"].word_vectorizer.get_feature_names_out().tolist()
    char_names = bundle["feature_bundle"].char_vectorizer.get_feature_names_out().tolist()
    dict_names = bundle["feature_bundle"].dict_vectorizer.get_feature_names_out().tolist()
    numeric_names = bundle["feature_bundle"].numeric_columns
    feature_names = word_names + char_names + dict_names + numeric_names
    coefs = np.abs(bundle["threat_model"].coef_)
    scores = coefs.mean(axis=0)
    top_idx = np.argsort(scores)[-20:]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i][:40] for i in top_idx], scores[top_idx], color="#1f77b4")
    ax.set_title("Top Feature Importance (Threat Head)")
    ax.set_xlabel("Mean Absolute Weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_bar_table(data: pd.DataFrame, x_col: str, y_col: str, title: str, path: Path) -> None:
    if data.empty or y_col not in data.columns:
        save_placeholder_figure(path, title, f"Missing data for {y_col}.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=data, x=x_col, y=y_col, hue=x_col, dodge=False, legend=False, ax=ax, palette="deep")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_pipeline_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    boxes = [
        (0.05, 0.5, "Raw Local Data"),
        (0.25, 0.5, "Preprocess + Evidence Extraction"),
        (0.47, 0.5, "DERG + CTI/MITRE Features"),
        (0.69, 0.5, "TrustSOC Models"),
        (0.88, 0.5, "Calibration + Report"),
    ]
    for x, y, text in boxes:
        ax.text(x, y, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="#dfe8f7", ec="#4c72b0"))
    for idx in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[idx + 1][0] - 0.08, boxes[idx + 1][1]), xytext=(boxes[idx][0] + 0.08, boxes[idx][1]), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("TrustSOC Research Pipeline")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_efficiency_table(data: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=data, x="Latency", y="Train Time", hue="Model", size="Params", ax=ax)
    ax.set_title("Efficiency Comparison")
    ax.set_xlabel("Latency (seconds/sample)")
    ax.set_ylabel("Train Time (minutes)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_adversarial_performance(data: pd.DataFrame, path: Path) -> None:
    plot_bar_table(data, "subset", "threat_f1_macro", "Adversarial Type Performance", path)


def plot_derg_graph(case_row: pd.Series, path: Path) -> None:
    graph, _ = graph_for_case(
        {
            **case_row.to_dict(),
            "raw_fields": json.loads(case_row["raw_fields_json"]) if "raw_fields_json" in case_row else {},
            "evidence_items": json.loads(case_row["evidence_items_json"]),
            "cti_matches": json.loads(case_row["cti_matches_json"]),
            "mitre_list": json.loads(case_row["mitre_list_json"]),
        }
    )
    pos = nx.spring_layout(graph, seed=42)
    fig, ax = plt.subplots(figsize=(10, 8))
    node_colors = []
    for node in graph.nodes:
        node_type = graph.nodes[node].get("node_type", "unknown")
        if node_type == "cti":
            node_colors.append("#d62728")
        elif node_type == "mitre":
            node_colors.append("#ff7f0e")
        elif node_type == "evidence":
            node_colors.append("#2ca02c")
        else:
            node_colors.append("#1f77b4")
    nx.draw_networkx(graph, pos=pos, ax=ax, with_labels=False, node_size=350, node_color=node_colors, arrows=True)
    label_map = {node: graph.nodes[node].get("node_type", "node") for node in graph.nodes}
    nx.draw_networkx_labels(graph, pos=pos, labels=label_map, font_size=7, ax=ax)
    ax.set_title("DERG Example Graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def create_model_figures(
    config: ProjectConfig,
    model_name: str,
    predictions_path: Path,
    metrics_path: Path,
    bundle_path: Path,
    training_history_path: Path | None = None,
    output_prefix: str | None = None,
    write_generic: bool = False,
) -> None:
    predictions = pd.read_csv(predictions_path)
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    history = pd.read_csv(training_history_path) if training_history_path and training_history_path.exists() else pd.DataFrame()

    base = config.model_figure_dir(output_prefix or model_name)
    base.mkdir(parents=True, exist_ok=True)
    prefix = ""

    def path_for(name: str) -> Path:
        return base / f"{prefix}{name}.png"

    plot_confusion(metrics["threat_type"], path_for("confusion_matrix"), normalized=False)
    plot_confusion(metrics["threat_type"], path_for("confusion_matrix_normalized"), normalized=True)
    if "severity" in metrics:
        plot_confusion(metrics["severity"], path_for("severity_confusion_matrix"), normalized=False)
    if "label" in metrics:
        plot_confusion(metrics["label"], path_for("label_confusion_matrix"), normalized=False)
    plot_distribution(predictions["trust_score"], path_for("trust_score_distribution"), "Trust Score Distribution", "Trust Score")
    plot_calibration(predictions, path_for("calibration_curve"), path_for("reliability_diagram"))
    plot_error_distribution(predictions, path_for("error_distribution"), path_for("residual_plot"))
    plot_predicted_vs_actual_trust(predictions, path_for("predicted_vs_actual_trust"))
    plot_true_vs_predicted_risk(predictions, path_for("risk_true_vs_predicted"))
    plot_roc_pr(predictions, path_for("roc_curve"), path_for("precision_recall_curve"))
    plot_feature_importance(bundle_path, path_for("feature_importance"))
    plot_training_history(history, "epoch", ["loss"], "Training Loss Curve", "Loss", path_for("training_loss_curve"))
    plot_training_history(history, "epoch", ["accuracy"], "Validation Accuracy Curve", "Accuracy", path_for("accuracy_curve"))
    plot_training_history(history, "epoch", ["f1"], "Validation F1 Curve", "F1", path_for("f1_score_curve"))
    plot_training_history(history, "epoch", ["joint_exact_match"], "Joint Exact Match Curve", "Joint Exact Match", path_for("joint_exact_curve"))
    plot_training_history(history, "epoch", ["risk_mae"], "Validation Risk MAE Curve", "Risk MAE", path_for("risk_mae_curve"))
    plot_training_history(
        history,
        "epoch",
        [col for col in ["accuracy", "f1", "joint_exact_match"] if col in history.columns],
        "Validation Accuracy/F1/Joint Curves",
        "Score",
        path_for("training_metrics_overview"),
    )
    plot_training_history(history, "epoch", ["validation_score"], "Composite Validation Score Curve", "Validation Score", path_for("validation_score_curve"))

    if write_generic:
        plot_confusion(metrics["threat_type"], base / "confusion_matrix.png", normalized=False)
        plot_confusion(metrics["threat_type"], base / "confusion_matrix_normalized.png", normalized=True)
        plot_distribution(predictions["trust_score"], base / "trust_score_distribution.png", "Trust Score Distribution", "Trust Score")
        plot_calibration(predictions, base / "calibration_curve.png", base / "reliability_diagram.png")
        plot_error_distribution(predictions, base / "error_distribution.png", base / "residual_plot.png")
        plot_predicted_vs_actual_trust(predictions, base / "predicted_vs_actual_trust.png")
        plot_true_vs_predicted_risk(predictions, base / "risk_true_vs_predicted.png")
        plot_roc_pr(predictions, base / "roc_curve.png", base / "precision_recall_curve.png")
        plot_feature_importance(bundle_path, base / "feature_importance.png")
        plot_training_history(history, "epoch", ["loss"], "Training Loss Curve", "Loss", base / "training_loss_curve.png")
        plot_training_history(history, "epoch", ["accuracy"], "Validation Accuracy Curve", "Accuracy", base / "accuracy_curve.png")
        plot_training_history(history, "epoch", ["f1"], "Validation F1 Curve", "F1", base / "f1_score_curve.png")
        plot_training_history(
            history,
            "epoch",
            [col for col in ["accuracy", "f1", "joint_exact_match"] if col in history.columns],
            "Validation Accuracy/F1/Joint Curves",
            "Score",
            base / "precision_recall_curve_epoch.png",
        )


def plot_model_radar_chart(df: pd.DataFrame, path: Path) -> None:
    """Generate polar radar chart comparing models on normalized dimensions."""
    models_to_compare = ["OpenSOC-AI", "TrustSOC-DERG"]
    sub_df = df[df["Model"].isin(models_to_compare)].copy()
    if sub_df.empty:
        save_placeholder_figure(path, "Model Comparison Radar Chart", "Main models not found in comparison data.")
        return

    categories = ["Accuracy", "Weighted F1", "Refusal Acc", "Risk R2", "Latency (Inv)"]
    N = len(categories)

    plot_data = {}
    for _, row in sub_df.iterrows():
        model = row["Model"]
        vals = []
        for cat in categories:
            val = 0.0
            if cat == "Accuracy":
                val = pd.to_numeric(row.get("Accuracy"), errors="coerce")
            elif cat == "Weighted F1":
                val = pd.to_numeric(row.get("Weighted F1"), errors="coerce")
            elif cat == "Refusal Acc":
                val = pd.to_numeric(row.get("Refusal Acc"), errors="coerce")
            elif cat == "Risk R2":
                val = max(0.0, pd.to_numeric(row.get("R2"), errors="coerce") or 0.0)
            elif cat == "Latency (Inv)":
                lat = pd.to_numeric(row.get("Latency"), errors="coerce")
                if lat and lat > 0:
                    val = min(1.0, 0.1 / lat)
                else:
                    val = 0.0
            
            if pd.isna(val) or val is None:
                val = 0.0
            vals.append(float(val))
        
        vals.append(vals[0])
        plot_data[model] = vals

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1.0)
    
    palette = sns.color_palette("deep", len(plot_data))
    for idx, (model, vals) in enumerate(plot_data.items()):
        color = palette[idx]
        ax.plot(angles, vals, linewidth=2, linestyle='solid', label=model, color=color)
        ax.fill(angles, vals, color=color, alpha=0.1)
        
    plt.title("Multi-Dimensional Model Comparison", size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
