from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .analysis_support import (
    compute_transformer_meta,
    encode_dataframe_for_transformer,
    load_transformer_analysis_context,
    run_transformer_inference,
)
from ..calibration_metrics import expected_calibration_error, reliability_diagram_data
from ..trust_calibration import decide_actions
from ..utils import get_logger, save_json

sns.set_theme(style="whitegrid")


def predict_trust_and_actions(context, df: pd.DataFrame) -> dict[str, Any]:
    encoded = encode_dataframe_for_transformer(df, context.bundle)
    outputs = run_transformer_inference(context.model, encoded, batch_size=32, collect_debug=False)
    meta_payload = compute_transformer_meta(df, outputs)
    trust_scores = context.calibrator.predict_proba(meta_payload["meta"])[:, 1]
    actions = decide_actions(
        trust_scores,
        meta_payload["uncertainty"],
        df["avg_reliability"].fillna(0.7).to_numpy(dtype=float),
        df["contradiction_score"].fillna(0.0).to_numpy(dtype=float),
        df["adversarial_noise_score"].fillna(0.0).to_numpy(dtype=float),
        np.clip(outputs["risk_pred"], 0.0, 100.0) / 100.0,
    )
    return {
        "encoded": encoded,
        "outputs": outputs,
        "meta": meta_payload,
        "trust_scores": trust_scores,
        "actions": actions,
    }


def subgroup_calibration_analysis(
    df: pd.DataFrame,
    trust_scores: np.ndarray,
    correctness: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        "all": np.ones(len(df), dtype=bool),
        "normal_case": df["adversarial_type"].eq("normal_case").to_numpy(),
        "adversarial": df["adversarial_type"].ne("normal_case").to_numpy(),
    }
    for adv_type in sorted(df["adversarial_type"].astype(str).unique()):
        groups[f"adv::{adv_type}"] = df["adversarial_type"].eq(adv_type).to_numpy()

    results: dict[str, Any] = {}
    plot_rows = []
    for name, mask in groups.items():
        if not mask.any():
            continue
        group_scores = trust_scores[mask]
        group_correct = correctness[mask].astype(int)
        results[name] = {
            "n": int(mask.sum()),
            "ece": float(expected_calibration_error(group_scores, group_correct)),
            "reliability_diagram": reliability_diagram_data(group_scores, group_correct),
        }
        for midpoint, acc, conf, count in zip(
            results[name]["reliability_diagram"]["midpoints"],
            results[name]["reliability_diagram"]["accuracies"],
            results[name]["reliability_diagram"]["confidences"],
            results[name]["reliability_diagram"]["counts"],
            strict=False,
        ):
            plot_rows.append(
                {
                    "group": name,
                    "midpoint": midpoint,
                    "accuracy": acc,
                    "confidence": conf,
                    "count": count,
                }
            )

    save_json(output_dir / "subgroup_calibration.json", results)
    plot_df = pd.DataFrame(plot_rows)
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        for group_name, group_df in plot_df[plot_df["group"].isin(["normal_case", "adversarial"])].groupby("group"):
            ax.plot(group_df["confidence"], group_df["accuracy"], marker="o", label=group_name)
        ax.set_title("Reliability Diagram: Normal vs Adversarial")
        ax.set_xlabel("Mean predicted trust")
        ax.set_ylabel("Empirical accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "reliability_normal_vs_adversarial.png", dpi=300)
        plt.close(fig)
    return results


def mc_dropout_uncertainty_analysis(context, df: pd.DataFrame, passes: int, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = encode_dataframe_for_transformer(df, context.bundle)
    action_prob_samples = []
    risk_samples = []
    for _ in range(passes):
        outputs = run_transformer_inference(context.model, encoded, batch_size=32, train_mode=True, collect_debug=False)
        logits = outputs["action_logits"]
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-6, None)
        action_prob_samples.append(probs)
        risk_samples.append(outputs["risk_pred"])

    prob_tensor = np.stack(action_prob_samples, axis=0)
    risk_tensor = np.stack(risk_samples, axis=0)
    mean_probs = prob_tensor.mean(axis=0)
    predictive_entropy = -np.sum(mean_probs * np.log(np.clip(mean_probs, 1e-8, 1.0)), axis=1)
    expected_entropy = -np.mean(np.sum(prob_tensor * np.log(np.clip(prob_tensor, 1e-8, 1.0)), axis=2), axis=0)
    epistemic = predictive_entropy - expected_entropy
    aleatoric = expected_entropy

    frame = pd.DataFrame(
        {
            "case_id": df.get("case_id", pd.Series(range(len(df)))).astype(str),
            "predictive_entropy": predictive_entropy,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "risk_std": risk_tensor.std(axis=0),
        }
    )
    frame.to_csv(output_dir / "mc_dropout_uncertainty.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(frame["epistemic_uncertainty"], kde=True, ax=ax, color="#7c3aed")
    ax.set_title("Epistemic Uncertainty Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "mc_dropout_epistemic.png", dpi=300)
    plt.close(fig)
    return frame


def numeric_counterfactual_analysis(context, df: pd.DataFrame, output_dir: Path, max_cases: int = 10) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = predict_trust_and_actions(context, df)
    trust_scores = base["trust_scores"]
    actions = base["actions"]
    candidate_rows = np.where(actions == "refuse")[0][:max_cases]
    quantiles = df[context.bundle["numeric_columns"]].quantile([0.05, 0.95]).to_dict()
    increase_cols = ["avg_reliability", "max_reliability", "evidence_consistency", "cti_match_score", "reliability_score"]
    decrease_cols = ["contradiction_score", "adversarial_noise_score", "conflicting_evidence_ratio", "high_risk_node_ratio"]

    rows: list[dict[str, Any]] = []
    for idx in candidate_rows:
        base_row = df.iloc[idx].copy()
        baseline_action = actions[idx]
        baseline_trust = trust_scores[idx]
        best: dict[str, Any] | None = None
        for column in increase_cols + decrease_cols:
            if column not in df.columns:
                continue
            current_value = float(base_row.get(column, 0.0))
            targets = []
            if column in increase_cols:
                hi = float(quantiles[column][0.95])
                targets = np.linspace(current_value, max(current_value, hi), 6)[1:]
            else:
                lo = float(quantiles[column][0.05])
                targets = np.linspace(current_value, min(current_value, lo), 6)[1:]
            for target_value in targets:
                candidate = base_row.copy()
                candidate[column] = float(target_value)
                candidate_df = pd.DataFrame([candidate])
                candidate_result = predict_trust_and_actions(context, candidate_df)
                new_action = str(candidate_result["actions"][0])
                new_trust = float(candidate_result["trust_scores"][0])
                if new_action == "conclude":
                    delta = abs(float(target_value) - current_value)
                    if best is None or delta < best["absolute_delta"]:
                        best = {
                            "case_index": int(idx),
                            "case_id": str(base_row.get("case_id", idx)),
                            "from_action": baseline_action,
                            "to_action": new_action,
                            "feature": column,
                            "original_value": current_value,
                            "counterfactual_value": float(target_value),
                            "absolute_delta": delta,
                            "baseline_trust": float(baseline_trust),
                            "counterfactual_trust": new_trust,
                        }
                    break
        if best is not None:
            rows.append(best)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "numeric_counterfactuals.csv", index=False, encoding="utf-8")
    return frame


def error_decomposition_analysis(predictions: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    threat_correct = predictions["threat_true"] == predictions["threat_pred"]
    severity_correct = predictions["severity_true"] == predictions["severity_pred"]
    label_correct = predictions["label_true"] == predictions["label_pred"]
    action_correct = predictions["expected_action_true"] == predictions["expected_action_pred"]

    summary = {
        "p_severity_correct_given_threat_correct": float(severity_correct[threat_correct].mean()) if threat_correct.any() else 0.0,
        "p_label_correct_given_threat_and_severity_correct": float(label_correct[threat_correct & severity_correct].mean()) if (threat_correct & severity_correct).any() else 0.0,
        "p_action_correct_given_joint_correct": float(action_correct[predictions["joint_correct"]].mean()) if predictions["joint_correct"].any() else 0.0,
    }
    tensor = (
        predictions.assign(
            threat_correct=threat_correct,
            severity_correct=severity_correct,
            label_correct=label_correct,
            action_correct=action_correct,
        )
        .groupby(["threat_correct", "severity_correct", "label_correct", "action_correct"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    tensor.to_csv(output_dir / "error_decomposition_tensor.csv", index=False, encoding="utf-8")
    save_json(output_dir / "error_decomposition_summary.json", summary)
    return summary


def _entity_substitute(text: str) -> str:
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "198.51.100.42", str(text))
    text = re.sub(r"\b[a-zA-Z0-9.-]+\.(?:com|net|org|local|io)\b", "example-security.local", text)
    text = re.sub(r"https?://\S+", "https://example-security.local", text)
    return text


def _semantic_equivalent(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text.replace(" - ", " ").replace(" ,", ",")


def behavioral_testing_analysis(context, df: pd.DataFrame, output_dir: Path, max_cases: int = 40) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = df.head(max_cases).copy()
    baseline = predict_trust_and_actions(context, subset)

    entity_df = subset.copy()
    entity_df["event_text"] = entity_df["event_text"].astype(str).map(_entity_substitute)
    entity_result = predict_trust_and_actions(context, entity_df)

    semantic_df = subset.copy()
    semantic_df["event_text"] = semantic_df["event_text"].astype(str).map(_semantic_equivalent)
    semantic_result = predict_trust_and_actions(context, semantic_df)

    monotonic_df = subset.copy()
    monotonic_df["avg_reliability"] = np.clip(monotonic_df["avg_reliability"].fillna(0.7) + 0.1, 0.0, 1.0)
    monotonic_df["evidence_consistency"] = np.clip(monotonic_df["evidence_consistency"].fillna(1.0) + 0.1, 0.0, 1.0)
    monotonic_result = predict_trust_and_actions(context, monotonic_df)

    summary = {
        "entity_substitution_invariance": float((baseline["actions"] == entity_result["actions"]).mean()),
        "semantic_equivalence_invariance": float((baseline["actions"] == semantic_result["actions"]).mean()),
        "monotonicity_pass_rate": float((monotonic_result["trust_scores"] >= baseline["trust_scores"] - 1e-6).mean()),
    }
    save_json(output_dir / "behavioral_testing.json", summary)
    return summary


def run_deep_analysis(config, model_name: str = "trustsoc_derg", passes: int = 30, fail_on_missing: bool = True) -> dict[str, Any]:
    logger = get_logger(config, "deep_analysis")
    output_dir = config.deep_analysis_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        context = load_transformer_analysis_context(config, model_name=model_name)
    except Exception as exc:
        summary = {
            "model_name": model_name,
            "status": "skipped",
            "reason": str(exc),
        }
        save_json(output_dir / "deep_analysis_summary.json", summary)
        logger.warning("Skipping deep-analysis suite for %s: %s", model_name, exc)
        if fail_on_missing:
            raise
        return summary

    base = predict_trust_and_actions(context, context.dataframe)
    predictions = pd.DataFrame(
        {
            "case_id": context.dataframe.get("case_id", pd.Series(range(len(context.dataframe)))).astype(str),
            "threat_true": context.dataframe["threat_type"].astype(str),
            "threat_pred": np.asarray(context.bundle["threat_classes"])[base["outputs"]["threat_logits"].argmax(axis=1)],
            "severity_true": context.dataframe["severity"].astype(str),
            "severity_pred": np.asarray(context.bundle["severity_classes"])[base["outputs"]["severity_logits"].argmax(axis=1)],
            "label_true": context.dataframe["label"].astype(str),
            "label_pred": np.asarray(context.bundle["label_classes"])[base["outputs"]["label_logits"].argmax(axis=1)],
            "expected_action_true": context.dataframe["expected_action_target"].astype(str),
            "expected_action_pred": base["actions"],
        }
    )
    correctness = (
        (predictions["threat_true"] == predictions["threat_pred"])
        & (predictions["severity_true"] == predictions["severity_pred"])
        & (predictions["label_true"] == predictions["label_pred"])
    ).to_numpy()
    predictions["joint_correct"] = correctness
    predictions.to_csv(output_dir / "analysis_predictions.csv", index=False, encoding="utf-8")

    calibration = subgroup_calibration_analysis(context.dataframe, base["trust_scores"], correctness, output_dir / "calibration")
    mc_dropout = mc_dropout_uncertainty_analysis(context, context.dataframe, passes=passes, output_dir=output_dir / "mc_dropout")
    counterfactuals = numeric_counterfactual_analysis(context, context.dataframe, output_dir / "counterfactuals")
    error_summary = error_decomposition_analysis(predictions, output_dir / "error_decomposition")
    behavioral = behavioral_testing_analysis(context, context.dataframe, output_dir / "behavioral_tests")

    summary = {
        "model_name": model_name,
        "n_samples": int(len(context.dataframe)),
        "subgroup_calibration_groups": sorted(calibration.keys()),
        "mc_dropout_rows": int(len(mc_dropout)),
        "counterfactual_rows": int(len(counterfactuals)),
        "error_decomposition": error_summary,
        "behavioral_testing": behavioral,
    }
    save_json(output_dir / "deep_analysis_summary.json", summary)
    logger.info("Saved deep-analysis outputs to %s", output_dir)
    return summary
