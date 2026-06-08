from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .analysis_support import (
    compute_transformer_meta,
    decode_predictions,
    encode_dataframe_for_transformer,
    load_transformer_analysis_context,
    run_transformer_inference,
)
from .case_study import export_case_studies
from ..data_loader import load_robustness_split
from ..trust_calibration import decide_actions
from ..utils import get_logger, save_json

sns.set_theme(style="whitegrid")


def _source_group(source_file: Any) -> str:
    source = str(source_file).lower()
    if "guide_test" in source:
        return "guide_test"
    if "guide_train" in source:
        return "guide_train"
    if "otx" in source:
        return "otx"
    if "cve" in source:
        return "cve"
    if "domains" in source:
        return "domains"
    if "ips" in source:
        return "ips"
    return "other"


def _safe_rate(mask: pd.Series | np.ndarray) -> float:
    arr = np.asarray(mask, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def _safe_subset_rate(values: pd.Series, condition: pd.Series) -> float:
    if int(condition.sum()) == 0:
        return 0.0
    return float(values.loc[condition].mean())


def _infer_predictions_for_dataframe(
    context: Any,
    dataframe: pd.DataFrame,
    scenario_name: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, float]:
    encoded = encode_dataframe_for_transformer(dataframe, context.bundle)
    started = time.perf_counter()
    outputs = run_transformer_inference(context.model, encoded, batch_size=64, collect_debug=False)
    elapsed = time.perf_counter() - started
    meta_payload = compute_transformer_meta(dataframe, outputs)
    trust_scores = context.calibrator.predict_proba(meta_payload["meta"])[:, 1]
    actions = decide_actions(
        trust_scores,
        meta_payload["uncertainty"],
        dataframe["avg_reliability"].fillna(0.7).to_numpy(dtype=float),
        dataframe["contradiction_score"].fillna(0.0).to_numpy(dtype=float),
        dataframe["adversarial_noise_score"].fillna(0.0).to_numpy(dtype=float),
        np.clip(outputs["risk_pred"], 0.0, 100.0) / 100.0,
    )

    threat_pred = decode_predictions(outputs["threat_logits"], context.bundle["threat_classes"]).astype(str)
    severity_pred = decode_predictions(outputs["severity_logits"], context.bundle["severity_classes"]).astype(str)
    label_pred = decode_predictions(outputs["label_logits"], context.bundle["label_classes"]).astype(str)

    predictions = pd.DataFrame(
        {
            "scenario": scenario_name,
            "case_id": dataframe["case_id"].astype(str).to_numpy(),
            "source_file": dataframe["source_file"].astype(str).to_numpy(),
            "source_group": dataframe["source_file"].map(_source_group).astype(str).to_numpy(),
            "threat_true": dataframe["threat_type"].astype(str).to_numpy(),
            "threat_pred": threat_pred,
            "severity_true": dataframe["severity"].astype(str).to_numpy(),
            "severity_pred": severity_pred,
            "label_true": dataframe["label"].astype(str).to_numpy(),
            "label_pred": label_pred,
            "expected_action_true": dataframe["expected_action_target"].astype(str).to_numpy(),
            "expected_action_pred": actions.astype(str),
            "adversarial_type": dataframe["adversarial_type"].astype(str).to_numpy(),
            "trust_score": trust_scores.astype(float),
            "uncertainty_score": meta_payload["uncertainty"].astype(float),
            "risk_true": dataframe["risk_score"].to_numpy(dtype=float),
            "risk_pred": outputs["risk_pred"].astype(float),
            "joint_correct": (
                (threat_pred == dataframe["threat_type"].astype(str).to_numpy())
                & (severity_pred == dataframe["severity"].astype(str).to_numpy())
                & (label_pred == dataframe["label"].astype(str).to_numpy())
            ),
        }
    )
    predictions.to_csv(output_dir / f"{scenario_name}_predictions.csv", index=False, encoding="utf-8")
    return predictions, elapsed


def _summarize_operational_scenario(
    predictions: pd.DataFrame,
    elapsed_seconds: float,
) -> dict[str, Any]:
    n_rows = len(predictions)
    conclude_mask = predictions["expected_action_pred"].eq("conclude")
    non_conclude_mask = ~conclude_mask
    true_non_conclude_mask = predictions["expected_action_true"].ne("conclude")
    adversarial_mask = predictions["adversarial_type"].ne("normal_case")
    high_severity_mask = predictions["severity_true"].isin(["HIGH", "CRITICAL"])

    return {
        "scenario": predictions["scenario"].iloc[0],
        "n_rows": n_rows,
        "threat_accuracy": _safe_rate(predictions["threat_true"].eq(predictions["threat_pred"])),
        "joint_accuracy": _safe_rate(predictions["joint_correct"]),
        "action_accuracy": _safe_rate(predictions["expected_action_true"].eq(predictions["expected_action_pred"])),
        "conclude_rate": _safe_rate(conclude_mask),
        "investigate_rate": _safe_rate(predictions["expected_action_pred"].eq("investigate")),
        "escalate_rate": _safe_rate(predictions["expected_action_pred"].eq("escalate")),
        "refuse_rate": _safe_rate(predictions["expected_action_pred"].eq("refuse")),
        "manual_review_rate": _safe_rate(non_conclude_mask),
        "manual_reviews_saved_per_1000": float(_safe_rate(conclude_mask) * 1000.0),
        "autonomous_precision": _safe_subset_rate(
            predictions["expected_action_true"].eq("conclude"),
            conclude_mask,
        ),
        "risky_auto_closure_rate": _safe_rate(
            conclude_mask & predictions["expected_action_true"].ne("conclude")
        ),
        "safe_handoff_rate": _safe_subset_rate(non_conclude_mask, true_non_conclude_mask),
        "adversarial_capture_rate": _safe_subset_rate(non_conclude_mask, adversarial_mask),
        "high_severity_review_rate": _safe_subset_rate(non_conclude_mask, high_severity_mask),
        "mean_trust_score": float(predictions["trust_score"].mean()),
        "median_trust_score": float(predictions["trust_score"].median()),
        "latency_ms_per_sample": float((elapsed_seconds / max(n_rows, 1)) * 1000.0),
    }


def _plot_action_distribution(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df[
        ["scenario", "conclude_rate", "investigate_rate", "escalate_rate", "refuse_rate"]
    ].melt(id_vars="scenario", var_name="action", value_name="rate")
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(summary_df), dtype=float)
    palette = {
        "conclude_rate": "#2563eb",
        "investigate_rate": "#f59e0b",
        "escalate_rate": "#dc2626",
        "refuse_rate": "#111827",
    }
    for action in ["conclude_rate", "investigate_rate", "escalate_rate", "refuse_rate"]:
        values = summary_df[action].to_numpy(dtype=float)
        ax.bar(summary_df["scenario"], values, bottom=bottom, label=action.replace("_rate", ""), color=palette[action])
        bottom += values
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Operational Action Distribution by Scenario")
    ax.legend(title="Predicted action")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_source_accuracy(source_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = source_df.melt(
        id_vars=["source_group", "n_rows"],
        value_vars=["threat_accuracy", "action_accuracy", "conclude_rate"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="source_group", y="value", hue="metric", ax=ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score / rate")
    ax.set_title("Cross-Source Deployment Audit")
    ax.set_xlabel("Source group")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_workload(workload_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = workload_df.melt(
        id_vars="scenario",
        value_vars=["auto_closed_alerts", "manual_review_alerts", "escalated_alerts", "refused_alerts"],
        var_name="bucket",
        value_name="alerts_per_1000",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="scenario", y="alerts_per_1000", hue="bucket", ax=ax)
    ax.set_ylabel("Alerts per 1000 incoming alerts")
    ax.set_title("Analyst Workload Simulation")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _build_source_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_group, part in predictions.groupby("source_group", dropna=False):
        rows.append(
            {
                "source_group": source_group,
                "n_rows": len(part),
                "threat_accuracy": _safe_rate(part["threat_true"].eq(part["threat_pred"])),
                "action_accuracy": _safe_rate(part["expected_action_true"].eq(part["expected_action_pred"])),
                "joint_accuracy": _safe_rate(part["joint_correct"]),
                "mean_trust_score": float(part["trust_score"].mean()),
                "conclude_rate": _safe_rate(part["expected_action_pred"].eq("conclude")),
                "investigate_rate": _safe_rate(part["expected_action_pred"].eq("investigate")),
                "escalate_rate": _safe_rate(part["expected_action_pred"].eq("escalate")),
                "refuse_rate": _safe_rate(part["expected_action_pred"].eq("refuse")),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_rows", "source_group"], ascending=[False, True])


def _build_adversarial_audit(predictions_by_scenario: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_name, predictions in predictions_by_scenario.items():
        subset = predictions[predictions["adversarial_type"].ne("normal_case")].copy()
        if subset.empty:
            continue
        for adv_type, part in subset.groupby("adversarial_type", dropna=False):
            non_conclude = part["expected_action_pred"].ne("conclude")
            rows.append(
                {
                    "scenario": scenario_name,
                    "adversarial_type": adv_type,
                    "n_rows": len(part),
                    "threat_accuracy": _safe_rate(part["threat_true"].eq(part["threat_pred"])),
                    "action_accuracy": _safe_rate(part["expected_action_true"].eq(part["expected_action_pred"])),
                    "non_conclude_rate": _safe_rate(non_conclude),
                    "refuse_rate": _safe_rate(part["expected_action_pred"].eq("refuse")),
                    "mean_trust_score": float(part["trust_score"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario", "n_rows"], ascending=[True, False])


def _build_workload_simulation(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        rows.append(
            {
                "scenario": row["scenario"],
                "auto_closed_alerts": float(row["conclude_rate"] * 1000.0),
                "manual_review_alerts": float(row["investigate_rate"] * 1000.0),
                "escalated_alerts": float(row["escalate_rate"] * 1000.0),
                "refused_alerts": float(row["refuse_rate"] * 1000.0),
                "risky_auto_closures": float(row["risky_auto_closure_rate"] * 1000.0),
            }
        )
    return pd.DataFrame(rows)


def _write_summary_markdown(
    summary_df: pd.DataFrame,
    source_df: pd.DataFrame,
    adversarial_df: pd.DataFrame,
    workload_df: pd.DataFrame,
    output_path: Path,
    case_study_paths: dict[str, str],
) -> None:
    clean_row = summary_df.loc[summary_df["scenario"].eq("clean_test")].iloc[0]
    noisy_row = summary_df.loc[summary_df["scenario"].eq("noisy_evidence")].iloc[0]
    top_source = source_df.sort_values("threat_accuracy", ascending=False).iloc[0]
    weakest_source = source_df.sort_values("threat_accuracy", ascending=True).iloc[0]

    lines = [
        "# Practical Experiments",
        "",
        "## Operational Triage",
        (
            f"- On clean held-out alerts, TrustSOC-DERG auto-concludes "
            f"{clean_row['conclude_rate']:.2%} of cases, reducing manual review by the same amount, "
            f"with action accuracy {clean_row['action_accuracy']:.2%} and autonomous precision "
            f"{clean_row['autonomous_precision']:.2%}."
        ),
        (
            f"- Under noisy evidence stress, manual review demand rises to "
            f"{noisy_row['manual_review_rate']:.2%}, while action accuracy drops to "
            f"{noisy_row['action_accuracy']:.2%} and refusal remains rare at "
            f"{noisy_row['refuse_rate']:.2%}."
        ),
        "",
        "## Cross-Source Deployment",
        (
            f"- The strongest source group is `{top_source['source_group']}` with threat accuracy "
            f"{top_source['threat_accuracy']:.2%}; the weakest is `{weakest_source['source_group']}` "
            f"with threat accuracy {weakest_source['threat_accuracy']:.2%}."
        ),
        (
            "- This split exposes a practical deployment gap: GUIDE-style incidents remain easy, "
            "while CTI-heavy source groups are often routed to investigate instead of conclude."
        ),
        "",
        "## Adversarial Audit",
        (
            f"- Adversarial detail rows exported: {len(adversarial_df)}. "
            "Use them to report where the model safely defers and where it still under-refuses."
        ),
        "",
        "## Workload Simulation",
        (
            f"- Clean-test simulation saves about {clean_row['manual_reviews_saved_per_1000']:.0f} "
            "manual reviews per 1000 alerts relative to a fully manual SOC baseline."
        ),
        "",
        "## Case Studies",
        f"- Clean-test cases: `{case_study_paths['clean_test']}`",
        f"- Noisy-evidence cases: `{case_study_paths['noisy_evidence']}`",
        "",
        "## Core Tables",
        "- `operational_scenarios.csv`",
        "- `source_audit.csv`",
        "- `adversarial_audit.csv`",
        "- `workload_simulation.csv`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_practical_experiments(
    config: Any,
    model_name: str = "trustsoc_derg",
    fail_on_missing: bool = True,
) -> dict[str, Any]:
    logger = get_logger(config, "practical_experiments")
    output_dir = config.artifacts_dir / "practical_experiments" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        context = load_transformer_analysis_context(config, model_name=model_name, split_name="test")
    except Exception as exc:
        summary = {
            "model_name": model_name,
            "status": "skipped",
            "reason": str(exc),
        }
        save_json(output_dir / "practical_experiments_summary.json", summary)
        logger.warning("Skipping practical experiments for %s: %s", model_name, exc)
        if fail_on_missing:
            raise
        return summary

    scenario_dataframes: dict[str, pd.DataFrame] = {"clean_test": context.dataframe}
    for split_name in ("adversarial", "missing_cti", "missing_mitre", "noisy_evidence"):
        try:
            scenario_dataframes[split_name] = load_robustness_split(config, split_name)
        except Exception as exc:
            logger.warning("Skipping robustness split %s: %s", split_name, exc)

    predictions_by_scenario: dict[str, pd.DataFrame] = {}
    operational_rows: list[dict[str, Any]] = []
    case_study_paths: dict[str, str] = {}

    for scenario_name, dataframe in scenario_dataframes.items():
        predictions, elapsed = _infer_predictions_for_dataframe(context, dataframe, scenario_name, output_dir)
        predictions_by_scenario[scenario_name] = predictions
        operational_rows.append(_summarize_operational_scenario(predictions, elapsed))

    summary_df = pd.DataFrame(operational_rows).sort_values("scenario")
    summary_df.to_csv(output_dir / "operational_scenarios.csv", index=False, encoding="utf-8")
    _plot_action_distribution(summary_df, output_dir / "operational_action_distribution.png")

    source_df = _build_source_audit(predictions_by_scenario["clean_test"])
    source_df.to_csv(output_dir / "source_audit.csv", index=False, encoding="utf-8")
    _plot_source_accuracy(source_df, output_dir / "source_audit.png")

    adversarial_df = _build_adversarial_audit(predictions_by_scenario)
    adversarial_df.to_csv(output_dir / "adversarial_audit.csv", index=False, encoding="utf-8")

    workload_df = _build_workload_simulation(summary_df)
    workload_df.to_csv(output_dir / "workload_simulation.csv", index=False, encoding="utf-8")
    _plot_workload(workload_df, output_dir / "workload_simulation.png")

    clean_cases_dir = output_dir / "case_studies" / "clean_test"
    noisy_cases_dir = output_dir / "case_studies" / "noisy_evidence"
    export_case_studies(
        scenario_dataframes["clean_test"],
        predictions_by_scenario["clean_test"]["trust_score"].to_numpy(dtype=float),
        predictions_by_scenario["clean_test"]["expected_action_pred"].to_numpy(dtype=str),
        clean_cases_dir,
        n_cases=6,
    )
    case_study_paths["clean_test"] = str((clean_cases_dir / "case_studies.md").resolve())

    if "noisy_evidence" in scenario_dataframes:
        export_case_studies(
            scenario_dataframes["noisy_evidence"],
            predictions_by_scenario["noisy_evidence"]["trust_score"].to_numpy(dtype=float),
            predictions_by_scenario["noisy_evidence"]["expected_action_pred"].to_numpy(dtype=str),
            noisy_cases_dir,
            n_cases=6,
        )
        case_study_paths["noisy_evidence"] = str((noisy_cases_dir / "case_studies.md").resolve())
    else:
        case_study_paths["noisy_evidence"] = ""

    _write_summary_markdown(
        summary_df,
        source_df,
        adversarial_df,
        workload_df,
        output_dir / "practical_experiments.md",
        case_study_paths,
    )

    clean_row = summary_df.loc[summary_df["scenario"].eq("clean_test")].iloc[0]
    noisy_row = summary_df.loc[summary_df["scenario"].eq("noisy_evidence")].iloc[0]
    summary = {
        "model_name": model_name,
        "status": "completed",
        "scenarios": summary_df["scenario"].tolist(),
        "highlights": {
            "clean_manual_reviews_saved_per_1000": float(clean_row["manual_reviews_saved_per_1000"]),
            "clean_action_accuracy": float(clean_row["action_accuracy"]),
            "clean_autonomous_precision": float(clean_row["autonomous_precision"]),
            "noisy_evidence_action_accuracy": float(noisy_row["action_accuracy"]),
            "noisy_evidence_refuse_rate": float(noisy_row["refuse_rate"]),
            "lowest_source_accuracy_group": str(source_df.sort_values("threat_accuracy", ascending=True).iloc[0]["source_group"]),
        },
        "artifacts": {
            "operational_scenarios": str((output_dir / "operational_scenarios.csv").resolve()),
            "source_audit": str((output_dir / "source_audit.csv").resolve()),
            "adversarial_audit": str((output_dir / "adversarial_audit.csv").resolve()),
            "workload_simulation": str((output_dir / "workload_simulation.csv").resolve()),
            "markdown_report": str((output_dir / "practical_experiments.md").resolve()),
            "clean_case_studies": case_study_paths["clean_test"],
            "noisy_evidence_case_studies": case_study_paths["noisy_evidence"],
        },
    }
    save_json(output_dir / "practical_experiments_summary.json", summary)
    logger.info("Saved practical experiment outputs to %s", output_dir)
    return summary
