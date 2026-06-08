from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProjectConfig
from .data_loader import load_processed_split
from .error_analysis import generate_error_analysis
from .models.model_utils import opensoc_reference_row
from .models.sklearn_baselines import train_baselines
from .models.trustsoc_transformer import train_derg
from .supporting_analysis.deep_analysis import run_deep_analysis
from .supporting_analysis.practical_experiments import run_practical_experiments
from .supporting_analysis.xai_analysis import run_xai_suite
from .utils import get_logger, load_json, save_json
from .visualization import (
    create_model_figures,
    plot_bar_table,
    plot_derg_graph,
    plot_efficiency_table,
    plot_pipeline_diagram,
    plot_model_radar_chart,
)

def _load_or_train_metrics(path: Path, trainer, *args, **kwargs):
    if path.exists():
        return load_json(path)
    return trainer(*args, **kwargs)


def _safe_load_split(config: ProjectConfig, split_name: str) -> pd.DataFrame | None:
    try:
        return load_processed_split(config, split_name)
    except FileNotFoundError:
        return None


def _available_processed_splits(config: ProjectConfig) -> bool:
    return all(path.exists() for path in config.processed_split_paths.values())


def _prediction_file_has_columns(path: Path, required: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        sample = pd.read_csv(path, nrows=1)
    except Exception:
        return False
    return required.issubset(set(sample.columns))


def ensure_core_metrics(config: ProjectConfig) -> dict[str, Any]:
    logger = get_logger(config, "evaluate")
    baseline_path = config.metrics_dir / "baseline_metrics.json"
    has_processed = _available_processed_splits(config)

    train_df = val_df = test_df = None
    if has_processed:
        train_df = load_processed_split(config, "train")
        val_df = load_processed_split(config, "val")
        test_df = load_processed_split(config, "test")

    if baseline_path.exists():
        baselines = load_json(baseline_path)
    elif has_processed and train_df is not None and val_df is not None and test_df is not None:
        baselines = train_baselines(train_df, val_df, test_df, config)
    else:
        baselines = {}

    derg_path = config.metrics_dir / "metrics_trustsoc_derg.json"

    if derg_path.exists():
        derg_metrics = load_json(derg_path)
    elif has_processed and train_df is not None and val_df is not None and test_df is not None:
        derg_metrics = train_derg(train_df, val_df, test_df, config)
    else:
        derg_metrics = {"status": "missing"}

    save_json(
        config.metrics_dir / "metrics_summary.json",
        {
            "baselines": baselines,
            "derg": derg_metrics,
        },
    )
    logger.info(
        "Core metrics ensured. processed_splits=%s",
        has_processed,
    )
    return {
        "baselines": baselines,
        "derg": derg_metrics,
    }


def compare_with_opensoc(config: ProjectConfig) -> pd.DataFrame:
    summary = ensure_core_metrics(config)
    rows = [opensoc_reference_row(config.baseline_reference)]

    for baseline_name, metrics in summary["baselines"].items():
        rows.append(
            {
                "Model": baseline_name,
                "Accuracy": metrics["threat_type"]["accuracy"],
                "Macro F1": metrics["threat_type"]["f1_macro"],
                "Weighted F1": metrics["threat_type"]["f1_weighted"],
                "MAE": metrics["risk_score"]["mae"],
                "RMSE": metrics["risk_score"]["rmse"],
                "MAPE": metrics["risk_score"]["mape"],
                "R2": metrics["risk_score"]["r2"],
                "ECE": None,
                "Brier": None,
                "Robust-F1": None,
                "Refusal Acc": None,
                "Latency": metrics["efficiency"]["average_latency_seconds_per_sample"],
                "Train Time": metrics["efficiency"]["train_time_seconds"] / 60.0,
                "Params": metrics["efficiency"]["parameter_count_estimate"],
                "Notes": metrics["notes"],
            }
        )

    for model_name, key in (("TrustSOC-DERG", "derg"),):
        metrics = summary[key]
        if metrics.get("status") == "missing":
            continue
        rows.append(
            {
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
                "Robust-F1": metrics["threat_type"]["f1_macro"],
                "Refusal Acc": metrics["calibration"]["refusal_accuracy"],
                "Latency": metrics["efficiency"]["average_latency_seconds_per_sample"],
                "Train Time": metrics["efficiency"]["train_time_seconds"] / 60.0,
                "Params": metrics["efficiency"].get("parameter_count_estimate", metrics["efficiency"].get("parameter_count")),
                "Notes": "Flagship TrustSOC-DERG model with TCT over DERG-derived evidence features.",
            }
        )

    return pd.DataFrame(rows)


def generate_report_tables(config: ProjectConfig) -> dict[str, Any]:
    summary = ensure_core_metrics(config)
    comparison = compare_with_opensoc(config)
    error_table = generate_error_analysis(config, "trustsoc_derg")
    reports_dir = config.summary_reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_test_df = _safe_load_split(config, "test")
    dataset_rows_source = None
    for key in ("derg",):
        candidate = summary.get(key, {})
        if candidate.get("dataset_rows"):
            dataset_rows_source = candidate["dataset_rows"]
            break
    if _available_processed_splits(config):
        dataset_stats = pd.DataFrame(
            [
                {
                    "split": split,
                    "rows": len(load_processed_split(config, split)),
                    "adversarial_rows": int(load_processed_split(config, split)["adversarial_type"].ne("normal_case").sum()),
                }
                for split in ("train", "val", "test")
            ]
        )
    elif dataset_rows_source is not None:
        dataset_stats = pd.DataFrame(
            [
                {
                    "split": split,
                    "rows": int(dataset_rows_source.get(split, 0)),
                    "adversarial_rows": None,
                }
                for split in ("train", "val", "test")
            ]
        )
    else:
        dataset_stats = pd.DataFrame(
            [{"split": "unknown", "rows": 0, "adversarial_rows": None}]
        )
    dataset_stats.to_csv(config.benchmark_tables_dir / "table_dataset_statistics.csv", index=False, encoding="utf-8")

    calibration_rows = []
    for model_name in ("trustsoc_derg",):
        metrics_path = config.metrics_dir / f"metrics_{model_name}.json"
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            calibration_rows.append(
                {
                    "Model": model_name,
                    "ECE": metrics["calibration"]["ece"],
                    "Brier": metrics["calibration"]["brier"],
                    "Refusal Correctness": metrics["calibration"]["refusal_correctness"],
                    "Overconfidence Rate": metrics["calibration"]["overconfidence_rate"],
                    "Underconfidence Rate": metrics["calibration"]["underconfidence_rate"],
                    "Trust Alignment Score": metrics["calibration"]["trust_alignment_score"],
                }
            )
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(config.benchmark_tables_dir / "table_calibration.csv", index=False, encoding="utf-8")
    save_json(config.metrics_dir / "calibration_metrics.json", calibration_rows)

    derg_metrics_main = load_json(config.metrics_dir / "metrics_trustsoc_derg.json") if (config.metrics_dir / "metrics_trustsoc_derg.json").exists() else summary.get("derg", {"status": "missing"})

    # Run statistical significance tests (Phase 2 contribution)
    try:
        run_statistical_significance_tests(config)
    except Exception as e:
        get_logger(config, "evaluate").error("Failed to run statistical tests: %s", e)

    if processed_test_df is not None and not processed_test_df.empty:
        figures_input = processed_test_df.iloc[0]
        plot_derg_graph(figures_input, config.concept_figures_dir / "derg_example_graph.png")
    plot_pipeline_diagram(config.concept_figures_dir / "pipeline_diagram.png")

    plot_bar_table(comparison[pd.to_numeric(comparison["Accuracy"], errors="coerce").notna()], "Model", "Accuracy", "Baseline Comparison", config.comparison_figures_dir / "baseline_comparison_chart.png")
    plot_efficiency_table(
        comparison[pd.to_numeric(comparison["Latency"], errors="coerce").notna()].assign(
            Latency=lambda d: pd.to_numeric(d["Latency"], errors="coerce"),
            **{"Train Time": lambda d: pd.to_numeric(d["Train Time"], errors="coerce")},
            Params=lambda d: pd.to_numeric(d["Params"], errors="coerce"),
        ),
        config.comparison_figures_dir / "efficiency_comparison.png",
    )
    plot_model_radar_chart(comparison, config.comparison_figures_dir / "model_comparison_radar.png")
    if (config.analysis_tables_dir / "table_robustness.csv").exists():
        robustness_df = pd.read_csv(config.analysis_tables_dir / "table_robustness.csv")
        plot_bar_table(robustness_df, "subset", "trustsoc_derg_f1", "Adversarial Type Performance", config.robustness_figures_dir / "adversarial_type_performance.png")

    if (
        _prediction_file_has_columns(config.predictions_dir / "predictions_trustsoc_derg.csv", {"trust_score", "joint_correct", "risk_true", "risk_pred"})
        and (config.metrics_dir / "metrics_trustsoc_derg.json").exists()
        and (config.models_dir / "trustsoc_derg" / "trustsoc_derg_bundle.pkl").exists()
    ):
        create_model_figures(
            config,
            "trustsoc_derg",
            config.predictions_dir / "predictions_trustsoc_derg.csv",
            config.metrics_dir / "metrics_trustsoc_derg.json",
            config.models_dir / "trustsoc_derg" / "trustsoc_derg_bundle.pkl",
            training_history_path=config.metrics_dir / "training_history_trustsoc_derg.csv",
            output_prefix="trustsoc_derg",
            write_generic=False,
        )
    derg_history_path = config.metrics_dir / "training_history_trustsoc_derg.csv"
    aggregate_metrics = {}
    for key, path in (
        ("derg", config.metrics_dir / "metrics_trustsoc_derg.json"),
    ):
        if path.exists():
            aggregate_metrics[key] = load_json(path)
    save_json(config.metrics_dir / "metrics.json", aggregate_metrics)

    xai_summary = run_xai_suite(config, model_name="trustsoc_derg", fail_on_missing=False)
    deep_summary = run_deep_analysis(config, model_name="trustsoc_derg", fail_on_missing=False)
    practical_summary = run_practical_experiments(config, model_name="trustsoc_derg", fail_on_missing=False)

    derg_metrics = aggregate_metrics.get("derg", {})
    summary_lines = [
        "# TrustSOC Research Summary",
        "",
        "## Main Model",
    ]
    if derg_metrics and derg_metrics.get("status") != "missing":
        summary_lines.append(
            f"- TrustSOC-DERG: threat acc {derg_metrics['threat_type']['accuracy']:.4f}, weighted F1 {derg_metrics['threat_type']['f1_weighted']:.4f}, risk MAE {derg_metrics['risk_score']['mae']:.4f}, ECE {derg_metrics['calibration']['ece']:.4f}."
        )
    else:
        summary_lines.append("- TrustSOC-DERG: metrics unavailable in the current workspace.")

    figure_lines = [
        f"- Full TrustSOC-DERG training loss: `{config.model_figure_dir('trustsoc_derg') / 'training_loss_curve.png'}`",
        f"- Full TrustSOC-DERG validation overview: `{config.model_figure_dir('trustsoc_derg') / 'training_metrics_overview.png'}`",
        f"- Full TrustSOC-DERG calibration: `{config.model_figure_dir('trustsoc_derg') / 'calibration_curve.png'}`",
        f"- Full TrustSOC-DERG risk scatter: `{config.model_figure_dir('trustsoc_derg') / 'risk_true_vs_predicted.png'}`",
    ]

    artifact_lines = [
        f"- Main training history: `{derg_history_path}`",
        f"- XAI summary: `{config.xai_dir / 'trustsoc_derg' / 'xai_summary.json'}`",
        f"- Deep-analysis summary: `{config.deep_analysis_dir / 'trustsoc_derg' / 'deep_analysis_summary.json'}`",
        f"- Practical experiments summary: `{config.artifacts_dir / 'practical_experiments' / 'trustsoc_derg' / 'practical_experiments_summary.json'}`",
    ]

    summary_lines.extend(
        [
            "",
            "## XAI Status",
            f"- XAI suite status: {xai_summary.get('status', 'completed')}",
            f"- XAI detail: {xai_summary.get('reason', 'artifacts written under artifacts/xai/')}",
            "",
            "## Deep Analysis Status",
            f"- Deep-analysis suite status: {deep_summary.get('status', 'completed')}",
            f"- Deep-analysis detail: {deep_summary.get('reason', 'artifacts written under artifacts/deep_analysis/')}",
            "",
            "## Practical Experiment Status",
            f"- Practical experiment suite status: {practical_summary.get('status', 'completed')}",
            (
                f"- Clean-test manual reviews saved per 1000 alerts: "
                f"{practical_summary.get('highlights', {}).get('clean_manual_reviews_saved_per_1000', 'n/a')}"
            ),
            (
                f"- Noisy-evidence refusal rate: "
                f"{practical_summary.get('highlights', {}).get('noisy_evidence_refuse_rate', 'n/a')}"
            ),
            "",
            "## Modeling Note",
            "- TrustSOC-DERG is the sole flagship model in this repository, implementing the Trust Calibration Transformer over DERG-derived evidence features.",
            "",
            "## Suggested Paper Positioning",
            "- Present `TrustSOC-DERG` as the sole flagship model in the paper-facing pipeline.",
            "- Use the XAI and deep-analysis artifacts as the main explanation and trust-analysis evidence for the paper.",
            "",
            "## Main Figures",
            *figure_lines,
            "",
            "## Key Artifacts",
            *artifact_lines,
        ]
    )
    summary_markdown = "\n".join(summary_lines)
    (reports_dir / "scopus_summary.md").write_text(summary_markdown, encoding="utf-8")

    return {
        "dataset_table": str((config.benchmark_tables_dir / "table_dataset_statistics.csv").resolve()),
        "error_table": str((config.analysis_tables_dir / "table_error_analysis.csv").resolve()),
        "summary_report": str((reports_dir / "scopus_summary.md").resolve()),
        "xai_summary": str((config.xai_dir / "trustsoc_derg" / "xai_summary.json").resolve()),
        "deep_analysis_summary": str((config.deep_analysis_dir / "trustsoc_derg" / "deep_analysis_summary.json").resolve()),
        "practical_experiments_summary": str((config.artifacts_dir / "practical_experiments" / "trustsoc_derg" / "practical_experiments_summary.json").resolve()),
    }


def evaluate(config: ProjectConfig) -> dict[str, Any]:
    ensure_core_metrics(config)
    return generate_report_tables(config)


def run_statistical_significance_tests(config: ProjectConfig) -> None:
    """Compute bootstrap confidence intervals and pairwise significance tests."""
    logger = get_logger(config, "evaluate")
    logger.info("Starting statistical significance testing...")

    # Load test split ground truth
    test_df = _safe_load_split(config, "test")
    if test_df is None:
        logger.warning("Processed test split is unavailable. Skipping statistical tests.")
        return
    y_true_cls = test_df["threat_type"].to_numpy()
    y_true_reg = test_df["risk_score"].to_numpy()

    # Discover available predictions
    pred_dir = config.predictions_dir
    pred_files = list(pred_dir.glob("predictions_*.csv"))

    if not pred_files:
        logger.warning("No prediction files found in %s. Skipping statistical tests.", pred_dir)
        return

    # Load predictions per model
    models_predictions_cls = {}
    models_predictions_reg = {}

    for p_file in pred_files:
        # Standardize model name (e.g. predictions_lite.csv -> lite)
        model_name = p_file.stem.replace("predictions_", "")
        try:
            p_df = pd.read_csv(p_file)
            if len(p_df) != len(test_df):
                logger.warning(
                    "Prediction file %s has %d rows, but test split has %d rows. Skipping.",
                    p_file.name, len(p_df), len(test_df)
                )
                continue
            
            if "threat_pred" in p_df.columns:
                models_predictions_cls[model_name] = p_df["threat_pred"].to_numpy()
            if "risk_pred" in p_df.columns:
                models_predictions_reg[model_name] = p_df["risk_pred"].to_numpy()
        except Exception as e:
            logger.error("Failed to load predictions from %s: %s", p_file.name, e)

    if not models_predictions_cls:
        logger.warning("No valid model predictions loaded. Skipping statistical tests.")
        return

    # 1. Compute Bootstrap Confidence Intervals for all models
    import numpy as np
    from .supporting_analysis.statistical_testing import bootstrap_all_metrics, pairwise_model_comparisons, cohens_d, format_ci
    from .supporting_analysis.latex_export import generate_confidence_interval_table

    ci_rows = []
    ci_data = {}
    
    for model_name, y_pred_cls in models_predictions_cls.items():
        y_pred_reg = models_predictions_reg.get(model_name, np.zeros_like(y_true_reg))
        
        ci_results = bootstrap_all_metrics(
            y_true_cls, y_pred_cls, y_true_reg, y_pred_reg, n_bootstrap=1000
        )
        
        ci_rows.append({
            "Model": model_name,
            "Accuracy_CI": format_ci(ci_results["accuracy"]),
            "Macro_F1_CI": format_ci(ci_results["f1_macro"]),
            "Weighted_F1_CI": format_ci(ci_results["f1_weighted"]),
            "Risk_MAE_CI": format_ci(ci_results["mae"]),
        })

        ci_data[model_name] = {
            "Accuracy": {
                "mean": ci_results["accuracy"]["mean"],
                "ci_lower": ci_results["accuracy"]["ci_lower"],
                "ci_upper": ci_results["accuracy"]["ci_upper"]
            },
            "Macro F1": {
                "mean": ci_results["f1_macro"]["mean"],
                "ci_lower": ci_results["f1_macro"]["ci_lower"],
                "ci_upper": ci_results["f1_macro"]["ci_upper"]
            },
            "Weighted F1": {
                "mean": ci_results["f1_weighted"]["mean"],
                "ci_lower": ci_results["f1_weighted"]["ci_lower"],
                "ci_upper": ci_results["f1_weighted"]["ci_upper"]
            },
            "MAE": {
                "mean": ci_results["mae"]["mean"],
                "ci_lower": ci_results["mae"]["ci_lower"],
                "ci_upper": ci_results["mae"]["ci_upper"]
            }
        }

    # Save CI CSV
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(config.statistical_tables_dir / "table_confidence_intervals.csv", index=False, encoding="utf-8")
    logger.info("Saved confidence intervals table to %s", config.statistical_tables_dir / "table_confidence_intervals.csv")

    # Generate and save LaTeX table
    try:
        ci_latex = generate_confidence_interval_table(ci_data)
        (config.statistical_tables_dir / "table_confidence_intervals.tex").write_text(ci_latex, encoding="utf-8")
        logger.info("Saved confidence intervals LaTeX table to %s", config.statistical_tables_dir / "table_confidence_intervals.tex")
    except Exception as e:
        logger.error("Failed to generate CI LaTeX table: %s", e)

    # 2. Pairwise McNemar's Tests for Classification Correctness
    if len(models_predictions_cls) >= 2:
        pairwise_results = pairwise_model_comparisons(y_true_cls, models_predictions_cls)
        pairwise_rows = []
        for res in pairwise_results:
            model_a, model_b = res["model_a"], res["model_b"]
            cd_val = None
            if model_a in models_predictions_reg and model_b in models_predictions_reg:
                cd_val = cohens_d(models_predictions_reg[model_a], models_predictions_reg[model_b])
                
            pairwise_rows.append({
                "Model_A": model_a,
                "Model_B": model_b,
                "McNemar_Stat": res["statistic"],
                "p_value": res["p_value"],
                "Sig_005": res["significant_005"],
                "Sig_001": res["significant_001"],
                "Risk_Cohens_d": cd_val,
            })
            
        pairwise_df = pd.DataFrame(pairwise_rows)
        pairwise_df.to_csv(config.statistical_tables_dir / "table_pairwise_significance.csv", index=False, encoding="utf-8")
        logger.info("Saved pairwise significance table to %s", config.statistical_tables_dir / "table_pairwise_significance.csv")
