from __future__ import annotations

import pandas as pd

from .config import ProjectConfig


def generate_error_analysis(config: ProjectConfig, model_name: str) -> pd.DataFrame:
    predictions_path = config.predictions_dir / f"predictions_{model_name}.csv"
    if not predictions_path.exists():
        summary = pd.DataFrame(
            [
                {
                    "issue": "predictions_missing",
                    "count": 0,
                    "note": f"Prediction file not found: {predictions_path.name}",
                }
            ]
        )
        out_path = config.analysis_tables_dir / "table_error_analysis.csv"
        summary.to_csv(out_path, index=False, encoding="utf-8")
        return summary

    predictions = pd.read_csv(predictions_path)
    required = {"joint_correct", "threat_true", "threat_pred", "severity_true", "severity_pred"}
    if not required.issubset(predictions.columns):
        summary = pd.DataFrame(
            [
                {
                    "issue": "predictions_invalid_schema",
                    "count": 0,
                    "note": f"Prediction file is missing required columns: {sorted(required - set(predictions.columns))}",
                }
            ]
        )
        out_path = config.analysis_tables_dir / "table_error_analysis.csv"
        summary.to_csv(out_path, index=False, encoding="utf-8")
        return summary
    errors = predictions[~predictions["joint_correct"]].copy()
    if errors.empty:
        summary = pd.DataFrame(
            [
                {
                    "issue": "no_joint_errors",
                    "count": 0,
                    "note": "No joint errors were observed on this split.",
                }
            ]
        )
    else:
        summary = (
            errors.groupby(["threat_true", "threat_pred", "severity_true", "severity_pred"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    out_path = config.analysis_tables_dir / "table_error_analysis.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8")
    return summary
