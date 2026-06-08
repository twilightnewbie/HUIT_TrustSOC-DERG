"""LaTeX table export for TrustSOC paper.

Generates publication-ready LaTeX tables with booktabs formatting,
bold best values, underlined second-best, and significance markers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _format_number(value: Any, precision: int = 4, bold: bool = False, underline: bool = False, significance: str = "") -> str:
    """Format a numeric value for LaTeX with optional bold/underline/significance."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, str):
        return value

    formatted = f"{float(value):.{precision}f}"

    if significance:
        formatted += significance

    if bold:
        formatted = f"\\textbf{{{formatted}}}"
    elif underline:
        formatted = f"\\underline{{{formatted}}}"

    return formatted


def _mark_best_in_column(
    df: pd.DataFrame,
    column: str,
    higher_is_better: bool = True,
) -> tuple[int | None, int | None]:
    """Find indices of best and second-best values in a column."""
    numeric_vals = pd.to_numeric(df[column], errors="coerce")
    valid = numeric_vals.dropna()
    if len(valid) < 2:
        best_idx = valid.idxmax() if higher_is_better else valid.idxmin()
        return best_idx, None

    if higher_is_better:
        sorted_idx = valid.nlargest(2).index.tolist()
    else:
        sorted_idx = valid.nsmallest(2).index.tolist()

    return sorted_idx[0], sorted_idx[1] if len(sorted_idx) > 1 else None


def dataframe_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    highlight_columns: dict[str, bool] | None = None,
    precision: int = 4,
    significance_data: dict[str, dict[str, str]] | None = None,
) -> str:
    """Convert a DataFrame to a publication-ready LaTeX table.
    
    Parameters
    ----------
    df : the data
    caption : table caption
    label : LaTeX label for referencing
    highlight_columns : dict mapping column_name -> higher_is_better
        Columns listed here will have best (bold) and 2nd-best (underline) marked.
    precision : decimal places
    significance_data : dict mapping column_name -> {row_index: "marker"}
        Markers like "*", "**", "***" for statistical significance.
    """
    if highlight_columns is None:
        highlight_columns = {}
    if significance_data is None:
        significance_data = {}

    # Find best/second-best per column
    best_map: dict[str, tuple[int | None, int | None]] = {}
    for col, higher_is_better in highlight_columns.items():
        if col in df.columns:
            best_map[col] = _mark_best_in_column(df, col, higher_is_better)

    # Build table
    n_cols = len(df.columns)
    col_spec = "l" + "c" * (n_cols - 1)

    lines = [
        f"\\begin{{table}}[htbp]",
        f"\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"\\toprule",
    ]

    # Header
    header = " & ".join([f"\\textbf{{{col}}}" for col in df.columns])
    lines.append(f"{header} \\\\")
    lines.append("\\midrule")

    # Data rows
    for row_idx, (_, row) in enumerate(df.iterrows()):
        cells = []
        for col in df.columns:
            value = row[col]
            is_best = best_map.get(col, (None, None))[0] == row_idx
            is_second = best_map.get(col, (None, None))[1] == row_idx
            sig = significance_data.get(col, {}).get(str(row_idx), "")

            if col in highlight_columns and not isinstance(value, str):
                cells.append(_format_number(value, precision, bold=is_best, underline=is_second, significance=sig))
            else:
                cells.append(_format_number(value, precision))
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-built Table Templates for TrustSOC Paper
# ---------------------------------------------------------------------------

def generate_baseline_comparison_table(comparison_df: pd.DataFrame) -> str:
    """Generate Table 1: Baseline comparison with OpenSOC-AI."""
    highlight = {
        "Accuracy": True,
        "Macro F1": True,
        "Weighted F1": True,
        "MAE": False,
        "RMSE": False,
        "R2": True,
        "ECE": False,
        "Brier": False,
    }
    return dataframe_to_latex(
        comparison_df,
        caption="Comparison of TrustSOC variants against OpenSOC-AI baseline. "
                "Best values are \\textbf{bold}, second-best are \\underline{underlined}.",
        label="tab:baseline_comparison",
        highlight_columns=highlight,
    )


def generate_robustness_table(robustness_df: pd.DataFrame) -> str:
    """Generate Table 3: Robustness evaluation across attack types."""
    highlight = {
        "trustsoc_derg_f1": True,
        "trustsoc_derg_refusal_accuracy": True,
    }
    return dataframe_to_latex(
        robustness_df,
        caption="Robustness evaluation across adversarial attack types. "
                "Refusal accuracy measures the model's ability to correctly refuse unreliable cases.",
        label="tab:robustness",
        highlight_columns=highlight,
    )


def generate_calibration_table(calibration_df: pd.DataFrame) -> str:
    """Generate Table 4: Trust calibration metrics."""
    highlight = {
        "ECE": False,
        "ACE": False,
        "Brier": False,
        "Trust Alignment Score": True,
        "AURC": False,
    }
    return dataframe_to_latex(
        calibration_df,
        caption="Trust calibration metrics for TrustSOC models. "
                "Lower ECE/ACE/Brier/AURC indicates better calibration.",
        label="tab:calibration",
        highlight_columns=highlight,
    )


def generate_confidence_interval_table(ci_data: dict[str, dict[str, dict[str, float]]]) -> str:
    """Generate Table 5: Results with 95% confidence intervals.
    
    Parameters
    ----------
    ci_data : nested dict {model_name: {metric_name: {mean, ci_lower, ci_upper}}}
    """
    rows = []
    for model_name, metrics in ci_data.items():
        row = {"Model": model_name}
        for metric_name, ci in metrics.items():
            row[metric_name] = f"{ci['mean']:.4f} ({ci['ci_lower']:.4f}–{ci['ci_upper']:.4f})"
        rows.append(row)

    df = pd.DataFrame(rows)
    return dataframe_to_latex(
        df,
        caption="Performance metrics with 95\\% bootstrap confidence intervals.",
        label="tab:confidence_intervals",
    )


# ---------------------------------------------------------------------------
# Export All Tables
# ---------------------------------------------------------------------------

def export_all_latex_tables(tables_dir: Path, output_dir: Path | None = None) -> dict[str, str]:
    """Read all CSV tables from tables_dir and generate LaTeX versions."""
    if output_dir is None:
        output_dir = tables_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    table_generators = {
        "table_robustness.csv": ("robustness", generate_robustness_table),
        "table_calibration.csv": ("calibration", generate_calibration_table),
    }

    for csv_name, (tex_name, generator) in table_generators.items():
        csv_path = tables_dir / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            latex = generator(df)
            tex_path = output_dir / f"{tex_name}.tex"
            tex_path.write_text(latex, encoding="utf-8")
            generated[tex_name] = str(tex_path)

    return generated
