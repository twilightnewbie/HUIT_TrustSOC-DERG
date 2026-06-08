from __future__ import annotations

from typing import Any

import pandas as pd

from .config import ProjectConfig
from .data_loader import load_processed_split, load_robustness_split
from .models.trustsoc_transformer import train_derg
from .utils import get_logger, save_json


def run_robustness(config: ProjectConfig) -> dict[str, Any]:
    logger = get_logger(config, "robustness")
    train_df = load_processed_split(config, "train")
    val_df = load_processed_split(config, "val")

    # Original 4 subsets + new attack types if available
    subset_names = ["adversarial", "missing_cti", "missing_mitre", "noisy_evidence"]
    for extra in ("evidence_poisoning", "evidence_suppression", "label_manipulation"):
        extra_path = config.data_splits_dir / f"robustness_{extra}.csv"
        if extra_path.exists():
            subset_names.append(extra)

    subsets = {}
    for subset_name in subset_names:
        test_df = load_robustness_split(config, subset_name)
        derg_metrics = train_derg(train_df, val_df, test_df, config, model_name=f"trustsoc_derg_{subset_name}")
        subsets[subset_name] = {
            "trustsoc_derg": {
                "threat_f1_macro": derg_metrics["threat_type"]["f1_macro"],
                "weighted_f1": derg_metrics["threat_type"]["f1_weighted"],
                "refusal_accuracy": derg_metrics["calibration"]["refusal_accuracy"],
            },
        }
        logger.info("Robustness subset %s evaluated", subset_name)

    save_json(config.metrics_dir / "robustness_metrics.json", subsets)
    records = []
    for subset_name, values in subsets.items():
        records.append(
            {
                "subset": subset_name,
                "trustsoc_derg_f1": values["trustsoc_derg"]["threat_f1_macro"],
                "trustsoc_derg_refusal_accuracy": values["trustsoc_derg"]["refusal_accuracy"],
            }
        )
    pd.DataFrame(records).to_csv(config.analysis_tables_dir / "table_robustness.csv", index=False, encoding="utf-8")
    return subsets
