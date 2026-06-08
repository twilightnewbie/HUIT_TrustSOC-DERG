from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProjectConfig


def copy_raw_files(config: ProjectConfig, logger) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for name, source in config.canonical_raw_files.items():
        destination = config.data_raw_dir / name
        if not source.exists():
            logger.warning("Raw source file missing and will be skipped: %s", source)
            continue
        if not destination.exists():
            shutil.copy2(source, destination)
            logger.info("Copied raw file %s -> %s", source, destination)
        copied[name] = destination
    return copied


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


def load_raw_datasets(config: ProjectConfig, logger) -> dict[str, pd.DataFrame]:
    copy_raw_files(config, logger)
    datasets: dict[str, pd.DataFrame] = {}
    for name in (
        "trustsoc_train.jsonl",
        "trustsoc_val.jsonl",
        "trustsoc_test.jsonl",
        "trustsoc_full.jsonl",
        "trustsoc_synthetic_adversarial_cases.jsonl",
        "1_otx_threat_intel.csv",
        "2_cve_vulnerabilities.csv",
        "3_malicious_domains.csv",
        "4_malicious_ips.csv",
    ):
        path = config.data_raw_dir / name
        if not path.exists():
            logger.warning("Expected raw dataset file not found: %s", path)
            continue
        if path.suffix == ".jsonl":
            datasets[name] = load_jsonl(path)
        else:
            datasets[name] = load_csv(path)
        logger.info("Loaded %s with shape %s", name, datasets[name].shape)
    return datasets


def load_processed_split(config: ProjectConfig, split_name: str) -> pd.DataFrame:
    path = config.processed_split_paths[split_name]
    return pd.read_csv(path, low_memory=False)


def load_robustness_split(config: ProjectConfig, split_name: str) -> pd.DataFrame:
    return pd.read_csv(config.robustness_split_paths[split_name], low_memory=False)
