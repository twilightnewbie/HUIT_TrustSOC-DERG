from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _detect_project_root() -> Path:
    """Auto-detect project root by walking up from this file until we find main.py."""
    current = Path(__file__).resolve().parent.parent
    if (current / "main.py").exists():
        return current
    # Fallback: check environment variable
    env_root = os.environ.get("TRUSTSOC_PROJECT_ROOT")
    if env_root:
        return Path(env_root)
    return current


@dataclass
class OpenSOCBaseline:
    threat_accuracy: float = 0.90
    threat_macro_f1: float | None = None
    threat_weighted_f1: float = 0.922
    severity_accuracy: float = 0.80
    severity_weighted_f1: float = 0.80
    mae: float = 5.62
    rmse: float = 10.4650
    mape: float = 14.80
    r2: float = 0.207
    latency_seconds_per_sample: float | None = None
    train_time_minutes: float = 27.29
    parameter_count: int = 12_615_680
    notes: str = (
        "TinyLlama-1.1B + QLoRA/LoRA, 10 epochs, fixed parser, "
        "OpenSOC-style eval, Precision=96.67%, Recall=90.00%, F1=92.20%."
    )


@dataclass
class ProjectConfig:
    project_root: Path = field(default_factory=_detect_project_root)
    # Source dataset paths — override via env or CLI if needed
    source_dataset_root: Path = field(default=None)
    source_trustsoc_root: Path = field(default=None)

    seed: int = 42
    small_dataset_threshold: int = 450
    text_max_word_features: int = 30000
    text_max_char_features: int = 20000
    baseline_reference: OpenSOCBaseline = field(default_factory=OpenSOCBaseline)

    def __post_init__(self) -> None:
        # Ensure project_root is a Path
        self.project_root = Path(self.project_root)

        # Derive all project-relative paths
        if self.source_dataset_root is None:
            env_val = os.environ.get("TRUSTSOC_DATASET_ROOT")
            self.source_dataset_root = Path(env_val) if env_val else self.project_root / "data" / "raw"
        else:
            self.source_dataset_root = Path(self.source_dataset_root)

        if self.source_trustsoc_root is None:
            env_val = os.environ.get("TRUSTSOC_SOURCE_ROOT")
            self.source_trustsoc_root = Path(env_val) if env_val else self.project_root / "data" / "raw"
        else:
            self.source_trustsoc_root = Path(self.source_trustsoc_root)

    @property
    def data_raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def data_splits_dir(self) -> Path:
        return self.project_root / "data" / "splits"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "artifacts" / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.project_root / "artifacts" / "metrics"

    @property
    def predictions_dir(self) -> Path:
        return self.project_root / "artifacts" / "predictions"

    @property
    def figures_dir(self) -> Path:
        return self.project_root / "artifacts" / "figures"

    @property
    def model_figures_dir(self) -> Path:
        return self.figures_dir / "models"

    @property
    def comparison_figures_dir(self) -> Path:
        return self.figures_dir / "comparison"

    @property
    def robustness_figures_dir(self) -> Path:
        return self.figures_dir / "robustness"

    @property
    def concept_figures_dir(self) -> Path:
        return self.figures_dir / "concepts"

    @property
    def xai_dir(self) -> Path:
        return self.project_root / "artifacts" / "xai"

    @property
    def deep_analysis_dir(self) -> Path:
        return self.project_root / "artifacts" / "deep_analysis"

    @property
    def tables_dir(self) -> Path:
        return self.project_root / "artifacts" / "tables"

    @property
    def benchmark_tables_dir(self) -> Path:
        return self.tables_dir / "benchmark"

    @property
    def analysis_tables_dir(self) -> Path:
        return self.tables_dir / "analysis"

    @property
    def statistical_tables_dir(self) -> Path:
        return self.tables_dir / "statistics"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "artifacts" / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "artifacts" / "reports"

    @property
    def summary_reports_dir(self) -> Path:
        return self.reports_dir / "summary"

    def model_figure_dir(self, model_name: str) -> Path:
        return self.model_figures_dir / model_name

    @property
    def canonical_raw_files(self) -> dict[str, Path]:
        return {
            "trustsoc_train.jsonl": self.source_trustsoc_root / "trustsoc_train.jsonl",
            "trustsoc_val.jsonl": self.source_trustsoc_root / "trustsoc_val.jsonl",
            "trustsoc_test.jsonl": self.source_trustsoc_root / "trustsoc_test.jsonl",
            "trustsoc_full.jsonl": self.source_trustsoc_root / "trustsoc_full.jsonl",
            "trustsoc_normal_cases.jsonl": self.source_trustsoc_root / "trustsoc_normal_cases.jsonl",
            "trustsoc_synthetic_adversarial_cases.jsonl": self.source_trustsoc_root / "trustsoc_synthetic_adversarial_cases.jsonl",
            "trustsoc_summary.csv": self.source_trustsoc_root / "trustsoc_summary.csv",
            "1_otx_threat_intel.csv": self.source_dataset_root / "1_otx_threat_intel.csv",
            "2_cve_vulnerabilities.csv": self.source_dataset_root / "2_cve_vulnerabilities.csv",
            "3_malicious_domains.csv": self.source_dataset_root / "3_malicious_domains.csv",
            "4_malicious_ips.csv": self.source_dataset_root / "4_malicious_ips.csv",
            "GUIDE_Train.csv": self.source_dataset_root / "GUIDE_Train.csv",
            "GUIDE_Test.csv": self.source_dataset_root / "GUIDE_Test.csv",
        }

    @property
    def processed_split_paths(self) -> dict[str, Path]:
        return {
            "train": self.data_processed_dir / "train_processed.csv",
            "val": self.data_processed_dir / "val_processed.csv",
            "test": self.data_processed_dir / "test_processed.csv",
            "full": self.data_processed_dir / "full_processed.csv",
        }

    @property
    def robustness_split_paths(self) -> dict[str, Path]:
        return {
            "adversarial": self.data_splits_dir / "robustness_adversarial.csv",
            "missing_cti": self.data_splits_dir / "robustness_missing_cti.csv",
            "missing_mitre": self.data_splits_dir / "robustness_missing_mitre.csv",
            "noisy_evidence": self.data_splits_dir / "robustness_noisy_evidence.csv",
        }


def get_config() -> ProjectConfig:
    return ProjectConfig()
