from __future__ import annotations

import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import ProjectConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories(config: ProjectConfig) -> None:
    for path in (
        config.project_root,
        config.data_raw_dir,
        config.data_processed_dir,
        config.data_splits_dir,
        config.models_dir,
        config.metrics_dir,
        config.predictions_dir,
        config.figures_dir,
        config.model_figures_dir,
        config.comparison_figures_dir,
        config.robustness_figures_dir,
        config.concept_figures_dir,
        config.xai_dir,
        config.deep_analysis_dir,
        config.tables_dir,
        config.benchmark_tables_dir,
        config.analysis_tables_dir,
        config.statistical_tables_dir,
        config.reports_dir,
        config.summary_reports_dir,
        config.logs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def get_logger(config: ProjectConfig, name: str) -> logging.Logger:
    ensure_directories(config)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = config.logs_dir / f"{name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def timed_call(logger: logging.Logger, label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    logger.info("%s finished in %.3f seconds", label, elapsed)
    return result, elapsed


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def maybe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default
