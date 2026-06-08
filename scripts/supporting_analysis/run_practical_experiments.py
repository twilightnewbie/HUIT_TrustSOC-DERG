from __future__ import annotations

from src.config import get_config
from src.supporting_analysis.practical_experiments import run_practical_experiments


if __name__ == "__main__":
    run_practical_experiments(get_config(), model_name="trustsoc_derg")
