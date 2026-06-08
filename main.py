from __future__ import annotations

import argparse

from src.config import get_config
from src.evaluation import compare_with_opensoc, evaluate, generate_report_tables
from src.models.sklearn_baselines import train_baselines
from src.models.trustsoc_transformer import train_derg
from src.preprocessing import preprocess_all
from src.robustness import run_robustness
from src.supporting_analysis.deep_analysis import run_deep_analysis
from src.supporting_analysis.practical_experiments import run_practical_experiments
from src.supporting_analysis.xai_analysis import run_xai_suite
from src.data_loader import load_processed_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrustSOC-Research local runner.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "preprocess",
            "train_baselines",
            "train_derg",
            "evaluate",
            "compare_opensoc",
            "robustness",
            "report",
            "xai",
            "deep_analysis",
            "practical_experiments",
            "full_analysis",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    if args.mode == "preprocess":
        preprocess_all(config)
        return

    if args.mode == "train_baselines":
        train_df = load_processed_split(config, "train")
        val_df = load_processed_split(config, "val")
        test_df = load_processed_split(config, "test")
        train_baselines(train_df, val_df, test_df, config)
        return

    if args.mode == "train_derg":
        train_df = load_processed_split(config, "train")
        val_df = load_processed_split(config, "val")
        test_df = load_processed_split(config, "test")
        train_derg(train_df, val_df, test_df, config, model_name="trustsoc_derg")
        return

    if args.mode == "evaluate":
        evaluate(config)
        return

    if args.mode == "compare_opensoc":
        compare_with_opensoc(config)
        return

    if args.mode == "robustness":
        run_robustness(config)
        return

    if args.mode == "report":
        generate_report_tables(config)
        return

    if args.mode == "full_analysis":
        generate_report_tables(config)
        return

    if args.mode == "xai":
        run_xai_suite(config, model_name="trustsoc_derg")
        return

    if args.mode == "deep_analysis":
        run_deep_analysis(config, model_name="trustsoc_derg")
        return

    if args.mode == "practical_experiments":
        run_practical_experiments(config, model_name="trustsoc_derg")
        return


if __name__ == "__main__":
    main()
