from src.config import get_config
from src.supporting_analysis.deep_analysis import run_deep_analysis


if __name__ == "__main__":
    run_deep_analysis(get_config())
