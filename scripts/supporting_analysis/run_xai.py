from src.config import get_config
from src.supporting_analysis.xai_analysis import run_xai_suite


if __name__ == "__main__":
    run_xai_suite(get_config())
