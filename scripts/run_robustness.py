from src.config import get_config
from src.robustness import run_robustness


if __name__ == "__main__":
    run_robustness(get_config())
