from src.config import get_config
from src.evaluation import evaluate


if __name__ == "__main__":
    evaluate(get_config())
