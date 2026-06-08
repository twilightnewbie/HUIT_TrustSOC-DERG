from src.config import get_config
from src.evaluation import compare_with_opensoc


if __name__ == "__main__":
    compare_with_opensoc(get_config())
