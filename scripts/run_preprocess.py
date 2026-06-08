from src.config import get_config
from src.preprocessing import preprocess_all


if __name__ == "__main__":
    preprocess_all(get_config())
