from src.config import get_config
from src.data_loader import load_processed_split
from src.models.sklearn_baselines import train_baselines


if __name__ == "__main__":
    config = get_config()
    train_baselines(
        load_processed_split(config, "train"),
        load_processed_split(config, "val"),
        load_processed_split(config, "test"),
        config,
    )
