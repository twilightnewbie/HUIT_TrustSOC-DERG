from src.config import get_config
from src.data_loader import load_processed_split
from src.models.trustsoc_transformer import train_derg


if __name__ == "__main__":
    config = get_config()
    train_derg(
        load_processed_split(config, "train"),
        load_processed_split(config, "val"),
        load_processed_split(config, "test"),
        config,
        model_name="trustsoc_derg",
    )
