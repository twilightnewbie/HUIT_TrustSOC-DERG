from src.config import get_config
from src.evaluation import generate_report_tables


if __name__ == "__main__":
    generate_report_tables(get_config())
