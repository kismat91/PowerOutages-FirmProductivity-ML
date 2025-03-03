# main.py

import argparse
import yaml
from scripts.data_preprocessing import preprocess_data
from scripts.visualization import generate_all_plots
from scripts.model_training import train_and_evaluate_models


def load_config(config_path="config.yaml"):
    """Utility function to load the config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(config_path="config.yaml"):
    # Load config
    config = load_config(config_path)

    # 1. Preprocess the data
    df = preprocess_data(config_path)

    # 2. Visualization / EDA
    generate_all_plots(df)

    # 3. Model Training
    best_models, results_df = train_and_evaluate_models(df, config)


#  python main.py --config config.yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration file (YAML).")
    args = parser.parse_args()

    main(args.config)
