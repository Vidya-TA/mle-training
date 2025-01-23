import argparse
import logging
import os
import sys


# Get the current directory (notebooks folder)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Move one step back to the parent directory
parent_dir = os.path.dirname(current_dir)

# Construct the path to the src directory
src_dir = os.path.join(parent_dir, "src")

# Add src directory to sys.path
sys.path.append(src_dir)

from ingest import fetch_housing_data, load_housing_data
from train import prepare_data, preprocess_data, train_model
from score import evaluate_model

def initialize_logger(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("housing_ml_pipeline")
    logger.setLevel(log_level)
    handler = logging.FileHandler(f"{output_dir}/pipeline.log")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger

def main(args):
    logger = initialize_logger(args.output_dir, args.log_level)

    # Fetch and load data
    fetch_housing_data(args.housing_url, args.housing_path, logger)
    housing = load_housing_data(args.housing_path, logger)

    # Prepare data
    strat_train_set, strat_test_set = prepare_data(housing, logger)
    train_data, train_labels = preprocess_data(strat_train_set, logger)
    test_data, test_labels = preprocess_data(strat_test_set, logger)

    # Train model
    model = train_model(train_data, train_labels, logger)

    # Evaluate model
    evaluate_model(model, test_data, test_labels, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--housing_url", default="https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz", help="URL of the housing data")
    parser.add_argument("--housing_path", default="datasets/housing", help="Path to store housing data")
    parser.add_argument("--output_dir", default="logs", help="Directory to store logs")
    parser.add_argument("--log_level", default="INFO", help="Logging level")

    args = parser.parse_args()
    main(args)
