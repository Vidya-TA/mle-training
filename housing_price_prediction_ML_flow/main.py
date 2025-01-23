import argparse
import os
import sys
sys.path.append(r'C:\Users\vidya.yedurumane\miniforge3\lib\site-packages')
import mlflow
import mlflow.sklearn  # If you are using sklearn models
from ingest import fetch_housing_data, load_housing_data
from train import prepare_data, preprocess_data, train_model
from score import evaluate_model

def main(args):
    # Check if there is already an active run
    active_run = mlflow.active_run()

    if active_run is None:
        # Start a new run if no active run exists
        with mlflow.start_run(run_name="Housing_ML_Pipeline") as run:
            # Log parameters
            mlflow.log_param("housing_url", args.housing_url)
            mlflow.log_param("housing_path", args.housing_path)

            # Fetch and load data
            fetch_housing_data(args.housing_url, args.housing_path)
            housing = load_housing_data(args.housing_path)

            # Prepare data
            strat_train_set, strat_test_set = prepare_data(housing)
            train_data, train_labels = preprocess_data(strat_train_set)
            test_data, test_labels = preprocess_data(strat_test_set)

            # Log some additional parameters
            mlflow.log_param("num_train_samples", len(train_data))
            mlflow.log_param("num_test_samples", len(test_data))

            # Train model
            model = train_model(train_data, train_labels)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Evaluate model and log metrics
            rmse = evaluate_model(model, test_data, test_labels)
            mlflow.log_metric("rmse", rmse)

            # Log any artifacts (e.g., plots, model file)
            artifact_path = "artifacts"
            if not os.path.exists(artifact_path):
                os.makedirs(artifact_path)


    else:
        print("An active run already exists.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--housing_url", default="https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz", help="URL of the housing data")
    parser.add_argument("--housing_path", default="datasets/housing", help="Path to store housing data")
    args = parser.parse_args()
    main(args)
