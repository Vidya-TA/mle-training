import os
import shutil
import argparse
import subprocess

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(parent_dir, "notebooks")
sys.path.append(src_dir)

def test_housing_pipeline():
    # Define test arguments
    housing_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
    housing_path = "C:/Users/vidya.yedurumane/Desktop/mle-training/datasets/housing"
    output_dir = "C:/Users/vidya.yedurumane/Desktop/mle-training/logs"

    # Clean up directories from previous tests, if they exist
    if os.path.exists(housing_path):
        shutil.rmtree(housing_path)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Prepare test directories
    os.makedirs(housing_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Run the main pipeline script
    try:
        # Simulate command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--housing_url", default=housing_url)
        parser.add_argument("--housing_path", default=housing_path)
        parser.add_argument("--output_dir", default=output_dir)
        parser.add_argument("--log_level", default="INFO")
        args = parser.parse_args([])  

        # Execute the pipeline
        import main  
        main.main(args)

        # Check if log file exists and contains relevant information
        log_file = os.path.join(output_dir, "pipeline.log")
        assert os.path.exists(log_file), "Log file was not created."

        # Check for specific log entries
        with open(log_file, "r") as log:
            logs = log.read()
            assert "INFO" in logs, "Logs do not contain INFO level messages."
            assert "Fetching housing data" in logs, "Data fetching step not logged."
            assert "Training model" in logs, "Training step not logged."
            assert "Evaluating model" in logs, "Evaluation step not logged."

        # Check if the dataset is downloaded
        assert os.path.exists(os.path.join(housing_path, "housing.csv")), "Dataset not downloaded."

        print("Functional test passed.")
    except AssertionError as e:
        print(f"Functional test failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up test directories
        shutil.rmtree(housing_path, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)

if __name__ == "__main__":
    test_housing_pipeline()
