import os
import tarfile
from six.moves import urllib
import pandas as pd
import mlflow

def fetch_housing_data(housing_url: str, housing_path: str):
    with mlflow.start_run(run_name="Fetch_Housing_Data", nested=True):
        # Log the URL and path as parameters
        mlflow.log_param("housing_url", housing_url)
        mlflow.log_param("housing_path", housing_path)

        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)

        with tarfile.open(tgz_path) as housing_tgz:
            housing_tgz.extractall(path=housing_path)
        mlflow.log_param("tgz_path", tgz_path)  # Log the tar file path

def load_housing_data(housing_path: str) -> pd.DataFrame:
    with mlflow.start_run(run_name="Load_Housing_Data", nested=True):
        csv_path = os.path.join(housing_path, "housing.csv")
        housing_data = pd.read_csv(csv_path)

        # Log data shape and path
        mlflow.log_param("csv_path", csv_path)
        mlflow.log_param("num_rows", housing_data.shape[0])
        mlflow.log_param("num_columns", housing_data.shape[1])
        return housing_data
