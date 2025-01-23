import os
import tarfile
import logging
import pandas as pd
from six.moves import urllib

def fetch_housing_data(housing_url: str, housing_path: str, logger: logging.Logger):
    logger.info("Fetching housing data...")
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)
    logger.info("Housing data fetched and extracted.")

def load_housing_data(housing_path: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading housing data...")
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
