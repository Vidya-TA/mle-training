import unittest
from unittest.mock import patch, MagicMock
import os
import logging
import pandas as pd
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(parent_dir, "src")
sys.path.append(src_dir)

# Import the functions to be tested
from ingest import fetch_housing_data, load_housing_data 

class TestHousingDataFunctions(unittest.TestCase):

    @patch("ingest.urllib.request.urlretrieve")
    @patch("ingest.tarfile.open")
    @patch("ingest.os.makedirs")
    def test_fetch_housing_data(self, mock_makedirs, mock_tarfile_open, mock_urlretrieve):
        # Arrange
        housing_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
        housing_path = "datasets/housing"
        logger = MagicMock()

        mock_urlretrieve.return_value = None  # Simulate successful download
        mock_tarfile = MagicMock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tarfile  # Mock context manager behavior

        # Act
        fetch_housing_data(housing_url, housing_path, logger)

        # Assert
        mock_makedirs.assert_called_once_with(housing_path, exist_ok=True)
        mock_urlretrieve.assert_called_once_with(housing_url, os.path.join(housing_path, "housing.tgz"))
        mock_tarfile_open.assert_called_once_with(os.path.join(housing_path, "housing.tgz"))

        # Ensure extractall was called with the correct path
        mock_tarfile.extractall.assert_called_once_with(path=housing_path)

        # Verify logger calls
        logger.info.assert_has_calls([
            unittest.mock.call("Fetching housing data..."),
            unittest.mock.call("Housing data fetched and extracted.")
        ])

    @patch("ingest.pd.read_csv")
    def test_load_housing_data(self, mock_read_csv):
        # Arrange
        housing_path = "C:/Users/vidya.yedurumane/Desktop/mle-training/datasets/housing"
        logger = MagicMock()
        mock_csv_path = os.path.join(housing_path, "housing.csv")
        fake_dataframe = pd.DataFrame({"column1": [1, 2], "column2": [3, 4]})
        mock_read_csv.return_value = fake_dataframe

        # Act
        result = load_housing_data(housing_path, logger)

        # Assert
        mock_read_csv.assert_called_once_with(mock_csv_path)
        self.assertTrue(result.equals(fake_dataframe))
        logger.info.assert_any_call("Loading housing data...")

if __name__ == "__main__":
    unittest.main()
