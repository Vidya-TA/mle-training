import unittest
from unittest.mock import MagicMock, patch 
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
import logging

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(parent_dir, "src")
sys.path.append(src_dir)

from train import prepare_data, preprocess_data, train_model  # Assuming train.py is in the same directory


class TestPipeline(unittest.TestCase):
    """Test class for the data preparation and training pipeline."""

    def setUp(self):
        data = { 'median_income': [1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5, 5.5, 5.5, 6.5, 6.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 1.5, 2.5, 3.5, 4.5, 5.5], 
                'total_rooms': [500, 600, 1500, 1600, 2500, 2600, 3500, 3600, 1200, 1300, 1800, 1900, 2400, 3200, 4000, 2200, 2800, 3600, 400, 1100, 2700, 3300, 4200], 
                'households': [150, 160, 300, 310, 400, 410, 500, 510, 350, 360, 450, 460, 600, 700, 800, 900, 1000, 1100, 130, 140, 370, 470, 520], 
                'total_bedrooms': [50, 60, 100, 110, 150, 160, 200, 210, 80, 90, 90, 100, 120, 130, 140, 150, 160, 170, 40, 50, 135, 145, 155], 
                'population': [1000, 1100, 2000, 2100, 3000, 3100, 4000, 4100, 1500, 1600, 1700, 1800, 2100, 2300, 2500, 2700, 2900, 3100, 1200, 1400, 2700, 3100, 4200], 
                'ocean_proximity': ['INLAND', 'INLAND', 'NEAR BAY', 'NEAR BAY', 'NEAR OCEAN', 'NEAR OCEAN', 'INLAND', 'INLAND', 'NEAR BAY', 'NEAR BAY', 'NEAR OCEAN', 'NEAR OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN', 'INLAND', 'NEAR BAY'], 
                'median_house_value': [100000, 110000, 200000, 210000, 300000, 310000, 400000, 410000, 150000, 160000, 180000, 190000, 220000, 240000, 260000, 280000, 300000, 320000, 140000, 150000, 270000, 290000, 310000] 
                }
        self.housing = pd.DataFrame(data)
        
        # Mock the logger
        self.mock_logger = MagicMock(spec=logging.Logger)

    def test_prepare_data_stratified_split(self):
        # Run the function
        strat_train_set, strat_test_set = prepare_data(self.housing, self.mock_logger)

        # Assert 'income_cat' was added correctly
        self.assertIn('income_cat', self.housing.columns)
        
        # Check if stratified split applied successfully
        log_messages = [call[0][0] for call in self.mock_logger.info.call_args_list]
        self.assertTrue("Data preparation completed." in log_messages, f"Logs: {log_messages}")
        
        # Assert that the 'income_cat' column is dropped in the final sets
        self.assertNotIn('income_cat', strat_train_set.columns)
        self.assertNotIn('income_cat', strat_test_set.columns)

        # Check the sizes of the split sets (80% and 20% based on test_size=0.2)
        self.assertEqual(len(strat_train_set), 18)  # 80% of 6 samples
        self.assertEqual(len(strat_test_set), 5)   # 20% of 6 samples

        # Check logger calls
        self.mock_logger.info.assert_any_call("Preparing data...")
        self.mock_logger.info.assert_any_call("Data preparation completed.")


    def test_imputer(self):
        """Test if the SimpleImputer is working as expected (filling missing values)."""
        data_with_missing = self.housing.copy()
        data_with_missing.loc[0, 'total_rooms'] = None  # Introduce missing value in 'total_rooms'

        # Run the preprocessing function
        data_prepared, _ = preprocess_data(data_with_missing, self.mock_logger)

        # Assert that the missing value in 'total_rooms' is imputed with the median (1500)
        imputed_value = data_prepared['total_rooms'].iloc[0]
        
        # Since median of 'total_rooms' might be different based on the data, check the median
        expected_median = data_with_missing['total_rooms'].median()
        self.assertEqual(imputed_value, expected_median)

    def test_preprocess_data(self):
        """Test the preprocess_data function, especially one-hot encoding."""
        # Run the preprocessing function
        data_prepared, target = preprocess_data(self.housing, self.mock_logger)

        # Check if the expected new features are created
        self.assertTrue("rooms_per_household" in data_prepared.columns)
        self.assertTrue("bedrooms_per_room" in data_prepared.columns)
        self.assertTrue("population_per_household" in data_prepared.columns)

        # Assert one-hot encoded columns (adjust for drop_first=True)
        expected_columns = ["ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN"]
        for col in expected_columns:
            self.assertIn(col, data_prepared.columns, f"Missing column: {col}")

        # Ensure dropped category is not included
        self.assertNotIn("ocean_proximity_INLAND", data_prepared.columns, "Unexpected column: ocean_proximity_INLAND")



    @patch('train.RandomizedSearchCV')
    @patch('train.RandomForestRegressor')
    def test_train_model(self, MockRandomForestRegressor, MockRandomizedSearchCV):
        """Test the train_model function."""
        # Prepare the mock for RandomForestRegressor
        mock_forest_reg = MagicMock(spec=RandomForestRegressor)
        MockRandomForestRegressor.return_value = mock_forest_reg

        # Prepare the mock for RandomizedSearchCV
        mock_rnd_search = MagicMock()
        MockRandomizedSearchCV.return_value = mock_rnd_search
        mock_rnd_search.best_estimator_ = mock_forest_reg

        # Separate features and labels
        strat_train_set, _ = prepare_data(self.housing.copy(), self.mock_logger)
        data_prepared, labels = preprocess_data(strat_train_set, self.mock_logger)

        # Call the train_model function
        model = train_model(data_prepared, labels, self.mock_logger)

        # Assertions
        MockRandomForestRegressor.assert_called_once_with(random_state=42)
        MockRandomizedSearchCV.assert_called_once()

        # Check if fit was called on the mock object with the correct data
        mock_rnd_search.fit.assert_called_once_with(data_prepared, labels)

        # Check if the returned model is the one expected
        self.assertEqual(model.best_estimator_, mock_forest_reg)

        # Check if the logger was called
        self.mock_logger.info.assert_any_call("Training model...")
        self.mock_logger.info.assert_any_call("Model training completed.")

if __name__ == '__main__':
    unittest.main()
