import unittest
from unittest.mock import MagicMock, Mock
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import logging

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(parent_dir, "src")
sys.path.append(src_dir)

class TestEvaluateModel(unittest.TestCase):
    def setUp(self):
        # Mock model
        self.mock_model = Mock()
        self.mock_model.best_estimator_ = MagicMock()
        self.mock_model.best_estimator_.predict = MagicMock(return_value=np.array([3.0, 4.0, 5.0]))

        # Mock logger
        self.mock_logger = Mock()

        # Test data
        self.test_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0]
        })
        self.test_labels = pd.Series([2.5, 4.5, 5.0])

    def test_evaluate_model(self):
        from score import evaluate_model  
        # Run the function
        rmse = evaluate_model(self.mock_model, self.test_data, self.test_labels, self.mock_logger)

        # Expected RMSE calculation
        expected_predictions = np.array([3.0, 4.0, 5.0])
        expected_mse = mean_squared_error(self.test_labels, expected_predictions)
        expected_rmse = np.sqrt(expected_mse)

        # Assertions
        self.assertEqual(rmse, expected_rmse, "The RMSE value is incorrect.")
        self.mock_model.best_estimator_.predict.assert_called_once_with(self.test_data)
        self.mock_logger.info.assert_any_call("Evaluating model...")
        self.mock_logger.info.assert_any_call(f"Model evaluation completed. RMSE: {expected_rmse}")

if __name__ == '__main__':
    unittest.main()