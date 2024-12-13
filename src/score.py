from sklearn.metrics import mean_squared_error
import numpy as np
import logging
import pandas as pd

def evaluate_model(model, test_data: pd.DataFrame, test_labels: pd.Series, logger: logging.Logger):
    logger.info("Evaluating model...")
    predictions = model.best_estimator_.predict(test_data)
    mse = mean_squared_error(test_labels, predictions)
    rmse = np.sqrt(mse)
    logger.info(f"Model evaluation completed. RMSE: {rmse}")
    return rmse
