from sklearn.metrics import mean_squared_error
import numpy as np
import mlflow
import pandas as pd

def evaluate_model(model, test_data: pd.DataFrame, test_labels: pd.Series):
    with mlflow.start_run(run_name="Evaluate_Model", nested=True):
        predictions = model.best_estimator_.predict(test_data)
        mse = mean_squared_error(test_labels, predictions)
        rmse = np.sqrt(mse)

        # Log evaluation metrics
        mlflow.log_metric("rmse", rmse)
        return rmse
