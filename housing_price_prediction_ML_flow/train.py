import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
import mlflow

def prepare_data(housing: pd.DataFrame):
    with mlflow.start_run(run_name="Prepare_Data", nested=True):
        housing["income_cat"] = pd.cut(
            housing["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        # Log dataset sizes
        mlflow.log_param("train_set_size", len(strat_train_set))
        mlflow.log_param("test_set_size", len(strat_test_set))
        return strat_train_set, strat_test_set


def preprocess_data(data: pd.DataFrame):
    with mlflow.start_run(run_name="Preprocess_Data", nested=True):
        data_num = data.drop(["ocean_proximity", "median_house_value"], axis=1)
        imputer = SimpleImputer(strategy="median")
        imputer.fit(data_num)
        data_prepared = pd.DataFrame(imputer.transform(data_num), columns=data_num.columns, index=data.index)

        data_prepared["rooms_per_household"] = data_prepared["total_rooms"] / data_prepared["households"]
        data_prepared["bedrooms_per_room"] = data_prepared["total_bedrooms"] / data_prepared["total_rooms"]
        data_prepared["population_per_household"] = data_prepared["population"] / data_prepared["households"]

        data_cat = pd.get_dummies(data[["ocean_proximity"]], drop_first=True)

        # Log preprocessing parameters
        mlflow.log_param("preprocessing_strategy", "median_imputation")
        return data_prepared.join(data_cat), data["median_house_value"].copy()


def train_model(data: pd.DataFrame, labels: pd.Series):
    with mlflow.start_run(run_name="Train_Model", nested=True):
        param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=20,
            cv=5,
            scoring='neg_mean_squared_error',
            random_state=42,
        )

        rnd_search.fit(data, labels)

        # Log model parameters
        best_params = rnd_search.best_params_
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        return rnd_search


