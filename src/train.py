import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
import logging

def prepare_data(housing: pd.DataFrame, logger: logging.Logger):
    logger.info("Preparing data...")
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    # Checking for minimal category samples 
    income_cat_counts = housing["income_cat"].value_counts() 
    for category, count in income_cat_counts.items(): 
        if count < 2: 
            print(f"Category {category} has less than 2 samples, which is too few for stratified split.")

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    logger.info("Data preparation completed.")
    return strat_train_set, strat_test_set


def preprocess_data(data: pd.DataFrame, logger: logging.Logger) -> (pd.DataFrame, pd.Series):
    logger.info("Preprocessing data...")
    data_num = data.drop(["ocean_proximity", "median_house_value"], axis=1)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(data_num)
    data_prepared = pd.DataFrame(imputer.transform(data_num), columns=data_num.columns, index=data.index)

    data_prepared["rooms_per_household"] = data_prepared["total_rooms"] / data_prepared["households"]
    data_prepared["bedrooms_per_room"] = data_prepared["total_bedrooms"] / data_prepared["total_rooms"]
    data_prepared["population_per_household"] = data_prepared["population"] / data_prepared["households"]

    data_cat = pd.get_dummies(data[["ocean_proximity"]], drop_first=True)
    logger.info("Preprocessing completed.")
    return data_prepared.join(data_cat), data["median_house_value"].copy()


def train_model(data: pd.DataFrame, labels: pd.Series, logger: logging.Logger):
    logger.info("Training model...")
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
    logger.info("Model training completed.")
    return rnd_search
