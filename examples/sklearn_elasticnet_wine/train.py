'''
    We use the data set from http://archive.ics.uci.edu/ml/datasets/Wine+Quality in this example.

    The goal is to model wine preferences by using their physicochemical properties.
    To do so, we use the ElasticNet regression algorithm.

    TO RUN:
    Execute the following command from the examples directory (Args optional):

    python sklearn_elasticnet_wine/train.py <alpha> <l1_ratio>
'''
import logging
import os
import sys
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def evaluate_metrics(actual, pred):
    """Provides key statistics provided the predicted and actual values.

    Parameters
    ----------
    actual:
        The array of actual values.

    pred:
        The array of values predicted by the model.

    Returns
    -------
    rmse, mae, r2:
        A tuple of the root mean squared error, mean absolute error,
        and the r2 score.
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def run_wine_quality_workflow():
    """Runs the wine quality workflow of retrieving data, training,
    and evaluating the model.
    """
    warnings.filterwarnings("ignore")
    np.random.seed(77)

    CSV_URL = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-red.csv"
    )

    # Gather wine-quality CSV file:
    try:
        src_df = pd.read_csv(CSV_URL, sep=";")
    except Exception as e:
        logger.exception(
            f"Unable to download source data CSV file from '{CSV_URL}'. Error: {e}"
        )

    # Train / test split:
    training_data, testing_data = train_test_split(
                                    src_df, 
                                    train_size=0.7,
                                    test_size=0.3
                                )

    # Target column:
    tgt_col = ["quality"]
    # Features:
    train_x = training_data.drop(tgt_col, axis=1)
    test_x = testing_data.drop(tgt_col, axis=1)
    # True data:
    train_y = training_data[tgt_col]
    test_y = testing_data[tgt_col]

    # Accept Hyperparameters + Validation:
    alpha = float(sys.argv[1] if len(sys.argv) > 1 else 0.5)
    l1_ratio = float(sys.argv[2] if len(sys.argv) > 2 else 0.5)

    # Run the model training + evaluation with MLflow:
    with mlflow.start_run():
        # Initialize & Train the model:
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Generate predicted values:
        predicted_y = lr.predict(test_x)

        # Determine metrics:
        (rmse, mae, r2) = evaluate_metrics(test_y, predicted_y)

        # Log the resulting metrics:
        logger.info(
            f"ElasticNet Model (alpha={alpha}, l1_ratio={l1_ratio}) \n"
            f" | RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f} "
        )

        # MLflow logs:
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_store = urlparse(mlflow.get_tracking_uri()).scheme

        # [CASE] Don't use model registry with file store:
        # Ref: https://mlflow.org/docs/latest/model-registry.html#api-workflow
        if tracking_url_store != "file":
            # Log the model:
            mlflow.sklearn.log_model(
                skmodel=lr,
                artifact_path="model", 
                registered_model_name="ElasticNet_WineQuality_Model"
            )

        else:
            mlflow.sklearn.log_model(
                sk_model=lr,
                artifact_path="model"
            )

if __name__ == "__main__":

    run_wine_quality_workflow()
