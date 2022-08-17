"""
    We use Prophet to predict the future retail sales.
"""
import json

import mlflow
import numpy as np
import pandas as pd
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics


ARTIFACT_PATH = "model"
np.random.seed(777)

sales_df = pd.read_csv(
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv"
)


def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}


with mlflow.start_run():

    model = Prophet().fit(sales_df)

    params = extract_params(model)

    metric_keys = ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]
    metrics_raw = cross_validation(
        model=model,
        horizon="365 days",
        period="180 days",
        initial="710 days",
        parallel="threads",
        disable_tqdm=True,
    )
    cv_metrics = performance_metrics(metrics_raw)
    metrics = { k: cv_metrics[k].mean() for k in metric_keys }

    print(f"Logged Metrics: \n{json.dumps(metrics, indent=2)} ")
    print(f"Logged Params: \n{json.dumps(params, indent=2)} ")

    mlflow.prophet.log_model(model, artifact_path=ARTIFACT_PATH)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    model_uri = mlflow.get_artifact_uri(artifact_path=ARTIFACT_PATH)
    print(f"Model artifact logged to: {model_uri}")


loaded_model = mlflow.prophet.load_model(model_uri)

forecast = loaded_model.predict(loaded_model.make_future_dataframe(periods=60))

print(f" Forecast:\n${forecast.head(30)} ")
