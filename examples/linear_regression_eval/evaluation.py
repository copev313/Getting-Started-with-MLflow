"""
    An example of how to track evaluation on a linear regression model
    using MLflow.
"""
import os

import mlflow
import numpy as np
import shap
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression


# Prepare our training data:
X, y = load_diabetes(return_X_y=True, as_frame=True)

X = X.iloc[:50, :4]
y = y.iloc[:50]

# Train the LinReg model:
model = LinearRegression()
model.fit(X, y)


# Log an explanation:
with mlflow.start_run() as run:
    mlflow.shap.log_explanation(model.predict, X)


# List Artifacts:
client = MlflowClient()
artifact_path = "model_explanations_shap"
artifacts = [x.path for x in client.list_artifacts(run.info.run_id, artifact_path)]

print("# artifacts: ")
print(artifacts)

# Load back the logged explanation:
dst_path = client.download_artifacts(run.info.run_id, artifact_path)
base_values = np.load(os.path.join(dst_path, "base_values.npy"))
shap_values = np.load(os.path.join(dst_path, "shap_values.npy"))

# show a force plot
shap.force_plot(float(base_values), shap_values[0, :], X.iloc[0, :], matplotlib=True)
