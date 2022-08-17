"""
    A sample of creating a classic iris classifier with LightGBM.
"""
import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient


def yield_artifacts(run_id, path=None):
    """Yield all artifacts in the specified run"""
    client = MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


def fetch_logged_data(run_id):
    """Fetch params, metrics, tags, and artifacts in the specified run"""
    client = MlflowClient()
    data = client.get_run(run_id).data
    # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = list(yield_artifacts(run_id))
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }

def main():
    # Prepare example dataset:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Enable auto logging - this includes lightgbm.sklearn estimators:
    mlflow.lightgbm.autolog()

    regressor = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.04,
        reg_lambda=1.0
    )
    regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = regressor.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="micro")
    run_id = mlflow.last_active_run().info.run_id
    print("Logged data and model in run {}".format(run_id))
    print("F1 score: {}".format(f1))

    # Show logged data:
    for key, data in fetch_logged_data(run_id).items():
        print("\n---------- logged {} ----------".format(key))
        print(data)



if __name__ == "__main__":
    main()
