import logging
import subprocess

import mlflow
import mlflow.deployments.cli
import pandas as pd
import requests
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from dataset import Dataset
from src.conf import Conf
from src.mlflow_pyfunc import MlflowPyfunc

CONF = Conf()


class MLOps:
    def __init__(self):
        self.mlflow_conda = {
            "channels": ["defaults"],
            "name": "conda",
            "dependencies": [
                "python=3.8",
                "pip",
                {"pip": ["mlflow", "scikit-learn", "cloudpickle", "pandas", "numpy"]},
            ],
        }

    def mlflow_eval_and_log(
        self, model_pipeline: Pipeline, validation_data: pd.DataFrame
    ) -> str:
        valid_x = validation_data.drop(columns=CONF.col_label)
        y_pred = model_pipeline.predict(valid_x)

        with mlflow.start_run():
            mlflow.log_metric(
                "accuracy",
                accuracy_score(validation_data[CONF.col_label].values, y_pred),
            )
            mlflow.log_metric(
                "precison",
                precision_score(validation_data[CONF.col_label].values, y_pred),
            )
            mlflow.log_metric(
                "recall", recall_score(validation_data[CONF.col_label].values, y_pred)
            )
            mlflow.log_metric(
                "roc_auc", roc_auc_score(validation_data[CONF.col_label].values, y_pred)
            )

            signature = infer_signature(valid_x, y_pred)

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=MlflowPyfunc(model=model_pipeline),
                conda_env=self.mlflow_conda,
                signature=signature,
            )

            mlflow.sklearn.log_model(
                artifact_path="model",
                sk_model=model_pipeline,
                conda_env=self.mlflow_conda,
                signature=signature,
            )
            run = mlflow.active_run()
            run_id = run.info.run_id
            logging.info("Active run_id: {}".format(run_id))
            return run_id

    @staticmethod
    def mlflow_serve(run_id: str):

        bash_command = (
            f"mlflow models serve -m {CONF.path_mlflow}/{run_id}/artifacts/model/"
        )

        logging.info(f"running bash_command: $ {bash_command}")
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output, error)
        return output, error

    @staticmethod
    def mlflow_inference(data: pd.DataFrame) -> str:
        host = "localhost"
        port = "5000"
        url = f"http://{host}:{port}/invocations"
        headers = {
            "Content-Type": "application/json",
        }

        feats = Dataset.get_feat_list(data)

        data_x = data[feats]
        http_data = data_x.to_json(orient="split")

        r = requests.post(url=url, headers=headers, data=http_data)
        print(f"Predictions: {r.text}")
        return r.text
