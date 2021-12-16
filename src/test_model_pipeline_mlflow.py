import mlflow
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from dataset import Dataset, get_data
from src.conf import Conf
from src.mlflow_pyfunc import MlflowPyfunc
from src.model_pipeline import ModelPipeline

CONF = Conf()


def train():
    dataset = Dataset(
        data_user=get_data("users_onboarding_paywall.tsv"),
        data_activity=get_data("activities_per_day.tsv"),
    )

    dataset.make_dataset()

    model = ModelPipeline()

    return model.train(dataset=dataset)


class TestModelPipelineMlflow:
    def test_train(self):
        assert [name for (name, _) in train().steps] == [
            "columntransformer",
            "lgbmclassifier",
        ]

    def test_mlflow(self):
        mlflow_conda = {
            "channels": ["defaults"],
            "name": "conda",
            "dependencies": [
                "python=3.8",
                "pip",
                {"pip": ["mlflow", "scikit-learn", "cloudpickle", "pandas", "numpy"]},
            ],
        }

        dataset = Dataset(
            data_user=get_data("users_onboarding_paywall.tsv"),
            data_activity=get_data("activities_per_day.tsv"),
        )

        dataset.make_dataset()

        train, valid, test = dataset.split()

        model_pipeline = ModelPipeline().train(dataset=dataset)

        valid_x = valid[dataset.get_feat_list()]

        y_pred = model_pipeline.predict(valid_x)

        with mlflow.start_run():
            mlflow.log_metric(
                "accuracy", accuracy_score(valid[CONF.col_label].values, y_pred)
            )
            mlflow.log_metric(
                "precison", precision_score(valid[CONF.col_label].values, y_pred)
            )
            mlflow.log_metric(
                "recall", recall_score(valid[CONF.col_label].values, y_pred)
            )
            mlflow.log_metric(
                "roc_auc", roc_auc_score(valid[CONF.col_label].values, y_pred)
            )

            signature = infer_signature(valid_x, y_pred)

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=MlflowPyfunc(model=model_pipeline),
                conda_env=mlflow_conda,
                signature=signature,
            )

            mlflow.sklearn.log_model(
                artifact_path="model",
                sk_model=model_pipeline,
                conda_env=mlflow_conda,
                signature=signature,
            )
            run = mlflow.active_run()
            print("Active run_id: {}".format(run.info.run_id))
