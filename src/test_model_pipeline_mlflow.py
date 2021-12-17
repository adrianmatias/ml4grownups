from dataset import Dataset, get_data
from src.conf import Conf
from src.mlops import MLOps
from src.model_pipeline import ModelPipeline

CONF = Conf()


def get_dataset() -> Dataset:
    dataset = Dataset(
        data_user=get_data("users_onboarding_paywall.tsv"),
        data_activity=get_data("activities_per_day.tsv"),
    )

    dataset.make_dataset()
    return dataset


class TestModelPipelineMlflow:
    dataset = get_dataset()
    train, valid, test = dataset.split()
    model_pipeline = ModelPipeline().train(train)

    def test_train(self):
        assert [name for (name, _) in self.model_pipeline.steps] == [
            "columntransformer",
            "lgbmclassifier",
        ]

    def test_mlflow_train_and_log(self):

        run_id = MLOps().mlflow_eval_and_log(
            model_pipeline=self.model_pipeline, validation_data=self.valid
        )

        assert run_id is not ""

    def test_mlflow_inference(self):

        preds = MLOps().mlflow_inference(data=self.test)
        print(preds)
        assert preds is not ""
