from dataset import Dataset, get_data
from src.model_pipeline_mlflow import ModelPipelineMlflow


class TestModelPipelineMlflow:
    def test_train(self):
        dataset = Dataset(
            data_user=get_data("users_onboarding_paywall.tsv"),
            data_activity=get_data("activities_per_day.tsv")
        )

        dataset.make_dataset()

        model = ModelPipelineMlflow()
        model_pipeline = model.train(dataset=dataset)

        assert model_pipeline.steps == ["columntransformer", "lgbmclassifier"]
