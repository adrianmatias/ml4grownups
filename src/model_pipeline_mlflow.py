import logging

import lightgbm as lgb
import mlflow.pyfunc
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.conf import Conf
from src.dataset import Dataset, get_data

CONF = Conf()

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)


class ModelPipelineMlflow(mlflow.pyfunc.PythonModel):

    @staticmethod
    def train(dataset: Dataset) -> Pipeline:
        feats_num = dataset.get_feat_list_num()
        feats_cat = dataset.get_feat_list_cat()

        logging.info(f"train on {len(feats_num)} features num: {feats_num}")
        logging.info(f"train on {len(feats_cat)} features cat: {feats_cat}")

        numeric_pipeline = make_pipeline(SimpleImputer(), OrdinalEncoder())
        categorical_pipeline = make_pipeline(SimpleImputer(strategy="constant"), OrdinalEncoder())

        full_processor = ColumnTransformer(transformers=[
            ('num', numeric_pipeline, feats_num),
            ('cat', categorical_pipeline, feats_cat)
        ])

        lgb_model = lgb.sklearn.LGBMClassifier(**CONF.fit_params)
        pipeline = make_pipeline(full_processor, lgb_model)

        logging.info(f"declared pipeline:\n{pipeline}")

        feats = feats_num + feats_cat

        pipeline.fit(X=dataset.df[feats], y=dataset.df[CONF.col_label])

        return pipeline

    # def load_context(self, context):
    #     import xgboost as xgb
    #     self.xgb_model = xgb.Booster()
    #     self.xgb_model.load_model(context.artifacts["xgb_model"])
    #
    # def predict(self, context, model_input):
    #     input_matrix = xgb.DMatrix(model_input.values)
    #     return self.xgb_model.predict(input_matrix)
