import logging

import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.conf import Conf
from src.dataset import Dataset

CONF = Conf()

logging.basicConfig(format=CONF.logging_pattern, level=logging.INFO)


class ModelPipeline:
    @staticmethod
    def train(dataset: Dataset) -> Pipeline:
        feats_num = dataset.get_feat_list_num()
        feats_cat = dataset.get_feat_list_cat()

        logging.info(f"train on {len(feats_num)} features num: {feats_num}")
        logging.info(f"train on {len(feats_cat)} features cat: {feats_cat}")

        pipeline_feat_engineering = ColumnTransformer(
            transformers=[
                ("num", make_pipeline(SimpleImputer()), feats_num),
                (
                    "cat",
                    make_pipeline(
                        # SimpleImputer does not support pd.NA, introduced at pandas==1.0.0
                        SimpleImputer(strategy="constant"),
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                    feats_cat,
                ),
            ]
        )

        pipeline = make_pipeline(
            pipeline_feat_engineering, lgb.sklearn.LGBMClassifier(**CONF.fit_params)
        )

        logging.info(f"declared pipeline:\n{pipeline}")

        pipeline.fit(
            X=dataset.df[dataset.get_feat_list()], y=dataset.df[CONF.col_label]
        )

        return pipeline
