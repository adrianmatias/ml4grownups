import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold

from conf import Conf

CONF = Conf()


class Model:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def split_train_test(self):
        """
        Train-validation-test split:
        No user datapoint contains information from future datapoints.
        Even if rolling aggregated features were present, this statement would remain valid.
        Thus, random split is safe against target leakage.
        """
        self.train, self.test = train_test_split(
            self.df,
            test_size=0.3,
            random_state=42
        )

    def train_and_evaluate(self):
        col_label = CONF.col_label

        train_x = self.train.drop(columns=col_label)
        train_y = self.train[col_label]
        test_x = self.test.drop(columns=col_label)
        test_y = self.test[col_label]

        train_set = lgb.Dataset(data=train_x, label=train_y)

        folds = KFold(n_splits=4)
        metric = "auc"
        params = {
            "objective": "binary",
            "metric": metric,
            "num_leaves": 127,
            "max_depth": -1,
            "lambda_l2": 5
        }
        # TODO: grid search parameters

        cv_result = lgb.cv(
            params=params,
            train_set=train_set,
            folds=folds,
            seed=42,
        )
        auc_cv = cv_result["auc-mean"][-1]

        lgb_model = lgb.LGBMClassifier(**params)
        lgb_model.fit(train_x, train_y)

        self.lgb_model = lgb_model

        auc_train = roc_auc_score(
            y_true=train_y,
            y_score=lgb_model.predict_proba(train_x)[:, 1]
        )
        auc_test = roc_auc_score(
            y_true=test_y,
            y_score=lgb_model.predict_proba(test_x)[:, 1]
        )

        self.summary = pd.DataFrame(
            [
                (auc_train, "train"),
                (auc_cv, "train_cv"),
                (auc_test, "test"),
            ] + [(
                    self.roc_auc_score_first_n_days_in_trial(n),
                    f"test_first_{n}_days"
            ) for n in range(1, 7)
            ],
            columns=[metric, "dataset"]
        ).set_index("dataset")

    def plot_summary(self):
        self.summary.plot(rot=90)

    def inspect_importance_info_gain(self):
        """
        This feature importance is computed over data known during training.
        Feature importance based on tree info gain is biased towards
        high cardinality features and may justify low generalizable conclusions.
        """
        lgb.plot_importance(self.lgb_model, height=0.8, max_num_features=20)

    def roc_auc_score_first_n_days_in_trial(self, n_days: int):
        df = self.test.copy()
        df = df[df[CONF.col_days_in_trial] <= n_days]
        df_x = df.drop(columns=CONF.col_label)
        df_y = df[CONF.col_label]
        return roc_auc_score(
            y_true=df_y,
            y_score=self.lgb_model.predict_proba(df_x)[:, 1]
        )

    def inspect_feat_cat(self, feat: str):
        col_label = CONF.col_label
        df = (
            self
                .df
                .groupby(feat)
                .agg({col_label: ["mean", "count"]})
                .sort_values((col_label, "count"))
        )
        df[col_label]["count"].plot(ylabel="count")
        df[col_label]["mean"].plot(kind='bar', secondary_y=True)
        plt.show()

    def inspect_feat_num(self, feat: str):
        sns.violinplot(x=CONF.col_label, y=feat, data=self.df)
        plt.show()

    def inspect_feat_importance(self):
        explainer = shap.TreeExplainer(self.lgb_model)

        test_x = self.test.drop(columns=CONF.col_label)
        shap.summary_plot(explainer.shap_values(test_x), test_x)

    def inspect_feat_impact(self):
        explainer = shap.TreeExplainer(self.lgb_model)

        test_x = self.test.drop(columns=CONF.col_label)
        shap_values = explainer.shap_values(test_x)

        shap.summary_plot(shap_values[1], test_x)
