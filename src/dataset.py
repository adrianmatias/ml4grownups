import os

import pandas as pd
import numpy as np
from conf import Conf
from typing import List
from sklearn.model_selection import train_test_split

CONF = Conf()


def main():
    dataset = Dataset(
        data_user=get_data("users_onboarding_paywall.tsv"),
        data_activity=get_data("activities_per_day.tsv"),
    )

    print(dataset.data_user.head())
    print(dataset.data_activity.head())

    dataset.make_dataset()

    print(dataset.df.head())


class Dataset:
    def __init__(self, data_user: pd.DataFrame, data_activity: pd.DataFrame):
        self.data_user = data_user
        self.data_activity = data_activity
        self.data_user_processed = pd.DataFrame
        self.data_activity_processed = pd.DataFrame
        self.df = pd.DataFrame

    def make_dataset(self):
        self.process_data_user()
        self.process_data_activity()
        self.join()

    def process_data_user(self):
        self.data_user_processed = (
            self.data_user.copy()
            .drop_duplicates(CONF.col_user)
            .set_index(CONF.col_user)
        )

    def split(self):
        train, test = train_test_split(self.df, test_size=0.2)
        train, valid = train_test_split(train, test_size=0.2)
        return train, valid, test


    @staticmethod
    def drop_label(df: pd.DataFrame):
        return df.drop(columns=CONF.col_label)

    def process_data_activity(self):
        cols_key = [CONF.col_user, CONF.col_days_in_trial]

        agg_total = dict(
            (col, sum) for col in self.data_activity.columns if "total_" in col
        )
        agg_mean = dict(
            (col, np.mean) for col in self.data_activity.columns if "mean_" in col
        )

        self.data_activity_processed = (
            self.data_activity.copy()
            .groupby(cols_key)
            .agg({**agg_total, **agg_mean})
            # TODO: apply rolling aggregation features
            # TODO: add record count
            .reset_index()
            .set_index(CONF.col_user)
        )

    def join(self):
        self.df = pd.merge(
            left=self.data_user_processed,
            right=self.data_activity_processed,
            on=CONF.col_user,
            how="left",
        ).reset_index()

    @staticmethod
    def get_feat_list_cat(df: pd.DataFrame) -> List[str]:
        return list(
            df.drop(columns=CONF.col_label).select_dtypes(include=["object"]).columns
        )

    @staticmethod
    def get_feat_list_num(df: pd.DataFrame) -> List[str]:
        return list(
            df.drop(columns=CONF.col_label).select_dtypes(exclude=["object"]).columns
        )

    @staticmethod
    def get_feat_list(df: pd.DataFrame) -> List[str]:
        return list(df.drop(columns=CONF.col_label).columns)


def get_data(filename: str) -> pd.DataFrame:
    path_filename = os.path.join(CONF.path_data_raw, filename)

    return pd.read_csv(filepath_or_buffer=path_filename, sep="\t")


if __name__ == "__main__":
    main()
