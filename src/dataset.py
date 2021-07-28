import os

import pandas as pd
import numpy as np
from conf import Conf

CONF = Conf()


def main():
    dataset = Dataset()

    print(dataset.data_user.head())
    print(dataset.data_activity.head())

    dataset.make_dataset()

    print(dataset.df.head())


class Dataset:
    def __init__(self):
        # labels = ""
        self.data_user: pd.DataFrame = self.get_data(
            "users_onboarding_paywall.tsv")
        self.data_activity: pd.DataFrame = self.get_data(
            "activities_per_day.tsv")
        self.data_user_processed = pd.DataFrame
        self.data_activity_processed = pd.DataFrame
        self.df = pd.DataFrame

    @staticmethod
    def get_data(filename: str) -> pd.DataFrame:
        path_filename = os.path.join(CONF.path_data_raw, filename)

        return pd.read_csv(
            filepath_or_buffer=path_filename,
            sep="\t"
        )

    def make_dataset(self):
        self.process_data_user()
        self.process_data_activity()
        self.join()

    def process_data_user(self):
        self.data_user_processed = (
            self
                .data_user
                .copy()
                .drop_duplicates(CONF.col_user)
                .set_index(CONF.col_user)
        )

    def process_data_activity(self):
        cols_key = [
            CONF.col_user,
            CONF.col_days_in_trial
        ]

        agg_total = dict(
            (col, sum)
            for col in self.data_activity.columns
            if "total_" in col
        )
        agg_mean = dict(
            (col, np.mean)
            for col in self.data_activity.columns
            if "mean_" in col
        )

        self.data_activity_processed = (
            self
                .data_activity
                .copy()
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


if __name__ == '__main__':
    main()
