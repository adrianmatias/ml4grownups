import pandas as pd

from conf import Conf

CONF = Conf()


class FeatureEngineering:
    def __init__(self, df_raw: pd.DataFrame):
        self.df_raw = df_raw
        self.other_category_name = "other"

    def featurize(self):
        feats_cat = [
            "os_name",
            "os_version",
            "country",
            "locale",
            "device_type",
            "device_model",
            "source",
            "level",
            "signup_provider",
            "currency",
            "payment_platform"
        ]

        feats_not = [
            CONF.col_label,
        ]

        for feat in feats_cat:
            self.reduce_cardinality(feat)

        feats = list(
            self
                .df_raw
                .drop(columns=feats_not)
                .select_dtypes(exclude=["object"])
                .columns
        )
        self.df = self.df_raw[feats + [CONF.col_label]]
        self.normalize_price()

    def reduce_cardinality(self, col):  # TODO: use label encoding
        other_freq_threshold = 0.01  # TODO: try different thresholds
        col_top = "top"
        freq = (
            self
                .df_raw[col]
                .value_counts(normalize=True)
        )
        top = (
            freq[freq.values > other_freq_threshold]
                .reset_index()
                .drop(columns=col)
                .rename(columns={"index": col})
        )
        top[col_top] = top

        df = pd.merge(
            left=self.df_raw,
            right=top,
            on=col,
            how="left"
        )
        df[col] = pd.Categorical(
            df[col_top].fillna(self.other_category_name),
            ordered=True  # TODO: provide frequency or mean label ordering
        )
        self.df_raw = df.drop(columns=col_top)

    def featurize_dates(self):
        # TODO: extract day of month, day of week, hour
        return

    def featurize_dates_diff(self):
        # TODO: create features for difference between dates
        return

    def normalize_price(self):
        col_price = "price"
        col_currency = "currency"
        rate_to_usd = {
            self.other_category_name: 1.0,
            "USD": 1.0,
            "EUR": 1.1950077693,
            "MXN": 0.0493444324,
            "BRL": 0.2016380856,
            "RUB": 0.0137762816,
            "GBP": 1.3976345213,
            "CNY": 0.1544614943,
            "PHP": 0.0205233231,
            "THB": 0.0314360538,
            "IDR": 0.0000692856,
            "TRY": 0.1156317003,
            "KRW": 0.0008819045,
            "CAD": 0.8138211946,
            "CLP": 0.0013607630,
            "AUD": 0.7588196347,
            "VND": 0.0000433363,
            "COP": 0.0002644818,
            "HKD": 0.1287821832
            # https://www.xe.com/currencytables/?from=USD
        }
        self.df[col_price] = self.df.apply(
            lambda row: rate_to_usd[row[col_currency]] * row[col_price],
            axis=1
        )
