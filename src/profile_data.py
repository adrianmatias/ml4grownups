import os
from pathlib import Path

import pandas as pd
import seaborn as sns
from pandas_profiling import ProfileReport

sns.set_theme(style="darkgrid")

PROJECT_DIR = Path(__file__).resolve().parents[1]

FOLDER_RAW = f"{PROJECT_DIR}/data/raw"
FOLDER_INTERIM = f"{PROJECT_DIR}/data/interim"


def main():
    profile_data(filename="users_onboarding_paywall.tsv")
    profile_data(filename="activities_per_day.tsv")


def profile_data(filename: str):
    filename_profile = filename.replace(".tsv", "_profile.html")

    df = pd.read_csv(
        os.path.join(FOLDER_RAW, filename),
        sep="\t"
    )
    print(df.head())

    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file(f"{FOLDER_INTERIM}/{filename_profile}")


if __name__ == '__main__':
    main()
