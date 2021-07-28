from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


# @dataclass
class Conf:
    path_data_raw: str = f"{ROOT_DIR}/data/raw"
    path_data_interim: str = f"{ROOT_DIR}/data/interim"

    col_user: str = "user_id"
    col_label: str = "customer"
    col_days_in_trial = "days_since_trial_start"
