from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


class Conf:
    path_data_raw: str = f"{ROOT_DIR}/data/raw"
    path_data_interim: str = f"{ROOT_DIR}/data/interim"
    path_models: str = f"{ROOT_DIR}/models"
    path_mlflow: str = f"{ROOT_DIR}/src/mlruns/0"

    col_user: str = "user_id"
    col_label: str = "customer"
    col_days_in_trial = "days_since_trial_start"

    metric = "auc"
    fit_params = {
        "objective": "binary",
        "metric": metric,
        "num_leaves": 127,
        "max_depth": -1,
        "lambda_l2": 5,
    }

    feats_not = [
        "onboarding_home_at",
        "signup_result_at",
        "subscription_enter_at",
        "subscription_at",
    ]

    logging_pattern = (
        "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
    )
