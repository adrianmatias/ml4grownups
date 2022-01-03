import pickle

import pandas as pd
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

app = FastAPI(title="Predicting Subscriber")


class User(BaseModel):

    user_id: float
    age: float
    price: float
    days_since_trial_start: float
    total_feature_14: float
    total_feature_15: float
    total_feature_16: float
    total_feature_17: float
    mean_feature_10: float
    mean_feature_11: float
    mean_feature_12: float
    mean_feature_13: float
    mean_feature_18: float
    mean_feature_19: float
    mean_feature_20: float
    mean_feature_21: float
    os_name: str
    os_version: str
    country: str
    timezone: str
    locale: str
    device_type: str
    device_model: str
    source: str
    level: str
    signup_provider: str
    currency: str
    payment_platform: str


@app.on_event("startup")
def load_pipeline():
    with open(f"./model/artifacts/model/model.pkl", "rb") as file:
        global clf
        clf = pickle.load(file)


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:80/docs"


@app.post("/predict")
def predict(user: User):

    df = pd.DataFrame(jsonable_encoder([user]))

    pred = clf.predict(df).tolist()[0]
    return {"Prediction": pred}
