# ml4grownups

This repository trains and deploys a supervised classification model via
- MLflow
- REST-API
- REST-API + Docker

## Data context

One mobile app offers a monthly subscription with one week trial period (at the end of which, the customer starts paying the subscription, unless they cancel). Using a sample of data, the aim of this repository is to extract insights and build a model to predict if a user would become a paying customer or not at the end of trial.
Further info at https://github.com/adrianmatias/app-trial-period-subscriber-classifier

## General overview of the app flow

Journey starts with installing the app on a user's device. When a user launches the app for the first time he needs to complete the onboarding. App asks for some information like the level or age of the children to personalize the experience. Right after, a paywall is shown. This is where user is ofered the chance to start a subscription with the mentioned 7 days trial period. After 7 days - and if they don’t cancel their subscription - they become paying customers and an invoice is recorded in the system. In the paywall they have the chance to not start a trial, so some users can use the app but in a limited way (would be a basic user). If they try to access features that are only available for paying customers, or if they reach daily limits, the paywall is shown again.

There is a key interest in knowing if a user would potentially convert into a customer after trial ends. Besides, knowing it as soon as possible in the trial period gives us some advantage to take other actions, so it is benefical to anticipate the prediction as much as possible, ideally in the first days of trial.

Main objectives are:

- Design and build a classification model that predicts if a user who has started a trial would become a paying customer.
- Provide some insights about which features have more impact in converting to customers.

## Data provided

- users_onboarding_paywall.tsv: information of user onboarding and subcription paywal users_onboarding_paywall.tsv
- activities_per_day.tsv: information of activities played during trial period.

All detailed info regarding structure and fields can be found here:

## How to reproduce analysis and use model

Is recommended to use `virtualenv` for development:

- Start by installing `virtualenv` if you don't have it
```
pip install virtualenv
```

- Once installed access the project folder
```
cd .../ml4grownups
```

- Create a virtual environment
```
virtualenv venv
```

- Enable the virtual environment
```
source venv/bin/activate
```

- Install the python dependencies on the virtual environment
```
pip install -r requirements.txt
```

- Run the test suite to train a model
```
python src/test_model_pipeline_mlflow.py
```

## Deployment and inference


### MLflow

serve and predict
```
mlflow models serve -m src/mlruns/0/<run_id>/artifacts/model

python src/dataset.py
curl -d @./data/interim/user_0.json -H 'Content-Type: application/json'  localhost:5000/invocations
```

### REST-API

- copy the model as rest-api resource
```
rm -r src/restapi/app/model
mkdir src/restapi/app/model
cp -r src/mlruns/0/{run_id}/. src/restapi/app/model
```

- Run app
```
cd ml4grownups/src/restapi/app
uvicorn main:app --reload
```

- post request to infer a sample user
```
python src/dataset.py
curl -X POST http://localhost:8000/predict -d @./user-examples/user_0.json -H "Content-Type: application/json"
```

### REST-API + Docker

- build and run locally docker image
```
docker build -t ml4grownups:rest-api .
docker run --rm -p 80:80 ml4grownups:rest-api
```

- post request to infer a sample user at docker exposed port 80
```
cd .../ml4grownups/src/restapi/app
curl -X POST http://localhost:80/predict -d @./user-examples/user_0.json -H "Content-Type: application/json"
```

### REST-API + Ray Serve

- Run app
```
cd .../ml4grownups/src/restapi/app
nohup python ray_wrapper.py > log_ray_wrapper.txt 2>&1 &
```
- post request to infer a sample user
```
cd .../ml4grownups/src/restapi/app
curl -X POST http://localhost:80/predict -d @./user-examples/user_0.json -H "Content-Type: application/json"
```

