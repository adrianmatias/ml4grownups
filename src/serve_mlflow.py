import requests

from dataset import Dataset, get_data

host = "localhost"
port = "5000"
url = f"http://{host}:{port}/invocations"
headers = {
    "Content-Type": "application/json",
}

dataset = Dataset(
    data_user=get_data("users_onboarding_paywall.tsv"),
    data_activity=get_data("activities_per_day.tsv"),
)

dataset.make_dataset()

train, valid, test = dataset.split()

test_x = test[dataset.get_feat_list()]

http_data = test_x.to_json(orient="split")

# pprint.pprint(json.loads(http_data))
print(test_x.info())

r = requests.post(url=url, headers=headers, data=http_data)
print(f"Predictions: {r.text}")
