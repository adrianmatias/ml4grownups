import json

import requests

from ray_wrapper import RayWrapper


class TestModelPipelineMlflow:

    RayWrapper.deploy()

    def test_deployment(self):
        with open(f"./user-examples/user_0.json", "rb") as file:
            user_dict = json.load(file)

        response = requests.post(url="http://127.0.0.1:8000/predict", json=user_dict)

        assert response.json() == {"Prediction": 1}
