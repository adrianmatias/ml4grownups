import json
import time

import requests

from ray_wrapper import RayWrapper


class TestRayWrapper:
    RayWrapper.deploy()

    def test_deployment(self):
        response = requests.post(url=f"{utils.host_port}/predict", json=utils.user_dict)

        assert response.json() == {"Prediction": 1}

    def test_async_latency(self):

        time_blocking = utils.time_request_list(endpoint="predict")
        time_async = utils.time_request_list(endpoint="predict_async")

        assert time_async * 2 < time_blocking


class Utils:

    host_port = "http://127.0.0.1:8000"
    with open(f"./user-examples/user_0.json", "rb") as file:
        user_dict = json.load(file)

    def time_request_list(self, endpoint: str):
        n_req_warm = 10
        n_req = 100

        self.post_request_round(endpoint, n_req_warm)

        start_time = time.time()

        self.post_request_round(endpoint, n_req)

        return time.time() - start_time

    def post_request_round(self, endpoint: str, n_req: int):
        [
            requests.post(url=f"{self.host_port}/{endpoint}", json=self.user_dict)
            for _ in range(n_req)
        ]


utils = Utils()
