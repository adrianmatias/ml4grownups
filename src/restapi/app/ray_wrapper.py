import ray
from ray import serve
from fastapi.encoders import jsonable_encoder
import pandas as pd
from main import app, User

ray.init(namespace="Predicting Subscriber", ignore_reinit_error=True)
serve.start(detached=True)


@serve.deployment(route_prefix="/", num_replicas=2)
@serve.ingress(app)
class RayWrapper:
    @app.post("/predict_async")
    def predict_async(self, user: User):
        df = pd.DataFrame(jsonable_encoder([user]))

        pred = self.clf.predict(df).tolist()[0]
        return {"Prediction": pred}


if __name__ == "__main__":
    RayWrapper.deploy()
    while True:
        pass
