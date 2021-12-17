import mlflow


class MlflowPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        model_input.columns = map(str.lower, model_input.columns)
        return self.model.predict_proba(model_input)[:, 1]
