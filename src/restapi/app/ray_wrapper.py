import ray
from ray import serve

from main import app

ray.init(namespace="summarizer", ignore_reinit_error=True)
serve.start(detached=True)


@serve.deployment(route_prefix="/", num_replicas=2)
@serve.ingress(app)
class RayWrapper:
    pass


if __name__ == '__main__':
    RayWrapper.deploy()
    while True:
        pass
