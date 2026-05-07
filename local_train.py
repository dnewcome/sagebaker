"""Run training in SageMaker Local Mode against a locally-built BYOC image.

Build the image first:
    docker build -t sage-baker-sklearn:latest .

This avoids any ECR pulls — `image_uri` points at the local image, and Local
Mode skips pulling when the tag has no registry prefix and is present locally.
"""
import os
from sagemaker.estimator import Estimator
from sagemaker.local import LocalSession

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "dummy")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "dummy")

# Snap-confined Docker can't bind-mount /tmp paths, so steer the SDK's
# scratch dirs (container_root + TMPDIR) under the project tree.
SCRATCH = os.path.abspath(".sm-scratch")
os.makedirs(SCRATCH, exist_ok=True)
os.environ["TMPDIR"] = SCRATCH

session = LocalSession()
session.config = {"local": {"local_code": True, "container_root": SCRATCH}}

# If MLFLOW_TRACKING_URI is set on the host, pass it through to the container.
# Rewrite localhost references because inside the container "localhost" is
# the container itself; host.docker.internal resolves to the host.
def _container_env():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri:
        return {}
    uri = uri.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal")
    return {"MLFLOW_TRACKING_URI": uri}

estimator = Estimator(
    image_uri="sage-baker-sklearn:latest",
    role="arn:aws:iam::000000000000:role/SageMakerRole",  # ignored locally
    instance_type="local",
    instance_count=1,
    hyperparameters={"n-estimators": 200, "max-depth": 4},
    environment=_container_env(),
    sagemaker_session=session,
)

estimator.fit({"train": "file://./data/"})
print("\nmodel artifact:", estimator.model_data)
