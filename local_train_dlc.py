"""Run training in SageMaker Local Mode against the official AWS DLC image.

Pulls the AWS scikit-learn Deep Learning Container the first time, which
requires real AWS credentials. Any account works — the image is publicly
readable, but ECR still demands a real auth token to issue the pull.

Recommended: AWS IAM Identity Center (SSO) with short-lived credentials.
    aws configure sso          # one-time, creates a profile
    aws sso login --profile <name>
    export AWS_PROFILE=<name>

Long-lived access keys also work (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
but AWS no longer recommends them for human users.

Benefits over the BYOC path:
  - edit train.py without rebuilding any image (entry_point flow)
  - AWS-tested framework + dep stack, with security patches
  - .deploy() on the returned estimator gives a working /ping + /invocations
    endpoint with no extra container work
"""
import os
import boto3
from sagemaker.local import LocalSession
from sagemaker.sklearn.estimator import SKLearn

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

if boto3.Session().get_credentials() is None:
    raise SystemExit(
        "Real AWS credentials required for DLC pulls.\n"
        "Run `aws configure` (access key) or `aws sso login --profile <name>`\n"
        "and ensure AWS_PROFILE points at the right profile if not [default]."
    )

# Snap-confined Docker can't bind-mount /tmp paths.
SCRATCH = os.path.abspath(".sm-scratch")
os.makedirs(SCRATCH, exist_ok=True)
os.environ["TMPDIR"] = SCRATCH

session = LocalSession()
session.config = {"local": {"local_code": True, "container_root": SCRATCH}}

# Same MLFLOW_TRACKING_URI passthrough as local_train.py — see comments there.
def _container_env():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri:
        return {}
    uri = uri.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal")
    return {"MLFLOW_TRACKING_URI": uri}

estimator = SKLearn(
    entry_point="train.py",
    source_dir="src",  # must NOT contain a requirements.txt — see README
    role="arn:aws:iam::000000000000:role/SageMakerRole",  # ignored locally
    instance_type="local",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    hyperparameters={"n-estimators": 200, "max-depth": 4},
    environment=_container_env(),
    sagemaker_session=session,
)

estimator.fit({"train": "file://./data/"})
print("\nmodel artifact:", estimator.model_data)
