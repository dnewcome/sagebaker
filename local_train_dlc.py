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
from sagemaker.local import LocalSession
from sagemaker.sklearn.estimator import SKLearn

if not (os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID")):
    raise SystemExit(
        "Real AWS credentials required for DLC pulls.\n"
        "Recommended: aws sso login --profile <name> && export AWS_PROFILE=<name>\n"
        "Or set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY directly."
    )
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# Snap-confined Docker can't bind-mount /tmp paths.
SCRATCH = os.path.abspath(".sm-scratch")
os.makedirs(SCRATCH, exist_ok=True)
os.environ["TMPDIR"] = SCRATCH

session = LocalSession()
session.config = {"local": {"local_code": True, "container_root": SCRATCH}}

estimator = SKLearn(
    entry_point="train.py",
    source_dir=".",
    role="arn:aws:iam::000000000000:role/SageMakerRole",  # ignored locally
    instance_type="local",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    hyperparameters={"n-estimators": 200, "max-depth": 4},
    sagemaker_session=session,
)

estimator.fit({"train": "file://./data/"})
print("\nmodel artifact:", estimator.model_data)
