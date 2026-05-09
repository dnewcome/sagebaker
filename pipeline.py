"""SageMaker Pipeline — production training equivalent of `make train`.

This is the cloud counterpart to local_train_dlc.py + local_train_feast_dlc.py.
Each step runs in a managed SageMaker container; the same `src/train.py`
runs unchanged inside the TrainingStep — only the channel URIs change
(S3 instead of `file://`).

Sketch — fill in the constants below for your account, then:

    .venv/bin/python pipeline.py upsert      # create / update the pipeline
    .venv/bin/python pipeline.py start       # kick off a run

Untested in this repo (no real AWS account targeted) — the structure is
the production-shape; you'll likely want to tweak instance types, IAM
permissions, and the prep step to match your real data flow.

The trainer image is the AWS scikit-learn DLC (no custom ECR push needed).
If you want a custom image later, swap `framework_version=` for
`image_uri=` on the SKLearn estimator.
"""
import sys

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# ===== fill in for your account ===========================================
PROJECT_NAME = "sage-baker"
BUCKET = "yourorg-ml-prod"
ROLE_ARN = "arn:aws:iam::ACCT:role/SageMakerExecutionRole"
MODEL_PACKAGE_GROUP = "sage-baker-sklearn"
FRAMEWORK_VERSION = "1.2-1"  # match src/train.py's sklearn pin

session = PipelineSession()

# --- runtime parameters ---------------------------------------------------
input_data = ParameterString(
    name="InputDataUri",
    default_value=f"s3://{BUCKET}/raw/sonar/",
)
metric_threshold = ParameterFloat(
    name="MetricThreshold",
    default_value=0.75,
)

# --- 1. PROCESSING: materialize features → S3 ----------------------------
# Replace `code=` with whatever your real prep does (BQ pull, Feast retrieval).
# Whatever it writes to /opt/ml/processing/output/ gets uploaded to S3.
processor = SKLearnProcessor(
    framework_version=FRAMEWORK_VERSION,
    role=ROLE_ARN,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=session,
)
prep_step = ProcessingStep(
    name="prepare",
    processor=processor,
    code="prep/prepare_bigquery.py",  # or your real prep script
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input/")],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/",
            destination=Join(on="/", values=[
                f"s3://{BUCKET}/training", ExecutionVariables.PIPELINE_EXECUTION_ID,
            ]),
        ),
    ],
)

# --- 2. TRAINING: AWS sklearn DLC running our src/train.py --------------
# Same trainer code as local. Only differences: image is pulled from
# AWS DLC repo, instance_type is a real EC2 type, channel is an S3 URI.
estimator = SKLearn(
    entry_point="train.py",
    source_dir="src",
    framework_version=FRAMEWORK_VERSION,
    py_version="py3",
    role=ROLE_ARN,
    instance_type="ml.c5.xlarge",
    hyperparameters={"n-estimators": 200, "max-depth": 4},
    sagemaker_session=session,
)
train_step = TrainingStep(
    name="train",
    estimator=estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=prep_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        ),
    },
)

# --- 3. EVAL: load the bundle, score, write metrics.json -----------------
# `evaluate.py` (not in this repo yet) reads the model.tar.gz + a holdout
# set and emits {"validation_accuracy": <float>, ...} to /opt/ml/processing/evaluation/.
eval_property = PropertyFile(
    name="EvaluationReport", output_name="evaluation", path="metrics.json"
)
eval_step = ProcessingStep(
    name="evaluate",
    processor=processor,
    code="evaluate.py",  # TODO: write this — small, ~30 lines
    inputs=[
        ProcessingInput(
            source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model/",
        ),
        ProcessingInput(
            source=prep_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            destination="/opt/ml/processing/test/",
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation/",
            destination=Join(on="/", values=[
                f"s3://{BUCKET}/evaluation", ExecutionVariables.PIPELINE_EXECUTION_ID,
            ]),
        ),
    ],
    property_files=[eval_property],
)

# --- 4. REGISTER: drop the model into the Model Package Group -----------
register_step = RegisterModel(
    name="register",
    estimator=estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name=MODEL_PACKAGE_GROUP,
    approval_status="PendingManualApproval",  # human gate before deploy
)

# --- 5. CONDITION: only register if validation_accuracy ≥ threshold ----
condition_step = ConditionStep(
    name="metric-gate",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=eval_step.name,
                property_file=eval_property,
                json_path="validation_accuracy",
            ),
            right=metric_threshold,
        ),
    ],
    if_steps=[register_step],
    else_steps=[],
)

# --- Pipeline definition -----------------------------------------------
pipeline = Pipeline(
    name=PROJECT_NAME,
    parameters=[input_data, metric_threshold],
    steps=[prep_step, train_step, eval_step, condition_step],
    sagemaker_session=session,
)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "upsert"
    if cmd == "upsert":
        pipeline.upsert(role_arn=ROLE_ARN)
        print(f"upserted pipeline: {pipeline.name}")
        print(f"start a run with: python pipeline.py start")
    elif cmd == "start":
        execution = pipeline.start()
        print(f"started: {execution.arn}")
    else:
        sys.exit(f"unknown command {cmd!r}; use 'upsert' or 'start'")
