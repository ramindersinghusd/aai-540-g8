"""Build the SageMaker continual-learning CI/CD notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "AAI540_SageMaker_Continual_Learning_CICD.ipynb"
CELL_COUNTER = 0


def _id(prefix: str) -> str:
    global CELL_COUNTER
    CELL_COUNTER += 1
    return f"{prefix}-{CELL_COUNTER:02d}"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _id("markdown"),
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _id("code"),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""
# AAI 540 continual learning and model CI/CD

This stage consumes the versioned data, champion-model, and monitoring receipts produced by stages 1-3. It builds a SageMaker Pipeline DAG that:

1. validates newly labeled feedback and splits it by product/question group;
2. appends feedback to the original train and validation channels without leakage;
3. trains a new candidate with the same immutable feature contract;
4. compares candidate and champion on the untouched test split;
5. blocks a regression with a condition gate; and
6. registers a passing model as `PendingManualApproval` (or writes an S3 promotion-candidate record when Model Registry is unavailable).

Promotion remains a separate, explicit human action. A monitoring alarm or successful training run never changes the champion automatically.
"""
    ),
    markdown(
        r"""
## Prerequisites, permissions, and cost boundary

Run the preceding three notebooks first from this project directory. The execution role needs S3 read/write, SageMaker Processing/Training/Pipelines, ECR pull, `iam:PassRole`, and - when enabled - Model Registry permissions. Set `ORCHESTRATION_MODE="scripted"` if the Learner Lab role does not permit SageMaker Pipelines; it uses the same scripts, manifests, and quality gate.

The default run uses small CPU instances and all Processing/Training compute terminates automatically. Model Registry and pipeline metadata do not create an endpoint. This notebook never deploys an always-on endpoint.
"""
    ),
    code(
        r"""
%pip install -q --upgrade \
    "boto3>=1.34" \
    "botocore>=1.34" \
    "sagemaker>=2.232,<3" \
    "pandas>=2.1,<3" \
    "pyarrow>=15,<22" \
    "scikit-learn==1.4.2" \
    "joblib>=1.3,<2" \
    "flask>=3,<4" \
    "pytest>=8,<9"
"""
    ),
    markdown(
        "> Restart the kernel only if the install cell reports an already-loaded package conflict. Then continue from the import cell."
    ),
    code(
        r"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

def locate_project_root():
    configured = os.getenv("AAI540_PROJECT_ROOT", "").strip()
    current = Path.cwd().resolve()
    candidates = ([Path(configured).expanduser().resolve()] if configured else [])
    candidates.extend([current, *current.parents])
    studio_home = Path("/home/sagemaker-user")
    if studio_home.exists():
        candidates.extend(
            marker.parent.parent
            for marker in studio_home.glob("**/scripts/validate_project.py")
        )
    checked = []
    for candidate in candidates:
        if candidate in checked:
            continue
        checked.append(candidate)
        if (
            (candidate / "src" / "train.py").is_file()
            and (candidate / "scripts" / "validate_project.py").is_file()
            and (candidate / "app" / "flask_app.py").is_file()
            and (candidate / "inference_code" / "retrieval.py").is_file()
            and (candidate / "tests" / "test_train.py").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate the AAI 540 project root. Upload the complete project "
        "folder (including src/, scripts/, tests/, app/, and inference_code/) to "
        "SageMaker, open the notebook from that folder, or set AAI540_PROJECT_ROOT."
    )

project_root = locate_project_root()
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print("Project root:", project_root)

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from IPython.display import display
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 160)
"""
    ),
    markdown("## 1 Configuration and AWS preflight"),
    code(
        r"""
# ---------- User-editable settings ----------
PROJECT_NAME = "product-qa"
AWS_REGION_OVERRIDE = ""
S3_BUCKET_OVERRIDE = ""
EXECUTION_ROLE_ARN_OVERRIDE = ""

ORCHESTRATION_MODE = "pipeline"         # "pipeline" or the Learner Lab fallback "scripted".
EXECUTE_CONTINUAL_RUN = True
REQUIRE_MONITORING_TRIGGER = False       # False permits an intentional course demonstration run.
ENABLE_MODEL_REGISTRY = True             # False writes a PendingManualApproval record to S3.

PROCESSING_INSTANCE_TYPE = "ml.m5.large"
TRAIN_INSTANCE_TYPE = "ml.m5.large"
SKLEARN_FRAMEWORK_VERSION = "1.4-2"
RANDOM_SEED = 540
MINIMUM_FEEDBACK_ROWS = 100
FEEDBACK_VALIDATION_FRACTION = 0.20
MINIMUM_MACRO_F1 = 0.40
MACRO_F1_REGRESSION_TOLERANCE = 0.02
NDCG_REGRESSION_TOLERANCE = 0.02
MODEL_PACKAGE_GROUP_NAME = "product-qa-ranker"
# -------------------------------------------

if ORCHESTRATION_MODE not in {"pipeline", "scripted"}:
    raise ValueError("ORCHESTRATION_MODE must be 'pipeline' or 'scripted'")

session = boto3.Session(region_name=AWS_REGION_OVERRIDE or None)
region = session.region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
if not region:
    raise RuntimeError("No AWS Region is configured. Set AWS_REGION_OVERRIDE.")

retry_config = Config(retries={"max_attempts": 10, "mode": "adaptive"})
s3 = session.client("s3", config=retry_config)
sm = session.client("sagemaker", config=retry_config)
sts = session.client("sts")
identity = sts.get_caller_identity()
account_id = identity["Account"]
caller_arn = identity["Arn"]

def infer_execution_role_arn(caller_identity_arn, account):
    if ":assumed-role/" in caller_identity_arn:
        role_part = caller_identity_arn.split(":assumed-role/", 1)[1].rsplit("/", 1)[0]
        return f"arn:aws:iam::{account}:role/{role_part}"
    if ":role/" in caller_identity_arn:
        return caller_identity_arn
    return ""

execution_role_arn = EXECUTION_ROLE_ARN_OVERRIDE or infer_execution_role_arn(caller_arn, account_id)
if not execution_role_arn:
    raise RuntimeError("Could not infer the SageMaker execution role. Set EXECUTION_ROLE_ARN_OVERRIDE.")

bucket = S3_BUCKET_OVERRIDE or f"sagemaker-{region}-{account_id}"
continual_started_at = datetime.now(timezone.utc)
continual_run_id = continual_started_at.strftime("%Y%m%d_%H%M%S")
continual_prefix = f"aai540/{PROJECT_NAME}/continual-learning/{continual_run_id}"
pipeline_artifact_root = f"s3://{bucket}/{continual_prefix}/pipeline-artifacts"
pipeline_name = f"aai540-{PROJECT_NAME}-continual-learning"
local_dir = Path("/tmp") / f"aai540_continual_{continual_run_id}"
local_dir.mkdir(parents=True, exist_ok=True)

print(json.dumps({
    "region": region,
    "caller_arn": caller_arn,
    "execution_role_arn": execution_role_arn,
    "bucket": bucket,
    "continual_run_id": continual_run_id,
    "orchestration_mode": ORCHESTRATION_MODE,
    "pipeline_name": pipeline_name,
    "model_package_group": MODEL_PACKAGE_GROUP_NAME if ENABLE_MODEL_REGISTRY else None,
}, indent=2))
"""
    ),
    markdown("## 2 Resolve the monitoring, modeling, and data lineage"),
    code(
        r"""
def split_s3_uri(uri):
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an S3 URI, got {uri}")
    bucket_name, _, key = uri[5:].partition("/")
    return bucket_name, key

def read_s3_json(uri):
    source_bucket, source_key = split_s3_uri(uri)
    return json.loads(s3.get_object(Bucket=source_bucket, Key=source_key)["Body"].read())

def put_s3_json(uri, payload):
    target_bucket, target_key = split_s3_uri(uri)
    s3.put_object(
        Bucket=target_bucket,
        Key=target_key,
        Body=json.dumps(payload, indent=2, allow_nan=False).encode("utf-8"),
        ContentType="application/json",
    )

monitoring_receipt_uri = f"s3://{bucket}/aai540/{PROJECT_NAME}/monitoring/latest.json"
monitoring_receipt = read_s3_json(monitoring_receipt_uri)
modeling_receipt = read_s3_json(monitoring_receipt["modeling_receipt"])
run_summary = read_s3_json(modeling_receipt["source_run_summary"])

if monitoring_receipt["model_run_id"] != modeling_receipt["model_run_id"]:
    raise ValueError("Monitoring and modeling receipts refer to different model runs")
required_splits = {"train", "validation", "test", "production"}
if set(run_summary.get("split_data", {})) != required_splits:
    raise ValueError(f"Expected split_data {sorted(required_splits)}")

monitoring_trigger = bool(monitoring_receipt["decision"]["retraining_recommended"])
execute_this_run = EXECUTE_CONTINUAL_RUN and (
    monitoring_trigger or not REQUIRE_MONITORING_TRIGGER
)
run_reason = (
    "monitoring threshold breached"
    if monitoring_trigger
    else "manual course demonstration override"
    if execute_this_run
    else "no monitoring trigger; candidate creation skipped"
)

train_uri = run_summary["split_data"]["train"]
validation_uri = run_summary["split_data"]["validation"]
test_uri = run_summary["split_data"]["test"]
feedback_uri = run_summary["split_data"]["production"]
champion_model_uri = modeling_receipt["candidate"]["model_artifact"]

lineage = {
    "monitoring_receipt": monitoring_receipt_uri,
    "modeling_receipt": monitoring_receipt["modeling_receipt"],
    "source_run_summary": modeling_receipt["source_run_summary"],
    "source_data_run_id": run_summary["run_id"],
    "champion_model_run_id": modeling_receipt["model_run_id"],
    "monitoring_trigger": monitoring_trigger,
    "execute_this_run": execute_this_run,
    "run_reason": run_reason,
    "inputs": {
        "train": train_uri,
        "validation": validation_uri,
        "test": test_uri,
        "new_labeled_feedback": feedback_uri,
        "champion_model": champion_model_uri,
    },
}
print(json.dumps(lineage, indent=2))
"""
    ),
    markdown("## 3 CI preflight: source, unit tests, and immutable contracts"),
    code(
        r"""
source_dir = project_root / "src"
required_source_files = [
    source_dir / "train.py",
    source_dir / "prepare_continual_data.py",
    source_dir / "evaluate_candidate.py",
    source_dir / "record_promotion.py",
    project_root / "app" / "__init__.py",
    project_root / "app" / "flask_app.py",
    project_root / "inference_code" / "__init__.py",
    project_root / "inference_code" / "inference.py",
    project_root / "inference_code" / "retrieval.py",
    project_root / "tests" / "test_flask_app.py",
    project_root / "tests" / "test_retrieval.py",
]
missing_source_files = [str(path) for path in required_source_files if not path.exists()]
if missing_source_files:
    raise FileNotFoundError(
        f"Run this notebook from the project root. Missing: {missing_source_files}"
    )

# The full local suite is also run before delivery. This in-notebook command is the CI entry gate.
import subprocess
import sys

test_environment = os.environ.copy()
existing_pythonpath = test_environment.get("PYTHONPATH", "")
test_environment["PYTHONPATH"] = (
    str(project_root)
    if not existing_pythonpath
    else f"{project_root}{os.pathsep}{existing_pythonpath}"
)
test_result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q"],
    cwd=project_root,
    env=test_environment,
    text=True,
    capture_output=True,
)
print(test_result.stdout)
if test_result.returncode != 0:
    print(test_result.stderr)
    raise RuntimeError("Local CI tests failed; no AWS jobs will be submitted")
print("CI preflight passed.")
"""
    ),
    markdown("## 4 Define the SageMaker Pipeline DAG"),
    code(
        r"""
pipeline_session = PipelineSession(boto_session=session, default_bucket=bucket)

if ENABLE_MODEL_REGISTRY:
    try:
        sm.describe_model_package_group(ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME)
        print(f"Using Model Package Group: {MODEL_PACKAGE_GROUP_NAME}")
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ValidationException":
            raise
        sm.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelPackageGroupDescription="AAI 540 Product QA ranker candidates",
            Tags=[
                {"Key": "project", "Value": PROJECT_NAME},
                {"Key": "course", "Value": "AAI-540"},
            ],
        )
        print(f"Created Model Package Group: {MODEL_PACKAGE_GROUP_NAME}")

train_param = ParameterString(name="BaselineTrainS3Uri", default_value=train_uri)
validation_param = ParameterString(name="BaselineValidationS3Uri", default_value=validation_uri)
feedback_param = ParameterString(name="LabeledFeedbackS3Uri", default_value=feedback_uri)
test_param = ParameterString(name="TestS3Uri", default_value=test_uri)
champion_param = ParameterString(name="ChampionModelS3Uri", default_value=champion_model_uri)
# Processing-job arguments are serialized as strings in the pipeline request.
# Declare CLI-bound values as String parameters and let argparse perform the
# int/float conversion inside the processing scripts.
minimum_feedback_param = ParameterString(
    name="MinimumFeedbackRows", default_value=str(MINIMUM_FEEDBACK_ROWS)
)
minimum_macro_f1_param = ParameterString(
    name="MinimumMacroF1", default_value=str(MINIMUM_MACRO_F1)
)

execution_root = Join(
    on="/",
    values=[pipeline_artifact_root, ExecutionVariables.PIPELINE_EXECUTION_ID],
)

prepare_processor = SKLearnProcessor(
    framework_version=SKLEARN_FRAMEWORK_VERSION,
    role=execution_role_arn,
    instance_count=1,
    instance_type=PROCESSING_INSTANCE_TYPE,
    base_job_name=f"aai540-{PROJECT_NAME}-prepare-feedback",
    sagemaker_session=pipeline_session,
)
prepare_args = prepare_processor.run(
    code=str(source_dir / "prepare_continual_data.py"),
    inputs=[
        ProcessingInput(source=train_param, destination="/opt/ml/processing/input/train"),
        ProcessingInput(source=validation_param, destination="/opt/ml/processing/input/validation"),
        ProcessingInput(source=feedback_param, destination="/opt/ml/processing/input/feedback"),
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
            destination=Join(on="/", values=[execution_root, "prepared", "train"]),
        ),
        ProcessingOutput(
            output_name="validation",
            source="/opt/ml/processing/output/validation",
            destination=Join(on="/", values=[execution_root, "prepared", "validation"]),
        ),
        ProcessingOutput(
            output_name="data-receipt",
            source="/opt/ml/processing/output/receipt",
            destination=Join(on="/", values=[execution_root, "prepared", "receipt"]),
        ),
    ],
    arguments=[
        "--train", "/opt/ml/processing/input/train",
        "--validation", "/opt/ml/processing/input/validation",
        "--feedback", "/opt/ml/processing/input/feedback",
        "--output-dir", "/opt/ml/processing/output",
        "--feedback-validation-fraction", str(FEEDBACK_VALIDATION_FRACTION),
        "--minimum-feedback-rows", minimum_feedback_param,
        "--random-seed", str(RANDOM_SEED),
    ],
)
step_prepare = ProcessingStep(name="PrepareContinualData", step_args=prepare_args)

candidate_estimator = SKLearn(
    entry_point="train.py",
    source_dir=str(source_dir),
    role=execution_role_arn,
    instance_count=1,
    instance_type=TRAIN_INSTANCE_TYPE,
    framework_version=SKLEARN_FRAMEWORK_VERSION,
    py_version="py3",
    hyperparameters={
        "model-type": "candidate",
        "random-state": RANDOM_SEED,
        "n-estimators": 300,
        "max-depth": 12,
        "min-samples-leaf": 2,
    },
    output_path=Join(on="/", values=[execution_root, "training"]),
    base_job_name=f"aai540-{PROJECT_NAME}-continual-candidate",
    sagemaker_session=pipeline_session,
)
train_args = candidate_estimator.fit(inputs={
    "train": TrainingInput(
        s3_data=step_prepare.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        content_type="text/csv",
    ),
    "validation": TrainingInput(
        s3_data=step_prepare.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
        content_type="text/csv",
    ),
})
step_train = TrainingStep(name="TrainContinualCandidate", step_args=train_args)

evaluation_report = PropertyFile(
    name="CandidateEvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)
evaluation_processor = SKLearnProcessor(
    framework_version=SKLEARN_FRAMEWORK_VERSION,
    role=execution_role_arn,
    instance_count=1,
    instance_type=PROCESSING_INSTANCE_TYPE,
    base_job_name=f"aai540-{PROJECT_NAME}-evaluate-candidate",
    sagemaker_session=pipeline_session,
)
evaluation_args = evaluation_processor.run(
    code=str(source_dir / "evaluate_candidate.py"),
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/input/candidate",
        ),
        ProcessingInput(source=champion_param, destination="/opt/ml/processing/input/champion"),
        ProcessingInput(source=test_param, destination="/opt/ml/processing/input/test"),
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/output/evaluation",
            destination=Join(on="/", values=[execution_root, "evaluation"]),
        )
    ],
    arguments=[
        "--candidate-model", "/opt/ml/processing/input/candidate",
        "--champion-model", "/opt/ml/processing/input/champion",
        "--test-data", "/opt/ml/processing/input/test",
        "--output-dir", "/opt/ml/processing/output/evaluation",
        "--minimum-macro-f1", minimum_macro_f1_param,
        "--macro-f1-regression-tolerance", str(MACRO_F1_REGRESSION_TOLERANCE),
        "--ndcg-regression-tolerance", str(NDCG_REGRESSION_TOLERANCE),
    ],
)
step_evaluate = ProcessingStep(
    name="EvaluateAgainstChampion",
    step_args=evaluation_args,
    property_files=[evaluation_report],
)

candidate_model = SKLearnModel(
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    role=execution_role_arn,
    entry_point="train.py",
    source_dir=str(source_dir),
    framework_version=SKLEARN_FRAMEWORK_VERSION,
    py_version="py3",
    sagemaker_session=pipeline_session,
)
evaluation_uri = Join(
    on="/",
    values=[
        step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
        "evaluation.json",
    ],
)

if ENABLE_MODEL_REGISTRY:
    step_promote_candidate = RegisterModel(
        name="RegisterPendingCandidate",
        model=candidate_model,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status="PendingManualApproval",
        model_metrics=ModelMetrics(
            model_statistics=MetricsSource(s3_uri=evaluation_uri, content_type="application/json")
        ),
    )
else:
    promotion_processor = SKLearnProcessor(
        framework_version=SKLEARN_FRAMEWORK_VERSION,
        role=execution_role_arn,
        instance_count=1,
        instance_type=PROCESSING_INSTANCE_TYPE,
        base_job_name=f"aai540-{PROJECT_NAME}-record-promotion",
        sagemaker_session=pipeline_session,
    )
    promotion_args = promotion_processor.run(
        code=str(source_dir / "record_promotion.py"),
        inputs=[ProcessingInput(
            source=step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
            destination="/opt/ml/processing/input/evaluation",
        )],
        outputs=[ProcessingOutput(
            output_name="promotion-candidate",
            source="/opt/ml/processing/output/promotion",
            destination=Join(on="/", values=[execution_root, "promotion"]),
        )],
        arguments=[
            "--evaluation", "/opt/ml/processing/input/evaluation",
            "--output-dir", "/opt/ml/processing/output/promotion",
        ],
    )
    step_promote_candidate = ProcessingStep(name="RecordPendingCandidate", step_args=promotion_args)

step_gate_failed = FailStep(
    name="CandidateQualityGateFailed",
    error_message="Candidate failed the macro-F1/NDCG release gate; champion is unchanged.",
)
step_quality_gate = ConditionStep(
    name="CandidateQualityGate",
    conditions=[ConditionEquals(
        left=JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="gate.passed",
        ),
        right=1,
    )],
    if_steps=[step_promote_candidate],
    else_steps=[step_gate_failed],
)

pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        train_param,
        validation_param,
        feedback_param,
        test_param,
        champion_param,
        minimum_feedback_param,
        minimum_macro_f1_param,
    ],
    steps=[step_prepare, step_train, step_evaluate, step_quality_gate],
    sagemaker_session=pipeline_session,
)

pipeline_definition = json.loads(pipeline.definition())
display(pd.DataFrame([
    {"step": item["Name"], "type": item["Type"]}
    for item in pipeline_definition["Steps"]
]))
print("Pipeline definition is valid JSON with", len(pipeline_definition["Steps"]), "top-level steps.")
"""
    ),
    markdown("## 5 Upsert and execute the DAG, or use the scripted fallback"),
    code(
        r"""
pipeline_execution_arn = None
pipeline_status = "Skipped"
registered_model_package_arn = None
evaluation_output_uri = None
candidate_model_artifact_uri = None
execution_steps = []

if ORCHESTRATION_MODE == "pipeline":
    upsert_response = pipeline.upsert(
        role_arn=execution_role_arn,
        description="AAI 540 Product QA continual-learning pipeline with review gate.",
        tags=[
            {"Key": "project", "Value": PROJECT_NAME},
            {"Key": "course", "Value": "AAI-540"},
        ],
    )
    print("Pipeline ARN:", upsert_response["PipelineArn"])
    if execute_this_run:
        execution = pipeline.start(parameters={
            "BaselineTrainS3Uri": train_uri,
            "BaselineValidationS3Uri": validation_uri,
            "LabeledFeedbackS3Uri": feedback_uri,
            "TestS3Uri": test_uri,
            "ChampionModelS3Uri": champion_model_uri,
            "MinimumFeedbackRows": str(MINIMUM_FEEDBACK_ROWS),
            "MinimumMacroF1": str(MINIMUM_MACRO_F1),
        })
        pipeline_execution_arn = execution.arn
        while True:
            description = sm.describe_pipeline_execution(
                PipelineExecutionArn=pipeline_execution_arn
            )
            pipeline_status = description["PipelineExecutionStatus"]
            print(f"Pipeline status: {pipeline_status}", end="\r")
            if pipeline_status in {"Succeeded", "Failed", "Stopped"}:
                break
            time.sleep(30)
        print(f"\nPipeline finished with status: {pipeline_status}")

        execution_steps = sm.list_pipeline_execution_steps(
            PipelineExecutionArn=pipeline_execution_arn,
            MaxResults=100,
        )["PipelineExecutionSteps"]
        display(pd.DataFrame([
            {
                "step": item["StepName"],
                "status": item["StepStatus"],
                "failure_reason": item.get("FailureReason"),
            }
            for item in execution_steps
        ]))
        for item in execution_steps:
            metadata = item.get("Metadata", {})
            if item["StepName"] == "EvaluateAgainstChampion" and "ProcessingJob" in metadata:
                job_name = metadata["ProcessingJob"]["Arn"].rsplit("/", 1)[-1]
                evaluation_job = sm.describe_processing_job(ProcessingJobName=job_name)
                evaluation_output_uri = next(
                    output["S3Output"]["S3Uri"]
                    for output in evaluation_job["ProcessingOutputConfig"]["Outputs"]
                    if output["OutputName"] == "evaluation"
                ) + "/evaluation.json"
            if item["StepName"] == "TrainContinualCandidate" and "TrainingJob" in metadata:
                job_name = metadata["TrainingJob"]["Arn"].rsplit("/", 1)[-1]
                candidate_model_artifact_uri = sm.describe_training_job(
                    TrainingJobName=job_name
                )["ModelArtifacts"]["S3ModelArtifacts"]
            register_metadata = metadata.get("RegisterModel", {})
            if register_metadata.get("Arn"):
                registered_model_package_arn = register_metadata["Arn"]
else:
    if execute_this_run:
        scripted_root = f"s3://{bucket}/{continual_prefix}/scripted"
        scripted_prepare = SKLearnProcessor(
            framework_version=SKLEARN_FRAMEWORK_VERSION,
            role=execution_role_arn,
            instance_count=1,
            instance_type=PROCESSING_INSTANCE_TYPE,
            base_job_name=f"aai540-{PROJECT_NAME}-prepare-feedback",
        )
        scripted_prepare.run(
            code=str(source_dir / "prepare_continual_data.py"),
            inputs=[
                ProcessingInput(source=train_uri, destination="/opt/ml/processing/input/train"),
                ProcessingInput(source=validation_uri, destination="/opt/ml/processing/input/validation"),
                ProcessingInput(source=feedback_uri, destination="/opt/ml/processing/input/feedback"),
            ],
            outputs=[
                ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train", destination=f"{scripted_root}/prepared/train"),
                ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation", destination=f"{scripted_root}/prepared/validation"),
                ProcessingOutput(output_name="data-receipt", source="/opt/ml/processing/output/receipt", destination=f"{scripted_root}/prepared/receipt"),
            ],
            arguments=[
                "--train", "/opt/ml/processing/input/train",
                "--validation", "/opt/ml/processing/input/validation",
                "--feedback", "/opt/ml/processing/input/feedback",
                "--output-dir", "/opt/ml/processing/output",
                "--feedback-validation-fraction", str(FEEDBACK_VALIDATION_FRACTION),
                "--minimum-feedback-rows", str(MINIMUM_FEEDBACK_ROWS),
                "--random-seed", str(RANDOM_SEED),
            ],
            wait=True,
            logs=True,
        )
        scripted_estimator = SKLearn(
            entry_point="train.py",
            source_dir=str(source_dir),
            role=execution_role_arn,
            instance_count=1,
            instance_type=TRAIN_INSTANCE_TYPE,
            framework_version=SKLEARN_FRAMEWORK_VERSION,
            py_version="py3",
            hyperparameters={
                "model-type": "candidate", "random-state": RANDOM_SEED,
                "n-estimators": 300, "max-depth": 12, "min-samples-leaf": 2,
            },
            output_path=f"{scripted_root}/training",
            base_job_name=f"aai540-{PROJECT_NAME}-continual-candidate",
        )
        scripted_estimator.fit({
            "train": TrainingInput(f"{scripted_root}/prepared/train", content_type="text/csv"),
            "validation": TrainingInput(f"{scripted_root}/prepared/validation", content_type="text/csv"),
        }, wait=True, logs=True)
        candidate_model_artifact_uri = scripted_estimator.model_data

        scripted_evaluator = SKLearnProcessor(
            framework_version=SKLEARN_FRAMEWORK_VERSION,
            role=execution_role_arn,
            instance_count=1,
            instance_type=PROCESSING_INSTANCE_TYPE,
            base_job_name=f"aai540-{PROJECT_NAME}-evaluate-candidate",
        )
        scripted_evaluator.run(
            code=str(source_dir / "evaluate_candidate.py"),
            inputs=[
                ProcessingInput(source=candidate_model_artifact_uri, destination="/opt/ml/processing/input/candidate"),
                ProcessingInput(source=champion_model_uri, destination="/opt/ml/processing/input/champion"),
                ProcessingInput(source=test_uri, destination="/opt/ml/processing/input/test"),
            ],
            outputs=[ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output/evaluation",
                destination=f"{scripted_root}/evaluation",
            )],
            arguments=[
                "--candidate-model", "/opt/ml/processing/input/candidate",
                "--champion-model", "/opt/ml/processing/input/champion",
                "--test-data", "/opt/ml/processing/input/test",
                "--output-dir", "/opt/ml/processing/output/evaluation",
                "--minimum-macro-f1", str(MINIMUM_MACRO_F1),
                "--macro-f1-regression-tolerance", str(MACRO_F1_REGRESSION_TOLERANCE),
                "--ndcg-regression-tolerance", str(NDCG_REGRESSION_TOLERANCE),
            ],
            wait=True,
            logs=True,
        )
        evaluation_output_uri = f"{scripted_root}/evaluation/evaluation.json"
        pipeline_status = "ScriptedCompleted"
"""
    ),
    markdown("## 6 Read the release evidence and perform optional fallback registration"),
    code(
        r"""
evaluation_report_payload = None
quality_gate_passed = False
if evaluation_output_uri:
    evaluation_report_payload = read_s3_json(evaluation_output_uri)
    quality_gate_passed = int(evaluation_report_payload["gate"]["passed"]) == 1
    display(pd.DataFrame([
        {"model": "champion", **evaluation_report_payload["champion"]},
        {"model": "candidate", **evaluation_report_payload["candidate"]},
    ]).set_index("model").style.format("{:.4f}"))
    print(json.dumps({
        "comparison": evaluation_report_payload["comparison"],
        "gate": evaluation_report_payload["gate"],
    }, indent=2))

if (
    ORCHESTRATION_MODE == "pipeline"
    and ENABLE_MODEL_REGISTRY
    and quality_gate_passed
    and not registered_model_package_arn
):
    # Pipeline step metadata varies slightly across SDK/service revisions. Resolve
    # the version by group and this run's start time when the ARN is not returned.
    versions = sm.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
        ModelPackageType="Versioned",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )["ModelPackageSummaryList"]
    matching_versions = [
        item for item in versions
        if item["CreationTime"] >= continual_started_at
    ]
    if len(matching_versions) != 1:
        raise RuntimeError(
            f"Expected one Model Package created by this run, found {len(matching_versions)}"
        )
    registered_model_package_arn = matching_versions[0]["ModelPackageArn"]

if ORCHESTRATION_MODE == "scripted" and execute_this_run and quality_gate_passed:
    if ENABLE_MODEL_REGISTRY:
        try:
            sm.describe_model_package_group(ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ValidationException":
                sm.create_model_package_group(
                    ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
                    ModelPackageGroupDescription="AAI 540 Product QA ranker candidates",
                    Tags=[{"Key": "project", "Value": PROJECT_NAME}, {"Key": "course", "Value": "AAI-540"}],
                )
            else:
                raise
        registered = scripted_estimator.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
            approval_status="PendingManualApproval",
            model_metrics=ModelMetrics(
                model_statistics=MetricsSource(
                    s3_uri=evaluation_output_uri,
                    content_type="application/json",
                )
            ),
        )
        registered_model_package_arn = registered.model_package_arn
    else:
        promotion_uri = f"s3://{bucket}/{continual_prefix}/scripted/promotion/promotion_candidate.json"
        put_s3_json(promotion_uri, {
            "schema_version": "aai540-promotion-candidate-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "approval_status": "PendingManualApproval",
            "candidate_model_artifact": candidate_model_artifact_uri,
            "evaluation": evaluation_report_payload,
        })
        print("Model Registry disabled; wrote", promotion_uri)

if execute_this_run and evaluation_output_uri is None:
    raise RuntimeError("Continual run started but no evaluation output could be resolved")
"""
    ),
    markdown("## 7 Persist the CI/CD receipt"),
    code(
        r"""
cicd_receipt = {
    "schema_version": "aai540-continual-learning-receipt-v1",
    "project": PROJECT_NAME,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "continual_run_id": continual_run_id,
    "lineage": lineage,
    "orchestration": {
        "mode": ORCHESTRATION_MODE,
        "pipeline_name": pipeline_name if ORCHESTRATION_MODE == "pipeline" else None,
        "pipeline_execution_arn": pipeline_execution_arn,
        "status": pipeline_status,
    },
    "candidate": {
        "model_artifact": candidate_model_artifact_uri,
        "evaluation": evaluation_output_uri,
        "quality_gate_passed": quality_gate_passed,
        "model_package_arn": registered_model_package_arn,
        "approval_status": "PendingManualApproval" if quality_gate_passed else "NotEligible",
    },
    "champion": {
        "model_run_id": modeling_receipt["model_run_id"],
        "model_artifact": champion_model_uri,
        "unchanged": True,
    },
    "thresholds": {
        "minimum_macro_f1": MINIMUM_MACRO_F1,
        "macro_f1_regression_tolerance": MACRO_F1_REGRESSION_TOLERANCE,
        "ndcg_regression_tolerance": NDCG_REGRESSION_TOLERANCE,
    },
}
cicd_receipt_uri = f"s3://{bucket}/{continual_prefix}/continual_learning_receipt.json"
put_s3_json(cicd_receipt_uri, cicd_receipt)
put_s3_json(f"s3://{bucket}/aai540/{PROJECT_NAME}/continual-learning/latest.json", cicd_receipt)

print(json.dumps(cicd_receipt, indent=2))
print("CI/CD receipt:", cicd_receipt_uri)
if quality_gate_passed:
    print("Candidate passed, but the champion is unchanged until a reviewer explicitly approves it.")
else:
    print("Candidate was not eligible for approval; the champion is unchanged.")
"""
    ),
    markdown(
        r"""
## 8 Explicit second-person approval (optional)

Leave this disabled during the automated run. After reviewing the evaluation report, monitoring evidence, lineage, cost, and rollback plan, a second team member may paste the candidate Model Package ARN, their name, and review note, then set `APPROVE_MODEL=True`. This changes only the registry approval state; it does not create an endpoint.
"""
    ),
    code(
        r"""
APPROVE_MODEL = False
MODEL_PACKAGE_ARN_TO_APPROVE = registered_model_package_arn or ""
APPROVER_NAME = ""
APPROVAL_NOTE = ""

if APPROVE_MODEL:
    if not quality_gate_passed:
        raise ValueError("The quality gate did not pass")
    if not ENABLE_MODEL_REGISTRY or not MODEL_PACKAGE_ARN_TO_APPROVE:
        raise ValueError("Model Registry and a Model Package ARN are required")
    if not APPROVER_NAME.strip() or not APPROVAL_NOTE.strip():
        raise ValueError("Record the second reviewer name and approval note")
    approved_at_utc = datetime.now(timezone.utc).isoformat()
    sm.update_model_package(
        ModelPackageArn=MODEL_PACKAGE_ARN_TO_APPROVE,
        ModelApprovalStatus="Approved",
        ApprovalDescription=(
            f"Approved by {APPROVER_NAME.strip()} at {approved_at_utc}: "
            f"{APPROVAL_NOTE.strip()}"
        )[:1024],
    )
    approved = sm.describe_model_package(ModelPackageName=MODEL_PACKAGE_ARN_TO_APPROVE)
    assert approved["ModelApprovalStatus"] == "Approved"
    approval_receipt_uri = f"s3://{bucket}/{continual_prefix}/approval_receipt.json"
    put_s3_json(approval_receipt_uri, {
        "schema_version": "aai540-model-approval-v1",
        "created_at_utc": approved_at_utc,
        "continual_learning_receipt": cicd_receipt_uri,
        "model_package_arn": MODEL_PACKAGE_ARN_TO_APPROVE,
        "model_artifact": candidate_model_artifact_uri,
        "approved_by": APPROVER_NAME.strip(),
        "approval_note": APPROVAL_NOTE.strip(),
        "approval_status": "Approved",
    })
    cicd_receipt["candidate"].update({
        "approval_status": "Approved",
        "approved_at_utc": approved_at_utc,
        "approved_by": APPROVER_NAME.strip(),
        "approval_note": APPROVAL_NOTE.strip(),
        "approval_receipt": approval_receipt_uri,
    })
    put_s3_json(
        f"s3://{bucket}/aai540/{PROJECT_NAME}/continual-learning/latest.json",
        cicd_receipt,
    )
    print("Approved model package:", MODEL_PACKAGE_ARN_TO_APPROVE)
    print("Immutable approval receipt:", approval_receipt_uri)
else:
    print("No approval performed. Candidate remains pending and champion remains unchanged.")
"""
    ),
    markdown(
        r"""
## 9 Rollback and cleanup controls

For this batch-first MVP, rollback means rejecting the newly approved Model Package and restoring the previous approved package for future Batch Transform runs. It does not require endpoint mutation. Supply both exact ARNs and enable the switch only after review.
"""
    ),
    code(
        r"""
ROLLBACK_REGISTRY_APPROVAL = False
NEW_MODEL_PACKAGE_ARN_TO_REJECT = ""
PREVIOUS_MODEL_PACKAGE_ARN_TO_RESTORE = ""

if ROLLBACK_REGISTRY_APPROVAL:
    if not NEW_MODEL_PACKAGE_ARN_TO_REJECT or not PREVIOUS_MODEL_PACKAGE_ARN_TO_RESTORE:
        raise ValueError("Both exact Model Package ARNs are required for rollback")
    sm.update_model_package(
        ModelPackageArn=NEW_MODEL_PACKAGE_ARN_TO_REJECT,
        ModelApprovalStatus="Rejected",
        ApprovalDescription="Rejected by reviewed AAI 540 rollback procedure.",
    )
    sm.update_model_package(
        ModelPackageArn=PREVIOUS_MODEL_PACKAGE_ARN_TO_RESTORE,
        ModelApprovalStatus="Approved",
        ApprovalDescription="Restored by reviewed AAI 540 rollback procedure.",
    )
    rollback_at_utc = datetime.now(timezone.utc).isoformat()
    rollback_receipt_uri = f"s3://{bucket}/{continual_prefix}/rollback_receipt.json"
    put_s3_json(rollback_receipt_uri, {
        "schema_version": "aai540-model-approval-rollback-v1",
        "created_at_utc": rollback_at_utc,
        "continual_learning_receipt": cicd_receipt_uri,
        "rejected_model_package_arn": NEW_MODEL_PACKAGE_ARN_TO_REJECT,
        "restored_model_package_arn": PREVIOUS_MODEL_PACKAGE_ARN_TO_RESTORE,
        "status": "Completed",
    })
    cicd_receipt["candidate"].update({
        "approval_status": "RejectedAfterRollback",
        "rollback_at_utc": rollback_at_utc,
        "restored_model_package_arn": PREVIOUS_MODEL_PACKAGE_ARN_TO_RESTORE,
        "rollback_receipt": rollback_receipt_uri,
    })
    put_s3_json(
        f"s3://{bucket}/aai540/{PROJECT_NAME}/continual-learning/latest.json",
        cicd_receipt,
    )
    print("Registry approval rollback completed. No endpoint resources were created.")
else:
    print("No rollback performed. Processing and training jobs have already terminated.")
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} cells")
