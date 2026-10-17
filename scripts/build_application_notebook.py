"""Build the temporary endpoint and application integration notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "AAI540_SageMaker_Application_Integration_Testing.ipynb"
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
# AAI 540 application integration, endpoint testing, and release evidence

Stage 6 packages the retrieval runtime behind SageMaker's HTTPS inference contract, optionally deploys a temporary real-time endpoint, exercises it through a small Flask API, measures latency and failure behavior, writes sanitized request telemetry, and deletes billable serving resources after the test.

The endpoint returns an extractive answer plus evidence or an explicit abstention. It never produces unsupported free-form text.
"""
    ),
    markdown(
        r"""
## Prerequisites and safety

Run stages 1-5 first. Endpoint deployment requires SageMaker model/endpoint permissions, ECR image pulls, S3 access, CloudWatch, and iam:PassRole. The model artifact already contains the MiniLM weights, so endpoint startup does not download model weights. Python runtime dependencies are installed from inference_code/requirements.txt during container startup.

Keep DELETE_ENDPOINT_AFTER_TESTS=True. A local in-process fallback is available when the Learner Lab role cannot create endpoints.
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
    "sentence-transformers>=3.0,<4" \
    "faiss-cpu>=1.8,<2" \
    "torch>=2.1,<3" \
    "flask>=3,<4"
"""
    ),
    markdown("> Restart the kernel only if an already-loaded dependency could not be upgraded, then continue from the import cell."),
    code(
        r"""
import hashlib
import json
import os
import sys
import tarfile
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
import numpy as np
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from IPython.display import display
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.sklearn.model import SKLearnModel

from app.flask_app import create_app
from inference_code.retrieval import RetrievalQAEngine

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

DEPLOY_TEMPORARY_ENDPOINT = True
ENDPOINT_INSTANCE_TYPE = "ml.m5.large"
CONTAINER_STARTUP_TIMEOUT_SECONDS = 1200
MODEL_DOWNLOAD_TIMEOUT_SECONDS = 900
TEST_REQUEST_COUNT = 20
P95_LATENCY_LIMIT_MS = 2500
DELETE_ENDPOINT_AFTER_TESTS = True
PUBLISH_CLOUDWATCH_METRICS = True
CREATE_INTEGRATION_MONITORING = True
# -------------------------------------------

session = boto3.Session(region_name=AWS_REGION_OVERRIDE or None)
region = session.region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
if not region:
    raise RuntimeError("No AWS Region is configured. Set AWS_REGION_OVERRIDE.")
retry_config = Config(retries={"max_attempts": 10, "mode": "adaptive"})
s3 = session.client("s3", config=retry_config)
sm = session.client("sagemaker", config=retry_config)
cloudwatch = session.client("cloudwatch", config=retry_config)
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
    raise RuntimeError("Could not infer the SageMaker execution role")

bucket = S3_BUCKET_OVERRIDE or f"sagemaker-{region}-{account_id}"
integration_run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
integration_prefix = f"aai540/{PROJECT_NAME}/integration/{integration_run_id}"
endpoint_name = f"aai540-{PROJECT_NAME}-{integration_run_id.replace('_', '-')}"[:63]
local_dir = Path("/tmp") / f"aai540_integration_{integration_run_id}"
local_dir.mkdir(parents=True, exist_ok=True)

print(json.dumps({
    "region": region,
    "caller_arn": caller_arn,
    "execution_role_arn": execution_role_arn,
    "bucket": bucket,
    "integration_run_id": integration_run_id,
    "endpoint_name": endpoint_name if DEPLOY_TEMPORARY_ENDPOINT else None,
}, indent=2))
"""
    ),
    markdown("## 2 Resolve and validate the retrieval runtime"),
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

def download_s3_file(uri, destination):
    source_bucket, source_key = split_s3_uri(uri)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(source_bucket, source_key, str(destination))
    return destination

retrieval_receipt_uri = f"s3://{bucket}/aai540/{PROJECT_NAME}/retrieval/latest.json"
retrieval_receipt = read_s3_json(retrieval_receipt_uri)
if not retrieval_receipt.get("evaluation", {}).get("passed"):
    raise ValueError("The latest retrieval runtime did not pass its known-query gate")
runtime_artifact_uri = retrieval_receipt["runtime"]["artifact"]

source_run_summary = read_s3_json(retrieval_receipt["lineage"]["source_run_summary"])
curated_bucket, curated_prefix = split_s3_uri(source_run_summary["curated_data"])
curated_objects = []
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=curated_bucket, Prefix=curated_prefix):
    curated_objects.extend(
        item for item in page.get("Contents", [])
        if item["Key"].endswith(".parquet")
    )
if not curated_objects:
    raise FileNotFoundError("No curated Parquet object available for integration fixtures")
fixture_path = local_dir / "fixture.parquet"
s3.download_file(curated_bucket, curated_objects[0]["Key"], str(fixture_path))
fixture_data = pd.read_parquet(fixture_path)
fixture_row = fixture_data[
    fixture_data["asin"].fillna("").astype(str).ne("")
].iloc[0]
valid_request = {
    "request_id": f"integration-{integration_run_id}",
    "asin": str(fixture_row["asin"]),
    "question": str(fixture_row["question"]),
}

print(json.dumps({
    "retrieval_receipt": retrieval_receipt_uri,
    "runtime_artifact": runtime_artifact_uri,
    "retrieval_run_id": retrieval_receipt["retrieval_run_id"],
    "fixture": {
        "request_id": valid_request["request_id"],
        "asin": valid_request["asin"],
        "question_characters": len(valid_request["question"]),
    },
}, indent=2))
"""
    ),
    markdown("## 3 Deploy a temporary SageMaker endpoint or initialize the local fallback"),
    code(
        r"""
predictor = None
sagemaker_model = None
deployment_status = "LocalFallback"
deployment_started_at = datetime.now(timezone.utc)
capture_uri = None  # Raw payload capture stays disabled; sanitized telemetry is written in section 6.

if DEPLOY_TEMPORARY_ENDPOINT:
    inference_source_dir = project_root / "inference_code"
    required_code = [
        inference_source_dir / "inference.py",
        inference_source_dir / "retrieval.py",
        inference_source_dir / "requirements.txt",
    ]
    if any(not path.exists() for path in required_code):
        raise FileNotFoundError("Run this notebook from the project root; inference_code is incomplete")
    sagemaker_model = SKLearnModel(
        model_data=runtime_artifact_uri,
        role=execution_role_arn,
        entry_point="inference.py",
        source_dir=str(inference_source_dir),
        framework_version="1.4-2",
        py_version="py3",
        name=f"{endpoint_name}-model",
        env={
            "SAGEMAKER_MODEL_SERVER_TIMEOUT": "300",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        },
    )
    predictor = sagemaker_model.deploy(
        initial_instance_count=1,
        instance_type=ENDPOINT_INSTANCE_TYPE,
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        model_data_download_timeout=MODEL_DOWNLOAD_TIMEOUT_SECONDS,
        container_startup_health_check_timeout=CONTAINER_STARTUP_TIMEOUT_SECONDS,
        wait=True,
        tags=[
            {"Key": "project", "Value": PROJECT_NAME},
            {"Key": "course", "Value": "AAI-540"},
            {"Key": "integration-run-id", "Value": integration_run_id},
        ],
    )
    deployment_description = sm.describe_endpoint(EndpointName=endpoint_name)
    deployment_status = deployment_description["EndpointStatus"]
    if deployment_status != "InService":
        raise RuntimeError(f"Endpoint did not reach InService: {deployment_status}")

    def invoke(payload):
        return predictor.predict(payload)
else:
    runtime_archive = download_s3_file(
        runtime_artifact_uri, local_dir / "product-qa-runtime.tar.gz"
    )
    local_runtime = local_dir / "runtime"
    local_runtime.mkdir(parents=True, exist_ok=True)
    with tarfile.open(runtime_archive, "r:gz") as archive:
        for member in archive.getmembers():
            target = (local_runtime / member.name).resolve()
            if local_runtime.resolve() not in target.parents or member.issym() or member.islnk():
                raise ValueError(f"Unsafe runtime archive member: {member.name}")
        archive.extractall(local_runtime)
    local_engine = RetrievalQAEngine.from_model_dir(local_runtime)

    def invoke(payload):
        return [local_engine.answer(payload)]

print("Inference backend status:", deployment_status)
"""
    ),
    markdown("## 4 Endpoint contract, abstention, grounding, and latency tests"),
    code(
        r"""
test_failures = []
telemetry_rows = []

def normalized_invoke(payload):
    started = time.perf_counter()
    result = invoke(payload)
    latency_ms = (time.perf_counter() - started) * 1000
    if isinstance(result, list) and len(result) == 1:
        result = result[0]
    return result, latency_ms

try:
    response, first_latency = normalized_invoke(valid_request)
    if response["request_id"] != valid_request["request_id"]:
        test_failures.append("response request_id does not match")
    if response["asin"] != valid_request["asin"]:
        test_failures.append("response ASIN does not match")
    if not response["abstained"] and response["answer"] not in [
        item["text"] for item in response["evidence"]
    ]:
        test_failures.append("answer is not copied from returned evidence")

    unknown_request = {
        "request_id": f"unknown-{integration_run_id}",
        "asin": "ASIN-NOT-IN-CORPUS",
        "question": valid_request["question"],
    }
    unknown_response, unknown_latency = normalized_invoke(unknown_request)
    if not unknown_response["abstained"] or unknown_response["evidence"]:
        test_failures.append("unknown product did not abstain with empty evidence")

    for index in range(TEST_REQUEST_COUNT):
        request_payload = {
            **valid_request,
            "request_id": f"load-{integration_run_id}-{index}",
        }
        result, latency_ms = normalized_invoke(request_payload)
        grounded = result["abstained"] or result["answer"] in [
            item["text"] for item in result["evidence"]
        ]
        telemetry_rows.append({
            "request_id": request_payload["request_id"],
            "asin_sha256": hashlib.sha256(
                request_payload["asin"].encode("utf-8")
            ).hexdigest()[:16],
            "question_char_count": len(request_payload["question"]),
            "latency_ms": latency_ms,
            "abstained": bool(result["abstained"]),
            "grounded": bool(grounded),
            "evidence_count": len(result["evidence"]),
            "retrieval_run_id": retrieval_receipt["retrieval_run_id"],
        })
except Exception as exc:
    test_failures.append(f"inference invocation failed: {type(exc).__name__}: {exc}")

telemetry = pd.DataFrame(telemetry_rows)
latency_metrics = {
    "requests": int(len(telemetry)),
    "p50_latency_ms": float(telemetry["latency_ms"].quantile(0.50)) if len(telemetry) else None,
    "p95_latency_ms": float(telemetry["latency_ms"].quantile(0.95)) if len(telemetry) else None,
    "abstention_rate": float(telemetry["abstained"].mean()) if len(telemetry) else None,
    "grounded_rate": float(telemetry["grounded"].mean()) if len(telemetry) else None,
}
if latency_metrics["p95_latency_ms"] is not None and latency_metrics["p95_latency_ms"] > P95_LATENCY_LIMIT_MS:
    test_failures.append(
        f"p95 latency {latency_metrics['p95_latency_ms']:.1f} ms exceeds "
        f"{P95_LATENCY_LIMIT_MS} ms"
    )
if latency_metrics["grounded_rate"] is not None and latency_metrics["grounded_rate"] != 1.0:
    test_failures.append("at least one answer was not grounded in returned evidence")

display(telemetry.head())
print(json.dumps(latency_metrics, indent=2))
print("Test failures:", test_failures or ["none"])
"""
    ),
    markdown("## 5 Flask API integration tests"),
    code(
        r"""
flask_app = create_app(
    endpoint_name=endpoint_name if DEPLOY_TEMPORARY_ENDPOINT else None,
    invoke=invoke,
)
flask_client = flask_app.test_client()
flask_results = []

health_response = flask_client.get("/health")
flask_results.append({
    "test": "health",
    "status": health_response.status_code,
    "passed": health_response.status_code == 200,
})

answer_response = flask_client.post("/answer", json=valid_request)
answer_payload = answer_response.get_json()
flask_results.append({
    "test": "valid answer request",
    "status": answer_response.status_code,
    "passed": (
        answer_response.status_code == 200
        and answer_payload.get("request_id") == valid_request["request_id"]
        and isinstance(answer_payload.get("evidence"), list)
    ),
})

invalid_response = flask_client.post(
    "/answer", json={"request_id": "invalid", "asin": valid_request["asin"]}
)
flask_results.append({
    "test": "missing question rejected",
    "status": invalid_response.status_code,
    "passed": invalid_response.status_code == 400,
})

flask_test_frame = pd.DataFrame(flask_results)
display(flask_test_frame)
if not flask_test_frame["passed"].all():
    test_failures.append("one or more Flask API contract tests failed")
"""
    ),
    markdown("## 6 Publish sanitized integration telemetry"),
    code(
        r"""
telemetry_uri = f"s3://{bucket}/{integration_prefix}/telemetry/sanitized_requests.csv"
telemetry_path = local_dir / "sanitized_requests.csv"
telemetry.to_csv(telemetry_path, index=False)
telemetry_bucket, telemetry_key = split_s3_uri(telemetry_uri)
s3.upload_file(str(telemetry_path), telemetry_bucket, telemetry_key)

cloudwatch_namespace = "AAI540/ProductQA"
integration_dashboard_name = None
integration_alarm_names = []
if PUBLISH_CLOUDWATCH_METRICS and latency_metrics["requests"]:
    dimensions = [
        {"Name": "Project", "Value": PROJECT_NAME},
        {"Name": "RetrievalRunId", "Value": retrieval_receipt["retrieval_run_id"]},
    ]
    metric_values = {
        "InferenceP50LatencyMs": latency_metrics["p50_latency_ms"],
        "InferenceP95LatencyMs": latency_metrics["p95_latency_ms"],
        "InferenceAbstentionRate": latency_metrics["abstention_rate"],
        "InferenceGroundedRate": latency_metrics["grounded_rate"],
        "IntegrationTestFailures": float(len(test_failures)),
    }
    cloudwatch.put_metric_data(
        Namespace=cloudwatch_namespace,
        MetricData=[
            {
                "MetricName": name,
                "Dimensions": dimensions,
                "Timestamp": datetime.now(timezone.utc),
                "Value": float(value),
                "Unit": "Milliseconds" if name.endswith("LatencyMs") else "None",
            }
            for name, value in metric_values.items()
            if value is not None
        ],
    )

    if CREATE_INTEGRATION_MONITORING:
        integration_dashboard_name = f"AAI540-{PROJECT_NAME}-application-integration"
        metric_dimension_map = {
            dimension["Name"]: dimension["Value"] for dimension in dimensions
        }
        dashboard_dimensions = [
            [dimension["Name"], dimension["Value"]] for dimension in dimensions
        ]
        dashboard_body = {
            "start": "-PT24H",
            "periodOverride": "inherit",
            "widgets": [
                {
                    "type": "text",
                    "x": 0,
                    "y": 0,
                    "width": 24,
                    "height": 2,
                    "properties": {
                        "markdown": (
                            f"# {PROJECT_NAME} application integration\n"
                            f"Latest retrieval run: `{retrieval_receipt['retrieval_run_id']}`"
                        )
                    },
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 2,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "title": "Inference latency",
                        "region": region,
                        "period": 300,
                        "stat": "Average",
                        "view": "timeSeries",
                        "metrics": [
                            [cloudwatch_namespace, "InferenceP50LatencyMs", *sum(dashboard_dimensions, [])],
                            [cloudwatch_namespace, "InferenceP95LatencyMs", *sum(dashboard_dimensions, [])],
                        ],
                        "yAxis": {"left": {"label": "Milliseconds", "showUnits": False}},
                    },
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 2,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "title": "Grounding, abstention, and integration failures",
                        "region": region,
                        "period": 300,
                        "stat": "Average",
                        "view": "timeSeries",
                        "metrics": [
                            [cloudwatch_namespace, "InferenceGroundedRate", *sum(dashboard_dimensions, [])],
                            [cloudwatch_namespace, "InferenceAbstentionRate", *sum(dashboard_dimensions, [])],
                            [cloudwatch_namespace, "IntegrationTestFailures", *sum(dashboard_dimensions, [])],
                        ],
                    },
                },
            ],
        }
        cloudwatch.put_dashboard(
            DashboardName=integration_dashboard_name,
            DashboardBody=json.dumps(dashboard_body),
        )

        alarm_specs = [
            {
                "suffix": "p95-latency",
                "metric": "InferenceP95LatencyMs",
                "threshold": float(P95_LATENCY_LIMIT_MS),
                "comparison": "GreaterThanOrEqualToThreshold",
                "unit": "Milliseconds",
                "description": "Integration p95 inference latency exceeded the release limit.",
            },
            {
                "suffix": "grounding",
                "metric": "InferenceGroundedRate",
                "threshold": 0.99,
                "comparison": "LessThanThreshold",
                "unit": "None",
                "description": "One or more non-abstaining responses were not grounded in returned evidence.",
            },
            {
                "suffix": "test-failures",
                "metric": "IntegrationTestFailures",
                "threshold": 1.0,
                "comparison": "GreaterThanOrEqualToThreshold",
                "unit": "None",
                "description": "Application integration tests reported one or more failures.",
            },
        ]
        for alarm_spec in alarm_specs:
            alarm_name = f"AAI540-{PROJECT_NAME}-{alarm_spec['suffix']}"
            cloudwatch.put_metric_alarm(
                AlarmName=alarm_name,
                AlarmDescription=alarm_spec["description"],
                Namespace=cloudwatch_namespace,
                MetricName=alarm_spec["metric"],
                Dimensions=[
                    {"Name": name, "Value": value}
                    for name, value in metric_dimension_map.items()
                ],
                Statistic="Average",
                Period=300,
                EvaluationPeriods=1,
                DatapointsToAlarm=1,
                Threshold=alarm_spec["threshold"],
                ComparisonOperator=alarm_spec["comparison"],
                TreatMissingData="notBreaching",
                Unit=alarm_spec["unit"],
                ActionsEnabled=False,
            )
            integration_alarm_names.append(alarm_name)
print("Sanitized telemetry:", telemetry_uri)
if integration_dashboard_name:
    print("CloudWatch dashboard:", integration_dashboard_name)
    print("CloudWatch alarms:", integration_alarm_names)
"""
    ),
    markdown("## 7 Persist deployment and test evidence"),
    code(
        r"""
integration_receipt = {
    "schema_version": "aai540-application-integration-v1",
    "project": PROJECT_NAME,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "integration_run_id": integration_run_id,
    "retrieval_receipt": retrieval_receipt_uri,
    "retrieval_run_id": retrieval_receipt["retrieval_run_id"],
    "runtime_artifact": runtime_artifact_uri,
    "deployment": {
        "mode": "temporary SageMaker real-time endpoint" if DEPLOY_TEMPORARY_ENDPOINT else "in-process fallback",
        "endpoint_name": endpoint_name if DEPLOY_TEMPORARY_ENDPOINT else None,
        "instance_type": ENDPOINT_INSTANCE_TYPE if DEPLOY_TEMPORARY_ENDPOINT else None,
        "status_at_test": deployment_status,
        "raw_data_capture": None,
        "sanitized_telemetry": telemetry_uri,
    },
    "tests": {
        "request_count": latency_metrics["requests"],
        "latency": latency_metrics,
        "flask_contract": flask_results,
        "failures": test_failures,
        "passed": not test_failures,
    },
    "telemetry": telemetry_uri,
    "monitoring": {
        "namespace": cloudwatch_namespace if PUBLISH_CLOUDWATCH_METRICS else None,
        "dashboard_name": integration_dashboard_name,
        "alarm_names": integration_alarm_names,
        "alarm_actions_enabled": False,
    },
    "rollback": "Delete the temporary endpoint; future deployments retain the prior approved runtime receipt.",
}
receipt_uri = f"s3://{bucket}/{integration_prefix}/application_integration_receipt.json"
receipt_body = json.dumps(integration_receipt, indent=2, allow_nan=False).encode("utf-8")
receipt_bucket, receipt_key = split_s3_uri(receipt_uri)
print(json.dumps(integration_receipt, indent=2))
print("Integration receipt will be finalized after endpoint cleanup:", receipt_uri)
"""
    ),
    markdown("## 8 Mandatory endpoint cleanup"),
    code(
        r"""
cleanup_status = "NotApplicable"
if DEPLOY_TEMPORARY_ENDPOINT:
    if DELETE_ENDPOINT_AFTER_TESTS:
        predictor.delete_endpoint(delete_endpoint_config=True)
        sagemaker_model.delete_model()
        cleanup_status = "EndpointConfigAndModelDeleteRequested"
        print(f"Deleted endpoint resources for {endpoint_name}.")
    else:
        cleanup_status = "EndpointRetainedByExplicitConfiguration"
        print(
            f"WARNING: endpoint {endpoint_name} is still billable. "
            "Set DELETE_ENDPOINT_AFTER_TESTS=True and rerun this cell."
        )
else:
    cleanup_status = "LocalFallbackNoAWSServingResources"
    print("No endpoint resources were created.")

# Write the run-scoped receipt only after cleanup so the immutable release record
# proves whether billable serving resources were removed.
integration_receipt["deployment"]["cleanup_status"] = cleanup_status
integration_receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
receipt_body = json.dumps(integration_receipt, indent=2, allow_nan=False).encode("utf-8")
s3.put_object(Bucket=receipt_bucket, Key=receipt_key, Body=receipt_body, ContentType="application/json")
s3.put_object(
    Bucket=bucket,
    Key=f"aai540/{PROJECT_NAME}/integration/latest.json",
    Body=receipt_body,
    ContentType="application/json",
)
print("Final cleanup status recorded in the integration receipt:", cleanup_status)
"""
    ),
    markdown("## 9 Final release gate"),
    code(
        r"""
if DEPLOY_TEMPORARY_ENDPOINT and DELETE_ENDPOINT_AFTER_TESTS:
    # DeleteEndpoint is asynchronous. A deleting/not-found endpoint is acceptable.
    try:
        current_status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        if current_status not in {"Deleting"}:
            print("Endpoint deletion has been requested; current status:", current_status)
    except ClientError:
        current_status = "Deleted"

if test_failures:
    raise RuntimeError(
        "Application integration did not pass. Serving resources were cleaned up. "
        f"Failures: {test_failures}"
    )
print("All application integration tests passed.")
print("Cleanup status:", cleanup_status)
print("The complete Product QA outcome is ready for demonstration.")
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} cells")
