"""Build the SageMaker observability notebook from reviewable source cells."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "AAI540_SageMaker_ML_System_Observability.ipynb"
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
# AAI 540 ML system observability

This notebook is stage 3 of the Product QA MLOps workflow. It discovers the latest successful model-development receipt, creates a validation baseline and a scored production batch, runs a custom SageMaker Processing monitor, publishes bounded custom metrics to CloudWatch, creates a dashboard and alarms, and writes an auditable monitoring receipt to S3.

The implementation uses a custom monitor because it works in AWS Academy Learner Lab accounts even when SageMaker Model Monitor is unavailable. It covers inference schema and nulls, feature and prediction drift (PSI), confidence, delayed-label macro F1/NDCG, and a deterministic retraining recommendation.
"""
    ),
    markdown(
        r"""
## Prerequisites and cost boundary

Run the data-lake notebook and model-development notebook first. Use the same AWS account, Region, project directory, and S3 bucket. The execution role needs S3 read/write, SageMaker Processing, ECR pull, CloudWatch `PutMetricData`, `PutDashboard`, `PutMetricAlarm`, and `iam:PassRole` permissions.

One small Processing job is the only compute created here and terminates automatically. CloudWatch custom metrics, dashboards, and alarms can incur small charges; cleanup toggles are provided at the end.
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
    "matplotlib>=3.8,<4" \
    "seaborn>=0.13,<1"
"""
    ),
    markdown(
        "> Restart the kernel only if the package cell reports that an already-loaded package could not be upgraded. Then continue from the import cell."
    ),
    code(
        r"""
import json
import os
import subprocess
import sys
import tarfile
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
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from botocore.config import Config
from IPython.display import display
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

from src.train import CANDIDATE_FEATURES, CLASS_LABELS, probability_matrix

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 140)
sns.set_theme(style="whitegrid", context="notebook")
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
SOURCE_MODEL_RUN_ID_OVERRIDE = ""       # Blank discovers the newest modeling_receipt.json.

RUN_PROCESSING_JOB = True                # False runs src/monitor.py in this notebook kernel.
CREATE_CLOUDWATCH_RESOURCES = True
INCLUDE_DELAYED_LABELS = True            # ePQA production holdout supplies simulated delayed labels.
PROCESSING_INSTANCE_TYPE = "ml.m5.large"
SKLEARN_FRAMEWORK_VERSION = "1.4-2"

PSI_THRESHOLD = 0.20
MISSING_RATE_THRESHOLD = 0.02
QUALITY_DROP_THRESHOLD = 0.05
LOW_CONFIDENCE_THRESHOLD = 0.60
CLOUDWATCH_NAMESPACE = "AAI540/ProductQA"
ALARM_ACTION_ARNS = []                   # Optional SNS topic ARNs; empty still creates visible alarms.
# -------------------------------------------

session = boto3.Session(region_name=AWS_REGION_OVERRIDE or None)
region = session.region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
if not region:
    raise RuntimeError("No AWS Region is configured. Set AWS_REGION_OVERRIDE and rerun.")

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
    raise RuntimeError("Could not infer the SageMaker execution role. Set EXECUTION_ROLE_ARN_OVERRIDE.")

bucket = S3_BUCKET_OVERRIDE or f"sagemaker-{region}-{account_id}"
monitor_run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
monitor_prefix = f"aai540/{PROJECT_NAME}/monitoring/{monitor_run_id}"
local_dir = Path("/tmp") / f"aai540_monitoring_{monitor_run_id}"
local_dir.mkdir(parents=True, exist_ok=True)

print(json.dumps({
    "region": region,
    "caller_arn": caller_arn,
    "execution_role_arn": execution_role_arn,
    "bucket": bucket,
    "monitor_run_id": monitor_run_id,
    "monitor_s3_root": f"s3://{bucket}/{monitor_prefix}/",
}, indent=2))
"""
    ),
    markdown("## 2 Discover and validate the model and data receipts"),
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

modeling_prefix = f"aai540/{PROJECT_NAME}/modeling/"
if SOURCE_MODEL_RUN_ID_OVERRIDE:
    modeling_receipt_key = f"{modeling_prefix}{SOURCE_MODEL_RUN_ID_OVERRIDE}/modeling_receipt.json"
else:
    candidates = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=modeling_prefix):
        candidates.extend(
            item for item in page.get("Contents", [])
            if item["Key"].endswith("/modeling_receipt.json")
        )
    if not candidates:
        raise FileNotFoundError(
            f"No modeling receipt below s3://{bucket}/{modeling_prefix}; run stage 2 first."
        )
    modeling_receipt_key = max(candidates, key=lambda item: item["LastModified"])["Key"]

modeling_receipt_uri = f"s3://{bucket}/{modeling_receipt_key}"
modeling_receipt = read_s3_json(modeling_receipt_uri)
required_receipt_paths = [
    ("candidate", "model_artifact"),
    ("deployment", "output"),
]
for parent, child in required_receipt_paths:
    if not modeling_receipt.get(parent, {}).get(child):
        raise ValueError(f"Modeling receipt is missing {parent}.{child}")
if modeling_receipt.get("deployment", {}).get("status") != "Completed":
    raise ValueError("The selected modeling receipt does not describe a completed Batch Transform job")

run_summary = read_s3_json(modeling_receipt["source_run_summary"])
for split_name in ["validation", "production"]:
    if split_name not in run_summary.get("split_data", {}):
        raise ValueError(f"Source run summary is missing split_data.{split_name}")

print(json.dumps({
    "modeling_receipt": modeling_receipt_uri,
    "model_run_id": modeling_receipt["model_run_id"],
    "source_data_run_id": modeling_receipt["source_data_run_id"],
    "candidate_artifact": modeling_receipt["candidate"]["model_artifact"],
    "production_predictions": modeling_receipt["deployment"]["output"],
}, indent=2))
"""
    ),
    markdown("## 3 Build the approved baseline and current scored batch"),
    code(
        r"""
validation_path = download_s3_file(
    run_summary["split_data"]["validation"], local_dir / "validation.parquet"
)
production_path = download_s3_file(
    run_summary["split_data"]["production"], local_dir / "production.parquet"
)
predictions_path = download_s3_file(
    modeling_receipt["deployment"]["output"], local_dir / "production.csv.out"
)
artifact_path = download_s3_file(
    modeling_receipt["candidate"]["model_artifact"], local_dir / "candidate-model.tar.gz"
)

validation = pd.read_parquet(validation_path)
production = pd.read_parquet(production_path)
prediction_columns = [
    "prediction", "expected_relevance", "probability_0", "probability_1", "probability_2"
]
production_predictions = pd.read_csv(predictions_path, header=None, names=prediction_columns)
if len(production_predictions) != len(production):
    raise ValueError(
        f"Production prediction count mismatch: {len(production_predictions)} vs {len(production)}"
    )

with tarfile.open(artifact_path, "r:gz") as archive:
    member = archive.getmember("model.joblib")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise FileNotFoundError("model.joblib is missing from the candidate artifact")
    model_bundle_path = local_dir / "candidate-model.joblib"
    model_bundle_path.write_bytes(extracted.read())
candidate_bundle = joblib.load(model_bundle_path)
if candidate_bundle["feature_columns"] != CANDIDATE_FEATURES:
    raise ValueError("Candidate artifact feature order does not match src/train.py")

baseline_features = validation[CANDIDATE_FEATURES]
baseline_probabilities = probability_matrix(candidate_bundle["model"], baseline_features)
baseline_predictions = candidate_bundle["model"].predict(baseline_features).astype(int)
baseline_scored = validation[["record_id", "qid", "asin", "label", *CANDIDATE_FEATURES]].copy()
baseline_scored["prediction"] = baseline_predictions
baseline_scored["expected_relevance"] = baseline_probabilities @ CLASS_LABELS.astype(float)
for index, column in enumerate(["probability_0", "probability_1", "probability_2"]):
    baseline_scored[column] = baseline_probabilities[:, index]

current_columns = ["record_id", "qid", "asin", "label", *CANDIDATE_FEATURES]
current_scored = pd.concat(
    [production[current_columns].reset_index(drop=True), production_predictions.reset_index(drop=True)],
    axis=1,
)
if not INCLUDE_DELAYED_LABELS:
    current_scored = current_scored.drop(columns="label")

for name, frame in [("baseline", baseline_scored), ("current", current_scored)]:
    path = local_dir / "monitor-input" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    key = f"{monitor_prefix}/input/{name}.csv"
    s3.upload_file(str(path), bucket, key)
    globals()[f"{name}_input_uri"] = f"s3://{bucket}/{key}"

display(pd.DataFrame({
    "dataset": ["baseline", "current"],
    "rows": [len(baseline_scored), len(current_scored)],
    "labels_available": ["label" in baseline_scored, "label" in current_scored],
    "input_uri": [baseline_input_uri, current_input_uri],
}))
"""
    ),
    markdown("## 4 Run the custom monitor"),
    code(
        r"""
report_output_uri = f"s3://{bucket}/{monitor_prefix}/report"
monitor_script = Path.cwd() / "src" / "monitor.py"
if not monitor_script.exists():
    raise FileNotFoundError(f"Run this notebook from the project root; missing {monitor_script}")

monitor_arguments = [
    "--baseline", "/opt/ml/processing/input/baseline",
    "--current", "/opt/ml/processing/input/current",
    "--output-dir", "/opt/ml/processing/output/report",
    "--psi-threshold", str(PSI_THRESHOLD),
    "--missing-rate-threshold", str(MISSING_RATE_THRESHOLD),
    "--quality-drop-threshold", str(QUALITY_DROP_THRESHOLD),
    "--low-confidence-threshold", str(LOW_CONFIDENCE_THRESHOLD),
]

if RUN_PROCESSING_JOB:
    processor = SKLearnProcessor(
        framework_version=SKLEARN_FRAMEWORK_VERSION,
        role=execution_role_arn,
        instance_count=1,
        instance_type=PROCESSING_INSTANCE_TYPE,
        base_job_name=f"aai540-{PROJECT_NAME}-monitor",
        tags=[
            {"Key": "project", "Value": PROJECT_NAME},
            {"Key": "course", "Value": "AAI-540"},
            {"Key": "model-run-id", "Value": modeling_receipt["model_run_id"]},
            {"Key": "monitor-run-id", "Value": monitor_run_id},
        ],
    )
    processor.run(
        code=str(monitor_script),
        inputs=[
            ProcessingInput(source=baseline_input_uri, destination="/opt/ml/processing/input/baseline"),
            ProcessingInput(source=current_input_uri, destination="/opt/ml/processing/input/current"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="report",
                source="/opt/ml/processing/output/report",
                destination=report_output_uri,
            )
        ],
        arguments=monitor_arguments,
        wait=True,
        logs=True,
    )
    processing_job_name = processor.latest_job.job_name
    processing_status = sm.describe_processing_job(
        ProcessingJobName=processing_job_name
    )["ProcessingJobStatus"]
else:
    local_report_dir = local_dir / "report"
    local_arguments = [
        "--baseline", str(local_dir / "monitor-input" / "baseline.csv"),
        "--current", str(local_dir / "monitor-input" / "current.csv"),
        "--output-dir", str(local_report_dir),
        "--psi-threshold", str(PSI_THRESHOLD),
        "--missing-rate-threshold", str(MISSING_RATE_THRESHOLD),
        "--quality-drop-threshold", str(QUALITY_DROP_THRESHOLD),
        "--low-confidence-threshold", str(LOW_CONFIDENCE_THRESHOLD),
    ]
    subprocess.run([sys.executable, str(monitor_script), *local_arguments], check=True)
    for output_file in local_report_dir.iterdir():
        s3.upload_file(str(output_file), bucket, f"{monitor_prefix}/report/{output_file.name}")
    processing_job_name = None
    processing_status = "LocalCompleted"

monitoring_report_uri = f"{report_output_uri}/monitoring_report.json"
feature_drift_uri = f"{report_output_uri}/feature_drift.csv"
monitoring_report = read_s3_json(monitoring_report_uri)
print(json.dumps(monitoring_report, indent=2))
"""
    ),
    markdown("## 5 Review drift and quality"),
    code(
        r"""
drift_path = download_s3_file(feature_drift_uri, local_dir / "feature_drift.csv")
feature_drift = pd.read_csv(drift_path).sort_values("psi", ascending=False)
display(feature_drift.style.format({"psi": "{:.4f}"}).background_gradient(subset=["psi"], cmap="YlOrRd"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=feature_drift, y="feature", x="psi", ax=axes[0], color="steelblue")
axes[0].axvline(PSI_THRESHOLD, color="red", linestyle="--", label=f"threshold={PSI_THRESHOLD}")
axes[0].set_title("Feature population stability index")
axes[0].legend()

quality_rows = []
for dataset in ["baseline", "current"]:
    labeled = monitoring_report[dataset].get("labeled_quality") or {}
    for metric in ["macro_f1", "balanced_accuracy", "mean_question_ndcg"]:
        if metric in labeled:
            quality_rows.append({"dataset": dataset, "metric": metric, "value": labeled[metric]})
quality_plot = pd.DataFrame(quality_rows)
if not quality_plot.empty:
    sns.barplot(data=quality_plot, x="metric", y="value", hue="dataset", ax=axes[1])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Delayed-label quality")
else:
    axes[1].text(0.5, 0.5, "No delayed labels supplied", ha="center", va="center")
    axes[1].set_axis_off()
plt.tight_layout()
plt.show()

decision = monitoring_report["decision"]
print("Retraining recommended:", decision["retraining_recommended"])
print("Reasons:", decision["reasons"] or ["No threshold breached"])
"""
    ),
    markdown("## 6 Publish CloudWatch metrics, dashboard, and alarms"),
    code(
        r"""
model_run_id = modeling_receipt["model_run_id"]
dimensions = [
    {"Name": "Project", "Value": PROJECT_NAME},
    {"Name": "ModelRunId", "Value": model_run_id},
]

current_confidence = monitoring_report["current"].get("confidence") or {}
metric_values = {
    "MissingRate": monitoring_report["current"]["data_quality"]["missing_rate"],
    "InvalidNumericRate": monitoring_report["current"]["data_quality"]["invalid_numeric_rate"],
    "MaxFeaturePSI": monitoring_report["drift"]["max_feature_psi"],
    "PredictionPSI": monitoring_report["drift"]["prediction_psi"],
    "MeanConfidence": current_confidence.get("mean_confidence"),
    "LowConfidenceRate": current_confidence.get("low_confidence_rate"),
    "MacroF1Delta": monitoring_report["quality_change"]["macro_f1_delta"],
    "MeanQuestionNDCGDelta": monitoring_report["quality_change"]["mean_question_ndcg_delta"],
    "RetrainingRecommended": float(monitoring_report["decision"]["retraining_recommended"]),
    "MonitoringJobSuccess": 1.0,
}
metric_data = [
    {
        "MetricName": name,
        "Dimensions": dimensions,
        "Timestamp": datetime.now(timezone.utc),
        "Value": float(value),
        "Unit": "None",
        "StorageResolution": 60,
    }
    for name, value in metric_values.items()
    if value is not None and np.isfinite(float(value))
]

dashboard_name = f"aai540-{PROJECT_NAME}-observability"
alarm_specs = [
    ("max-feature-psi", "MaxFeaturePSI", "GreaterThanOrEqualToThreshold", PSI_THRESHOLD),
    ("missing-rate", "MissingRate", "GreaterThanOrEqualToThreshold", MISSING_RATE_THRESHOLD),
    ("macro-f1-drop", "MacroF1Delta", "LessThanOrEqualToThreshold", -QUALITY_DROP_THRESHOLD),
]
alarm_names = [f"aai540-{PROJECT_NAME}-{suffix}" for suffix, *_ in alarm_specs]

if CREATE_CLOUDWATCH_RESOURCES:
    for start in range(0, len(metric_data), 20):
        cloudwatch.put_metric_data(
            Namespace=CLOUDWATCH_NAMESPACE,
            MetricData=metric_data[start:start + 20],
        )

    dashboard_body = {
        "widgets": [
            {
                "type": "text", "x": 0, "y": 0, "width": 24, "height": 2,
                "properties": {"markdown": (
                    f"# AAI 540 Product QA observability\n"
                    f"Model run `{model_run_id}`; latest monitor run `{monitor_run_id}`. "
                    "This dashboard is managed by the observability notebook."
                )},
            },
            {
                "type": "metric", "x": 0, "y": 2, "width": 12, "height": 6,
                "properties": {
                    "title": "Data and prediction drift",
                    "region": region,
                    "period": 300,
                    "stat": "Maximum",
                    "view": "timeSeries",
                    "metrics": [
                        [CLOUDWATCH_NAMESPACE, "MaxFeaturePSI", "Project", PROJECT_NAME, "ModelRunId", model_run_id],
                        [CLOUDWATCH_NAMESPACE, "PredictionPSI", "Project", PROJECT_NAME, "ModelRunId", model_run_id],
                        [CLOUDWATCH_NAMESPACE, "MissingRate", "Project", PROJECT_NAME, "ModelRunId", model_run_id],
                    ],
                    "annotations": {"horizontal": [{"value": PSI_THRESHOLD, "label": "PSI alert", "color": "#d62728"}]},
                },
            },
            {
                "type": "metric", "x": 12, "y": 2, "width": 12, "height": 6,
                "properties": {
                    "title": "Confidence and delayed-label quality",
                    "region": region,
                    "period": 300,
                    "stat": "Average",
                    "view": "timeSeries",
                    "metrics": [
                        [CLOUDWATCH_NAMESPACE, "MeanConfidence", "Project", PROJECT_NAME, "ModelRunId", model_run_id],
                        [CLOUDWATCH_NAMESPACE, "LowConfidenceRate", "Project", PROJECT_NAME, "ModelRunId", model_run_id],
                        [CLOUDWATCH_NAMESPACE, "MacroF1Delta", "Project", PROJECT_NAME, "ModelRunId", model_run_id],
                        [CLOUDWATCH_NAMESPACE, "MeanQuestionNDCGDelta", "Project", PROJECT_NAME, "ModelRunId", model_run_id],
                    ],
                },
            },
        ]
    }
    dashboard_response = cloudwatch.put_dashboard(
        DashboardName=dashboard_name,
        DashboardBody=json.dumps(dashboard_body),
    )
    if dashboard_response.get("DashboardValidationMessages"):
        raise RuntimeError(dashboard_response["DashboardValidationMessages"])

    for suffix, metric_name, comparison, threshold in alarm_specs:
        cloudwatch.put_metric_alarm(
            AlarmName=f"aai540-{PROJECT_NAME}-{suffix}",
            AlarmDescription=f"AAI 540 Product QA monitor: {metric_name} breached its release threshold.",
            ActionsEnabled=bool(ALARM_ACTION_ARNS),
            AlarmActions=ALARM_ACTION_ARNS,
            Namespace=CLOUDWATCH_NAMESPACE,
            MetricName=metric_name,
            Dimensions=dimensions,
            Statistic="Maximum" if metric_name != "MacroF1Delta" else "Minimum",
            Period=300,
            EvaluationPeriods=1,
            DatapointsToAlarm=1,
            Threshold=float(threshold),
            ComparisonOperator=comparison,
            TreatMissingData="notBreaching",
        )

    configured = cloudwatch.describe_alarms(AlarmNames=alarm_names)["MetricAlarms"]
    if {alarm["AlarmName"] for alarm in configured} != set(alarm_names):
        raise RuntimeError("Not all CloudWatch alarms were created")
    print(f"Published {len(metric_data)} metrics, dashboard {dashboard_name}, and {len(configured)} alarms.")
else:
    configured = []
    print("CloudWatch publication skipped by configuration.")
"""
    ),
    markdown("## 7 Persist the monitoring receipt for continual learning"),
    code(
        r"""
monitoring_receipt = {
    "schema_version": "aai540-monitoring-receipt-v1",
    "project": PROJECT_NAME,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "monitor_run_id": monitor_run_id,
    "model_run_id": model_run_id,
    "source_data_run_id": modeling_receipt["source_data_run_id"],
    "modeling_receipt": modeling_receipt_uri,
    "processing": {
        "mode": "SageMaker Processing" if RUN_PROCESSING_JOB else "local notebook fallback",
        "job_name": processing_job_name,
        "status": processing_status,
    },
    "inputs": {"baseline": baseline_input_uri, "current": current_input_uri},
    "outputs": {
        "monitoring_report": monitoring_report_uri,
        "feature_drift": feature_drift_uri,
    },
    "decision": monitoring_report["decision"],
    "cloudwatch": {
        "enabled": CREATE_CLOUDWATCH_RESOURCES,
        "namespace": CLOUDWATCH_NAMESPACE,
        "dashboard": dashboard_name if CREATE_CLOUDWATCH_RESOURCES else None,
        "alarms": alarm_names if CREATE_CLOUDWATCH_RESOURCES else [],
    },
}
receipt_key = f"{monitor_prefix}/monitoring_receipt.json"
latest_key = f"aai540/{PROJECT_NAME}/monitoring/latest.json"
receipt_body = json.dumps(monitoring_receipt, indent=2, allow_nan=False).encode("utf-8")
for key in [receipt_key, latest_key]:
    s3.put_object(Bucket=bucket, Key=key, Body=receipt_body, ContentType="application/json")

print(json.dumps(monitoring_receipt, indent=2))
print(f"Monitoring receipt: s3://{bucket}/{receipt_key}")
print(f"Latest pointer: s3://{bucket}/{latest_key}")
"""
    ),
    markdown(
        r"""
## Completion and cleanup

The next notebook reads `monitoring/latest.json`. A monitoring recommendation is a trigger to create a candidate; it is not permission to promote a model. Keep alarms for the demonstration, or use the explicit cleanup cell after capturing evidence.
"""
    ),
    code(
        r"""
DELETE_CLOUDWATCH_RESOURCES = False

if DELETE_CLOUDWATCH_RESOURCES:
    cloudwatch.delete_alarms(AlarmNames=alarm_names)
    cloudwatch.delete_dashboards(DashboardNames=[dashboard_name])
    print(f"Deleted dashboard {dashboard_name} and alarms: {alarm_names}")
else:
    print("No CloudWatch resources deleted.")
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
