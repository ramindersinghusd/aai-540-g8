"""Build the SageMaker model-development notebook from reviewable source cells."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "AAI540_SageMaker_Model_Development_Deployment.ipynb"
CELL_COUNTER = 0


def next_cell_id(prefix: str) -> str:
    global CELL_COUNTER
    CELL_COUNTER += 1
    return f"{prefix}-{CELL_COUNTER:02d}"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": next_cell_id("markdown"),
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": next_cell_id("code"),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""
# AAI 540 SageMaker benchmark, model development, evaluation, and deployment

This notebook is the modeling continuation of `AAI540_SageMaker_DataLake_FeatureStore.ipynb` for the **Retrieval-Augmented Product Question Answering** project. It:

1. discovers a completed, versioned data run in Amazon S3;
2. trains a deliberately small **two-feature benchmark** in a SageMaker training job;
3. trains and debugs a class-balanced **random-forest candidate** in a second SageMaker training job;
4. compares both frozen artifacts on the untouched product-grouped test split using macro F1 and per-question NDCG; and
5. deploys the candidate through **SageMaker Batch Transform** against the production split.

Batch Transform was selected instead of an always-on endpoint because the first product-QA MVP can score candidate evidence asynchronously, the job automatically releases compute, and it avoids idle endpoint cost.

**Leakage controls:** the upstream notebook groups splits by ASIN (with qid fallback); this notebook rechecks group disjointness. Features derived from the gold `answer` field are excluded from both models because that field is not guaranteed to exist at inference time.
"""
    ),
    markdown(
        r"""
## Prerequisites and permissions

Run this notebook in a SageMaker AI JupyterLab Space from the same project directory as `src/train.py`. First complete the data-lake notebook and retain its S3 run artifacts.

The execution role needs access to the project S3 prefix plus permissions for SageMaker training, model creation, Batch Transform, CloudWatch Logs, ECR image pulls, and `iam:PassRole`. The default instance types are intentionally small; change them only if your lab allow-list requires a different type.

The notebook pins the SageMaker Scikit-learn container to framework version `1.4-2` and installs the matching local Scikit-learn version so downloaded model artifacts can be evaluated without a serialization-version mismatch.
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
        r"""
> Restart the kernel only if the package cell reports that a loaded package could not be upgraded. Training and Batch Transform create billable AWS jobs; the configuration flags make those actions explicit and easy to disable during review.
"""
    ),
    code(
        r"""
import json
import os
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
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn
from sklearn.metrics import confusion_matrix

from src.train import (
    BENCHMARK_FEATURES,
    CANDIDATE_FEATURES,
    CLASS_LABELS,
    evaluate_model,
    probability_matrix,
)

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 120)
sns.set_theme(style="whitegrid", context="notebook")
"""
    ),
    markdown("## 1 Configuration and AWS preflight"),
    code(
        r"""
# ---------- User-editable settings ----------
PROJECT_NAME = "product-qa"
AWS_REGION_OVERRIDE = ""              # Blank uses the current AWS session Region.
S3_BUCKET_OVERRIDE = ""               # Blank uses sagemaker-<region>-<account-id>.
EXECUTION_ROLE_ARN_OVERRIDE = ""       # Set if automatic role inference is not correct.
SOURCE_RUN_ID_OVERRIDE = ""            # Blank discovers the newest run_summary.json.
RANDOM_SEED = 540

TRAIN_INSTANCE_TYPE = "ml.m5.large"
TRANSFORM_INSTANCE_TYPE = "ml.m5.large"
SKLEARN_FRAMEWORK_VERSION = "1.4-2"

RUN_TRAINING_JOBS = True
RUN_BATCH_TRANSFORM = True
REQUIRE_BENCHMARK_IMPROVEMENT = True

# Only used when RUN_TRAINING_JOBS=False.
BENCHMARK_JOB_NAME = ""
CANDIDATE_JOB_NAME = ""
# -------------------------------------------

session = boto3.Session(region_name=AWS_REGION_OVERRIDE or None)
region = session.region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
if not region:
    raise RuntimeError("No AWS Region is configured. Set AWS_REGION_OVERRIDE and rerun this cell.")

retry_config = Config(retries={"max_attempts": 10, "mode": "adaptive"})
s3 = session.client("s3", config=retry_config)
sm_client = session.client("sagemaker", config=retry_config)
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
model_run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
model_prefix = f"aai540/{PROJECT_NAME}/modeling/{model_run_id}"
local_dir = Path("/tmp") / f"aai540_modeling_{model_run_id}"
local_dir.mkdir(parents=True, exist_ok=True)

print(json.dumps({
    "region": region,
    "caller_arn": caller_arn,
    "execution_role_arn": execution_role_arn,
    "bucket": bucket,
    "model_run_id": model_run_id,
    "model_s3_root": f"s3://{bucket}/{model_prefix}/",
}, indent=2))
"""
    ),
    markdown("## 2 Discover and validate the upstream data run"),
    code(
        r"""
def split_s3_uri(uri):
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an S3 URI, got: {uri}")
    bucket_name, _, key = uri[5:].partition("/")
    return bucket_name, key

data_runs_prefix = f"aai540/{PROJECT_NAME}/runs/"
if SOURCE_RUN_ID_OVERRIDE:
    summary_key = f"{data_runs_prefix}{SOURCE_RUN_ID_OVERRIDE}/run_summary.json"
    selected_summary = {"Key": summary_key, "LastModified": None}
else:
    summaries = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=data_runs_prefix):
        summaries.extend(
            item for item in page.get("Contents", [])
            if item["Key"].endswith("/run_summary.json")
        )
    if not summaries:
        raise FileNotFoundError(
            f"No run_summary.json found below s3://{bucket}/{data_runs_prefix}. "
            "Run the data-lake notebook first or set SOURCE_RUN_ID_OVERRIDE."
        )
    selected_summary = max(summaries, key=lambda item: item["LastModified"])
    summary_key = selected_summary["Key"]

run_summary = json.loads(s3.get_object(Bucket=bucket, Key=summary_key)["Body"].read())
required_splits = {"train", "validation", "test", "production"}
available_splits = set(run_summary.get("split_data", {}))
if available_splits != required_splits:
    raise ValueError(f"Expected split_data keys {sorted(required_splits)}, got {sorted(available_splits)}")

source_run_id = run_summary["run_id"]
print(f"Selected upstream run: {source_run_id}")
print(f"Run summary: s3://{bucket}/{summary_key}")
display(pd.Series(run_summary, name="value").to_frame())
"""
    ),
    code(
        r"""
def download_s3_file(uri, destination):
    source_bucket, source_key = split_s3_uri(uri)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(source_bucket, source_key, str(destination))
    return destination

splits = {}
for split_name, split_uri in run_summary["split_data"].items():
    local_path = download_s3_file(split_uri, local_dir / f"{split_name}.parquet")
    splits[split_name] = pd.read_parquet(local_path)

required_columns = set(
    ["record_id", "asin", "qid", "label"] +
    BENCHMARK_FEATURES + CANDIDATE_FEATURES
)
for split_name, frame in splits.items():
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{split_name} split is missing modeling columns: {missing}")
    if frame.empty:
        raise ValueError(f"{split_name} split is empty")
    if frame[list(required_columns)].isna().any().any():
        raise ValueError(f"{split_name} contains nulls in required modeling columns")
    if not set(frame["label"].astype(int)).issubset(set(CLASS_LABELS)):
        raise ValueError(f"{split_name} contains labels outside {CLASS_LABELS.tolist()}")

def modeling_group_keys(frame):
    asin = frame["asin"].fillna("").astype(str)
    qid = frame["qid"].fillna("").astype(str)
    return set(np.where(asin.ne("") & asin.ne("UNKNOWN"), "asin:" + asin, "qid:" + qid))

group_sets = {name: modeling_group_keys(frame) for name, frame in splits.items()}
for left_index, left_name in enumerate(sorted(group_sets)):
    for right_name in sorted(group_sets)[left_index + 1:]:
        overlap = group_sets[left_name] & group_sets[right_name]
        if overlap:
            raise ValueError(f"Group leakage between {left_name} and {right_name}: {list(overlap)[:5]}")

split_profile = pd.DataFrame([
    {
        "split": name,
        "rows": len(frame),
        "groups": len(group_sets[name]),
        "label_0_pct": frame["label"].eq(0).mean(),
        "label_1_pct": frame["label"].eq(1).mean(),
        "label_2_pct": frame["label"].eq(2).mean(),
    }
    for name, frame in splits.items()
]).set_index("split")
display(split_profile.style.format({column: "{:.1%}" for column in split_profile if column.endswith("_pct")}))
print("Leakage check passed: all four product/fallback-question group sets are disjoint.")
"""
    ),
    markdown(
        r"""
## 3 Materialize versioned training channels

The training CSVs contain IDs for leakage checks, the label, and the candidate feature superset. Feature selection happens inside the versioned training script. Batch inference receives only the 14 candidate features, in a fixed order and without a header.
"""
    ),
    code(
        r"""
training_columns = list(dict.fromkeys(
    ["record_id", "asin", "qid", "label"] + BENCHMARK_FEATURES + CANDIDATE_FEATURES
))
training_input_uris = {}
for split_name in ["train", "validation"]:
    path = local_dir / "channels" / split_name / f"{split_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    splits[split_name][training_columns].to_csv(path, index=False)
    key = f"{model_prefix}/input/{split_name}/{split_name}.csv"
    s3.upload_file(str(path), bucket, key)
    training_input_uris[split_name] = f"s3://{bucket}/{key}"

production_input_path = local_dir / "batch-input" / "production.csv"
production_input_path.parent.mkdir(parents=True, exist_ok=True)
splits["production"][CANDIDATE_FEATURES].to_csv(production_input_path, index=False, header=False)
production_input_key = f"{model_prefix}/batch-input/production.csv"
s3.upload_file(str(production_input_path), bucket, production_input_key)
production_input_uri = f"s3://{bucket}/{production_input_key}"

print(json.dumps({
    "training_channels": training_input_uris,
    "batch_input": production_input_uri,
    "training_columns": training_columns,
    "candidate_inference_columns_in_order": CANDIDATE_FEATURES,
}, indent=2))
"""
    ),
    markdown("## 4 Configure SageMaker-managed training"),
    code(
        r"""
metric_names = [
    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
    "log_loss", "mean_question_ndcg",
]
metric_definitions = [
    {
        "Name": f"{split_name}:{metric_name}",
        "Regex": rf"{split_name}:{metric_name}=([0-9.eE+-]+)",
    }
    for split_name in ["train", "validation"]
    for metric_name in metric_names
]

source_dir = project_root / "src"
if not (source_dir / "train.py").exists():
    raise FileNotFoundError(
        f"Expected {source_dir / 'train.py'}. Run this notebook from the project root."
    )

training_channels = {
    name: TrainingInput(uri, content_type="text/csv", input_mode="File")
    for name, uri in training_input_uris.items()
}

def make_estimator(model_type):
    hyperparameters = {
        "model-type": model_type,
        "random-state": RANDOM_SEED,
        "min-samples-leaf": 25 if model_type == "benchmark" else 2,
    }
    if model_type == "candidate":
        hyperparameters.update({"n-estimators": 300, "max-depth": 12})
    return SKLearn(
        entry_point="train.py",
        source_dir=str(source_dir),
        role=execution_role_arn,
        instance_count=1,
        instance_type=TRAIN_INSTANCE_TYPE,
        framework_version=SKLEARN_FRAMEWORK_VERSION,
        py_version="py3",
        hyperparameters=hyperparameters,
        metric_definitions=metric_definitions,
        output_path=f"s3://{bucket}/{model_prefix}/artifacts/{model_type}",
        base_job_name=f"aai540-{PROJECT_NAME}-{model_type}",
        sagemaker_session=None,
        tags=[
            {"Key": "project", "Value": PROJECT_NAME},
            {"Key": "course", "Value": "AAI-540"},
            {"Key": "model-type", "Value": model_type},
            {"Key": "model-run-id", "Value": model_run_id},
            {"Key": "source-run-id", "Value": source_run_id},
        ],
    )
"""
    ),
    markdown(
        r"""
## 5 Train the two-feature benchmark

The benchmark is a depth-two decision tree using only hashed cosine similarity and token Jaccard. It is intentionally interpretable and capacity-limited so it defines a credible minimum bar rather than competing with the main model on complexity.
"""
    ),
    code(
        r"""
if RUN_TRAINING_JOBS:
    benchmark_estimator = make_estimator("benchmark")
    benchmark_estimator.fit(training_channels, wait=True, logs=True)
    BENCHMARK_JOB_NAME = benchmark_estimator.latest_training_job.name
else:
    if not BENCHMARK_JOB_NAME:
        raise ValueError("Set BENCHMARK_JOB_NAME when RUN_TRAINING_JOBS=False")
    benchmark_estimator = SKLearn.attach(BENCHMARK_JOB_NAME)

benchmark_description = sm_client.describe_training_job(TrainingJobName=BENCHMARK_JOB_NAME)
print(json.dumps({
    "training_job": BENCHMARK_JOB_NAME,
    "status": benchmark_description["TrainingJobStatus"],
    "model_artifact": benchmark_description["ModelArtifacts"]["S3ModelArtifacts"],
    "billable_seconds": benchmark_description.get("BillableTimeInSeconds"),
}, indent=2))
"""
    ),
    markdown(
        r"""
## 6 Build, train, and debug the first model iteration

The candidate is a deterministic, class-balanced random forest over 14 low-cost scalar features. Training validates the input schema, label domain, null policy, unique record separation, and product/question group separation before fitting. It prints train and validation metrics to CloudWatch and packages the ordered feature schema with the estimator.
"""
    ),
    code(
        r"""
if RUN_TRAINING_JOBS:
    candidate_estimator = make_estimator("candidate")
    candidate_estimator.fit(training_channels, wait=True, logs=True)
    CANDIDATE_JOB_NAME = candidate_estimator.latest_training_job.name
else:
    if not CANDIDATE_JOB_NAME:
        raise ValueError("Set CANDIDATE_JOB_NAME when RUN_TRAINING_JOBS=False")
    candidate_estimator = SKLearn.attach(CANDIDATE_JOB_NAME)

candidate_description = sm_client.describe_training_job(TrainingJobName=CANDIDATE_JOB_NAME)
print(json.dumps({
    "training_job": CANDIDATE_JOB_NAME,
    "status": candidate_description["TrainingJobStatus"],
    "model_artifact": candidate_description["ModelArtifacts"]["S3ModelArtifacts"],
    "billable_seconds": candidate_description.get("BillableTimeInSeconds"),
}, indent=2))
"""
    ),
    markdown("## 7 Evaluate frozen artifacts on the untouched test split"),
    code(
        r"""
def load_model_bundle(model_artifact_uri, model_name):
    archive_path = download_s3_file(model_artifact_uri, local_dir / "artifacts" / f"{model_name}.tar.gz")
    bundle_path = local_dir / "artifacts" / f"{model_name}.joblib"
    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            member = archive.getmember("model.joblib")
        except KeyError as exc:
            raise FileNotFoundError("model.joblib is missing from the SageMaker artifact") from exc
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError("Could not read model.joblib from the SageMaker artifact")
        bundle_path.write_bytes(extracted.read())
    bundle = joblib.load(bundle_path)
    expected_keys = {"model", "model_type", "feature_columns", "metrics", "schema_version"}
    missing_keys = sorted(expected_keys - set(bundle))
    if missing_keys:
        raise ValueError(f"{model_name} artifact is missing keys: {missing_keys}")
    return bundle

benchmark_bundle = load_model_bundle(
    benchmark_description["ModelArtifacts"]["S3ModelArtifacts"], "benchmark"
)
candidate_bundle = load_model_bundle(
    candidate_description["ModelArtifacts"]["S3ModelArtifacts"], "candidate"
)

evaluation_rows = []
for model_name, bundle in [("benchmark", benchmark_bundle), ("candidate", candidate_bundle)]:
    validation_metrics = bundle["metrics"]["validation"]
    test_metrics = evaluate_model(bundle["model"], splits["test"], bundle["feature_columns"])
    for dataset_name, metrics in [("validation", validation_metrics), ("test", test_metrics)]:
        evaluation_rows.append({"model": model_name, "dataset": dataset_name, **metrics})

evaluation = pd.DataFrame(evaluation_rows).set_index(["model", "dataset"]).sort_index()
display(evaluation.style.format("{:.4f}"))

comparison_metrics = ["macro_f1", "balanced_accuracy", "mean_question_ndcg", "log_loss"]
plot_frame = evaluation.reset_index().melt(
    id_vars=["model", "dataset"],
    value_vars=comparison_metrics,
    var_name="metric",
    value_name="value",
)
g = sns.catplot(
    data=plot_frame,
    x="model", y="value", hue="dataset", col="metric",
    kind="bar", col_wrap=2, sharey=False, height=3.5,
)
g.fig.suptitle("Benchmark versus candidate", y=1.03)
plt.show()

validation_macro_delta = float(
    evaluation.loc[("candidate", "validation"), "macro_f1"]
    - evaluation.loc[("benchmark", "validation"), "macro_f1"]
)
validation_ndcg_delta = float(
    evaluation.loc[("candidate", "validation"), "mean_question_ndcg"]
    - evaluation.loc[("benchmark", "validation"), "mean_question_ndcg"]
)
deployment_gate_passed = bool(
    validation_macro_delta >= 0 and validation_ndcg_delta >= -0.02
)
print(json.dumps({
    "validation_macro_f1_delta": validation_macro_delta,
    "validation_mean_question_ndcg_delta": validation_ndcg_delta,
    "deployment_gate_passed": deployment_gate_passed,
}, indent=2))

if REQUIRE_BENCHMARK_IMPROVEMENT and not deployment_gate_passed:
    raise RuntimeError(
        "Candidate did not pass the predeclared validation gate. Debug/tune it before deployment, "
        "or set REQUIRE_BENCHMARK_IMPROVEMENT=False only with a documented justification."
    )
"""
    ),
    markdown("### Error analysis and model interpretation"),
    code(
        r"""
test_frame = splits["test"].copy()
candidate_model = candidate_bundle["model"]
test_predictions = candidate_model.predict(test_frame[CANDIDATE_FEATURES]).astype(int)
test_probabilities = probability_matrix(candidate_model, test_frame[CANDIDATE_FEATURES])
test_expected_relevance = test_probabilities @ CLASS_LABELS.astype(float)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
matrix = confusion_matrix(test_frame["label"].astype(int), test_predictions, labels=CLASS_LABELS)
sns.heatmap(
    matrix, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=axes[0],
)
axes[0].set(title="Candidate test confusion matrix", xlabel="Predicted", ylabel="Actual")

importance = pd.Series(
    candidate_model.feature_importances_, index=CANDIDATE_FEATURES
).sort_values(ascending=True)
importance.tail(14).plot.barh(ax=axes[1], color="steelblue")
axes[1].set(title="Candidate feature importance", xlabel="Impurity-based importance")
plt.tight_layout()
plt.show()

error_sample = test_frame.loc[test_predictions != test_frame["label"].to_numpy(), [
    "record_id", "qid", "asin", "source", "question_type", "label",
]].copy()
error_sample["prediction"] = test_predictions[test_predictions != test_frame["label"].to_numpy()]
error_sample["expected_relevance"] = test_expected_relevance[
    test_predictions != test_frame["label"].to_numpy()
]
display(error_sample.head(20))
print(f"Candidate test errors: {len(error_sample):,} of {len(test_frame):,}")
"""
    ),
    markdown(
        r"""
## 8 Deploy the candidate with SageMaker Batch Transform

The production input contains one headerless CSV record per candidate pair. SageMaker splits on line boundaries and sends multiple records per request. The custom inference functions enforce the 14-column schema and emit five columns per record: predicted class, expected relevance, and probabilities for classes 0, 1, and 2.
"""
    ),
    code(
        r"""
batch_output_uri = f"s3://{bucket}/{model_prefix}/batch-output/"

if RUN_BATCH_TRANSFORM:
    transformer = candidate_estimator.transformer(
        instance_count=1,
        instance_type=TRANSFORM_INSTANCE_TYPE,
        strategy="MultiRecord",
        assemble_with="Line",
        output_path=batch_output_uri,
        accept="text/csv",
        max_payload=6,
        env={"SAGEMAKER_MODEL_SERVER_TIMEOUT": "120"},
        tags=[
            {"Key": "project", "Value": PROJECT_NAME},
            {"Key": "course", "Value": "AAI-540"},
            {"Key": "model-run-id", "Value": model_run_id},
        ],
    )
    transformer.transform(
        data=production_input_uri,
        data_type="S3Prefix",
        content_type="text/csv",
        split_type="Line",
        wait=True,
        logs=True,
    )
    transform_job_name = transformer.latest_transform_job.name
    transform_description = sm_client.describe_transform_job(TransformJobName=transform_job_name)
    if transform_description["TransformJobStatus"] != "Completed":
        raise RuntimeError(f"Batch Transform did not complete: {transform_description}")
else:
    raise RuntimeError(
        "RUN_BATCH_TRANSFORM=False: training/evaluation completed, but deployment was intentionally skipped."
    )

prediction_key = f"{model_prefix}/batch-output/production.csv.out"
prediction_path = local_dir / "batch-output" / "production.csv.out"
prediction_path.parent.mkdir(parents=True, exist_ok=True)
s3.download_file(bucket, prediction_key, str(prediction_path))
prediction_columns = [
    "prediction", "expected_relevance", "probability_0", "probability_1", "probability_2"
]
production_predictions = pd.read_csv(prediction_path, header=None, names=prediction_columns)
if len(production_predictions) != len(splits["production"]):
    raise ValueError(
        f"Expected {len(splits['production']):,} predictions, got {len(production_predictions):,}"
    )

scored_production = pd.concat([
    splits["production"][["record_id", "qid", "asin"]].reset_index(drop=True),
    production_predictions.reset_index(drop=True),
], axis=1)
assert np.allclose(
    scored_production[["probability_0", "probability_1", "probability_2"]].sum(axis=1),
    1.0,
    atol=1e-5,
)
display(scored_production.head(10))
print(json.dumps({
    "transform_job": transform_job_name,
    "status": transform_description["TransformJobStatus"],
    "input": production_input_uri,
    "output": f"s3://{bucket}/{prediction_key}",
    "rows_scored": len(scored_production),
}, indent=2))
"""
    ),
    markdown("## 9 Persist an auditable modeling and deployment receipt"),
    code(
        r"""
evaluation_records = []
for (model_name, dataset_name), row in evaluation.iterrows():
    evaluation_records.append({
        "model": model_name,
        "dataset": dataset_name,
        **{metric: float(value) for metric, value in row.items()},
    })

receipt = {
    "project": PROJECT_NAME,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "model_run_id": model_run_id,
    "source_data_run_id": source_run_id,
    "source_run_summary": f"s3://{bucket}/{summary_key}",
    "framework": {"name": "scikit-learn", "sagemaker_version": SKLEARN_FRAMEWORK_VERSION},
    "benchmark": {
        "description": "Depth-two decision tree using two lexical similarity features",
        "features": BENCHMARK_FEATURES,
        "training_job": BENCHMARK_JOB_NAME,
        "model_artifact": benchmark_description["ModelArtifacts"]["S3ModelArtifacts"],
    },
    "candidate": {
        "description": "Class-balanced random forest over inference-time scalar features",
        "features": CANDIDATE_FEATURES,
        "training_job": CANDIDATE_JOB_NAME,
        "model_artifact": candidate_description["ModelArtifacts"]["S3ModelArtifacts"],
    },
    "evaluation": evaluation_records,
    "deployment_gate": {
        "validation_macro_f1_delta": float(validation_macro_delta),
        "validation_mean_question_ndcg_delta": float(validation_ndcg_delta),
        "passed": bool(deployment_gate_passed),
    },
    "deployment": {
        "mode": "SageMaker Batch Transform",
        "transform_job": transform_job_name,
        "status": transform_description["TransformJobStatus"],
        "instance_type": TRANSFORM_INSTANCE_TYPE,
        "input": production_input_uri,
        "output": f"s3://{bucket}/{prediction_key}",
        "rows_scored": int(len(scored_production)),
    },
}
receipt_key = f"{model_prefix}/modeling_receipt.json"
s3.put_object(
    Bucket=bucket,
    Key=receipt_key,
    Body=json.dumps(receipt, indent=2).encode("utf-8"),
    ContentType="application/json",
)
print(json.dumps(receipt, indent=2))
print(f"\nModeling receipt: s3://{bucket}/{receipt_key}")
"""
    ),
    markdown(
        r"""
## Completion checklist and cost control

- **Benchmark in SageMaker:** a managed training job produced a versioned two-feature benchmark artifact.
- **First model iteration:** a separate managed training job validated inputs, trained the random forest, logged metrics, and packaged its inference schema.
- **Evaluation:** validation and untouched test results compare both artifacts; the deployment gate uses validation only, preserving the test split for final reporting.
- **Deployment:** a completed Batch Transform job scored every production record and stored ordered predictions in S3.

Batch Transform releases its compute when the job completes, so there is no real-time endpoint to delete. S3 artifacts, SageMaker model metadata, training histories, transform history, and CloudWatch logs remain as reproducibility evidence and incur only their normal storage/log charges. Delete those artifacts through your lab's approved cleanup process when they are no longer needed.
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

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Wrote {OUTPUT} with {len(cells)} cells")
