"""Build the semantic retrieval and grounded inference notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "AAI540_SageMaker_Retrieval_RAG_Inference.ipynb"
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
# AAI 540 semantic retrieval and grounded Product QA inference

Stage 5 completes the core user outcome: a caller supplies a request ID, product ASIN, and English question; the system retrieves product evidence using MiniLM embeddings and FAISS, reranks candidates with the trained classifier, then returns an extractive answer with ranked evidence or abstains when relevance confidence is low.

No free-form answer generation is used. The returned answer is always copied from retrieved evidence, keeping the MVP grounded and auditable.
"""
    ),
    markdown(
        r"""
## Prerequisites and cost boundary

Run stages 1-4 first. This notebook downloads the approved data/model lineage, builds the embedding index in the notebook kernel, and uploads one portable runtime artifact. It creates no endpoint. The first MiniLM load downloads public model weights; subsequent loads use the local cache.
"""
    ),
    code(
        r"""
%pip install -q --upgrade \
    "boto3>=1.34" \
    "botocore>=1.34" \
    "pandas>=2.1,<3" \
    "pyarrow>=15,<22" \
    "scikit-learn==1.4.2" \
    "joblib>=1.3,<2" \
    "sentence-transformers>=3.0,<4" \
    "faiss-cpu>=1.8,<2" \
    "torch>=2.1,<3"
"""
    ),
    markdown("> Restart the kernel only if an already-loaded dependency could not be upgraded, then continue from the import cell."),
    code(
        r"""
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
import joblib
import numpy as np
import pandas as pd
from botocore.config import Config
from IPython.display import display

from inference_code.retrieval import (
    RetrievalQAEngine,
    build_artifacts,
    build_corpus,
    make_candidate_id,
)

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
MODEL_ARTIFACT_OVERRIDE = ""            # Optional approved/champion model.tar.gz URI.
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 128
MAX_CORPUS_ROWS = None                   # Keep None for the complete curated subset.
RETRIEVAL_K = 50
EVIDENCE_K = 3
MINIMUM_PROBABILITY_2 = 0.35
KNOWN_QUERY_SAMPLE_SIZE = 50
MINIMUM_KNOWN_QUERY_RECALL_AT_3 = 0.40
# -------------------------------------------

session = boto3.Session(region_name=AWS_REGION_OVERRIDE or None)
region = session.region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
if not region:
    raise RuntimeError("No AWS Region is configured. Set AWS_REGION_OVERRIDE.")
retry_config = Config(retries={"max_attempts": 10, "mode": "adaptive"})
s3 = session.client("s3", config=retry_config)
sts = session.client("sts")
identity = sts.get_caller_identity()
account_id = identity["Account"]
bucket = S3_BUCKET_OVERRIDE or f"sagemaker-{region}-{account_id}"
retrieval_run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
retrieval_prefix = f"aai540/{PROJECT_NAME}/retrieval/{retrieval_run_id}"
local_dir = Path("/tmp") / f"aai540_retrieval_{retrieval_run_id}"
local_dir.mkdir(parents=True, exist_ok=True)

print(json.dumps({
    "region": region,
    "caller_arn": identity["Arn"],
    "bucket": bucket,
    "retrieval_run_id": retrieval_run_id,
    "embedding_model": EMBEDDING_MODEL_ID,
}, indent=2))
"""
    ),
    markdown("## 2 Resolve the immutable data and model lineage"),
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

def download_s3_prefix(uri, destination):
    source_bucket, prefix = split_s3_uri(uri)
    destination = Path(destination)
    downloaded = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=source_bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            if item["Key"].endswith("/"):
                continue
            relative = item["Key"][len(prefix):].lstrip("/") or Path(item["Key"]).name
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(source_bucket, item["Key"], str(target))
            downloaded.append(target)
    if not downloaded:
        raise FileNotFoundError(f"No objects found below {uri}")
    return downloaded

monitoring_receipt_uri = f"s3://{bucket}/aai540/{PROJECT_NAME}/monitoring/latest.json"
continual_receipt_uri = f"s3://{bucket}/aai540/{PROJECT_NAME}/continual-learning/latest.json"
monitoring_receipt = read_s3_json(monitoring_receipt_uri)
continual_receipt = read_s3_json(continual_receipt_uri)
modeling_receipt = read_s3_json(monitoring_receipt["modeling_receipt"])
run_summary = read_s3_json(modeling_receipt["source_run_summary"])

if continual_receipt.get("lineage", {}).get("monitoring_receipt") != monitoring_receipt_uri:
    raise ValueError(
        "The latest continual-learning receipt is not based on the latest monitoring receipt. "
        "Run stages 3 and 4 again before building retrieval artifacts."
    )
if continual_receipt.get("champion", {}).get("model_run_id") != modeling_receipt["model_run_id"]:
    raise ValueError("The continual-learning and modeling receipts disagree on the champion model")

continual_candidate = continual_receipt.get("candidate", {})
approved_continual_artifact = (
    continual_candidate.get("model_artifact")
    if continual_candidate.get("approval_status") == "Approved"
    else None
)
if MODEL_ARTIFACT_OVERRIDE:
    model_artifact_uri = MODEL_ARTIFACT_OVERRIDE
    source_model_role = "explicit approved/champion override"
elif approved_continual_artifact:
    model_artifact_uri = approved_continual_artifact
    source_model_role = "approved continual-learning candidate"
else:
    model_artifact_uri = modeling_receipt["candidate"]["model_artifact"]
    source_model_role = "existing champion"
curated_data_uri = run_summary["curated_data"]

lineage = {
    "source_data_run_id": run_summary["run_id"],
    "source_run_summary": modeling_receipt["source_run_summary"],
    "source_model_run_id": modeling_receipt["model_run_id"],
    "source_model_artifact": model_artifact_uri,
    "source_model_role": source_model_role,
    "monitor_run_id": monitoring_receipt["monitor_run_id"],
    "monitoring_receipt": monitoring_receipt_uri,
    "continual_learning_run_id": continual_receipt["continual_run_id"],
    "continual_learning_receipt": continual_receipt_uri,
    "continual_candidate_approval_status": continual_receipt["candidate"]["approval_status"],
    "curated_data": curated_data_uri,
}
print(json.dumps(lineage, indent=2))
"""
    ),
    markdown("## 3 Build the product evidence corpus"),
    code(
        r"""
curated_files = download_s3_prefix(curated_data_uri, local_dir / "curated")
parquet_files = [path for path in curated_files if path.suffix.lower() == ".parquet"]
if not parquet_files:
    raise FileNotFoundError("The curated S3 prefix contains no Parquet files")
curated = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)

model_archive = download_s3_file(model_artifact_uri, local_dir / "source-model.tar.gz")
with tarfile.open(model_archive, "r:gz") as archive:
    member = archive.getmember("model.joblib")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise FileNotFoundError("model.joblib is missing from the model artifact")
    source_model_path = local_dir / "source-model.joblib"
    source_model_path.write_bytes(extracted.read())
model_bundle = joblib.load(source_model_path)

corpus = build_corpus(curated)
if MAX_CORPUS_ROWS is not None and len(corpus) > MAX_CORPUS_ROWS:
    corpus = corpus.sort_values("candidate_id").head(MAX_CORPUS_ROWS).reset_index(drop=True)

display(pd.DataFrame({
    "metric": ["curated rows", "candidate corpus rows", "unique products", "sources"],
    "value": [len(curated), len(corpus), corpus["asin"].nunique(), corpus["source"].nunique()],
}))
display(corpus.head())
"""
    ),
    markdown("## 4 Generate MiniLM embeddings and the FAISS index"),
    code(
        r"""
runtime_dir = local_dir / "runtime"
manifest = build_artifacts(
    corpus,
    model_bundle,
    runtime_dir,
    embedding_model_id=EMBEDDING_MODEL_ID,
    batch_size=EMBEDDING_BATCH_SIZE,
)
required_runtime_files = [
    runtime_dir / "model.joblib",
    runtime_dir / "corpus.parquet",
    runtime_dir / "embeddings.npy",
    runtime_dir / "faiss.index",
    runtime_dir / "retrieval_manifest.json",
    runtime_dir / "embedding_model",
]
missing_runtime_files = [str(path) for path in required_runtime_files if not path.exists()]
if missing_runtime_files:
    raise FileNotFoundError(f"Runtime build is incomplete: {missing_runtime_files}")
print(json.dumps(manifest, indent=2))
"""
    ),
    markdown("## 5 Exercise the complete query-to-answer path"),
    code(
        r"""
engine = RetrievalQAEngine.from_model_dir(
    runtime_dir,
    retrieval_k=RETRIEVAL_K,
    evidence_k=EVIDENCE_K,
    minimum_probability_2=MINIMUM_PROBABILITY_2,
)

demo_row = curated[curated["asin"].astype(str).ne("")].iloc[0]
demo_request = {
    "request_id": f"demo-{retrieval_run_id}",
    "asin": str(demo_row["asin"]),
    "question": str(demo_row["question"]),
}
started = time.perf_counter()
demo_response = engine.answer(demo_request)
demo_latency_ms = (time.perf_counter() - started) * 1000
print(json.dumps(demo_response, indent=2))
print(f"In-kernel latency: {demo_latency_ms:.1f} ms")

assert demo_response["request_id"] == demo_request["request_id"]
assert demo_response["asin"] == demo_request["asin"]
assert isinstance(demo_response["evidence"], list)
if not demo_response["abstained"]:
    assert demo_response["answer"] == demo_response["evidence"][0]["text"]
"""
    ),
    markdown("## 6 Known-query retrieval and grounding evaluation"),
    code(
        r"""
positive = curated[
    curated["label"].astype(int).eq(2)
    & curated["asin"].fillna("").astype(str).ne("")
].drop_duplicates(["qid", "asin"]).copy()
sample_size = min(KNOWN_QUERY_SAMPLE_SIZE, len(positive))
if sample_size == 0:
    raise ValueError("No fully relevant labeled queries are available for the known-query test")
known_queries = positive.sample(n=sample_size, random_state=540)

evaluation_rows = []
for index, row in enumerate(known_queries.itertuples(index=False)):
    request_payload = {
        "request_id": f"known-{index}",
        "asin": str(row.asin),
        "question": str(row.question),
    }
    started = time.perf_counter()
    response = engine.answer(request_payload)
    latency_ms = (time.perf_counter() - started) * 1000
    expected_id = make_candidate_id(row.asin, row.candidate, row.title, row.source)
    returned_ids = [item["candidate_id"] for item in response["evidence"]]
    evaluation_rows.append({
        "request_id": request_payload["request_id"],
        "asin": request_payload["asin"],
        "expected_candidate_recalled_at_3": expected_id in returned_ids,
        "answered": not response["abstained"],
        "grounded": response["abstained"] or response["answer"] in [
            item["text"] for item in response["evidence"]
        ],
        "latency_ms": latency_ms,
    })

known_query_results = pd.DataFrame(evaluation_rows)
known_query_metrics = {
    "sample_rows": int(len(known_query_results)),
    "recall_at_3": float(known_query_results["expected_candidate_recalled_at_3"].mean()),
    "answer_rate": float(known_query_results["answered"].mean()),
    "grounded_response_rate": float(known_query_results["grounded"].mean()),
    "p50_latency_ms": float(known_query_results["latency_ms"].quantile(0.50)),
    "p95_latency_ms": float(known_query_results["latency_ms"].quantile(0.95)),
}
display(known_query_results.head(10))
print(json.dumps(known_query_metrics, indent=2))

if known_query_metrics["recall_at_3"] < MINIMUM_KNOWN_QUERY_RECALL_AT_3:
    raise RuntimeError(
        f"Known-query recall@3 {known_query_metrics['recall_at_3']:.3f} is below "
        f"{MINIMUM_KNOWN_QUERY_RECALL_AT_3:.3f}; do not package this index."
    )
if known_query_metrics["grounded_response_rate"] != 1.0:
    raise RuntimeError("At least one non-abstained answer was not copied from returned evidence")
"""
    ),
    markdown("## 7 Package and publish the portable inference runtime"),
    code(
        r'''
model_card = f"""# Product QA retrieval runtime

- Retrieval run: {retrieval_run_id}
- Data run: {run_summary['run_id']}
- Ranker run: {modeling_receipt['model_run_id']}
- Ranker source role: {source_model_role}
- Continual-learning run: {continual_receipt['continual_run_id']}
- Embedding model: {EMBEDDING_MODEL_ID}
- Corpus rows: {manifest['corpus_rows']}
- Known-query recall@3: {known_query_metrics['recall_at_3']:.4f}
- Grounded response rate: {known_query_metrics['grounded_response_rate']:.4f}

The system returns extractive evidence only and abstains below the configured fully-relevant probability threshold. Evaluation uses a labeled subset of ePQA and is not evidence of performance on all products, languages, or question types.
"""
(runtime_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")

runtime_archive = local_dir / "product-qa-runtime.tar.gz"
with tarfile.open(runtime_archive, "w:gz") as archive:
    for path in sorted(runtime_dir.rglob("*")):
        if path.is_file():
            archive.add(path, arcname=str(path.relative_to(runtime_dir)))

artifact_key = f"{retrieval_prefix}/artifacts/product-qa-runtime.tar.gz"
s3.upload_file(str(runtime_archive), bucket, artifact_key)
runtime_artifact_uri = f"s3://{bucket}/{artifact_key}"

evaluation_key = f"{retrieval_prefix}/evaluation/known_query_results.csv"
evaluation_path = local_dir / "known_query_results.csv"
known_query_results.to_csv(evaluation_path, index=False)
s3.upload_file(str(evaluation_path), bucket, evaluation_key)

print(json.dumps({
    "runtime_artifact": runtime_artifact_uri,
    "artifact_bytes": runtime_archive.stat().st_size,
    "evaluation": f"s3://{bucket}/{evaluation_key}",
}, indent=2))
'''
    ),
    markdown("## 8 Persist the retrieval receipt"),
    code(
        r"""
retrieval_receipt = {
    "schema_version": "aai540-retrieval-receipt-v1",
    "project": PROJECT_NAME,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "retrieval_run_id": retrieval_run_id,
    "lineage": lineage,
    "runtime": {
        "artifact": runtime_artifact_uri,
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "corpus_rows": manifest["corpus_rows"],
        "unique_products": manifest["unique_products"],
        "retrieval_k": RETRIEVAL_K,
        "evidence_k": EVIDENCE_K,
        "minimum_probability_2": MINIMUM_PROBABILITY_2,
    },
    "evaluation": {
        **known_query_metrics,
        "details": f"s3://{bucket}/{evaluation_key}",
        "minimum_recall_at_3": MINIMUM_KNOWN_QUERY_RECALL_AT_3,
        "passed": True,
    },
    "limitations": [
        "English ePQA subset only",
        "Extractive answers only",
        "No guarantee for products absent from the indexed corpus",
        "The random-forest reranker is not a calibrated language model",
    ],
}
receipt_body = json.dumps(retrieval_receipt, indent=2, allow_nan=False).encode("utf-8")
receipt_key = f"{retrieval_prefix}/retrieval_receipt.json"
latest_key = f"aai540/{PROJECT_NAME}/retrieval/latest.json"
for key in [receipt_key, latest_key]:
    s3.put_object(Bucket=bucket, Key=key, Body=receipt_body, ContentType="application/json")

print(json.dumps(retrieval_receipt, indent=2))
print(f"Retrieval receipt: s3://{bucket}/{receipt_key}")
print(f"Latest pointer: s3://{bucket}/{latest_key}")
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
