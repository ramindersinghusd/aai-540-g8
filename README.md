# AAI 540 Product QA MLOps project

This repository contains a six-stage, SageMaker-ready implementation of the Retrieval-Augmented Product Question Answering project. The completed MVP prepares ePQA data, trains and evaluates a ranker, scores and monitors production-style data, creates gated continual-learning candidates, builds a MiniLM/FAISS evidence index, and serves grounded extractive answers through a temporary SageMaker endpoint and Flask API contract.

No notebook embeds AWS credentials. Every stage writes a versioned JSON receipt to S3, and each downstream stage discovers and validates the upstream receipt before creating AWS resources.

## Run order

Run the notebooks from the repository root in this exact order:

1. `AAI540_SageMaker_DataLake_FeatureStore.ipynb`
2. `AAI540_SageMaker_Model_Development_Deployment.ipynb`
3. `AAI540_SageMaker_ML_System_Observability.ipynb`
4. `AAI540_SageMaker_Continual_Learning_CICD.ipynb`
5. `AAI540_SageMaker_Retrieval_RAG_Inference.ipynb`
6. `AAI540_SageMaker_Application_Integration_Testing.ipynb`

Upload the complete project directory to SageMaker Studio—not only the six notebook files. The notebooks require `src/`, `scripts/`, `tests/`, `app/`, and `inference_code/`. Stages 2-6 automatically search the current directory, its parents, and `/home/sagemaker-user`, then print the resolved project root. If the project is stored elsewhere, set the `AAI540_PROJECT_ROOT` environment variable before the import cell.

Use one SageMaker Studio/JupyterLab kernel, AWS account, Region, project name, and S3 bucket for all six. Review each notebook's user-editable configuration cell before running it. The defaults use the current AWS session Region and `sagemaker-<region>-<account-id>` bucket.

## Stage contracts

| Stage | Main AWS services | Required upstream input | Versioned output |
|---|---|---|---|
| Data lake and features | S3, Glue, Athena, Feature Store | Official ePQA CSV files | `runs/<run-id>/run_summary.json` plus four group-isolated Parquet splits |
| Model development | SageMaker Training, Batch Transform, S3 | `run_summary.json` | `modeling/<run-id>/modeling_receipt.json`, candidate artifact, production predictions |
| Observability | SageMaker Processing, CloudWatch, S3 | Modeling receipt and predictions | `monitoring/<run-id>/monitoring_receipt.json`, detailed drift report, dashboard, alarms |
| Continual learning | SageMaker Pipelines, Processing, Training, optional Model Registry | All prior receipts plus labeled feedback | `continual-learning/<run-id>/continual_learning_receipt.json` and a gated pending candidate |
| Retrieval and grounded inference | MiniLM, FAISS, S3 | Curated corpus, champion ranker, monitoring receipt, and continual-learning receipt | `retrieval/<run-id>/retrieval_receipt.json` and portable runtime artifact |
| Application integration | Temporary SageMaker endpoint, Flask, CloudWatch, S3 | Retrieval runtime | `integration/<run-id>/application_integration_receipt.json`, sanitized telemetry, and test evidence |

Stages 3-6 update a `latest.json` state pointer in their respective S3 prefix. Run-scoped execution, approval, rollback, and integration receipts remain versioned audit evidence; the pointer reflects the most recent reviewed state.

## What the final workflow enforces

- Product/ASIN grouping, with qid fallback, prevents cross-split leakage.
- Training and inference share the same ordered 14-feature schema stored in the model bundle.
- Gold-answer-derived features are excluded from inference.
- Benchmark and candidate models are compared using macro F1 and per-question NDCG.
- Batch Transform is used instead of an idle endpoint for the cost-conscious MVP.
- Monitoring computes missing/invalid rates, feature and prediction PSI, confidence, macro-F1 change, and NDCG change.
- PSI above `0.20`, missing rate above `0.02`, or quality degradation beyond `0.05` recommends retraining.
- New labeled production feedback is split deterministically by product/question group before being appended to training and validation.
- The continual candidate must meet an absolute macro-F1 floor and remain within declared macro-F1/NDCG regression tolerances.
- A passing model is only `PendingManualApproval`; the champion never changes without an explicit reviewer action.
- If the reviewer runs the approval cell, stage 4 updates its immutable/latest receipts and stage 5 automatically adopts that approved candidate; otherwise it retains the prior champion. `MODEL_ARTIFACT_OVERRIDE` remains available for an explicitly reviewed model artifact.
- Retrieval refuses to build unless the latest continual-learning, monitoring, and champion-model lineage is consistent.
- MiniLM embeddings and a FAISS inner-product index retrieve product-specific evidence.
- The supervised ranker reranks semantically retrieved candidates with the exact training feature order.
- Answers are extractive copies of returned evidence, with an explicit low-confidence abstention path.
- The final notebook tests the SageMaker HTTPS inference contract, Flask health/answer routes, unknown-product behavior, grounding, p50/p95 latency, telemetry, and cleanup.

## Permissions and Learner Lab fallbacks

The execution role normally needs S3, Glue, Athena, SageMaker Feature Store, Training, Processing, Batch Transform, Pipelines, Model Registry, CloudWatch, ECR pull, and `iam:PassRole` permissions for the selected role.

The design anticipates restricted AWS Academy roles:

- The observability notebook uses a custom Processing monitor rather than depending on SageMaker Model Monitor. Set `RUN_PROCESSING_JOB=False` only if Processing jobs are unavailable; the identical script runs in the notebook kernel and publishes the same S3 report.
- Set `ORCHESTRATION_MODE="scripted"` in stage 4 if SageMaker Pipelines is unavailable. The same preparation, training, evaluation, and gate scripts run as a controlled sequence.
- Set `ENABLE_MODEL_REGISTRY=False` if Model Registry is unavailable. A versioned S3 promotion-candidate record is written with `PendingManualApproval` semantics.
- CloudWatch publication can be disabled with `CREATE_CLOUDWATCH_RESOURCES=False`, although doing so omits the dashboard/alarm deliverable.
- Set `DEPLOY_TEMPORARY_ENDPOINT=False` in stage 6 if endpoint creation is denied. The same runtime and Flask contracts execute in-process; the receipt records that fallback.

Permission failures are not silently ignored. Change a fallback only after confirming that the corresponding Learner Lab action is denied.

## Validation performed locally

The project includes tests for:

- benchmark/candidate training artifacts and batch inference;
- stable and drifted monitoring batches;
- deterministic continual-data group isolation; and
- candidate-versus-champion release gating;
- training/inference scalar-feature parity and JSON serving handlers;
- semantic retrieval, evidence grounding, and unknown-product abstention; and
- Flask health, validation, and answer-route contracts.

Run them in SageMaker after the install cell has completed:

```bash
python -m pytest -q
```

All Python modules and every non-magic code cell in all six notebooks have also been syntax-checked. AWS-managed jobs cannot be executed outside the user's account, so the final cloud validation is the ordered notebook run itself. Every AWS job is followed by an explicit terminal-status, row-count, schema, or receipt assertion.

## Cost and cleanup

- Feature Store is offline-only; its ingestion is the longest part of stage 1.
- Training, Processing, and Batch Transform instances terminate when their jobs complete.
- No always-on endpoint is created.
- The data notebook can delete the feature-group metadata without deleting S3 evidence.
- The observability notebook provides an explicit CloudWatch dashboard/alarm cleanup switch.
- The final notebook retains an application dashboard and three action-disabled alarms while deleting the temporary endpoint, endpoint configuration, and model.
- S3 evidence is retained intentionally; delete versioned prefixes only after submission or demonstration evidence has been captured.

## Source layout

- `src/train.py`: shared SageMaker training and batch-inference contract.
- `src/monitor.py`: portable observability calculations and report writer.
- `src/prepare_continual_data.py`: feedback validation, deterministic group split, and channel creation.
- `src/evaluate_candidate.py`: champion comparison and release gate.
- `src/record_promotion.py`: S3 fallback when Model Registry is unavailable.
- `inference_code/retrieval.py`: shared semantic retrieval, feature parity, reranking, evidence, and abstention runtime.
- `inference_code/inference.py`: SageMaker JSON inference handlers.
- `app/flask_app.py`: health and answer API contract around the inference backend.
- `scripts/build_*_notebook.py`: reviewable sources for generated notebooks.
- `tests/`: local unit and smoke tests.
