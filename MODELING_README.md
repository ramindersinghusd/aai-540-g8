# SageMaker model development and deployment

> The complete six-stage runbook is in `README.md`. This file documents stage 2 only.

Use `AAI540_SageMaker_Model_Development_Deployment.ipynb` after completing the
data-lake notebook. The modeling notebook discovers the latest successful data
run (or accepts an explicit run ID), trains both models in SageMaker, evaluates
them on the untouched product-grouped test split, and deploys the candidate as
a SageMaker Batch Transform job.

## Models

- **Benchmark:** depth-two decision tree using only hashed cosine similarity
  and token Jaccard similarity.
- **First iteration:** class-balanced random forest using 14 inexpensive,
  inference-time scalar features. Gold-answer-derived features are excluded.

The primary classification metric is macro F1 because the three relevance
classes may be imbalanced. Mean per-question NDCG is also reported because the
downstream use case ranks candidate evidence for product questions.

## Run order

1. Run `AAI540_SageMaker_DataLake_FeatureStore.ipynb` in SageMaker Studio.
2. Keep its S3 artifacts, especially `run_summary.json` and the four split
   Parquet files.
3. Open and run `AAI540_SageMaker_Model_Development_Deployment.ipynb` from the
   same project directory. Review the configuration cell before starting jobs.
4. Retain the generated modeling receipt in S3 as assignment evidence. Batch
   Transform creates no always-on endpoint, so there is no endpoint to delete.

The SageMaker entry point is `src/train.py`. A local smoke test is available at
`tests/test_train.py`.
