"""Train and serve the product-QA benchmark and candidate classifiers.

This module is used as the SageMaker Scikit-learn entry point.  It intentionally
keeps feature selection inside the model artifact so training, evaluation, and
batch inference all use the same ordered schema.
"""

from __future__ import annotations

import argparse
import json
import os
from io import StringIO
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    ndcg_score,
)
from sklearn.tree import DecisionTreeClassifier


LABEL_COLUMN = "label"
ID_COLUMNS = ["record_id", "asin", "qid"]
CLASS_LABELS = np.array([0, 1, 2], dtype=int)

# A deliberately small benchmark: two inexpensive lexical-similarity features
# and a depth-two decision tree.
BENCHMARK_FEATURES = [
    "hash_cosine_similarity",
    "token_jaccard",
]

# These features are available from the question, candidate, and product title
# at inference time.  Gold-answer-derived fields are deliberately excluded.
CANDIDATE_FEATURES = [
    "question_char_count",
    "candidate_char_count",
    "title_char_count",
    "hash_cosine_similarity",
    "question_word_count",
    "candidate_word_count",
    "title_word_count",
    "token_overlap_count",
    "token_jaccard",
    "number_overlap_count",
    "question_has_number",
    "candidate_has_number",
    "candidate_in_title",
    "missing_title",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["benchmark", "candidate"], required=True)
    parser.add_argument("--random-state", type=int, default=540)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument(
        "--train",
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    parser.add_argument(
        "--validation",
        default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"),
    )
    return parser.parse_args()


def read_channel(channel_path: str | Path) -> pd.DataFrame:
    """Read and concatenate header-bearing CSV files from a SageMaker channel."""
    path = Path(channel_path)
    files = [path] if path.is_file() else sorted(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in training channel: {path}")
    frames = [pd.read_csv(file_path) for file_path in files]
    return pd.concat(frames, ignore_index=True)


def group_key(frame: pd.DataFrame) -> pd.Series:
    asin = frame["asin"].fillna("").astype(str)
    qid = frame["qid"].fillna("").astype(str)
    return np.where(asin.ne("") & asin.ne("UNKNOWN"), "asin:" + asin, "qid:" + qid)


def validate_training_data(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: Iterable[str],
) -> None:
    required = set(ID_COLUMNS + [LABEL_COLUMN] + list(feature_columns))
    for name, frame in [("train", train), ("validation", validation)]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} data is missing required columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} data is empty")
        if frame[list(feature_columns)].isna().any().any():
            raise ValueError(f"{name} feature data contains missing values")
        labels = set(pd.to_numeric(frame[LABEL_COLUMN], errors="coerce").dropna().astype(int))
        if not labels or not labels.issubset(set(CLASS_LABELS)):
            raise ValueError(f"{name} contains invalid labels: {sorted(labels)}")

    duplicate_records = set(train["record_id"]) & set(validation["record_id"])
    if duplicate_records:
        raise ValueError("Train/validation record_id leakage detected")
    overlapping_groups = set(group_key(train)) & set(group_key(validation))
    if overlapping_groups:
        raise ValueError("Train/validation product or fallback-question leakage detected")


def probability_matrix(model, features: pd.DataFrame) -> np.ndarray:
    """Return P(class) in stable 0, 1, 2 column order."""
    observed = model.predict_proba(features)
    probabilities = np.zeros((len(features), len(CLASS_LABELS)), dtype=float)
    class_to_index = {int(label): index for index, label in enumerate(CLASS_LABELS)}
    for source_index, label in enumerate(model.classes_):
        probabilities[:, class_to_index[int(label)]] = observed[:, source_index]
    return probabilities


def mean_group_ndcg(
    labels: Iterable[int],
    scores: Iterable[float],
    qids: Iterable[str],
) -> float:
    ranking_frame = pd.DataFrame({"label": labels, "score": scores, "qid": qids})
    group_scores = []
    for _, group in ranking_frame.groupby("qid", sort=False):
        if len(group) < 2:
            continue
        group_scores.append(
            ndcg_score(
                group[["label"]].to_numpy(dtype=float).T,
                group[["score"]].to_numpy(dtype=float).T,
            )
        )
    # A zero fallback keeps metrics JSON standards-compliant for an unusual
    # input where every qid has only one candidate (the ePQA data has groups).
    return float(np.mean(group_scores)) if group_scores else 0.0


def evaluate_model(model, frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    features = frame[feature_columns]
    labels = frame[LABEL_COLUMN].astype(int).to_numpy()
    predictions = model.predict(features).astype(int)
    probabilities = probability_matrix(model, features)
    expected_relevance = probabilities @ CLASS_LABELS.astype(float)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=CLASS_LABELS)),
        "mean_question_ndcg": mean_group_ndcg(labels, expected_relevance, frame["qid"]),
    }


def build_model(args: argparse.Namespace):
    if args.model_type == "benchmark":
        return DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=max(2, args.min_samples_leaf),
            class_weight="balanced",
            random_state=args.random_state,
        )
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.random_state,
    )


def train(args: argparse.Namespace) -> dict:
    train_frame = read_channel(args.train)
    validation_frame = read_channel(args.validation)
    feature_columns = BENCHMARK_FEATURES if args.model_type == "benchmark" else CANDIDATE_FEATURES
    validate_training_data(train_frame, validation_frame, feature_columns)

    model = build_model(args)
    model.fit(train_frame[feature_columns], train_frame[LABEL_COLUMN].astype(int))
    metrics = {
        "train": evaluate_model(model, train_frame, feature_columns),
        "validation": evaluate_model(model, validation_frame, feature_columns),
    }

    # These stable lines are captured by the metric regexes in the notebook.
    for split_name, split_metrics in metrics.items():
        for metric_name, value in split_metrics.items():
            print(f"{split_name}:{metric_name}={value:.8f}", flush=True)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "model_type": args.model_type,
        "feature_columns": feature_columns,
        "class_labels": CLASS_LABELS.tolist(),
        "schema_version": "epqa-scalar-features-v1",
        "metrics": metrics,
    }
    joblib.dump(bundle, model_dir / "model.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (model_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "schema_version": bundle["schema_version"],
                "model_type": args.model_type,
                "feature_columns": feature_columns,
                "class_labels": CLASS_LABELS.tolist(),
            },
            indent=2,
        )
    )
    return bundle


def model_fn(model_dir: str):
    return joblib.load(Path(model_dir) / "model.joblib")


def input_fn(request_body, content_type: str):
    if content_type.split(";", 1)[0].strip().lower() != "text/csv":
        raise ValueError(f"Unsupported content type: {content_type}; expected text/csv")
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    # Batch input has no header; model feature order is applied in predict_fn.
    return pd.read_csv(StringIO(request_body), header=None)


def predict_fn(input_data: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    feature_columns = bundle["feature_columns"]
    if input_data.shape[1] != len(feature_columns):
        raise ValueError(
            f"Expected {len(feature_columns)} columns in model order, got {input_data.shape[1]}"
        )
    input_data = input_data.copy()
    input_data.columns = feature_columns
    model = bundle["model"]
    predictions = model.predict(input_data).astype(int)
    probabilities = probability_matrix(model, input_data)
    expected_relevance = probabilities @ CLASS_LABELS.astype(float)
    return pd.DataFrame(
        {
            "prediction": predictions,
            "expected_relevance": expected_relevance,
            "probability_0": probabilities[:, 0],
            "probability_1": probabilities[:, 1],
            "probability_2": probabilities[:, 2],
        }
    )


def output_fn(prediction: pd.DataFrame, accept: str):
    if accept.split(";", 1)[0].strip().lower() != "text/csv":
        raise ValueError(f"Unsupported accept type: {accept}; expected text/csv")
    return prediction.to_csv(index=False, header=False, float_format="%.8f")


if __name__ == "__main__":
    train(parse_args())
