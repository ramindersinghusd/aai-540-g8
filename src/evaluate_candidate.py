"""Compare a continual-learning candidate with the current champion."""

from __future__ import annotations

import argparse
import json
import math
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    ndcg_score,
)


CLASS_LABELS = np.array([0, 1, 2], dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--champion-model", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--minimum-macro-f1", type=float, default=0.40)
    parser.add_argument("--macro-f1-regression-tolerance", type=float, default=0.02)
    parser.add_argument("--ndcg-regression-tolerance", type=float, default=0.02)
    return parser.parse_args()


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    files = [source] if source.is_file() else sorted(
        item for item in source.rglob("*")
        if item.is_file() and item.suffix.lower() in {".csv", ".parquet"}
    )
    if not files:
        raise FileNotFoundError(f"No CSV or Parquet test data found at {source}")
    frames = [
        pd.read_parquet(item) if item.suffix.lower() == ".parquet" else pd.read_csv(item)
        for item in files
    ]
    return pd.concat(frames, ignore_index=True)


def _safe_extract_model(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember("model.joblib")
        target = (destination / member.name).resolve()
        if destination.resolve() not in target.parents:
            raise ValueError("Unsafe model archive member path")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError("model.joblib could not be read from model archive")
        target.write_bytes(extracted.read())
    return destination / "model.joblib"


def load_bundle(path: str | Path, label: str) -> dict:
    source = Path(path)
    if source.is_file() and source.name == "model.joblib":
        model_path = source
    else:
        candidates = sorted(source.rglob("model.joblib")) if source.is_dir() else []
        if candidates:
            model_path = candidates[0]
        else:
            archives = sorted(source.rglob("*.tar.gz")) if source.is_dir() else [source]
            archives = [item for item in archives if item.is_file()]
            if not archives:
                raise FileNotFoundError(f"No model.joblib or tar.gz found for {label} at {source}")
            model_path = _safe_extract_model(archives[0], Path("/tmp") / f"{label}-model")
    bundle = joblib.load(model_path)
    required = {"model", "model_type", "feature_columns", "schema_version"}
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"{label} model bundle is missing keys: {missing}")
    return bundle


def probability_matrix(model, features: pd.DataFrame) -> np.ndarray:
    observed = model.predict_proba(features)
    probabilities = np.zeros((len(features), len(CLASS_LABELS)), dtype=float)
    index = {int(label): position for position, label in enumerate(CLASS_LABELS)}
    for source_index, label in enumerate(model.classes_):
        probabilities[:, index[int(label)]] = observed[:, source_index]
    return probabilities


def mean_question_ndcg(labels, scores, qids) -> float:
    frame = pd.DataFrame({"label": labels, "score": scores, "qid": qids})
    values = []
    for _, group in frame.groupby("qid", sort=False):
        if len(group) >= 2:
            values.append(
                ndcg_score(
                    group[["label"]].to_numpy(dtype=float).T,
                    group[["score"]].to_numpy(dtype=float).T,
                )
            )
    return float(np.mean(values)) if values else 0.0


def evaluate(bundle: dict, frame: pd.DataFrame) -> dict[str, float]:
    columns = list(bundle["feature_columns"])
    missing = sorted(set(["label", "qid", *columns]) - set(frame.columns))
    if missing:
        raise ValueError(f"Evaluation data is missing columns: {missing}")
    features = frame[columns]
    labels = frame["label"].astype(int).to_numpy()
    predictions = bundle["model"].predict(features).astype(int)
    probabilities = probability_matrix(bundle["model"], features)
    expected_relevance = probabilities @ CLASS_LABELS.astype(float)
    return {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities, labels=CLASS_LABELS)),
        "mean_question_ndcg": mean_question_ndcg(labels, expected_relevance, frame["qid"]),
    }


def compare_models(
    candidate_bundle: dict,
    champion_bundle: dict,
    test_frame: pd.DataFrame,
    *,
    minimum_macro_f1: float = 0.40,
    macro_f1_regression_tolerance: float = 0.02,
    ndcg_regression_tolerance: float = 0.02,
) -> dict:
    if candidate_bundle["schema_version"] != champion_bundle["schema_version"]:
        raise ValueError("Candidate and champion feature schema versions differ")
    if candidate_bundle["feature_columns"] != champion_bundle["feature_columns"]:
        raise ValueError("Candidate and champion feature columns differ")
    candidate = evaluate(candidate_bundle, test_frame)
    champion = evaluate(champion_bundle, test_frame)
    macro_delta = candidate["macro_f1"] - champion["macro_f1"]
    ndcg_delta = candidate["mean_question_ndcg"] - champion["mean_question_ndcg"]
    reasons = []
    if candidate["macro_f1"] < minimum_macro_f1:
        reasons.append(
            f"candidate macro F1 {candidate['macro_f1']:.4f} is below {minimum_macro_f1:.4f}"
        )
    if macro_delta < -macro_f1_regression_tolerance:
        reasons.append(
            f"macro F1 delta {macro_delta:.4f} is below -{macro_f1_regression_tolerance:.4f}"
        )
    if ndcg_delta < -ndcg_regression_tolerance:
        reasons.append(
            f"NDCG delta {ndcg_delta:.4f} is below -{ndcg_regression_tolerance:.4f}"
        )
    return {
        "schema_version": "aai540-model-evaluation-v1",
        "candidate": candidate,
        "champion": champion,
        "comparison": {
            "macro_f1_delta": float(macro_delta),
            "mean_question_ndcg_delta": float(ndcg_delta),
        },
        "thresholds": {
            "minimum_macro_f1": float(minimum_macro_f1),
            "macro_f1_regression_tolerance": float(macro_f1_regression_tolerance),
            "ndcg_regression_tolerance": float(ndcg_regression_tolerance),
        },
        "gate": {"passed": int(not reasons), "reasons": reasons},
    }


def _validate_finite(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _validate_finite(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Evaluation report contains a non-finite number")


def main() -> None:
    args = parse_args()
    candidate = load_bundle(args.candidate_model, "candidate")
    champion = load_bundle(args.champion_model, "champion")
    test_frame = read_table(args.test_data)
    report = compare_models(
        candidate,
        champion,
        test_frame,
        minimum_macro_f1=args.minimum_macro_f1,
        macro_f1_regression_tolerance=args.macro_f1_regression_tolerance,
        ndcg_regression_tolerance=args.ndcg_regression_tolerance,
    )
    _validate_finite(report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"candidate:macro_f1={report['candidate']['macro_f1']:.8f}", flush=True)
    print(f"candidate:mean_question_ndcg={report['candidate']['mean_question_ndcg']:.8f}", flush=True)
    print(f"gate:passed={report['gate']['passed']:.8f}", flush=True)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
