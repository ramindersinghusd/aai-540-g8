"""Generate data-drift and delayed-label quality reports for product-QA batches.

The module is intentionally independent of SageMaker APIs.  It can run inside a
SageMaker Processing job or locally, while the notebook owns S3 and CloudWatch
publication.  Baseline and current inputs are scored row-level CSV/Parquet files
with the same inference-time feature schema.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, ndcg_score


CLASS_LABELS = np.array([0, 1, 2], dtype=int)
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
PROBABILITY_COLUMNS = ["probability_0", "probability_1", "probability_2"]
SCORED_COLUMNS = ["prediction", "expected_relevance", *PROBABILITY_COLUMNS]
IDENTIFIER_COLUMNS = ["record_id", "qid", "asin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--psi-threshold", type=float, default=0.20)
    parser.add_argument("--missing-rate-threshold", type=float, default=0.02)
    parser.add_argument("--quality-drop-threshold", type=float, default=0.05)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.60)
    return parser.parse_args()


def read_table(path: str | Path) -> pd.DataFrame:
    """Read one file or concatenate all supported files below a directory."""
    source = Path(path)
    if source.is_file():
        files = [source]
    else:
        files = sorted(
            candidate
            for candidate in source.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in {".csv", ".parquet"}
        )
    if not files:
        raise FileNotFoundError(f"No CSV or Parquet files found at {source}")
    frames = []
    for file_path in files:
        if file_path.suffix.lower() == ".parquet":
            frames.append(pd.read_parquet(file_path))
        else:
            frames.append(pd.read_csv(file_path))
    return pd.concat(frames, ignore_index=True)


def _finite_numeric(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def population_stability_index(
    baseline: Iterable[float], current: Iterable[float], bins: int = 10
) -> float | None:
    """Compute PSI with baseline quantile bins and small-count smoothing."""
    expected = _finite_numeric(pd.Series(baseline))
    actual = _finite_numeric(pd.Series(current))
    if not len(expected) or not len(actual):
        return None

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) == 1:
        center = float(edges[0])
        width = max(abs(center) * 0.01, 1e-6)
        edges = np.array([center - width, center + width])
    edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    epsilon = 1e-6
    expected_pct = np.clip(expected_counts / expected_counts.sum(), epsilon, None)
    actual_pct = np.clip(actual_counts / actual_counts.sum(), epsilon, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def categorical_psi(
    baseline: Iterable[object], current: Iterable[object]
) -> float | None:
    expected = pd.Series(baseline).fillna("__MISSING__").astype(str)
    actual = pd.Series(current).fillna("__MISSING__").astype(str)
    if expected.empty or actual.empty:
        return None
    categories = sorted(set(expected) | set(actual))
    epsilon = 1e-6
    expected_pct = expected.value_counts(normalize=True).reindex(categories, fill_value=0)
    actual_pct = actual.value_counts(normalize=True).reindex(categories, fill_value=0)
    expected_pct = expected_pct.clip(lower=epsilon)
    actual_pct = actual_pct.clip(lower=epsilon)
    return float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())


def _mean_question_ndcg(frame: pd.DataFrame) -> float | None:
    if not {"label", "expected_relevance", "qid"}.issubset(frame.columns):
        return None
    valid = frame.copy()
    valid["label"] = pd.to_numeric(valid["label"], errors="coerce")
    valid["expected_relevance"] = pd.to_numeric(valid["expected_relevance"], errors="coerce")
    valid = valid[
        valid["label"].isin(CLASS_LABELS)
        & np.isfinite(valid["expected_relevance"])
        & valid["qid"].notna()
    ]
    scores = []
    for _, group in valid.groupby("qid", sort=False):
        if len(group) >= 2:
            scores.append(
                ndcg_score(
                    group[["label"]].to_numpy(dtype=float).T,
                    group[["expected_relevance"]].to_numpy(dtype=float).T,
                )
            )
    return float(np.mean(scores)) if scores else None


def labeled_quality(frame: pd.DataFrame) -> dict[str, float] | None:
    required = {"label", "prediction", *PROBABILITY_COLUMNS}
    if not required.issubset(frame.columns):
        return None
    work = frame.copy()
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    work["prediction"] = pd.to_numeric(work["prediction"], errors="coerce")
    for column in PROBABILITY_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    valid = work[
        work["label"].isin(CLASS_LABELS)
        & work["prediction"].isin(CLASS_LABELS)
        & np.isfinite(work[PROBABILITY_COLUMNS]).all(axis=1)
    ]
    if valid.empty:
        return None
    labels = valid["label"].astype(int).to_numpy()
    predictions = valid["prediction"].astype(int).to_numpy()
    probabilities = valid[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        probabilities,
        row_sums,
        out=np.full_like(probabilities, 1 / len(CLASS_LABELS)),
        where=row_sums > 0,
    )
    result = {
        "rows_with_labels": int(len(valid)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "log_loss": float(log_loss(labels, probabilities, labels=CLASS_LABELS)),
    }
    ndcg = _mean_question_ndcg(valid)
    if ndcg is not None:
        result["mean_question_ndcg"] = ndcg
    return result


def _data_quality(frame: pd.DataFrame, required_columns: list[str]) -> dict:
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    present = [column for column in required_columns if column in frame.columns]
    missing_cells = int(frame[present].isna().sum().sum()) if present else len(frame)
    denominator = max(len(frame) * max(len(required_columns), 1), 1)
    numeric_columns = [
        column
        for column in [*CANDIDATE_FEATURES, *SCORED_COLUMNS]
        if column in frame.columns
    ]
    invalid_numeric = 0
    for column in numeric_columns:
        converted = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        invalid_numeric += int((~np.isfinite(converted)).sum())
    numeric_denominator = max(len(frame) * max(len(numeric_columns), 1), 1)
    duplicate_rate = (
        float(frame["record_id"].duplicated().mean()) if "record_id" in frame and len(frame) else 0.0
    )
    return {
        "rows": int(len(frame)),
        "missing_columns": missing_columns,
        "missing_rate": float((missing_cells + len(frame) * len(missing_columns)) / denominator),
        "invalid_numeric_rate": float(invalid_numeric / numeric_denominator),
        "duplicate_record_rate": duplicate_rate,
    }


def _confidence_summary(frame: pd.DataFrame, threshold: float) -> dict[str, float] | None:
    if not set(PROBABILITY_COLUMNS).issubset(frame.columns):
        return None
    probabilities = frame[PROBABILITY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    confidence = probabilities.max(axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if confidence.empty:
        return None
    return {
        "mean_confidence": float(confidence.mean()),
        "p10_confidence": float(confidence.quantile(0.10)),
        "low_confidence_rate": float(confidence.lt(threshold).mean()),
    }


def generate_report(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    *,
    psi_threshold: float = 0.20,
    missing_rate_threshold: float = 0.02,
    quality_drop_threshold: float = 0.05,
    low_confidence_threshold: float = 0.60,
) -> tuple[dict, pd.DataFrame]:
    required_columns = [*IDENTIFIER_COLUMNS, *CANDIDATE_FEATURES, *SCORED_COLUMNS]
    baseline_quality = _data_quality(baseline, required_columns)
    current_quality = _data_quality(current, required_columns)

    drift_rows = []
    for feature in CANDIDATE_FEATURES:
        value = None
        if feature in baseline.columns and feature in current.columns:
            value = population_stability_index(baseline[feature], current[feature])
        drift_rows.append({"feature": feature, "psi": value})
    drift_frame = pd.DataFrame(drift_rows)
    finite_psi = drift_frame["psi"].dropna().astype(float)
    max_feature_psi = float(finite_psi.max()) if not finite_psi.empty else None
    mean_feature_psi = float(finite_psi.mean()) if not finite_psi.empty else None
    prediction_psi = (
        categorical_psi(baseline["prediction"], current["prediction"])
        if "prediction" in baseline and "prediction" in current
        else None
    )

    baseline_labeled = labeled_quality(baseline)
    current_labeled = labeled_quality(current)
    quality_delta = None
    ndcg_delta = None
    if baseline_labeled and current_labeled:
        quality_delta = current_labeled["macro_f1"] - baseline_labeled["macro_f1"]
        if "mean_question_ndcg" in baseline_labeled and "mean_question_ndcg" in current_labeled:
            ndcg_delta = (
                current_labeled["mean_question_ndcg"]
                - baseline_labeled["mean_question_ndcg"]
            )

    reasons = []
    if current_quality["missing_columns"]:
        reasons.append("required monitoring columns are missing")
    if current_quality["missing_rate"] > missing_rate_threshold:
        reasons.append(
            f"missing rate {current_quality['missing_rate']:.4f} exceeds {missing_rate_threshold:.4f}"
        )
    if current_quality["invalid_numeric_rate"] > 0:
        reasons.append("current batch contains non-finite numeric values")
    if max_feature_psi is not None and max_feature_psi > psi_threshold:
        reasons.append(f"maximum feature PSI {max_feature_psi:.4f} exceeds {psi_threshold:.4f}")
    if prediction_psi is not None and prediction_psi > psi_threshold:
        reasons.append(f"prediction PSI {prediction_psi:.4f} exceeds {psi_threshold:.4f}")
    if quality_delta is not None and quality_delta < -quality_drop_threshold:
        reasons.append(
            f"macro F1 changed by {quality_delta:.4f}, below -{quality_drop_threshold:.4f}"
        )
    if ndcg_delta is not None and ndcg_delta < -quality_drop_threshold:
        reasons.append(
            f"mean question NDCG changed by {ndcg_delta:.4f}, below -{quality_drop_threshold:.4f}"
        )

    report = {
        "schema_version": "aai540-monitoring-v1",
        "thresholds": {
            "psi": float(psi_threshold),
            "missing_rate": float(missing_rate_threshold),
            "quality_drop": float(quality_drop_threshold),
            "low_confidence": float(low_confidence_threshold),
        },
        "baseline": {
            "data_quality": baseline_quality,
            "confidence": _confidence_summary(baseline, low_confidence_threshold),
            "labeled_quality": baseline_labeled,
        },
        "current": {
            "data_quality": current_quality,
            "confidence": _confidence_summary(current, low_confidence_threshold),
            "labeled_quality": current_labeled,
        },
        "drift": {
            "max_feature_psi": max_feature_psi,
            "mean_feature_psi": mean_feature_psi,
            "prediction_psi": prediction_psi,
        },
        "quality_change": {
            "macro_f1_delta": quality_delta,
            "mean_question_ndcg_delta": ndcg_delta,
        },
        "decision": {
            "retraining_recommended": bool(reasons),
            "reasons": reasons,
        },
    }
    return report, drift_frame


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    baseline = read_table(args.baseline)
    current = read_table(args.current)
    report, drift_frame = generate_report(
        baseline,
        current,
        psi_threshold=args.psi_threshold,
        missing_rate_threshold=args.missing_rate_threshold,
        quality_drop_threshold=args.quality_drop_threshold,
        low_confidence_threshold=args.low_confidence_threshold,
    )
    report = _json_safe(report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "monitoring_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    drift_frame.to_csv(output_dir / "feature_drift.csv", index=False)

    metric_values = {
        "current:missing_rate": report["current"]["data_quality"]["missing_rate"],
        "drift:max_feature_psi": report["drift"]["max_feature_psi"],
        "drift:prediction_psi": report["drift"]["prediction_psi"],
        "quality:macro_f1_delta": report["quality_change"]["macro_f1_delta"],
        "decision:retraining_recommended": float(
            report["decision"]["retraining_recommended"]
        ),
    }
    for name, value in metric_values.items():
        if value is not None:
            print(f"{name}={float(value):.8f}", flush=True)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
