"""Validate and merge newly labeled feedback into continual-learning channels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
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
REQUIRED_COLUMNS = ["record_id", "asin", "qid", "label", *FEATURE_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--feedback-validation-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-feedback-rows", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=540)
    return parser.parse_args()


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    files = [source] if source.is_file() else sorted(
        item for item in source.rglob("*")
        if item.is_file() and item.suffix.lower() in {".csv", ".parquet"}
    )
    if not files:
        raise FileNotFoundError(f"No CSV or Parquet data found at {source}")
    frames = [
        pd.read_parquet(item) if item.suffix.lower() == ".parquet" else pd.read_csv(item)
        for item in files
    ]
    return pd.concat(frames, ignore_index=True)


def group_key(frame: pd.DataFrame) -> pd.Series:
    asin = frame["asin"].fillna("").astype(str)
    qid = frame["qid"].fillna("").astype(str)
    return pd.Series(
        np.where(asin.ne("") & asin.ne("UNKNOWN"), "asin:" + asin, "qid:" + qid),
        index=frame.index,
    )


def validate_frame(name: str, frame: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} is empty")
    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(f"{name} contains nulls in required columns")
    labels = set(pd.to_numeric(frame["label"], errors="coerce").dropna().astype(int))
    if not labels or not labels.issubset({0, 1, 2}):
        raise ValueError(f"{name} contains invalid labels: {sorted(labels)}")


def stable_group_order(groups: list[str], seed: int) -> list[str]:
    return sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )


def prepare_channels(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feedback: pd.DataFrame,
    *,
    feedback_validation_fraction: float = 0.20,
    minimum_feedback_rows: int = 100,
    random_seed: int = 540,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not 0 < feedback_validation_fraction < 1:
        raise ValueError("feedback_validation_fraction must be between 0 and 1")
    for name, frame in [("train", train), ("validation", validation), ("feedback", feedback)]:
        validate_frame(name, frame)

    feedback = feedback.drop_duplicates("record_id", keep="last").copy()
    feedback["label"] = pd.to_numeric(feedback["label"], errors="coerce")
    feedback = feedback[feedback["label"].isin([0, 1, 2])].copy()
    if len(feedback) < minimum_feedback_rows:
        raise ValueError(
            f"Only {len(feedback)} valid feedback rows; require at least {minimum_feedback_rows}"
        )

    group_sets = {
        "train": set(group_key(train)),
        "validation": set(group_key(validation)),
        "feedback": set(group_key(feedback)),
    }
    for left, right in [("train", "validation"), ("train", "feedback"), ("validation", "feedback")]:
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise ValueError(f"Group leakage between {left} and {right}: {sorted(overlap)[:5]}")

    feedback = feedback.assign(_group=group_key(feedback))
    ordered_groups = stable_group_order(sorted(feedback["_group"].unique()), random_seed)
    if len(ordered_groups) < 2:
        raise ValueError("Feedback must contain at least two product/question groups")
    validation_group_count = max(1, round(len(ordered_groups) * feedback_validation_fraction))
    validation_group_count = min(validation_group_count, len(ordered_groups) - 1)
    feedback_validation_groups = set(ordered_groups[:validation_group_count])
    feedback_validation = feedback[feedback["_group"].isin(feedback_validation_groups)].drop(
        columns="_group"
    )
    feedback_train = feedback[~feedback["_group"].isin(feedback_validation_groups)].drop(
        columns="_group"
    )

    output_columns = list(dict.fromkeys(REQUIRED_COLUMNS))
    continual_train = pd.concat(
        [train[output_columns], feedback_train[output_columns]], ignore_index=True
    )
    continual_validation = pd.concat(
        [validation[output_columns], feedback_validation[output_columns]], ignore_index=True
    )
    if set(group_key(continual_train)) & set(group_key(continual_validation)):
        raise AssertionError("Prepared train and validation channels overlap by group")
    if continual_train["record_id"].duplicated().any() or continual_validation["record_id"].duplicated().any():
        raise AssertionError("Prepared channels contain duplicate record IDs")

    receipt = {
        "schema_version": "aai540-continual-data-v1",
        "random_seed": random_seed,
        "feedback_validation_fraction": feedback_validation_fraction,
        "rows": {
            "baseline_train": int(len(train)),
            "baseline_validation": int(len(validation)),
            "feedback_total": int(len(feedback)),
            "feedback_train": int(len(feedback_train)),
            "feedback_validation": int(len(feedback_validation)),
            "continual_train": int(len(continual_train)),
            "continual_validation": int(len(continual_validation)),
        },
        "groups": {
            "feedback_total": int(len(ordered_groups)),
            "feedback_train": int(feedback_train.assign(_group=group_key(feedback_train))["_group"].nunique()),
            "feedback_validation": int(feedback_validation.assign(_group=group_key(feedback_validation))["_group"].nunique()),
        },
        "feature_columns": FEATURE_COLUMNS,
    }
    return continual_train, continual_validation, receipt


def main() -> None:
    args = parse_args()
    train = read_table(args.train)
    validation = read_table(args.validation)
    feedback = read_table(args.feedback)
    continual_train, continual_validation, receipt = prepare_channels(
        train,
        validation,
        feedback,
        feedback_validation_fraction=args.feedback_validation_fraction,
        minimum_feedback_rows=args.minimum_feedback_rows,
        random_seed=args.random_seed,
    )
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    validation_dir = output_dir / "validation"
    receipt_dir = output_dir / "receipt"
    for directory in [train_dir, validation_dir, receipt_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    continual_train.to_csv(train_dir / "train.csv", index=False)
    continual_validation.to_csv(validation_dir / "validation.csv", index=False)
    (receipt_dir / "data_receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
