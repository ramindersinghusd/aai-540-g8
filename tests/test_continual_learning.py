from argparse import Namespace

import numpy as np
import pandas as pd

from src.evaluate_candidate import compare_models
from src.prepare_continual_data import FEATURE_COLUMNS, group_key, prepare_channels
from src.train import train


def make_frame(prefix: str, groups: int = 12) -> pd.DataFrame:
    rows = groups * 3
    labels = np.tile([0, 1, 2], groups)
    rng = np.random.default_rng(sum(map(ord, prefix)))
    frame = pd.DataFrame({
        "record_id": [f"{prefix}-record-{index}" for index in range(rows)],
        "asin": [f"{prefix}-product-{index // 3}" for index in range(rows)],
        "qid": [f"{prefix}-question-{index // 3}" for index in range(rows)],
        "label": labels,
    })
    for index, feature in enumerate(FEATURE_COLUMNS):
        frame[feature] = labels * (index + 1) + rng.normal(0, 0.1, rows)
    return frame


def train_bundle(tmp_path, name, train_frame, validation_frame):
    train_dir = tmp_path / name / "train"
    validation_dir = tmp_path / name / "validation"
    model_dir = tmp_path / name / "model"
    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    train_frame.to_csv(train_dir / "train.csv", index=False)
    validation_frame.to_csv(validation_dir / "validation.csv", index=False)
    return train(Namespace(
        model_type="candidate",
        random_state=540,
        n_estimators=30,
        max_depth=6,
        min_samples_leaf=1,
        model_dir=str(model_dir),
        train=str(train_dir),
        validation=str(validation_dir),
    ))


def test_prepare_channels_preserves_group_isolation():
    train_frame = make_frame("train")
    validation_frame = make_frame("validation")
    feedback_frame = make_frame("feedback")
    prepared_train, prepared_validation, receipt = prepare_channels(
        train_frame,
        validation_frame,
        feedback_frame,
        minimum_feedback_rows=10,
    )

    assert not (set(group_key(prepared_train)) & set(group_key(prepared_validation)))
    assert receipt["rows"]["feedback_total"] == len(feedback_frame)
    assert len(prepared_train) + len(prepared_validation) == (
        len(train_frame) + len(validation_frame) + len(feedback_frame)
    )


def test_candidate_evaluation_gate_accepts_equivalent_model(tmp_path):
    train_frame = make_frame("train")
    validation_frame = make_frame("validation")
    test_frame = make_frame("test")
    champion = train_bundle(tmp_path, "champion", train_frame, validation_frame)
    candidate = train_bundle(tmp_path, "candidate", train_frame, validation_frame)

    report = compare_models(
        candidate,
        champion,
        test_frame,
        minimum_macro_f1=0.0,
    )
    assert report["gate"]["passed"] == 1
    assert report["comparison"]["macro_f1_delta"] == 0.0
