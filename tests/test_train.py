from argparse import Namespace

import numpy as np
import pandas as pd

from src.train import (
    BENCHMARK_FEATURES,
    CANDIDATE_FEATURES,
    input_fn,
    model_fn,
    output_fn,
    predict_fn,
    train,
)


def make_frame(prefix: str, asin_prefix: str, rows: int = 36) -> pd.DataFrame:
    rng = np.random.default_rng(sum(map(ord, prefix)))
    labels = np.tile([0, 1, 2], rows // 3)
    frame = pd.DataFrame(
        {
            "record_id": [f"{prefix}-{index}" for index in range(rows)],
            "asin": [f"{asin_prefix}-{index // 3}" for index in range(rows)],
            "qid": [f"{prefix}-q-{index // 3}" for index in range(rows)],
            "label": labels,
        }
    )
    for index, feature in enumerate(CANDIDATE_FEATURES):
        frame[feature] = labels * (index + 1) + rng.normal(0, 0.2, rows)
    frame["hash_cosine_similarity"] = labels / 2 + rng.normal(0, 0.03, rows)
    frame["token_jaccard"] = labels / 3 + rng.normal(0, 0.03, rows)
    return frame


def test_training_artifact_and_batch_inference(tmp_path):
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    train_dir.mkdir()
    validation_dir.mkdir()
    train_frame = make_frame("train", "train-product")
    validation_frame = make_frame("validation", "validation-product")
    train_frame.to_csv(train_dir / "train.csv", index=False)
    validation_frame.to_csv(validation_dir / "validation.csv", index=False)

    for model_type, features in [
        ("benchmark", BENCHMARK_FEATURES),
        ("candidate", CANDIDATE_FEATURES),
    ]:
        model_dir = tmp_path / model_type
        args = Namespace(
            model_type=model_type,
            random_state=540,
            n_estimators=20,
            max_depth=5,
            min_samples_leaf=1,
            model_dir=str(model_dir),
            train=str(train_dir),
            validation=str(validation_dir),
        )
        train(args)
        bundle = model_fn(str(model_dir))
        csv_payload = validation_frame[features].head(4).to_csv(index=False, header=False)
        parsed = input_fn(csv_payload, "text/csv")
        predictions = predict_fn(parsed, bundle)
        rendered = output_fn(predictions, "text/csv")

        assert bundle["model_type"] == model_type
        assert predictions.shape == (4, 5)
        assert set(predictions["prediction"]).issubset({0, 1, 2})
        assert len(rendered.strip().splitlines()) == 4
