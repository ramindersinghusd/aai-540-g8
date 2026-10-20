import numpy as np
import pandas as pd

from src.monitor import CANDIDATE_FEATURES, generate_report


def make_scored_frame(rows=120, seed=540):
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1, 2], rows // 3)
    frame = pd.DataFrame({
        "record_id": [f"record-{seed}-{index}" for index in range(rows)],
        "qid": [f"question-{index // 3}" for index in range(rows)],
        "asin": [f"product-{index // 6}" for index in range(rows)],
        "label": labels,
        "prediction": labels,
        "expected_relevance": labels + rng.normal(0, 0.05, rows),
        "probability_0": np.where(labels == 0, 0.8, 0.1),
        "probability_1": np.where(labels == 1, 0.8, 0.1),
        "probability_2": np.where(labels == 2, 0.8, 0.1),
    })
    for index, feature in enumerate(CANDIDATE_FEATURES):
        frame[feature] = rng.normal(index, 1, rows)
    return frame


def test_stable_batch_does_not_trigger_retraining():
    baseline = make_scored_frame(seed=540)
    current = baseline.copy()
    current["record_id"] = [f"current-{index}" for index in range(len(current))]
    report, drift = generate_report(baseline, current)

    assert report["decision"]["retraining_recommended"] is False
    assert report["drift"]["max_feature_psi"] == 0.0
    assert report["quality_change"]["macro_f1_delta"] == 0.0
    assert len(drift) == len(CANDIDATE_FEATURES)


def test_feature_drift_triggers_retraining():
    baseline = make_scored_frame(seed=540)
    current = make_scored_frame(seed=541)
    current[CANDIDATE_FEATURES[0]] += 50
    report, _ = generate_report(baseline, current)

    assert report["decision"]["retraining_recommended"] is True
    assert report["drift"]["max_feature_psi"] > report["thresholds"]["psi"]
    assert any("PSI" in reason for reason in report["decision"]["reasons"])
