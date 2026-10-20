import numpy as np
import pandas as pd

from inference_code.inference import input_fn, output_fn
from inference_code.retrieval import (
    CANDIDATE_FEATURES,
    RetrievalQAEngine,
    build_corpus,
    scalar_features,
)


class FakeEmbedder:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.array([
                float("red" in lowered),
                float("blue" in lowered),
                float("battery" in lowered),
            ])
            if not vector.any():
                vector[:] = 1
            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        return np.asarray(vectors, dtype="float32")


class FakeRanker:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, features):
        similarity = np.clip(features["token_jaccard"].to_numpy(dtype=float), 0, 1)
        probability_2 = 0.4 + 0.5 * similarity
        probability_0 = 0.4 * (1 - similarity)
        probability_1 = 1 - probability_0 - probability_2
        return np.column_stack([probability_0, probability_1, probability_2])


def test_scalar_feature_contract():
    features = scalar_features("Is this red?", "This item is red.", "Red item")
    assert list(features) == CANDIDATE_FEATURES
    assert features["token_overlap_count"] >= 1
    assert 0 <= features["hash_cosine_similarity"] <= 1


def test_json_inference_contract():
    payload = input_fn(
        '{"request_id":"r1","asin":"A1","question":"Is it red?"}',
        "application/json",
    )
    assert isinstance(payload, list) and payload[0]["asin"] == "A1"
    rendered = output_fn([{"answer": "red", "abstained": False}], "application/json")
    assert '"answer": "red"' in rendered


def test_retrieval_returns_grounded_evidence_and_abstains_unknown_product():
    curated = pd.DataFrame([
        {"asin": "A1", "candidate": "The product is red.", "title": "Red widget", "source": "qa"},
        {"asin": "A1", "candidate": "The battery lasts ten hours.", "title": "Red widget", "source": "review"},
        {"asin": "B2", "candidate": "The product is blue.", "title": "Blue widget", "source": "qa"},
    ])
    corpus = build_corpus(curated)
    embedder = FakeEmbedder()
    embeddings = embedder.encode(
        (corpus["title"] + " [SEP] " + corpus["candidate"]).tolist()
    )
    bundle = {
        "model": FakeRanker(),
        "feature_columns": CANDIDATE_FEATURES,
        "schema_version": "epqa-scalar-features-v1",
    }
    engine = RetrievalQAEngine(
        corpus,
        embeddings,
        bundle,
        embedder,
        minimum_probability_2=0.35,
    )

    result = engine.answer({"request_id": "r1", "asin": "A1", "question": "Is it red?"})
    assert result["abstained"] is False
    assert result["answer"] == result["evidence"][0]["text"]
    assert result["evidence"][0]["candidate_id"] in set(corpus["candidate_id"])

    missing = engine.answer({"request_id": "r2", "asin": "UNKNOWN-ASIN", "question": "Is it red?"})
    assert missing["abstained"] is True
    assert missing["evidence"] == []
