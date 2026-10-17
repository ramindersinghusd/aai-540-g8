"""Semantic retrieval plus scalar-feature reranking for Product QA."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


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
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(value: Any) -> str:
    text = CONTROL_PATTERN.sub(" ", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text).lower())


def hashing_cosine(question: str, candidate: str, dimensions: int = 2**14) -> float:
    """Match the notebook's unsigned unigram/bigram HashingVectorizer cosine."""
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(
        n_features=dimensions,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.transform([question, candidate])
    return float(matrix[0].multiply(matrix[1]).sum())


def scalar_features(question: str, candidate: str, title: str) -> dict[str, float | int]:
    question = normalize_text(question)
    candidate = normalize_text(candidate)
    title = normalize_text(title)
    question_tokens = tokens(question)
    candidate_tokens = tokens(candidate)
    title_tokens = tokens(title)
    question_set, candidate_set = set(question_tokens), set(candidate_tokens)
    intersection = question_set & candidate_set
    union = question_set | candidate_set
    question_numbers = set(NUMBER_PATTERN.findall(question))
    candidate_numbers = set(NUMBER_PATTERN.findall(candidate))
    features = {
        "question_char_count": len(question),
        "candidate_char_count": len(candidate),
        "title_char_count": len(title),
        "hash_cosine_similarity": hashing_cosine(question, candidate),
        "question_word_count": len(question_tokens),
        "candidate_word_count": len(candidate_tokens),
        "title_word_count": len(title_tokens),
        "token_overlap_count": len(intersection),
        "token_jaccard": len(intersection) / max(len(union), 1),
        "number_overlap_count": len(question_numbers & candidate_numbers),
        "question_has_number": int(bool(question_numbers)),
        "candidate_has_number": int(bool(candidate_numbers)),
        "candidate_in_title": int(bool(candidate) and candidate.lower() in title.lower()),
        "missing_title": int(not bool(title)),
    }
    return {name: features[name] for name in CANDIDATE_FEATURES}


def make_candidate_id(asin: str, candidate: str, title: str, source: str) -> str:
    material = "\x1f".join(map(normalize_text, [asin, candidate, title, source]))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def build_corpus(curated: pd.DataFrame) -> pd.DataFrame:
    required = {"asin", "candidate", "title", "source"}
    missing = sorted(required - set(curated.columns))
    if missing:
        raise ValueError(f"Curated data is missing corpus columns: {missing}")
    corpus = curated[["asin", "candidate", "title", "source"]].copy()
    for column in corpus.columns:
        corpus[column] = corpus[column].map(normalize_text)
    corpus["asin"] = corpus["asin"].replace("", "UNKNOWN")
    corpus = corpus[corpus["candidate"].ne("")].drop_duplicates(
        ["asin", "candidate", "title", "source"]
    )
    corpus.insert(
        0,
        "candidate_id",
        corpus.apply(
            lambda row: make_candidate_id(row.asin, row.candidate, row.title, row.source),
            axis=1,
        ),
    )
    corpus = corpus.drop_duplicates("candidate_id").reset_index(drop=True)
    if corpus.empty:
        raise ValueError("Candidate corpus is empty")
    return corpus


def build_artifacts(
    corpus: pd.DataFrame,
    model_bundle: dict,
    output_dir: str | Path,
    *,
    embedding_model_id: str,
    batch_size: int = 128,
) -> dict:
    """Encode the corpus, build FAISS, save the embedder, and write a manifest."""
    import faiss
    from sentence_transformers import SentenceTransformer

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if list(model_bundle.get("feature_columns", [])) != CANDIDATE_FEATURES:
        raise ValueError("Ranker bundle does not use the expected inference feature schema")

    embedder = SentenceTransformer(embedding_model_id)
    documents = (
        corpus["title"].fillna("").astype(str)
        + " [SEP] "
        + corpus["candidate"].fillna("").astype(str)
    ).tolist()
    embeddings = embedder.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    corpus.to_parquet(output / "corpus.parquet", index=False, compression="snappy")
    np.save(output / "embeddings.npy", embeddings, allow_pickle=False)
    faiss.write_index(index, str(output / "faiss.index"))
    joblib.dump(model_bundle, output / "model.joblib")
    embedder.save(str(output / "embedding_model"))

    manifest = {
        "schema_version": "aai540-retrieval-runtime-v1",
        "embedding_model_id": embedding_model_id,
        "embedding_dimensions": int(embeddings.shape[1]),
        "corpus_rows": int(len(corpus)),
        "unique_products": int(corpus["asin"].nunique()),
        "ranker_schema_version": model_bundle.get("schema_version"),
        "ranker_feature_columns": list(model_bundle["feature_columns"]),
        "retrieval": "FAISS inner-product search over L2-normalized MiniLM embeddings",
        "answer_policy": "Return the highest reranked extractive candidate or abstain",
    }
    (output / "retrieval_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


class RetrievalQAEngine:
    """Validated query -> semantic retrieval -> supervised rerank -> answer/abstain."""

    def __init__(
        self,
        corpus: pd.DataFrame,
        embeddings: np.ndarray,
        model_bundle: dict,
        embedder,
        *,
        faiss_index=None,
        retrieval_k: int = 50,
        evidence_k: int = 3,
        minimum_probability_2: float = 0.35,
    ) -> None:
        if len(corpus) != len(embeddings):
            raise ValueError("Corpus and embedding row counts differ")
        if list(model_bundle.get("feature_columns", [])) != CANDIDATE_FEATURES:
            raise ValueError("Model feature schema is incompatible with retrieval runtime")
        self.corpus = corpus.reset_index(drop=True)
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.model_bundle = model_bundle
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.retrieval_k = int(retrieval_k)
        self.evidence_k = int(evidence_k)
        self.minimum_probability_2 = float(minimum_probability_2)

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, **kwargs):
        import faiss
        from sentence_transformers import SentenceTransformer

        root = Path(model_dir)
        corpus = pd.read_parquet(root / "corpus.parquet")
        embeddings = np.load(root / "embeddings.npy", allow_pickle=False)
        model_bundle = joblib.load(root / "model.joblib")
        embedder = SentenceTransformer(str(root / "embedding_model"))
        index = faiss.read_index(str(root / "faiss.index"))
        return cls(corpus, embeddings, model_bundle, embedder, faiss_index=index, **kwargs)

    def _validate_request(self, payload: dict) -> tuple[str, str, str]:
        if not isinstance(payload, dict):
            raise ValueError("Request must be a JSON object")
        request_id = normalize_text(payload.get("request_id"))
        asin = normalize_text(payload.get("asin"))
        question = normalize_text(payload.get("question"))
        if not request_id or len(request_id) > 128:
            raise ValueError("request_id is required and must be at most 128 characters")
        if not asin or len(asin) > 64:
            raise ValueError("asin is required and must be at most 64 characters")
        if not question or len(question) > 2000:
            raise ValueError("question is required and must be at most 2000 characters")
        return request_id, asin, question

    def _semantic_candidates(self, asin: str, question_embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        product_indices = np.flatnonzero(self.corpus["asin"].to_numpy(dtype=str) == asin)
        if not len(product_indices):
            return np.array([], dtype=int), np.array([], dtype=float)

        selected: list[int] = []
        score_by_index: dict[int, float] = {}
        if self.faiss_index is not None:
            search_k = min(len(self.corpus), max(self.retrieval_k * 50, 1000))
            scores, indices = self.faiss_index.search(question_embedding[None, :], search_k)
            product_set = set(product_indices.tolist())
            for index, score in zip(indices[0], scores[0]):
                index = int(index)
                if index in product_set:
                    selected.append(index)
                    score_by_index[index] = float(score)
                    if len(selected) >= self.retrieval_k:
                        break

        if len(selected) < min(self.retrieval_k, len(product_indices)):
            exact_scores = self.embeddings[product_indices] @ question_embedding
            order = np.argsort(-exact_scores)[: self.retrieval_k]
            selected = product_indices[order].astype(int).tolist()
            score_by_index = {
                int(index): float(exact_scores[position])
                for position, index in zip(order, product_indices[order])
            }
        return np.asarray(selected, dtype=int), np.asarray(
            [score_by_index[index] for index in selected], dtype=float
        )

    def answer(self, payload: dict) -> dict:
        request_id, asin, question = self._validate_request(payload)
        encoded = self.embedder.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        )
        question_embedding = np.asarray(encoded[0], dtype="float32")
        candidate_indices, semantic_scores = self._semantic_candidates(asin, question_embedding)
        if not len(candidate_indices):
            return {
                "request_id": request_id,
                "asin": asin,
                "answer": None,
                "abstained": True,
                "abstention_reason": "unknown_product_or_no_candidates",
                "evidence": [],
                "model_schema_version": self.model_bundle.get("schema_version"),
            }

        candidates = self.corpus.iloc[candidate_indices].copy().reset_index(drop=True)
        feature_frame = pd.DataFrame([
            scalar_features(question, row.candidate, row.title)
            for row in candidates.itertuples(index=False)
        ])[CANDIDATE_FEATURES]
        model = self.model_bundle["model"]
        observed = model.predict_proba(feature_frame)
        probabilities = np.zeros((len(feature_frame), 3), dtype=float)
        for source_index, label in enumerate(model.classes_):
            probabilities[:, int(label)] = observed[:, source_index]
        expected_relevance = probabilities @ np.array([0.0, 1.0, 2.0])
        blended_score = 0.75 * (expected_relevance / 2.0) + 0.25 * np.clip(
            semantic_scores, 0.0, 1.0
        )
        order = np.argsort(-blended_score)[: self.evidence_k]

        evidence = []
        for rank, position in enumerate(order, start=1):
            row = candidates.iloc[int(position)]
            evidence.append({
                "rank": rank,
                "candidate_id": row["candidate_id"],
                "text": row["candidate"],
                "title": row["title"],
                "source": row["source"],
                "semantic_similarity": float(semantic_scores[position]),
                "rerank_score": float(blended_score[position]),
                "probability_0": float(probabilities[position, 0]),
                "probability_1": float(probabilities[position, 1]),
                "probability_2": float(probabilities[position, 2]),
            })
        top = evidence[0]
        abstained = top["probability_2"] < self.minimum_probability_2
        return {
            "request_id": request_id,
            "asin": asin,
            "answer": None if abstained else top["text"],
            "abstained": bool(abstained),
            "abstention_reason": "low_relevance_confidence" if abstained else None,
            "evidence": evidence,
            "model_schema_version": self.model_bundle.get("schema_version"),
        }
