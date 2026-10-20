"""SageMaker inference handlers for the Product QA retrieval runtime."""

from __future__ import annotations

import json

try:
    from .retrieval import RetrievalQAEngine
except ImportError:  # SageMaker places the source directory directly on sys.path.
    from retrieval import RetrievalQAEngine


def model_fn(model_dir: str):
    return RetrievalQAEngine.from_model_dir(model_dir)


def input_fn(request_body, content_type: str):
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise ValueError(f"Unsupported content type {content_type}; expected application/json")
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    payload = json.loads(request_body)
    if isinstance(payload, list):
        return payload
    return [payload]


def predict_fn(input_data: list[dict], engine: RetrievalQAEngine):
    return [engine.answer(payload) for payload in input_data]


def output_fn(prediction, accept: str):
    media_type = accept.split(";", 1)[0].strip().lower()
    if media_type not in {"application/json", "*/*"}:
        raise ValueError(f"Unsupported accept type {accept}; expected application/json")
    return json.dumps(prediction, allow_nan=False)
