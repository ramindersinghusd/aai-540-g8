"""Small Flask contract around a SageMaker Product QA endpoint."""

from __future__ import annotations

import json
import os
from typing import Callable

from flask import Flask, jsonify, request


def create_app(
    endpoint_name: str | None = None,
    *,
    invoke: Callable[[dict], dict | list] | None = None,
    runtime_client=None,
) -> Flask:
    app = Flask(__name__)
    resolved_endpoint = endpoint_name or os.getenv("PRODUCT_QA_ENDPOINT_NAME", "")

    if invoke is None:
        if not resolved_endpoint:
            raise ValueError("An endpoint name or invoke function is required")
        if runtime_client is None:
            import boto3

            runtime_client = boto3.client("sagemaker-runtime")

        def invoke(payload):
            response = runtime_client.invoke_endpoint(
                EndpointName=resolved_endpoint,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload).encode("utf-8"),
            )
            return json.loads(response["Body"].read())

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "endpoint": resolved_endpoint or "in-process"})

    @app.post("/answer")
    def answer():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400
        required = {"request_id", "asin", "question"}
        missing = sorted(
            key for key in required
            if not isinstance(payload.get(key), str) or not payload[key].strip()
        )
        if missing:
            return jsonify({"error": f"Missing non-empty string fields: {missing}"}), 400
        if len(payload["request_id"]) > 128 or len(payload["asin"]) > 64 or len(payload["question"]) > 2000:
            return jsonify({"error": "One or more request fields exceed the supported length"}), 400
        try:
            result = invoke(payload)
            if isinstance(result, list) and len(result) == 1:
                result = result[0]
            return jsonify(result), 200
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            app.logger.exception("Product QA invocation failed")
            return jsonify({"error": "Inference service unavailable"}), 502

    return app
