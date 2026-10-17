from app.flask_app import create_app


def test_flask_contract():
    def invoke(payload):
        return [{
            "request_id": payload["request_id"],
            "asin": payload["asin"],
            "answer": "Evidence answer",
            "abstained": False,
            "evidence": [{"rank": 1, "text": "Evidence answer"}],
        }]

    app = create_app(invoke=invoke)
    client = app.test_client()

    assert client.get("/health").status_code == 200
    response = client.post(
        "/answer",
        json={"request_id": "request-1", "asin": "A1", "question": "Does it fit?"},
    )
    assert response.status_code == 200
    assert response.get_json()["answer"] == "Evidence answer"
    assert client.post("/answer", data="not-json").status_code == 400
