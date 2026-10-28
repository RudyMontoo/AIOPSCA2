import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
sample = json.load(open("sample_request.json"))


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict():
    resp = client.post("/predict", json=sample)
    assert resp.status_code == 200
    body = resp.json()
    assert "churn" in body and "churn_probability" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
