"""Tests for the FastAPI /predict and /health endpoints."""

import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 35,
    "annual_income": 60000,
    "credit_score": 700,
    "existing_debt": 5000,
    "loan_amount_requested": 10000,
    "employment_years": 5,
    "debt_to_income_ratio": 0.08,
    "utilization_3mo_avg": 30,
    "utilization_6mo_avg": 28,
    "utilization_12mo_avg": 25,
    "age_woe": 0.02,
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    data = response.json()
    assert "default_probability" in data
    assert "decision" in data
    assert "threshold" in data
    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["decision"] in ("approve", "reject")


def test_predict_decision_matches_threshold():
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()

    if data["default_probability"] >= data["threshold"]:
        assert data["decision"] == "reject"
    else:
        assert data["decision"] == "approve"


def test_predict_missing_field_rejected():
    incomplete_payload = VALID_PAYLOAD.copy()
    del incomplete_payload["credit_score"]

    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422  # FastAPI/pydantic validation error


def test_predict_invalid_type_rejected():
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["age"] = "not_a_number"

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
