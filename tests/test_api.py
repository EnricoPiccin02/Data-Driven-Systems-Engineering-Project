"""
Sprint 6 — API Integration tests.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient

from src.app.api import app

client = TestClient(app)


def _sample_payload():
    now = datetime(2024, 1, 15, 18, 0, tzinfo=timezone.utc)
    readings = [
        {"timestamp": (now - timedelta(minutes=30 * i)).isoformat(), "consumption_kwh": 0.5 + 0.1 * (i % 3)}
        for i in range(400, 0, -1)
    ]
    return {
        "household_id": "HH000",
        "forecast_timestamp": now.isoformat(),
        "recent_readings": readings,
        "temperature_c": 5.0,
    }


def test_health_endpoint_reports_status():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_forecast_endpoint_requires_production_model_or_503():
    resp = client.post("/forecast", json=_sample_payload())
    # Either a Production model is loaded (200, with a numeric forecast) or
    # none has been promoted yet in this environment (503) — both are valid,
    # well-defined API behaviours; a 500 (unhandled crash) would not be.
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert "predicted_consumption_kwh" in body


def test_recommendations_endpoint_shape_when_model_available():
    health = client.get("/health").json()
    if not health["production_model_loaded"]:
        pytest.skip("No Production model loaded in this environment — see scripts/train_models.py")
    resp = client.post("/recommendations", json=_sample_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["household_id"] == "HH000"
    assert len(body["recommendations"]) >= 1
