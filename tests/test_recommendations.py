"""
Sprint 7 — Optimisation Suggestions tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.recommendations import generate_recommendations


def test_peak_hour_high_consumption_triggers_shift_recommendation():
    recs = generate_recommendations(
        hour=18, is_weekend=False, heating_degree=0.0,
        predicted_consumption_kwh=2.5, recent_mean_kwh=1.0,
    )
    titles = [r.title for r in recs]
    assert any("evening peak" in t.lower() for t in titles)


def test_cold_snap_triggers_heating_recommendation():
    recs = generate_recommendations(
        hour=10, is_weekend=False, heating_degree=10.0,
        predicted_consumption_kwh=0.8, recent_mean_kwh=0.8,
    )
    titles = [r.title for r in recs]
    assert any("cold snap" in t.lower() for t in titles)


def test_no_flags_returns_no_action_needed():
    recs = generate_recommendations(
        hour=10, is_weekend=False, heating_degree=0.0,
        predicted_consumption_kwh=0.5, recent_mean_kwh=0.5,
    )
    assert len(recs) == 1
    assert recs[0].title == "No action needed"


def test_spike_vs_recent_average_triggers_alert():
    recs = generate_recommendations(
        hour=10, is_weekend=False, heating_degree=0.0,
        predicted_consumption_kwh=3.0, recent_mean_kwh=1.0,
    )
    titles = [r.title for r in recs]
    assert any("above your recent average" in t.lower() for t in titles)
