"""
Sprint 8 - Online Drift Detection tests.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("river")

from src.mlops.streaming_drift import OnlineDriftMonitor


def test_no_drift_on_stable_stream():
    monitor = OnlineDriftMonitor(feature_names=["x"])
    for i in range(200):
        _drifted = monitor.update({"x": 1.0 + (i % 3) * 0.01})  # Tiny stable noise
    status = monitor.status()
    assert status["n_updates"] == 200
    assert status["n_drift_events_total"] == 0


def test_drift_detected_on_abrupt_shift():
    monitor = OnlineDriftMonitor(feature_names=["x"])
    for _ in range(100):
        monitor.update({"x": 1.0})
    triggered_anywhere = False
    for _ in range(300):
        drifted = monitor.update({"x": 10.0})  # Abrupt, sustained shift
        if drifted:
            triggered_anywhere = True
    status = monitor.status()
    assert triggered_anywhere
    assert status["n_drift_events_total"] >= 1
    assert "x" in status["features_with_drift_ever"]


def test_missing_feature_in_reading_is_skipped_not_errored():
    monitor = OnlineDriftMonitor(feature_names=["x", "y"])
    drifted = monitor.update({"x": 1.0})  # "y" absent from this reading
    assert drifted == []  # No crash, no false drift


def test_reset_clears_single_detector():
    monitor = OnlineDriftMonitor(feature_names=["x"])
    for _ in range(100):
        monitor.update({"x": 1.0})
    for _ in range(300):
        monitor.update({"x": 10.0})
    assert monitor.status()["n_drift_events_total"] >= 1
    monitor.reset("x")
    # Sfter reset the detector is fresh; window should have shrunk back down.
    assert monitor.detectors["x"].width <= 1
