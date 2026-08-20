"""
Sprint 8 — Monitoring & Maintenance (Batch Drift Detection) tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("evidently")

from src.mlops.monitoring import (
    dataset_drift_summary,
    detect_feature_drift,
    detect_prediction_drift,
)


def test_detect_feature_drift_flags_clearly_shifted_feature():
    rng = np.random.default_rng(1)
    reference = pd.DataFrame({"temperature_c": rng.normal(10, 3, 500)})
    current = pd.DataFrame({"temperature_c": rng.normal(25, 3, 500)})  # Large mean shift
    results = detect_feature_drift(reference, current, ["temperature_c"])
    assert len(results) == 1
    assert results[0].drift_detected
    assert results[0].severity == "CRITICAL"


def test_detect_feature_drift_no_flag_for_identical_distributions():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    current = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    results = detect_feature_drift(reference, current, ["x"])
    assert len(results) == 1
    assert not results[0].drift_detected
    assert results[0].severity == "OK"


def test_detect_feature_drift_skips_missing_columns():
    reference = pd.DataFrame({"a": [1, 2, 3]})
    current = pd.DataFrame({"b": [1, 2, 3]})
    results = detect_feature_drift(reference, current, ["a"])
    assert results == []


def test_dataset_drift_summary_shape():
    rng = np.random.default_rng(2)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(5, 1, 500)})
    current = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(50, 1, 500)})
    summary = dataset_drift_summary(reference, current, ["a", "b"])
    assert "share_drifted_columns" in summary
    assert 0.0 <= summary["share_drifted_columns"] <= 1.0
    assert summary["n_drifted_columns"] >= 1  # "b" shifted hard


def test_detect_prediction_drift_on_residuals():
    rng = np.random.default_rng(3)
    reference_errors = pd.Series(rng.normal(0, 0.05, 500))
    current_errors = pd.Series(rng.normal(0.3, 0.05, 500))  # Residuals got much worse
    result = detect_prediction_drift(reference_errors, current_errors)
    assert result.drift_detected
