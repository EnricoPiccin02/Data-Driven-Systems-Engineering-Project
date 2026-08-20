"""
Sprint 1 — Data Validation tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("great_expectations")

from src.data_engineering.validate import (
    _check_timestamp_regularity,
    has_critical_failure,
    run_suite,
)


def test_smart_meter_suite_passes_on_clean_data():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="30min"),
            "household_id": ["HH000"] * 10,
            "consumption_kwh": np.linspace(0.3, 0.8, 10),
        }
    )
    results = run_suite(df, "smart_meter")
    assert not has_critical_failure(results)


def test_smart_meter_suite_flags_out_of_range_spike():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="30min"),
            "household_id": ["HH000"] * 5,
            "consumption_kwh": [0.5, 0.6, 999.0, 0.4, 0.3],
        }
    )
    results = run_suite(df, "smart_meter")
    assert has_critical_failure(results)


def test_smart_meter_suite_flags_duplicate_rows():
    ts = pd.date_range("2024-01-01", periods=3, freq="30min")
    df = pd.DataFrame(
        {
            "timestamp": list(ts) + [ts[0]],
            "household_id": ["HH000"] * 4,
            "consumption_kwh": [0.5, 0.6, 0.4, 0.5],
        }
    )
    results = run_suite(df, "smart_meter")
    assert has_critical_failure(results)


def test_weather_suite_flags_impossible_temperature():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1h"),
            "temperature_c": [10.0, 200.0, 12.0],  # 200°C is impossible
            "humidity_pct": [50.0, 55.0, 60.0],
            "wind_speed_ms": [3.0, 4.0, 5.0],
        }
    )
    results = run_suite(df, "weather")
    assert has_critical_failure(results)


def test_calendar_suite_flags_duplicate_dates():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "day_of_week": [0, 0, 1],
            "is_weekend": [False, False, False],
            "is_holiday": [True, True, False],
        }
    )
    results = run_suite(df, "calendar")
    assert has_critical_failure(results)


def test_timestamp_regularity_check_detects_gap():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00", "2024-01-01 00:30", "2024-01-01 02:00"],
            "household_id": ["HH000"] * 3,
        }
    )
    result = _check_timestamp_regularity(df, freq_minutes=30)
    assert not result.success
    assert "n_gaps=1" in result.observed
