"""
Sprint 3 — Feature Engineering tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import PipelineConfig
from src.data_engineering.feature_engineering import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_feature_table,
)


def _toy_df(n=20):
    ts = pd.date_range("2024-01-01", periods=n, freq="30min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "household_id": ["HH000"] * n,
            "consumption_kwh": np.arange(n, dtype=float),
            "temperature_c": np.full(n, 10.0),
            "is_weekend": [False] * n,
            "is_holiday": [False] * n,
        }
    )


def test_lag_feature_shifts_correctly():
    df = _toy_df()
    config = PipelineConfig(lag_steps=(1, 2), rolling_windows=(4,))
    out = add_lag_features(df, config)
    # lag_1 at row i should equal consumption at row i-1
    assert out["lag_1"].iloc[5] == out["consumption_kwh"].iloc[4]
    assert pd.isna(out["lag_1"].iloc[0])


def test_rolling_features_do_not_leak_future():
    df = _toy_df()
    config = PipelineConfig(lag_steps=(1,), rolling_windows=(4,))
    out = add_rolling_features(df, config)
    # roll_mean at row i must only be a function of rows < i (shift(1) applied)
    # so it should differ from a naive centred rolling mean including row i
    naive_incl_current = df["consumption_kwh"].rolling(4, min_periods=1).mean()
    assert not out["roll_mean_4"].equals(naive_incl_current)


def test_calendar_features_are_bounded():
    df = _toy_df()
    out = add_calendar_features(df)
    assert out["hour_sin"].between(-1, 1).all()
    assert out["hour_cos"].between(-1, 1).all()


def test_build_feature_table_is_deterministic():
    df = _toy_df()
    config = PipelineConfig(lag_steps=(1, 2), rolling_windows=(4,))
    out1 = build_feature_table(df, config)
    out2 = build_feature_table(df, config)
    pd.testing.assert_frame_equal(out1, out2)


def test_build_feature_table_adds_expected_columns():
    df = _toy_df()
    config = PipelineConfig(lag_steps=(1, 2), rolling_windows=(4,))
    out = build_feature_table(df, config)
    for col in ["lag_1", "lag_2", "roll_mean_4", "roll_std_4", "hour_sin", "heating_degree"]:
        assert col in out.columns
