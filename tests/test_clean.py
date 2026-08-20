"""
Sprint 3: Data Cleaning tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import PipelineConfig
from src.data_engineering.clean import MAX_PLAUSIBLE_KWH, clean_smart_meter


def _toy_meter_df():
    ts = pd.date_range("2024-01-01", periods=6, freq="30min")
    df = pd.DataFrame(
        {
            "timestamp": list(ts) + [ts[0]],  # Inject one duplicate row
            "household_id": ["HH000"] * 7,
            "consumption_kwh": [0.5, 0.6, 100.0, 0.4, np.nan, 0.3, 0.5],
        }
    )
    return df


def test_clean_removes_exact_duplicates():
    df = _toy_meter_df()
    cleaned = clean_smart_meter(df, PipelineConfig(freq_minutes=30))
    # One duplicate (timestamp, household_id) pair should collapse
    assert cleaned.duplicated(subset=["timestamp", "household_id"]).sum() == 0


def test_clean_winsorises_spikes():
    df = _toy_meter_df()
    cleaned = clean_smart_meter(df, PipelineConfig(freq_minutes=30))
    assert cleaned["consumption_kwh"].max() <= MAX_PLAUSIBLE_KWH


def test_clean_is_deterministic():
    df = _toy_meter_df()
    config = PipelineConfig(freq_minutes=30)
    out1 = clean_smart_meter(df, config)
    out2 = clean_smart_meter(df, config)
    pd.testing.assert_frame_equal(out1, out2)


def test_clean_fills_short_gap():
    df = _toy_meter_df()
    cleaned = clean_smart_meter(df, PipelineConfig(freq_minutes=30))
    # The NaN in the toy data is a single-step gap -> should be forward-filled
    assert cleaned["consumption_kwh"].isna().sum() == 0
