"""
Sprint 3 — Transformation stage.

Merges the three cleaned sources (smart meter, weather, calendar) onto a
single per-(household, timestamp) analytical table. Weather (hourly) and
calendar (daily) are upsampled/broadcast onto the half-hourly meter grid
via an as-of merge.
"""
from __future__ import annotations

import pandas as pd

from src.common.logging_config import get_logger, stage

logger = get_logger(__name__)


def merge_sources(meter: pd.DataFrame, weather: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    with stage(logger, "transform:merge_sources", n_meter_rows=len(meter)):
        meter = meter.sort_values("timestamp").copy()
        weather = weather.sort_values("timestamp").copy()
        calendar = calendar.copy()

        # All three sources may carry their own `is_real_data` provenance
        # flag. We only care about the smart-meter's provenance downstream
        # (it's the one that determines whether a row came from the real
        # household or a synthetic peer), so drop the weather/calendar
        # copies before merging.
        if "is_real_data" in meter.columns:
            meter = meter.rename(columns={"is_real_data": "is_real_meter_data"})
        weather = weather.drop(columns=["is_real_data"], errors="ignore")
        calendar = calendar.drop(columns=["is_real_data"], errors="ignore")

        merged = pd.merge_asof(
            meter, weather, on="timestamp", direction="backward",
            tolerance=pd.Timedelta("1h"),
        )

        ts = merged["timestamp"]
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        merged["date"] = ts.dt.normalize()

        calendar = calendar.rename(columns={"date": "date"})
        merged = merged.merge(calendar, on="date", how="left")

        n_missing_weather = merged["temperature_c"].isna().sum()
        if n_missing_weather:
            logger.warning(f"{n_missing_weather} rows have no matching weather reading")

        return merged.drop(columns=["date"])
