"""
Sprint 3 — Feature Engineering.

Pure functions, one feature family per function, so each is independently
unit-testable and independently reusable at inference time (Sprint 6).
The API will import these same functions rather than reimplementing them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import DEFAULT_CONFIG, PipelineConfig
from src.common.logging_config import get_logger, stage

logger = get_logger(__name__)


def resample_to_config_grid(df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Collapse arbitrarily-spaced readings onto the canonical
    `config.freq_minutes` grid, per household, before any lag/rolling
    feature is computed.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    other_cols = [c for c in df.columns if c not in numeric_cols and c not in ("timestamp", "household_id")]

    out = []
    for household_id, g in df.groupby("household_id"):
        g = g.set_index("timestamp").sort_index()
        resampled = g[numeric_cols].resample(f"{config.freq_minutes}min").mean()
        if other_cols:
            resampled = resampled.join(g[other_cols].resample(f"{config.freq_minutes}min").last())
        resampled["household_id"] = household_id
        out.append(resampled.reset_index())

    return pd.concat(out, ignore_index=True).sort_values(["household_id", "timestamp"]).reset_index(drop=True)


def add_lag_features(df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    df = df.sort_values(["household_id", "timestamp"]).copy()
    g = df.groupby("household_id")["consumption_kwh"]
    for lag in config.lag_steps:
        df[f"lag_{lag}"] = g.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    df = df.sort_values(["household_id", "timestamp"]).copy()
    g = df.groupby("household_id")["consumption_kwh"]
    for window in config.rolling_windows:
        # shift(1) first, since rolling stats must only use past information
        shifted = g.shift(1)
        df[f"roll_mean_{window}"] = shifted.groupby(df["household_id"]).transform(
            lambda s, window=window: s.rolling(window, min_periods=max(1, window // 4)).mean()
        )
        df[f"roll_std_{window}"] = shifted.groupby(df["household_id"]).transform(
            lambda s, window=window: s.rolling(window, min_periods=max(1, window // 4)).std()
        )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * (df["hour"] + df["minute"] / 60) / 24)
    df["hour_cos"] = np.cos(2 * np.pi * (df["hour"] + df["minute"] / 60) / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    if "is_weekend" in df.columns:
        df["is_weekend"] = df["is_weekend"].astype("boolean").fillna(False).astype(int)
    if "is_holiday" in df.columns:
        df["is_holiday"] = df["is_holiday"].astype("boolean").fillna(False).astype(int)
    return df


def add_weather_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "temperature_c" in df.columns:
        # Heating/cooling-degree-style features: consumption tends to rise
        # both when it's cold (heating) and when hot.
        df["heating_degree"] = (18 - df["temperature_c"]).clip(lower=0)
        df["cooling_degree"] = (df["temperature_c"] - 22).clip(lower=0)
    return df


def build_feature_table(df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    with stage(logger, "feature_engineering:build_feature_table", n_rows_in=len(df)):
        # Enforce the canonical grid unconditionally, so that any future
        # caller (offline or online) inherits the correct cadence handling.
        df = resample_to_config_grid(df, config)
        df = add_calendar_features(df)
        df = add_weather_interaction_features(df)
        df = add_lag_features(df, config)
        df = add_rolling_features(df, config)
        logger.info(f"feature table shape: {df.shape}")
        return df