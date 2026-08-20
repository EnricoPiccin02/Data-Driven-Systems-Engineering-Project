"""
Sprint 3 — Cleaning stage.

Deterministic: identical input bytes -> identical output bytes. No use of
wall-clock time, unseeded randomness, or non-deterministic set/dict ordering
in anything that affects output values.

Responsibilities:
 1. Drop exact duplicate (timestamp, household_id) rows (keep first).
 2. Align/clip readings onto the canonical half-hourly.
 3. Winsorise implausible spikes flagged by `validate.py`'s CRITICAL range check.
 4. Forward-fill short gaps (<=2 missed reads) per household; longer gaps
    are left as NaN and explicitly flagged.
"""
from __future__ import annotations

import pandas as pd

from src.common.config import DEFAULT_CONFIG, PipelineConfig
from src.common.logging_config import get_logger, stage

logger = get_logger(__name__)

MAX_PLAUSIBLE_KWH = 20.0  # Above this, treat as sensor/transmission fault
SHORT_GAP_LIMIT = 2  # Consecutive missing 30-min reads eligible for ffill


def clean_smart_meter(df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    with stage(logger, "clean:smart_meter", n_rows_in=len(df)):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        n_before = len(df)
        df = df.drop_duplicates(subset=["timestamp", "household_id"], keep="first")
        logger.info(f"dropped {n_before - len(df)} exact duplicate rows")

        n_spikes = (df["consumption_kwh"] > MAX_PLAUSIBLE_KWH).sum()
        df["consumption_kwh"] = df["consumption_kwh"].clip(upper=MAX_PLAUSIBLE_KWH)
        df["is_winsorised"] = df["consumption_kwh"] >= MAX_PLAUSIBLE_KWH
        logger.info(f"winsorised {n_spikes} implausible spikes at {MAX_PLAUSIBLE_KWH} kWh")

        full_grid = pd.date_range(
            df["timestamp"].min(), df["timestamp"].max(),
            freq=f"{config.freq_minutes}min",
        )
        aligned_parts = []
        for hh_id, g in df.groupby("household_id", sort=True):
            g = g.set_index("timestamp").reindex(full_grid)
            g["household_id"] = hh_id
            g.index.name = "timestamp"
            aligned_parts.append(g)
        df = pd.concat(aligned_parts).reset_index()

        # `is_real_data` is constant per household. The above grid alignment
        # can introduce NaN for it on newly-added rows, so backfill/forward-fill
        # it within each household group rather than losing real/synthetic
        # provenance on realigned rows.
        if "is_real_data" in df.columns:
            df["is_real_data"] = (
                df.groupby("household_id")["is_real_data"].transform(lambda s: s.ffill().bfill())
            ).fillna(False)

        df["was_missing"] = df["consumption_kwh"].isna()
        df["consumption_kwh"] = (
            df.groupby("household_id")["consumption_kwh"]
            .transform(lambda s: s.ffill(limit=SHORT_GAP_LIMIT))
        )
        remaining_na = df["consumption_kwh"].isna().sum()
        logger.info(
            f"filled short gaps (<= {SHORT_GAP_LIMIT} steps); "
            f"{remaining_na} rows remain NaN (long gaps, left explicit)"
        )

        df["is_winsorised"] = df["is_winsorised"].fillna(False)
        return df.sort_values(["household_id", "timestamp"]).reset_index(drop=True)


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    with stage(logger, "clean:weather", n_rows_in=len(df)):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.set_index("timestamp").sort_index()
        df = df.interpolate(method="time", limit=3)
        return df.reset_index()


def clean_calendar(df: pd.DataFrame) -> pd.DataFrame:
    with stage(logger, "clean:calendar", n_rows_in=len(df)):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
