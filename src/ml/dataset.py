"""
Sprint 4 — ML dataset preparation.

Time-series forecasting can't use arandom train/test split (which can
leak future information into training via nearby rows.
Hence, a strict chronological split is used instead, with an
explicit purge gap equal to the longest lookback window so that no training
row's features were computed using any point in the test period.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.common.config import DEFAULT_CONFIG, PipelineConfig
from src.common.logging_config import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "consumption_kwh"
ENTITY_KEYS = ["household_id", "timestamp"]


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]


def chronological_split(
    df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> SplitResult:
    """
    Split by timestamp, with a purge gap.

    The purge gap is `max(lag_steps)` rows: any row whose feature window
    would reach into the next split is dropped from the earlier split.
    """
    df = df.sort_values("timestamp").copy()
    df = df.dropna(subset=[TARGET_COLUMN])  # Can't train/evaluate against a missing target

    purge = max(config.lag_steps) if config.lag_steps else 0
    timestamps = df["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    n = len(timestamps)
    test_start_idx = int(n * (1 - test_frac))
    val_start_idx = int(n * (1 - test_frac - val_frac))

    val_boundary = timestamps.iloc[val_start_idx]
    test_boundary = timestamps.iloc[test_start_idx]

    train = df[df["timestamp"] < val_boundary]
    val = df[(df["timestamp"] >= val_boundary) & (df["timestamp"] < test_boundary)]
    test = df[df["timestamp"] >= test_boundary]

    # Purge: drop the last `purge` half-hour steps of train/val per household,
    # so no row's lag/rolling features reach across the split boundary.
    def _purge_tail(part: pd.DataFrame) -> pd.DataFrame:
        if purge == 0:
            return part
        
        kept = []
        for _, g in part.groupby("household_id", sort=False):
            g = g.sort_values("timestamp")
            kept.append(g.iloc[:-purge] if len(g) > purge else g.iloc[0:0])
        return pd.concat(kept, ignore_index=True) if kept else part.iloc[0:0]

    train = _purge_tail(train)
    val = _purge_tail(val)

    exclude = set(
        ENTITY_KEYS
        + [
            TARGET_COLUMN,
            "is_winsorised",
            "was_missing",
            "is_real_data",
            "is_real_meter_data",  # Provenance metadata
        ]
    )
    feature_columns = [c for c in df.columns if c not in exclude]

    logger.info(
        f"chronological_split: train={len(train):,} val={len(val):,} test={len(test):,} "
        f"(purge={purge} steps, {len(feature_columns)} feature columns)"
    )
    return SplitResult(train=train, val=val, test=test, feature_columns=feature_columns)


def Xy(part: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = part[feature_columns].copy()
    # Tree/linear models don't accept NaN; a simple median-impute is
    # applied at train time only to avoid leaking val/test statistics.
    y = part[TARGET_COLUMN]
    return X, y
