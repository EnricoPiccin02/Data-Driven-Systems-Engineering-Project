"""
Sprint 4 — Baseline model.

"Beat the baseline" is the first, non-negotiable ML kernel milestone.
The baseline is deliberately trivial: a naive persistence forecast
(which predict the same-time-last-week value, i.e. `lag_336`).
Any model that can't beat this has learned nothing useful, and in 
energy forecasting persistence baselines are notoriously strong
(consumption is highly autocorrelated week-over-week), so this is a
meaningful bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.evaluate import evaluate


class NaivePersistenceBaseline:
    """
    Predicts consumption_kwh(t) = consumption_kwh(t - 1 week).

    Falls back to `lag_48` (same time yesterday) where `lag_336` is
    unavailable (e.g. first week of data), and to the household's train-set
    mean as a last resort — every row must get a prediction.
    """

    def __init__(self):
        self.household_mean_: dict[str, float] | None = None

    def fit(self, train_df: pd.DataFrame) -> NaivePersistenceBaseline:
        self.household_mean_ = train_df.groupby("household_id")["consumption_kwh"].mean().to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = df["lag_336"].copy()
        pred = pred.fillna(df["lag_48"])
        fallback = df["household_id"].map(self.household_mean_).fillna(
            np.mean(list(self.household_mean_.values()))
        )
        pred = pred.fillna(fallback)
        return pred.to_numpy()

    def evaluate(self, df: pd.DataFrame) -> dict:
        y_pred = self.predict(df)
        return evaluate(df["consumption_kwh"], y_pred)
