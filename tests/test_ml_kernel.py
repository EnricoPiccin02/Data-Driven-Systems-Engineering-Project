"""
Sprints 4,5 — ML Kernel tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import PipelineConfig
from src.ml.baseline import NaivePersistenceBaseline
from src.ml.dataset import chronological_split
from src.ml.evaluate import evaluate, mae, mape, relative_improvement, rmse
from src.ml.explain import compute_shap_values


def _toy_features_df(n_households=2, n_steps=1500):
    frames = []
    for h in range(n_households):
        ts = pd.date_range("2024-01-01", periods=n_steps, freq="30min")
        y = 1.0 + 0.3 * np.sin(np.arange(n_steps) / 48 * 2 * np.pi) + np.random.RandomState(h).normal(0, 0.05, n_steps)
        df = pd.DataFrame({"timestamp": ts, "household_id": f"HH{h:03d}", "consumption_kwh": y})
        df["lag_1"] = df["consumption_kwh"].shift(1)
        df["lag_336"] = df["consumption_kwh"].shift(336)
        df["lag_48"] = df["consumption_kwh"].shift(48)
        df["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_chronological_split_respects_time_order():
    df = _toy_features_df()
    config = PipelineConfig(lag_steps=(1, 48))
    split = chronological_split(df, config, val_frac=0.2, test_frac=0.2)
    assert split.train["timestamp"].max() < split.val["timestamp"].min()
    assert split.val["timestamp"].max() < split.test["timestamp"].min()


def test_chronological_split_purges_tail():
    df = _toy_features_df()
    config = PipelineConfig(lag_steps=(1, 48))
    split = chronological_split(df, config, val_frac=0.2, test_frac=0.2)
    # purge = max(lag_steps) = 48 steps = 24h: train's tail should be pulled
    # back from the val boundary by roughly that amount.
    gap = split.val["timestamp"].min() - split.train["timestamp"].max()
    assert gap >= pd.Timedelta(hours=20)


def test_naive_baseline_beats_random_guessing():
    df = _toy_features_df()
    config = PipelineConfig(lag_steps=(1, 48))
    split = chronological_split(df, config, val_frac=0.2, test_frac=0.2)
    baseline = NaivePersistenceBaseline().fit(split.train)
    metrics = baseline.evaluate(split.val)
    # Sanity: MAE should be small and finite for this smooth toy signal
    assert metrics["mae"] < 1.0
    assert metrics["n_samples"] == len(split.val)


def test_metrics_basic_properties():
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert mae(y_true, y_pred) == 0.0
    assert rmse(y_true, y_pred) == 0.0
    assert mape(y_true, y_pred) == 0.0

    y_pred_off = np.array([2.0, 2.0, 2.0])
    m = evaluate(y_true, y_pred_off)
    assert m["mae"] > 0


def test_relative_improvement_direction():
    baseline = {"mae": 1.0}
    better_model = {"mae": 0.5}
    worse_model = {"mae": 2.0}
    assert relative_improvement(baseline, better_model) == 50.0
    assert relative_improvement(baseline, worse_model) == -100.0


def test_compute_shap_values_supports_xgboost():
    pytest.importorskip("xgboost")
    pytest.importorskip("shap")

    from xgboost import XGBRegressor

    X = pd.DataFrame({
        "x1": np.linspace(0, 1, 40),
        "x2": np.linspace(1, 2, 40),
        "x3": np.linspace(-1, 1, 40),
    })
    y = 2.0 * X["x1"] + 0.5 * X["x2"] - 1.2 * X["x3"] + 0.1
    model = XGBRegressor(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X, y)

    shap_values = compute_shap_values(model, X)
    assert shap_values is not None
    assert hasattr(shap_values, "values")
    assert tuple(shap_values.values.shape[:2]) == (len(X), len(X.columns))