"""
Sprint 4 — Evaluation metrics.

A standalone module so the exact same functions are used for 
baseline comparison (Sprint 4), model comparison (Sprint 5),
and production monitoring (Sprint 8).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: pd.Series, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    """Mean Absolute Percentage Error, guarded against near-zero true values
    (household consumption legitimately dips near zero overnight)."""
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "n_samples": len(y_true),
    }


def relative_improvement(baseline_metrics: dict, model_metrics: dict, metric: str = "mae") -> float:
    """% improvement of `model_metrics` over `baseline_metrics` for `metric`
    (positive = better). Used against the KPI "beat naive persistence by >=15%")."""
    base = baseline_metrics[metric]
    if base == 0:
        return 0.0
    return (base - model_metrics[metric]) / base * 100
