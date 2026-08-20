"""
Sprint 5 — Explainability.

`SHAP` gives per-prediction (local) attributions, i.e., a materially
richer answer to "why did the model predict this?".

`shap.TreeExplainer` is used for the three tree-based candidates
(RandomForest, XGBoost, LightGBM), i.e., the recommended
explainer for tree ensembles.
`shap.LinearExplainer` is used for LinearRegression.
`explain_model()` picks the right explainer automatically based
on the model type, so callers (e.g., `train.py`, the dashboard)
don't need to know which.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless: no display server assumed (CI, containers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.common.config import REPORTS_DIR
from src.common.logging_config import get_logger, stage

logger = get_logger(__name__)

TREE_MODEL_TYPES = (RandomForestRegressor, XGBRegressor, LGBMRegressor)


def _build_explainer(model, X_background: pd.DataFrame):
    """Build the appropriate `SHAP` explainer for a supported model.

    Tree-based models use `TreeExplainer` because `SHAP` can exploit their
    internal tree structure and compute Tree `SHAP` efficiently and exactly
    under the selected feature-dependence assumptions.

    Linear models use `LinearExplainer`.

    Unknown models fall back to the generic `SHAP` Explainer.
    """
    if isinstance(model, TREE_MODEL_TYPES):
        logger.info(
            "explain: using TreeExplainer for %s",
            type(model).__name__,
        )
        return shap.TreeExplainer(model)

    if isinstance(model, LinearRegression):
        logger.info(
            "explain: using LinearExplainer for %s",
            type(model).__name__,
        )
        return shap.LinearExplainer(model, X_background)

    logger.warning(
        "explain: no specialised SHAP explainer for %s; "
        "falling back to model-agnostic shap.Explainer",
        type(model).__name__,
    )

    return shap.Explainer(model.predict, X_background)


def compute_shap_values(model, X: pd.DataFrame, X_background: pd.DataFrame | None = None):
    """Returns a `shap.Explanation` object (or raw values array for older
    `SHAP` explainer APIs) covering every row in `X`."""
    with stage(logger, "explain:compute_shap_values", n_rows=len(X), model_type=type(model).__name__):
        background = X_background if X_background is not None else X.sample(
            n=min(100, len(X)), random_state=42
        )
        explainer = _build_explainer(model, background)
        shap_values = explainer(X)
        return shap_values


def summarise_top_features(shap_values, feature_names: list[str] | None = None, top_n: int = 10) -> pd.DataFrame:
    """Global importance = mean(|SHAP value|) per feature — the standard
    `SHAP` summary statistic, computed from the same per-prediction values
    used for local explanations (so global and local views are always
    consistent with each other)."""
    values = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
    names = feature_names or (list(shap_values.feature_names) if hasattr(shap_values, "feature_names") else None)
    mean_abs = np.abs(values).mean(axis=0)
    df = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def save_summary_plot(shap_values, out_path: Path | None = None) -> Path:
    """Saves `SHAP`'s standard beeswarm summary plot to disk."""
    out_path = out_path or (REPORTS_DIR / "shap_summary_plot.png")
    plt.figure()
    shap.summary_plot(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info(f"explain: saved SHAP summary plot -> {out_path}")
    return out_path


def explain_local_prediction(shap_values, row_index: int, feature_names: list[str] | None = None) -> pd.DataFrame:
    """Per-prediction attribution for a single row."""
    row = shap_values[row_index]
    names = feature_names or (list(row.feature_names) if hasattr(row, "feature_names") else None)
    df = pd.DataFrame({"feature": names, "shap_value": row.values, "feature_value": row.data})
    df["abs_shap_value"] = df["shap_value"].abs()
    return df.sort_values("abs_shap_value", ascending=False).drop(columns="abs_shap_value").reset_index(drop=True)
