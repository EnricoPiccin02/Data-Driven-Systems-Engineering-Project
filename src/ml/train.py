"""
Sprints 4,5 — ML Kernel: Baseline + Learned models comparison.

Models are trained on the Feature Store's materialised table,
chronologically split (as per `dataset.py`), and compared against the
`NaivePersistenceBaseline` using the shared metrics module (`evaluate.py`).
Every run is logged via `MLflow` and, if it's one of the learned models,
registered in the `MLflow`-backed `ModelRegistry` (Change 4) with an
additional `Pickle` artifact.

Models included:
  - LinearRegression        — interpretable, fast, a sane second baseline
  - RandomForestRegressor   — bagging ensemble, robust default for tabular data
  - XGBRegressor            — gradient-boosted trees
  - LGBMRegressor           — gradient-boosted trees, histogram-based; typically
                              faster to train than XGBoost at this data volume
"""
from __future__ import annotations

import argparse

import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.common.config import DEFAULT_CONFIG
from src.common.logging_config import get_logger, stage
from src.feature_store.store import FeatureStore
from src.ml.baseline import NaivePersistenceBaseline
from src.ml.dataset import Xy, chronological_split
from src.ml.evaluate import evaluate, relative_improvement
from src.ml.experiment_tracking import MLflowExperimentTracker
from src.ml.explain import (
    compute_shap_values,
    save_summary_plot,
    summarise_top_features,
)
from src.ml.registry import ModelRegistry

logger = get_logger(__name__)

MODEL_NAME = "consumption_forecaster"

# How many validation rows to actually run through `SHAP`` for
# the per-run summary plot/top-features log.
SHAP_SUMMARY_SAMPLE_SIZE = 300


def _impute(X_train: pd.DataFrame, *others: pd.DataFrame) -> list[pd.DataFrame]:
    """Median-impute NaNs (from early-window lag/rolling features), fit on
    train only, applied to every split.
    Prevents val/test statistics leaking into the imputation.
    XGBoost/LightGBM can natively handle NaN, but the imputation is kept
    uniform across all four models so the model comparison isolates the
    model choice, not a difference in missing-value handling."""
    medians = X_train.median(numeric_only=True)
    out = [X_train.fillna(medians)]
    for other in others:
        out.append(other.fillna(medians))
    return out


MODEL_FACTORIES = {
    "linear_regression": lambda: LinearRegression(),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42
    ),
    "xgboost": lambda: XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
        objective="reg:squarederror",
    ),
    "lightgbm": lambda: LGBMRegressor(
        n_estimators=300, max_depth=-1, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
    ),
}


def train_and_compare(config=DEFAULT_CONFIG, models: list[str] | None = None) -> dict:
    models = models or list(MODEL_FACTORIES.keys())

    with stage(logger, "ml:load_features"):
        store = FeatureStore()
        features = store.read_features()
        manifest = store.get_manifest()

    split = chronological_split(features, config)
    X_train, y_train = Xy(split.train, split.feature_columns)
    X_val, y_val = Xy(split.val, split.feature_columns)
    X_test, _y_test = Xy(split.test, split.feature_columns)
    X_train, X_val, X_test = _impute(X_train, X_val, X_test)

    tracker = MLflowExperimentTracker("consumption_forecasting")
    registry = ModelRegistry()
    results = {}

    # Baseline always run first: everything else must beat this.
    with tracker.start_run(run_name="baseline_naive_persistence"):
        baseline = NaivePersistenceBaseline().fit(split.train)
        baseline_metrics = baseline.evaluate(split.val)
        tracker.log_params({"model_type": "naive_persistence"})
        tracker.log_metrics(baseline_metrics)
        results["baseline"] = baseline_metrics
        logger.info(f"baseline (val): {baseline_metrics}")

    # Learned models
    for model_key in models:
        with stage(logger, f"ml:train:{model_key}"), tracker.start_run(run_name=model_key):
            model = MODEL_FACTORIES[model_key]()
            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            val_metrics = evaluate(y_val, val_pred)
            improvement = relative_improvement(baseline_metrics, val_metrics, "mae")

            tracker.log_params({"model_type": model_key, **_safe_params(model)})
            tracker.log_metrics({**val_metrics, "improvement_vs_baseline_mae_pct": improvement})

            # Explainability: log a `SHAP` summary per run.
            try:
                shap_sample = X_val.sample(n=min(SHAP_SUMMARY_SAMPLE_SIZE, len(X_val)), random_state=42)
                shap_values = compute_shap_values(model, shap_sample)
                summary_path = save_summary_plot(shap_values)
                mlflow.log_artifact(str(summary_path), artifact_path="explainability")
                top_features = summarise_top_features(shap_values, feature_names=list(shap_sample.columns))
                logger.info(f"{model_key}: top SHAP feature = '{top_features.iloc[0]['feature']}' "
                            f"(computed on {len(shap_sample)}/{len(X_val)} val rows)")
            except Exception as exc:  # noqa: BLE001 — explainability must not block training
                logger.warning(f"{model_key}: SHAP explanation failed "
                                f"({type(exc).__name__}: {exc}); continuing without it")

            meta = registry.register_model(
                MODEL_NAME + f"__{model_key}",
                model,
                metrics=val_metrics,
                params=_safe_params(model),
                feature_columns=split.feature_columns,
                training_data_manifest_sha256=manifest.get("content_sha256"),
            )
            results[model_key] = {**val_metrics, "improvement_vs_baseline_mae_pct": improvement,
                                   "registry_version": meta.version}
            logger.info(f"{model_key} (val): MAE={val_metrics['mae']:.4f} "
                        f"(+{improvement:.1f}% vs baseline), registered as v{meta.version}")

    # Promote the champion (lowest val MAE among learned models) to Staging.
    champion_key = min(models, key=lambda k: results[k]["mae"])
    champion_version = results[champion_key]["registry_version"]
    registry.transition_stage(MODEL_NAME + f"__{champion_key}", champion_version, "Staging")
    results["champion"] = champion_key
    logger.info(f"Champion model: {champion_key} (v{champion_version}) -> Staging")

    return results


def _safe_params(model) -> dict:
    """Extract a JSON-serialisable subset of the estimator's get_params()."""
    params = model.get_params() if hasattr(model, "get_params") else {}
    return {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool, type(None)))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None, choices=list(MODEL_FACTORIES.keys()))
    args = parser.parse_args()
    results = train_and_compare(models=args.models)
    import json
    print(json.dumps(results, indent=2, default=str))