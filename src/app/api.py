"""
Sprint 6 — Prediction Service (App Engineering / Serving).

Loads the Production-stage model from the local Model Registry at startup
and exposes:
  GET  /health            — liveness + which model is loaded
  POST /forecast          — single-interval consumption forecast
  POST /recommendations   — optimisation suggestions derived from a forecast

To achieve train/serve parity `_compute_online_features` calls the
exact same functions used to build the offline Feature Store, applied
to the caller-supplied `recent_readings` window. This is the concrete
mechanism that prevents training/serving skew.

Usage:
    uvicorn src.app.api:app --reload --port 8000
"""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.app.recommendations import generate_recommendations
from src.app.schemas import (
    FeatureContribution,
    ForecastExplanation,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    RecommendationItem,
    RecommendationResponse,
)
from src.common.config import DEFAULT_CONFIG
from src.common.logging_config import get_logger
from src.data_engineering.feature_engineering import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_weather_interaction_features,
    resample_to_config_grid,
)
from src.ml.explain import compute_shap_values, explain_local_prediction
from src.ml.registry import ModelRegistry
from src.ml.train import MODEL_FACTORIES, MODEL_NAME

logger = get_logger(__name__)
app = FastAPI(title="Smart Energy Consumption Forecasting API", version="0.7.0")

CANDIDATE_MODEL_NAMES = [f"{MODEL_NAME}__{key}" for key in MODEL_FACTORIES]

_registry = ModelRegistry()
_model = None
_model_meta = None
# SHAP background sample, built once at startup from real training-time
# feature data
_explainer_background: pd.DataFrame | None = None


@app.on_event("startup")
def _load_production_model() -> None:
    global _model, _model_meta, _explainer_background
    try:
        _model, _model_meta = _registry.find_production_model(CANDIDATE_MODEL_NAMES)
        logger.info(f"api: loaded Production model {_model_meta.model_name} v{_model_meta.version}")

        from src.feature_store.store import FeatureStore
        store = FeatureStore()
        bg = store.read_features(columns=_model_meta.feature_columns).dropna()
        if not bg.empty:
            _explainer_background = bg.sample(n=min(100, len(bg)), random_state=42)
            logger.info(f"api: built SHAP background sample ({len(_explainer_background)} rows)")
        else:
            logger.warning("api: no usable rows for SHAP background sample — /forecast/explain "
                            "will fall back to per-request sampling from the request itself.")
    except LookupError:
        logger.warning(
            "api: no Production-stage model found among any family. "
            "Run scripts/train_models.py then scripts/promote_champion.py before serving."
        )
    except RuntimeError as exc:
        logger.error(f"api: ambiguous production state, refusing to pick one: {exc}")


def _compute_online_features(req: ForecastRequest, feature_columns: list[str]) -> pd.DataFrame:
    """Reconstruct a single feature row for `req.forecast_timestamp` using
    the caller-supplied history — same functions as offline training."""
    history = pd.DataFrame([r.model_dump() for r in req.recent_readings])
    history["household_id"] = req.household_id
    target_row = pd.DataFrame(
        [{"timestamp": req.forecast_timestamp, "household_id": req.household_id, "consumption_kwh": None}]
    )
    combined = pd.concat([history, target_row], ignore_index=True).sort_values("timestamp")
    combined["temperature_c"] = req.temperature_c

    # Collapse whatever raw cadence the caller polled at (e.g.
    # every few seconds from a live dashboard) onto the canonical
    # freq_minutes grid before computing lag/rolling features.
    # Otherwise lag_N/roll_*_N silently mean "N raw readings back"
    # instead of "N config.freq_minutes intervals back", diverging
    # from what the model was trained on.
    combined = resample_to_config_grid(combined, DEFAULT_CONFIG)

    combined = add_calendar_features(combined)
    combined = add_weather_interaction_features(combined)
    combined = add_lag_features(combined)
    combined = add_rolling_features(combined)

    row = combined.iloc[[-1]]
    missing = [c for c in feature_columns if c not in row.columns]
    for c in missing:
        row[c] = None
    return row[feature_columns]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        production_model_loaded=_model is not None,
        model_name=_model_meta.model_name if _model_meta is not None else None,
        model_version=_model_meta.version if _model_meta is not None else None,
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="No Production model loaded.",
        )
    X = _compute_online_features(req, _model_meta.feature_columns)
    X = X.fillna(0.0)  # request-time features shouldn't be NaN often; documented fallback
    y_pred = float(_model.predict(X)[0])
    return ForecastResponse(
        household_id=req.household_id,
        forecast_timestamp=req.forecast_timestamp,
        predicted_consumption_kwh=round(y_pred, 4),
        model_name=_model_meta.model_name,
        model_version=_model_meta.version,
        model_stage=_model_meta.stage,
    )


@app.post("/forecast/explain", response_model=ForecastExplanation)
def forecast_explain(req: ForecastRequest) -> ForecastExplanation:
    """Per-prediction SHAP attribution for a single forecast, answering
    "why did the model predict this number" rather than only a global
    feature-importance ranking."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="No Production model loaded.",
        )
    X = _compute_online_features(req, _model_meta.feature_columns)
    X = X.fillna(0.0)
    y_pred = float(_model.predict(X)[0])

    # Falls back to sampling background from X itself only if startup
    # couldn't build one from the FeatureStore (e.g. empty store) — a
    # single-row background is a poor explainer background in general, but
    # keeps this endpoint from hard-failing rather than degrading gracefully.
    background = _explainer_background if _explainer_background is not None else X
    shap_values = compute_shap_values(_model, X, X_background=background)
    contributions = explain_local_prediction(shap_values, row_index=0, feature_names=_model_meta.feature_columns)

    return ForecastExplanation(
        household_id=req.household_id,
        forecast_timestamp=req.forecast_timestamp,
        predicted_consumption_kwh=round(y_pred, 4),
        top_contributions=[
            FeatureContribution(
                feature=r.feature,
                shap_value=float(r.shap_value),
                feature_value=(None if pd.isna(r.feature_value) else float(r.feature_value)),
            )
            for r in contributions.head(8).itertuples()
        ],
    )


@app.post("/recommendations", response_model=RecommendationResponse)
def recommendations(req: ForecastRequest) -> RecommendationResponse:
    forecast_resp = forecast(req)
    recent_values = [r.consumption_kwh for r in req.recent_readings[-48:]]
    recent_mean = sum(recent_values) / len(recent_values) if recent_values else 0.0

    hour = req.forecast_timestamp.hour
    is_weekend = req.forecast_timestamp.weekday() >= 5
    heating_degree = max(18 - req.temperature_c, 0) if req.temperature_c is not None else 0.0

    recs = generate_recommendations(
        hour=hour,
        is_weekend=is_weekend,
        heating_degree=heating_degree,
        predicted_consumption_kwh=forecast_resp.predicted_consumption_kwh,
        recent_mean_kwh=recent_mean,
    )
    return RecommendationResponse(
        household_id=req.household_id,
        recommendations=[
            RecommendationItem(title=r.title, description=r.description, estimated_saving_kwh=r.estimated_saving_kwh)
            for r in recs
        ],
    )


@app.get("/monitoring/reference-sample")
def monitoring_reference_sample(n: int = 500, columns: str | None = None):
    """A small sample of the training-time Feature Store,
    exposed over REST so the dashboard's live Evidently drift comparison
    stays a REST-only client rather than reading `data/feature_store/`
    off disk directly.
    Keeping the response small matters since this may be called from
    a browser-side dashboard.
    """
    from src.feature_store.store import FeatureStore

    default_columns = ["temperature_c", "heating_degree", "lag_1", "lag_48", "roll_mean_48", "roll_std_48"]
    wanted = columns.split(",") if columns else default_columns

    store = FeatureStore()
    df = store.read_features(columns=wanted)
    sample = df[wanted].dropna().sample(n=min(n, len(df)), random_state=42)
    return {"columns": wanted, "n_rows": len(sample), "records": sample.to_dict(orient="records")}