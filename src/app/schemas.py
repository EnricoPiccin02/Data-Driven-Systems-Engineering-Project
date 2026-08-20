"""
Sprint 6 — API schemas.

Kept separate from `api.py` so the contract (what a client sends/receives)
is reviewable independently of the routing/serving logic.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RecentReading(BaseModel):
    """One historical consumption reading, used to compute lag/rolling
    features online at request time."""

    timestamp: datetime
    consumption_kwh: float


class ForecastRequest(BaseModel):
    household_id: str = Field(..., examples=["HH000"])
    forecast_timestamp: datetime = Field(
        ..., description="The half-hour interval to forecast consumption for."
    )
    recent_readings: list[RecentReading] = Field(
        ..., description="At least the last 7 days of half-hourly readings, "
        "most recent last, so lag_336/roll_*_336 can be computed."
    )
    temperature_c: float | None = Field(
        None, description="Forecast/observed temperature for forecast_timestamp, if known."
    )


class ForecastResponse(BaseModel):
    household_id: str
    forecast_timestamp: datetime
    predicted_consumption_kwh: float
    model_name: str
    model_version: int
    model_stage: str


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    feature_value: float | None = None


class ForecastExplanation(BaseModel):
    household_id: str
    forecast_timestamp: datetime
    predicted_consumption_kwh: float
    top_contributions: list[FeatureContribution]


class RecommendationItem(BaseModel):
    title: str
    description: str
    estimated_saving_kwh: float


class RecommendationResponse(BaseModel):
    household_id: str
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    production_model_loaded: bool
    model_name: str | None = None
    model_version: int | None = None
