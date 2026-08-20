"""
Sprint 7 — Optimisation Suggestions (App Engineering).

Implemented as a small, transparent rule engine over the model's
forecast that consumes the same feature vocabulary produced by
`feature_engineering.py` (hour, is_weekend, heating_degree, ...)
plus the model's forecast, so recommendations stay consistent with
whatever the model is reacting to.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.common.config import DEFAULT_CONFIG


@dataclass
class Recommendation:
    title: str
    description: str
    estimated_saving_kwh: float


PEAK_HOURS = set(range(17, 21))  # 17:00-20:59, matches the evening-peak shape
OFF_PEAK_HOURS = set(range(6)) | set(range(22, 24))

HIGH_CONSUMPTION_THRESHOLD_KWH = 0.85 * DEFAULT_CONFIG.live_meter_scale
HEATING_DEGREE_ALERT = 8.0  # heating_degree = max(18 - temp, 0); >8 means quite cold


def generate_recommendations(
    hour: int,
    is_weekend: bool,
    heating_degree: float,
    predicted_consumption_kwh: float,
    recent_mean_kwh: float,
) -> list[Recommendation]:
    recs: list[Recommendation] = []

    if hour in PEAK_HOURS and predicted_consumption_kwh > HIGH_CONSUMPTION_THRESHOLD_KWH:
        recs.append(
            Recommendation(
                title="Shift flexible loads out of the evening peak",
                description=(
                    "Your forecast consumption in this evening-peak window is high. "
                    "Shifting appliances such as a dishwasher, washing machine, or EV "
                    "charging to an off-peak window (22:00-06:00) reduces both cost "
                    "under time-of-use tariffs and grid-level peak demand."
                ),
                estimated_saving_kwh=round(predicted_consumption_kwh * 0.15, 3),
            )
        )

    if heating_degree > HEATING_DEGREE_ALERT:
        recs.append(
            Recommendation(
                title="Cold snap — check heating schedule efficiency",
                description=(
                    "Outdoor temperature is significantly below your comfort threshold, "
                    "which is driving heating-related demand up. Consider a 1-2°C "
                    "thermostat setback overnight or pre-heating during off-peak hours "
                    "instead of during the evening peak."
                ),
                estimated_saving_kwh=round(0.05 * heating_degree, 3),
            )
        )

    if predicted_consumption_kwh > recent_mean_kwh * 1.5 and recent_mean_kwh > 0:
        recs.append(
            Recommendation(
                title="Consumption forecast well above your recent average",
                description=(
                    "The forecast for this interval is more than 50% above your recent "
                    "average — worth checking for an appliance left running or a change "
                    "in occupancy before it becomes a billing surprise."
                ),
                estimated_saving_kwh=round(predicted_consumption_kwh - recent_mean_kwh, 3),
            )
        )

    if not recs:
        recs.append(
            Recommendation(
                title="No action needed",
                description="Forecast consumption for this interval looks typical for your household.",
                estimated_saving_kwh=0.0,
            )
        )

    return recs