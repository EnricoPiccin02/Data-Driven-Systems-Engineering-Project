"""
Sprint 7 — Real-Time Online Data Client.

Two real-time sources are combined:
  1. Real live weather — `LiveWeatherClient` calls Open-Meteo's
     `forecast` endpoint with `current=...`, which returns genuinely live,
     no-API-key, right-now weather for any lat/lon.
  2. Live meter stream — no free, unauthenticated, real-time household
     smart-meter feed exists publicly. `LiveMeterSimulator` generates a
     physically-plausible reading synchronised to actual wall-clock time.
     Explicitly labelled `is_real_data=False` in every reading.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.common.config import DEFAULT_CONFIG
from src.common.logging_config import get_logger

logger = get_logger(__name__)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CONNECTIVITY_CHECK_URL = "https://api.open-meteo.com/v1/forecast"
CONNECTIVITY_TIMEOUT_SECONDS = 3


def is_online(timeout: float = CONNECTIVITY_TIMEOUT_SECONDS) -> bool:
    """Cheap and fast connectivity probe."""
    try:
        resp = requests.get(
            CONNECTIVITY_CHECK_URL,
            params={
                "latitude": DEFAULT_CONFIG.weather_latitude,
                "longitude": DEFAULT_CONFIG.weather_longitude,
                "current_weather": True,
            },
            timeout=timeout,
        )
        return resp.status_code == 200
    except requests.RequestException as exc:
        logger.info(f"live_data: connectivity check failed ({type(exc).__name__}) — "
                    f"dashboard will fall back to CSV upload")
        return False


class LiveWeatherClient:
    """Real, live weather via Open-Meteo's Forecast API (no API key)."""

    def __init__(
        self,
        latitude: float = DEFAULT_CONFIG.weather_latitude,
        longitude: float = DEFAULT_CONFIG.weather_longitude,
    ):
        self.latitude = latitude
        self.longitude = longitude

        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )

        self.session = requests.Session()
        self.session.mount(
            "https://",
            HTTPAdapter(max_retries=retry),
        )

    def fetch_current(self) -> dict:
        resp = self.session.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": self.latitude,
                "longitude": self.longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "wind_speed_10m"
                ),
                "timezone": "UTC",
            },
            timeout=(10, 30),
        )
        resp.raise_for_status()

        current = resp.json()["current"]
        wind_speed = current["wind_speed_10m"]

        return {
            "timestamp": current["time"],
            "temperature_c": current["temperature_2m"],
            "humidity_pct": current["relative_humidity_2m"],
            "wind_speed_ms": (
                wind_speed / 3.6
                if wind_speed is not None
                else 0.0
            ),
            "is_real_data": True,
        }


class LiveMeterSimulator:
    """Wall-clock-synchronised live household reading."""

    def __init__(
        self,
        household_id: str = DEFAULT_CONFIG.live_household_id,
        scale: float = DEFAULT_CONFIG.live_meter_scale,
        seed: int = DEFAULT_CONFIG.live_meter_seed,
    ):
        self.household_id = household_id
        self.scale = scale
        self._rng = np.random.default_rng(seed)

    def fetch_current(self) -> dict:
        now = datetime.now(timezone.utc)
        hour = now.hour + now.minute / 60.0
        is_weekend = now.weekday() >= 5

        morning_peak = math.exp(-((hour - 7.5) ** 2) / (2 * 1.2 ** 2))
        evening_peak = math.exp(-((hour - 19.0) ** 2) / (2 * 2.0 ** 2))
        shape = 0.25 + 0.55 * morning_peak + 0.9 * evening_peak
        weekend_uplift = 1.15 if is_weekend else 1.0
        noise = float(self._rng.normal(0, 0.06))

        consumption = max(0.02, self.scale * shape * weekend_uplift * (1 + noise))
        return {
            "timestamp": now.isoformat(),
            "household_id": self.household_id,
            "consumption_kwh": consumption,
            "is_real_data": False,
        }


@dataclass
class LiveDataStreamer:
    """Combines live weather + the live meter simulator into one reading per
    call, and keeps a rolling in-memory buffer (most recent `maxlen` readings)."""

    household_id: str = DEFAULT_CONFIG.live_household_id
    maxlen: int = 400  # > 336 (the longest lag window) with headroom
    weather_client: LiveWeatherClient = field(default_factory=LiveWeatherClient)
    meter_client: LiveMeterSimulator = field(default_factory=lambda: LiveMeterSimulator())
    buffer: deque = field(default_factory=lambda: deque(maxlen=400))

    def poll(self) -> dict:
        """Fetch one new combined reading, append it to the buffer, and
        return it. Call this once per dashboard refresh tick."""
        weather = self.weather_client.fetch_current()
        meter = self.meter_client.fetch_current()
        reading = {
            "timestamp": meter["timestamp"],
            "household_id": self.household_id,
            "consumption_kwh": meter["consumption_kwh"],
            "is_real_data": meter["is_real_data"],
            "temperature_c": weather["temperature_c"],
            "humidity_pct": weather["humidity_pct"],
            "wind_speed_ms": weather["wind_speed_ms"],
            "is_real_weather": weather["is_real_data"],
            "source_latitude": self.weather_client.latitude,
            "source_longitude": self.weather_client.longitude,
        }
        self.buffer.append(reading)
        return reading

    def history_dataframe(self):
        import pandas as pd
        df = pd.DataFrame(list(self.buffer))
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df