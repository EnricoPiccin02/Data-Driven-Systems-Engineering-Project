"""
Central configuration for the Smart Energy Consumption Forecasting pipeline.
This acts as the single source of truth for every parameter describing the
household/location/date-range for both the offline training pipeline and
the live streaming client.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_STORE_DIR = PROJECT_ROOT / "data" / "feature_store"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, FEATURE_STORE_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PipelineConfig:
    """Deterministic pipeline configuration (seeded for reproducibility)."""

    random_seed: int = 42

    # Household population.
    # HH_REAL_000 is the real external household.
    # HH001 ... HH024 are the synthetic peer households.
    n_households: int = 25
    real_household_id: str = "HH_REAL_000"

    start_date: str = "2025-01-01"
    end_date: str = "2025-12-01"  # Exclusive upper bound -> first 11 months of 2025

    # Canonical consumption frequency used by the rest of the pipeline.
    freq_minutes: int = 30

    # Calendar.
    country_holidays: str = "IT"

    lag_steps: tuple = (1, 2, 48, 336)  # 30min, 1h, 1 day, 1 week (in 30-min steps)
    rolling_windows: tuple = (48, 336)  # 1 day, 1 week (in 30-min steps)

    # Real-data switch.
    # True  -> try the real external source.
    # False -> explicitly use synthetic data.
    use_real_data: bool = True

    # Real household electricity source.
    # Zenodo: "Household Electricity Energy Comsumption" dataset (19183126 records).
    # The dataset contains one Slovenian household with 15-minute
    # electricity consumption measurements covering 2024 and 2025.
    household_electricity_url: str = (
        "https://zenodo.org/records/19183126/files/"
        "Electricity_Consumption_dataset.zip?download=1"
    )

    # Location: University of Trieste, Piazzale Europa campus, Trieste, Italy.
    # Used both for training-time weather and as the default live-streaming
    # location.
    location_label: str = (
        "Trieste, Italy "
        "(weather scenario; real household electricity reference from Slovenia)"
    )
    weather_latitude: float = 45.6495
    weather_longitude: float = 13.7768
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"

    # Live-streaming defaults.
    live_household_id: str = "HH_LIVE_000"
    live_meter_scale: float = 1.0
    live_meter_seed: int = 42

    @property
    def raw_meter_path(self) -> Path:
        return DATA_RAW_DIR / "smart_meter.csv"

    @property
    def raw_weather_path(self) -> Path:
        return DATA_RAW_DIR / "weather.csv"

    @property
    def raw_calendar_path(self) -> Path:
        return DATA_RAW_DIR / "calendar.csv"

    @property
    def processed_path(self) -> Path:
        return DATA_PROCESSED_DIR / "cleaned_consumption.csv"

    @property
    def feature_table_path(self) -> Path:
        return FEATURE_STORE_DIR / "consumption_features.csv"


DEFAULT_CONFIG = PipelineConfig()