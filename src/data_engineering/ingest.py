"""
Sprint 1 - Data Collection.
"""
from __future__ import annotations

import abc
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.common.config import DEFAULT_CONFIG, PipelineConfig
from src.common.logging_config import get_logger, stage

logger = get_logger(__name__)


class DataSource(abc.ABC):
    """Common interface every raw data source must implement."""

    name: str

    @abc.abstractmethod
    def fetch(self, config: PipelineConfig) -> pd.DataFrame:
        ...

    def save(self, config: PipelineConfig, out_path: Path) -> pd.DataFrame:
        with stage(logger, f"ingest:{self.name}", out=out_path.name):
            df = self.fetch(config)
            df.to_csv(out_path, index=False)
            logger.info(f"{self.name}: wrote {len(df):,} rows -> {out_path}")
        return df


class RealOrFallbackSource(DataSource):
    """Wraps a real DataSource with a synthetic fallback.
    Callers always get a DataFrame back, but `is_real_data` tells every
    downstream consumer exactly which rows came from the real API/dataset
    and which are synthetic stand-ins."""

    def __init__(self, name: str, real: DataSource, fallback: DataSource):
        self.name = name
        self.real = real
        self.fallback = fallback

    def fetch(self, config: PipelineConfig) -> pd.DataFrame:
        if not config.use_real_data:
            logger.info(f"{self.name}: use_real_data=False, using synthetic fallback by config")
            df = self.fallback.fetch(config)
            df["is_real_data"] = False
            return df
        try:
            df = self.real.fetch(config)
            df["is_real_data"] = True
            logger.info(f"{self.name}: fetched REAL data ({len(df):,} rows)")
            return df
        except Exception as exc:  # noqa: BLE001 — Deliberately broad: any network/parse
            # Failure must degrade gracefully, not crash the pipeline.
            logger.warning(
                f"{self.name}: real source failed ({type(exc).__name__}: {exc}); "
                f"falling back to synthetic generator. Pipeline continues, but "
                f"downstream consumers should check the is_real_data column."
            )
            df = self.fallback.fetch(config)
            df["is_real_data"] = False
            return df


# --------------------------------------------------------------------------- #
# Real sources
# --------------------------------------------------------------------------- #

class HouseholdElectricity2025Source(DataSource):
    """
    Real household electricity data for the 2025 target period.

    Source:
        Fatima, Aziz & Bernard, Ženko.
        "Household Electricity Energy Comsumption"
        Zenodo record 19183126.

    The dataset contains one Slovenian household with 15-minute
    electricity-consumption measurements covering 2024 and 2025.

    The source is downloaded as a small ZIP containing CSV data.
    The data are converted to the pipeline's canonical schema:

        timestamp
        household_id
        consumption_kwh

    The original measurements are in kWh per 15-minute interval.
    They are aggregated to config.freq_minutes by summing the
    energy consumed within each interval.
    """

    name = "household_electricity_2025"

    def fetch(self, config: PipelineConfig) -> pd.DataFrame:
        max_attempts = 3

        start = pd.Timestamp(config.start_date)
        end = pd.Timestamp(config.end_date)

        target_year = start.year

        # Download the ZIP with retries.
        resp = None

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(
                    f"{self.name}: downloading real household "
                    f"electricity dataset "
                    f"(attempt {attempt}/{max_attempts})"
                )

                resp = requests.get(
                    config.household_electricity_url,
                    timeout=60,
                )

                resp.raise_for_status()

                logger.info(
                    f"{self.name}: dataset download succeeded "
                    f"on attempt {attempt}/{max_attempts}"
                )

                break

            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
            ) as exc:
                if attempt == max_attempts:
                    logger.warning(
                        f"{self.name}: all {max_attempts} download "
                        f"attempts failed."
                    )
                    raise

                wait_seconds = 2 ** (attempt - 1)

                logger.warning(
                    f"{self.name}: download failed on attempt "
                    f"{attempt}/{max_attempts} "
                    f"({type(exc).__name__}: {exc}); "
                    f"retrying in {wait_seconds}s..."
                )

                import time

                time.sleep(wait_seconds)

        if resp is None:
            raise RuntimeError(
                f"{self.name}: download failed unexpectedly."
            )

        # Open the ZIP and select the CSV corresponding to the target year.
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            all_members = zf.namelist()

            logger.info(
                f"{self.name}: ZIP files: {all_members}"
            )

            csv_names = sorted(
                name
                for name in all_members
                if name.lower().endswith(".csv")
            )

            if not csv_names:
                raise ValueError(
                    f"{self.name}: dataset ZIP contains no CSV files."
                )

            # Look for Electricity_Consumption_dataset/2025_UTC.csv
            target_year_string = str(target_year)

            matching_csv_names = [
                name
                for name in csv_names
                if target_year_string in Path(name).stem
            ]

            if not matching_csv_names:
                raise ValueError(
                    f"{self.name}: could not find a CSV for target "
                    f"year {target_year}. Available CSV files: "
                    f"{csv_names}"
                )

            if len(matching_csv_names) > 1:
                raise ValueError(
                    f"{self.name}: multiple CSV files match target "
                    f"year {target_year}: {matching_csv_names}"
                )

            csv_name = matching_csv_names[0]

            logger.info(
                f"{self.name}: selected source file "
                f"'{csv_name}' for target year {target_year}"
            )

            with zf.open(csv_name) as f:
                raw = pd.read_csv(f)

        if raw.empty:
            raise ValueError(
                f"{self.name}: selected CSV '{csv_name}' is empty."
            )

        # Normalize column names.
        raw.columns = [
            str(column).strip().lower()
            for column in raw.columns
        ]

        logger.info(
            f"{self.name}: source columns after normalization: "
            f"{list(raw.columns)}"
        )

        # Identify timestamp column.
        timestamp_candidates = [
            "timestamp",
            "datetime",
            "date_time",
            "date",
            "time",
        ]

        timestamp_columns = [
            column
            for column in raw.columns
            if column in timestamp_candidates
        ]

        if not timestamp_columns:
            raise ValueError(
                f"{self.name}: could not identify timestamp column "
                f"in household electricity dataset. Available "
                f"columns: {list(raw.columns)}"
            )

        timestamp_column = timestamp_columns[0]

        # Identify electricity-consumption column.
        consumption_candidates = [
            "consumption_kwh",
            "electricity_consumption_kwh",
            "electricity_consumption",
            "consumption",
            "kwh",
            "energy_kwh",
            "energy",
            "energy_a_plus",
        ]

        consumption_column = next(
            (
                column
                for column in consumption_candidates
                if column in raw.columns
            ),
            None,
        )

        if consumption_column is None:
            # Fall back to identifying a numeric column that is not the timestamp.
            numeric_candidates = [
                column
                for column in raw.columns
                if column != timestamp_column
                and pd.api.types.is_numeric_dtype(raw[column])
            ]

            if len(numeric_candidates) == 1:
                consumption_column = numeric_candidates[0]
            else:
                raise ValueError(
                    f"{self.name}: could not identify "
                    f"electricity-consumption column. Available "
                    f"columns: {list(raw.columns)}"
                )

        logger.info(
            f"{self.name}: using timestamp column "
            f"'{timestamp_column}' and consumption column "
            f"'{consumption_column}'"
        )

        # Parse timestamps.
        raw["timestamp"] = pd.to_datetime(
            raw[timestamp_column],
            errors="coerce",
            utc=True,
        )

        logger.info(
            f"{self.name}: raw timestamp range after parsing: "
            f"{raw['timestamp'].min()} -> {raw['timestamp'].max()}"
        )

        logger.info(
            f"{self.name}: valid timestamp rows: "
            f"{raw['timestamp'].notna().sum():,} / {len(raw):,}"
        )

        # Parse consumption values.
        raw["consumption_kwh"] = pd.to_numeric(
            raw[consumption_column],
            errors="coerce",
        )

        raw = raw[
            ["timestamp", "consumption_kwh"]
        ].dropna()

        if raw.empty:
            raise ValueError(
                f"{self.name}: no valid timestamp/consumption "
                f"rows remain after parsing."
            )

        # Convert UTC-aware timestamps to naive timestamps, matching the
        # rest of the existing pipeline.
        raw["timestamp"] = raw["timestamp"].dt.tz_localize(None)

        raw = raw.sort_values("timestamp")

        # Electricity consumption cannot be negative. Negative values would
        # normally indicate generation/export rather than consumption.
        raw = raw[raw["consumption_kwh"] >= 0]

        if raw.empty:
            raise ValueError(
                f"{self.name}: no non-negative consumption "
                f"observations remain."
            )

        logger.info(
            f"{self.name}: raw timestamp range after parsing: "
            f"{raw['timestamp'].min()} -> "
            f"{raw['timestamp'].max()}"
        )

        logger.info(
            f"{self.name}: valid timestamp rows: "
            f"{len(raw):,}"
        )

        # Filter against the pipeline's target period.
        raw = raw[
            (raw["timestamp"] >= start)
            & (raw["timestamp"] < end)
        ]

        if raw.empty:
            raise ValueError(
                f"{self.name}: no observations available in "
                f"[{start}, {end}). Source file '{csv_name}' "
                f"was selected for target year {target_year}, "
                f"but its parsed timestamps do not overlap the "
                f"requested period."
            )

        # Resample from the source frequency to the canonical frequency.
        # Energy_A_plus is already energy in kWh per source interval,
        # so multiple intervals are added together.
        resampled = (
            raw.set_index("timestamp")[
                "consumption_kwh"
            ]
            .resample(
                f"{config.freq_minutes}min"
            )
            .sum()
            .rename("consumption_kwh")
            .to_frame()
            .reset_index()
        )

        # Assign the canonical real-household ID.
        resampled["household_id"] = (
            config.real_household_id
        )

        result = resampled[
            [
                "timestamp",
                "household_id",
                "consumption_kwh",
            ]
        ]

        logger.info(
            f"{self.name}: produced {len(result):,} "
            f"canonical {config.freq_minutes}-minute observations "
            f"for {config.real_household_id}"
        )

        return result


class OpenMeteoWeatherSource(DataSource):
    """Real historical hourly weather from the Open-Meteo Archive API."""

    name = "open_meteo_weather"

    def fetch(self, config: PipelineConfig) -> pd.DataFrame:
        params = {
            "latitude": config.weather_latitude,
            "longitude": config.weather_longitude,
            "start_date": config.start_date,
            "end_date": (
                pd.Timestamp(config.end_date)
                - pd.Timedelta(days=1)
            ).date().isoformat(),
            "hourly": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "wind_speed_10m"
            ),
            "timezone": "UTC",
        }

        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(
                    f"{self.name}: requesting real weather data "
                    f"(attempt {attempt}/{max_attempts})"
                )

                resp = requests.get(
                    config.open_meteo_archive_url,
                    params=params,
                    timeout=30,
                )

                resp.raise_for_status()
                payload = resp.json()
                hourly = payload["hourly"]

                df = pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            hourly["time"]
                        ),
                        "temperature_c": hourly[
                            "temperature_2m"
                        ],
                        "humidity_pct": hourly[
                            "relative_humidity_2m"
                        ],
                        "wind_speed_ms": [
                            w / 3.6
                            for w in hourly[
                                "wind_speed_10m"
                            ]
                        ],
                    }
                )

                if df.empty:
                    raise ValueError(
                        "Open-Meteo returned an empty hourly dataset."
                    )

                logger.info(
                    f"{self.name}: successfully retrieved "
                    f"{len(df):,} hourly observations"
                )

                return df

            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
            ) as exc:
                if attempt == max_attempts:
                    logger.warning(
                        f"{self.name}: all {max_attempts} attempts "
                        f"failed."
                    )
                    raise

                wait_seconds = 2 ** (attempt - 1)

                logger.warning(
                    f"{self.name}: request failed on attempt "
                    f"{attempt}/{max_attempts} "
                    f"({type(exc).__name__}: {exc}); "
                    f"retrying in {wait_seconds}s..."
                )

                import time

                time.sleep(wait_seconds)

            except (KeyError, ValueError) as exc:
                logger.warning(
                    f"{self.name}: invalid API response "
                    f"({type(exc).__name__}: {exc})"
                )
                raise


# --------------------------------------------------------------------------- #
# Synthetic sources used as fallback/peer household generator
# --------------------------------------------------------------------------- #

def _timestamp_index(config: PipelineConfig) -> pd.DatetimeIndex:
    return pd.date_range(
        start=config.start_date,
        end=config.end_date,
        freq=f"{config.freq_minutes}min",
        inclusive="left",
    )


class SyntheticSmartMeterGenerator(DataSource):
    """
    Half-hourly household electricity consumption (kWh), synthetic peers.

    Used to
        (a) populate the multi-household population alongside the one real household, and
        (b) as the offline fallback for the real household itself the network is unreachable.
    """

    name = "smart_meter"

    def __init__(self, n_households: int | None = None, id_offset: int = 0):
        self._n_households = n_households
        self._id_offset = id_offset

    def fetch(self, config: PipelineConfig) -> pd.DataFrame:
        rng = np.random.default_rng(config.random_seed)
        ts = _timestamp_index(config)
        n = len(ts)
        n_households = self._n_households if self._n_households is not None else config.n_households

        hour = ts.hour + ts.minute / 60.0
        is_weekend = ts.dayofweek >= 5
        weekend_uplift = np.where(is_weekend, 1.15, 1.0)

        rows = []
        for i in range(n_households):
            hh = i + self._id_offset
            household_id = f"HH{hh:03d}"
            scale = rng.uniform(0.6, 2.2)
            phase_jitter = rng.normal(0, 0.15)
            noise = rng.normal(0, 0.06, size=n)
            meter_dropout = rng.random(n) < 0.001

            shape = 0.25 + 0.55 * np.exp(
                -((hour - (7.5 + phase_jitter)) ** 2) / (2 * 1.2 ** 2)
            ) + 0.9 * np.exp(-((hour - (19.0 + phase_jitter)) ** 2) / (2 * 2.0 ** 2))

            consumption = scale * shape * weekend_uplift * (1 + noise)
            consumption = np.clip(consumption, 0.02, None)
            consumption = np.where(meter_dropout, np.nan, consumption)

            rows.append(
                pd.DataFrame(
                    {"timestamp": ts, "household_id": household_id, "consumption_kwh": consumption}
                )
            )

        df = pd.concat(rows, ignore_index=True)

        dup_sample = df.sample(frac=0.0005, random_state=config.random_seed)
        df = pd.concat([df, dup_sample], ignore_index=True)
        spike_idx = df.sample(n=3, random_state=config.random_seed).index
        df.loc[spike_idx, "consumption_kwh"] = df.loc[spike_idx, "consumption_kwh"] * 50

        return df.sort_values(["household_id", "timestamp"]).reset_index(drop=True)


class SyntheticWeatherGenerator(DataSource):
    """Fallback weather generator (used only if Open-Meteo is unreachable)."""

    name = "weather"

    def fetch(self, config: PipelineConfig) -> pd.DataFrame:
        rng = np.random.default_rng(config.random_seed + 1)
        ts = pd.date_range(start=config.start_date, end=config.end_date, freq="1h", inclusive="left")
        n = len(ts)
        day_of_year = ts.dayofyear.values
        hour = ts.hour.values

        annual = -6 * np.cos(2 * np.pi * day_of_year / 365.25)
        diurnal = 3 * np.sin(2 * np.pi * (hour - 6) / 24)
        walk = np.cumsum(rng.normal(0, 0.3, size=n))
        walk -= walk.mean()
        temperature = 8 + annual + diurnal + walk

        humidity = np.clip(70 - 0.8 * diurnal + rng.normal(0, 5, size=n), 20, 100)
        wind_speed = np.clip(4 + rng.normal(0, 2, size=n), 0, None)

        return pd.DataFrame(
            {"timestamp": ts, "temperature_c": temperature, "humidity_pct": humidity, "wind_speed_ms": wind_speed}
        )


class CalendarSource(DataSource):
    """Generate calendar features from the holidays package."""

    name = "calendar"

    def fetch(self, config: PipelineConfig) -> pd.DataFrame:
        import holidays as holidays_lib

        days = pd.date_range(
            config.start_date,
            config.end_date,
            freq="1D",
            inclusive="left",
        )

        df = pd.DataFrame({"date": days})

        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"] >= 5

        years = sorted({d.year for d in days})

        country_holidays = holidays_lib.country_holidays(
            config.country_holidays,
            years=years,
        )

        holiday_dates = {
            d.isoformat()
            for d in country_holidays
        }

        df["is_holiday"] = (
            df["date"]
            .dt.strftime("%Y-%m-%d")
            .isin(holiday_dates)
        )

        # These are generated from an authoritative holiday calendar,
        # not synthetic measurements.
        df["is_real_data"] = True

        return df


# --------------------------------------------------------------------------- #
# Source registry / orchestration
# --------------------------------------------------------------------------- #

class _CombinedSmartMeterSource(DataSource):
    """
    Real household + synthetic peer population.

    HH_REAL_000 comes from the real external household dataset.
    HH001 ... HH024 are synthetic peer households.
    """

    name = "smart_meter"

    def fetch(self, config: PipelineConfig) -> pd.DataFrame:
        real_or_fallback_real_hh = RealOrFallbackSource(
            "smart_meter:real_household",
            real=HouseholdElectricity2025Source(),
            fallback=SyntheticSmartMeterGenerator(
                n_households=1,
                id_offset=0,
            ),
        )

        real_household_df = real_or_fallback_real_hh.fetch(config)

        # When real data were successfully obtained, explicitly assign
        # the canonical real-household ID.
        if real_household_df["is_real_data"].iloc[0]:
            real_household_df["household_id"] = config.real_household_id

        # All remaining households are synthetic peers.
        n_peers = max(config.n_households - 1, 0)

        peers_df = SyntheticSmartMeterGenerator(
            n_households=n_peers,
            id_offset=1,
        ).fetch(config)

        peers_df["is_real_data"] = False

        combined = pd.concat(
            [real_household_df, peers_df],
            ignore_index=True,
        )

        return (
            combined
            .sort_values(["household_id", "timestamp"])
            .reset_index(drop=True)
        )


SOURCES = {
    "smart_meter": _CombinedSmartMeterSource(),

    "weather": RealOrFallbackSource(
        "weather",
        real=OpenMeteoWeatherSource(),
        fallback=SyntheticWeatherGenerator(),
    ),

    "calendar": CalendarSource(),
}


def ingest_all(config: PipelineConfig = DEFAULT_CONFIG) -> dict[str, pd.DataFrame]:
    results = {}
    results["smart_meter"] = SOURCES["smart_meter"].save(config, config.raw_meter_path)
    results["weather"] = SOURCES["weather"].save(config, config.raw_weather_path)
    results["calendar"] = SOURCES["calendar"].save(config, config.raw_calendar_path)
    return results


if __name__ == "__main__":
    ingest_all()
