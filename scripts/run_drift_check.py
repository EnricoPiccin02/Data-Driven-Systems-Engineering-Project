"""
Sprint 8 - Automated Drift Check.

Also runs the streaming ADWIN detector across every polled live point, so
the script's output reports both the batch (Evidently) signal and the online (ADWIN) signal.

Usage:
    python3 scripts/run_drift_check.py                                          # defaults: 30 live polls, 1s apart
    python3 scripts/run_drift_check.py --n-polls 60 --poll-interval-seconds 2
    python3 scripts/run_drift_check.py --no-auto-retrain                        # report only, never retrain
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.promote_champion import promote_champion
from src.common.config import DEFAULT_CONFIG
from src.common.logging_config import get_logger
from src.data_engineering.clean import clean_smart_meter, clean_weather
from src.data_engineering.feature_engineering import build_feature_table
from src.data_engineering.ingest import CalendarSource
from src.data_engineering.transform import merge_sources
from src.feature_store.store import FeatureStore
from src.ml.train import train_and_compare
from src.mlops.monitoring import dataset_drift_summary, detect_feature_drift
from src.mlops.retrain_trigger import decide_retrain
from src.mlops.streaming_drift import OnlineDriftMonitor
from src.streaming.live_data import LiveDataStreamer, is_online

logger = get_logger("run_drift_check")

NUMERIC_FEATURES_TO_MONITOR = [
    "temperature_c", "heating_degree", "lag_1", "lag_48", "roll_mean_48", "roll_std_48",
]


def collect_live_window(n_polls: int, poll_interval_seconds: float) -> tuple[pd.DataFrame, list]:
    """Polls real online data `n_polls` times, `poll_interval_seconds` apart,
    feeding every point through the streaming ADWIN monitor as it arrives,
    and returns (feature_engineered_dataframe, adwin_drift_events)."""
    streamer = LiveDataStreamer()
    online_monitor = OnlineDriftMonitor()
    drift_events_seen = []

    for i in range(n_polls):
        reading = streamer.poll()
        newly_drifted = online_monitor.update(
            {"consumption_kwh": reading["consumption_kwh"], "temperature_c": reading["temperature_c"]}
        )
        if newly_drifted:
            drift_events_seen.append({"poll": i, "features": newly_drifted, "timestamp": reading["timestamp"]})
        if i < n_polls - 1:
            time.sleep(poll_interval_seconds)

    live_df = streamer.history_dataframe()
    live_df["timestamp"] = pd.to_datetime(live_df["timestamp"])

    # Run the same offline feature-engineering path over the live window so
    # the comparison against the training-time reference is aligned.
    calendar = CalendarSource().fetch(DEFAULT_CONFIG)
    meter_clean = clean_smart_meter(live_df[["timestamp", "household_id", "consumption_kwh", "is_real_data"]], DEFAULT_CONFIG)
    weather_clean = clean_weather(live_df[["timestamp", "temperature_c", "humidity_pct", "wind_speed_ms"]])
    merged = merge_sources(meter_clean, weather_clean, calendar)
    features = build_feature_table(merged, DEFAULT_CONFIG)

    return features, online_monitor.status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-polls", type=int, default=30)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--no-auto-retrain", action="store_true",
                        help="Report the drift/retrain decision without acting on it.")
    args = parser.parse_args()

    if not is_online():
        logger.error(
            "run_drift_check: no network access detected. This job requires real "
            "online data (Open-Meteo) by design — it does not fall back to "
            "comparing two halves of a static file. Aborting."
        )
        return 2

    logger.info(f"run_drift_check: polling {args.n_polls} live readings, "
                f"{args.poll_interval_seconds}s apart...")
    current_features, adwin_status = collect_live_window(args.n_polls, args.poll_interval_seconds)

    store = FeatureStore()
    reference_features = store.read_features()

    drift_results = detect_feature_drift(reference_features, current_features, NUMERIC_FEATURES_TO_MONITOR)
    dataset_summary = dataset_drift_summary(reference_features, current_features, NUMERIC_FEATURES_TO_MONITOR)
    decision = decide_retrain(drift_results, days_since_last_train=1)

    report = {
        "reference_window": "training-time Feature Store (data/feature_store/consumption_features.csv)",
        "current_window": f"{args.n_polls} live online readings (Open-Meteo + live meter stream)",
        "batch_drift_results (evidently)": [r.__dict__ for r in drift_results],
        "dataset_drift_summary (evidently)": dataset_summary,
        "streaming_drift_status (river ADWIN)": adwin_status,
        "retrain_decision": decision.__dict__,
    }
    print(json.dumps(report, indent=2, default=str))

    if decision.should_retrain and not args.no_auto_retrain:
        logger.warning(f"run_drift_check: should_retrain=True (urgency={decision.urgency}) — "
                        f"automatically retraining and promoting the new champion.")
        train_and_compare()
        promotion_result = promote_champion()
        print(json.dumps({"auto_retrain_promotion_result": promotion_result}, indent=2, default=str))
    elif decision.should_retrain:
        logger.info("run_drift_check: should_retrain=True but --no-auto-retrain set — reporting only.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
