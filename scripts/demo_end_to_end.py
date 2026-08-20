"""
Sprint 9 - Runs every layer of the MLOps workflow in-process, in the order a live system would

1. data
2. features
3. train/register
4. promote
5. serve one forecast
6. recommend
7. drift-check

without needing the API/dashboard processes up.

Usage:
    python3 scripts/demo_end_to_end.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.promote_champion import promote_champion
from src.app.recommendations import generate_recommendations
from src.common.config import DEFAULT_CONFIG
from src.common.logging_config import get_logger
from src.data_engineering.clean import clean_calendar, clean_smart_meter, clean_weather
from src.data_engineering.feature_engineering import build_feature_table
from src.data_engineering.ingest import ingest_all
from src.data_engineering.transform import merge_sources
from src.data_engineering.validate import validate_raw
from src.feature_store.store import FeatureStore
from src.ml.registry import ModelRegistry
from src.ml.train import train_and_compare
from src.mlops.monitoring import detect_feature_drift
from src.mlops.retrain_trigger import decide_retrain

logger = get_logger("demo")


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main() -> None:
    config = DEFAULT_CONFIG

    banner("STAGE 1/6 — Data Engineering: ingest + validate")
    raw = ingest_all(config)
    quality_report = validate_raw(config)
    print(json.dumps(quality_report["summary"], indent=2))
    print(
        f"Real-data provenance: "
        f"{raw['smart_meter']['is_real_data'].mean():.1%} of meter rows are real."
    )

    banner("STAGE 2/6 — Data Engineering: clean + transform + features")
    meter_clean = clean_smart_meter(raw["smart_meter"], config)
    weather_clean = clean_weather(raw["weather"])
    calendar_clean = clean_calendar(raw["calendar"])
    merged = merge_sources(meter_clean, weather_clean, calendar_clean)
    features = build_feature_table(merged, config)
    store = FeatureStore()
    store.write_features(features, config)
    print(f"Feature store: {len(features):,} rows, {len(store.get_feature_names())} features")

    banner("STAGE 3/6 — ML Engineering: train, compare, register")
    results = train_and_compare(config)
    print(json.dumps(results, indent=2, default=str))

    banner("STAGE 4/6 — ML Engineering: promote champion to Production")
    registry = ModelRegistry()
    promotion_result = promote_champion()  # standalone step
    model_name = promotion_result["model_name"]
    model, meta = registry.get_production_model(model_name)
    print(f"Production model: {model_name} v{meta.version} (val MAE={meta.metrics['mae']:.4f})")

    banner("STAGE 5/6 — App Engineering: one forecast + recommendation (in-process)")
    sample = features.dropna(subset=meta.feature_columns).iloc[-1:]
    X = sample[meta.feature_columns]
    y_pred = float(model.predict(X)[0])
    row = sample.iloc[0]
    recs = generate_recommendations(
        hour=int(row["hour"]), is_weekend=bool(row["is_weekend"]),
        heating_degree=float(row["heating_degree"]), predicted_consumption_kwh=y_pred,
        recent_mean_kwh=float(row["roll_mean_48"]) if pd.notna(row["roll_mean_48"]) else y_pred,
    )
    print(f"Household {row['household_id']} @ {row['timestamp']}: forecast={y_pred:.3f} kWh")
    for r in recs:
        print(f"  - [{r.title}] {r.description[:90]}... (est. saving {r.estimated_saving_kwh} kWh)")

    banner("STAGE 6/6 — MLOps: drift check against real online data + retraining decision")
    print("Polling live Open-Meteo weather + the wall-clock-synced live meter stream.")
    from scripts.run_drift_check import NUMERIC_FEATURES_TO_MONITOR, collect_live_window
    current_features, adwin_status = collect_live_window(n_polls=10, poll_interval_seconds=1.0)
    drift_results = detect_feature_drift(features, current_features, NUMERIC_FEATURES_TO_MONITOR)
    decision = decide_retrain(drift_results, days_since_last_train=1)
    print(f"Drift flags (Evidently, vs. real live data): "
          f"{sum(r.drift_detected for r in drift_results)}/{len(drift_results)} features")
    print(f"Streaming drift (River ADWIN): {adwin_status['n_drift_events_total']} events "
          f"across {adwin_status['n_updates']} live updates")
    print(f"Retrain decision: should_retrain={decision.should_retrain} urgency={decision.urgency}")
    for reason in decision.reasons:
        print(f"  - {reason}")

    banner("DEMO COMPLETE")


if __name__ == "__main__":
    main()
