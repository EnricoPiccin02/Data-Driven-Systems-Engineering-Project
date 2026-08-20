"""
Sprints 1,2,3 - Orchestrates the full data pipeline pipeline:

1. Ingest
2. Validate
3. Clean
4. Transform
5. Feature Engineering
6. Feature Store

Usage:
    python3 scripts/run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.common.config import DEFAULT_CONFIG
from src.common.logging_config import get_logger, stage
from src.common.versioning import write_manifest
from src.data_engineering.clean import clean_calendar, clean_smart_meter, clean_weather
from src.data_engineering.feature_engineering import build_feature_table
from src.data_engineering.ingest import ingest_all
from src.data_engineering.transform import merge_sources
from src.data_engineering.validate import validate_raw
from src.feature_store.store import FeatureStore

logger = get_logger("pipeline")


def main() -> int:
    config = DEFAULT_CONFIG

    with stage(logger, "PIPELINE: Sprint 1 - Ingest"):
        raw = ingest_all(config)

    with stage(logger, "PIPELINE: Sprint 1 - Validate"):
        report = validate_raw(config)
        for suite, summary in report["summary"].items():
            if summary["n_critical_failures"] > 0:
                logger.warning(
                    f"Suite '{suite}' has {summary['n_critical_failures']} CRITICAL "
                    f"failures. Continuing (cleaning stage will remediate what it can) "
                    f"— see reports/data_quality_report.json"
                )

    with stage(logger, "PIPELINE: Sprint 2 - Version raw data"):
        write_manifest(
            [config.raw_meter_path, config.raw_weather_path, config.raw_calendar_path],
            config.raw_meter_path.parent / "raw_manifest.json",
        )

    with stage(logger, "PIPELINE: Sprint 3 - Clean"):
        meter_clean = clean_smart_meter(raw["smart_meter"], config)
        weather_clean = clean_weather(raw["weather"])
        calendar_clean = clean_calendar(raw["calendar"])
        meter_clean.to_csv(config.processed_path, index=False)

    with stage(logger, "PIPELINE: Sprint 3 - Transform (merge sources)"):
        merged = merge_sources(meter_clean, weather_clean, calendar_clean)

    with stage(logger, "PIPELINE: Sprint 3 - Feature engineering"):
        features = build_feature_table(merged, config)

    with stage(logger, "PIPELINE: Sprint 3 - Write Feature Store"):
        store = FeatureStore()
        store.write_features(features, config)

    logger.info("Pipeline complete. Feature store ready for Sprint 4 (modelling).")
    logger.info(f"  raw:       {config.raw_meter_path}")
    logger.info(f"  processed: {config.processed_path}")
    logger.info(f"  features:  {store.table_path} ({len(store.get_feature_names())} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
