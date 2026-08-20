"""
Sprint 3 — Feature Store.

A deliberately minimal local Feature Store, composed of a CSV feature
table plus a JSON manifest recording the feature list, dtypes, generation
config, and a content hash.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common.config import DEFAULT_CONFIG, FEATURE_STORE_DIR, PipelineConfig
from src.common.logging_config import get_logger, stage
from src.common.versioning import file_sha256

logger = get_logger(__name__)

ENTITY_KEYS = ["household_id", "timestamp"]
TARGET_COLUMN = "consumption_kwh"


@dataclass
class FeatureStore:
    base_dir: Path = FEATURE_STORE_DIR

    @property
    def table_path(self) -> Path:
        return self.base_dir / "consumption_features.csv"

    @property
    def manifest_path(self) -> Path:
        return self.base_dir / "feature_manifest.json"

    def write_features(self, df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        with stage(logger, "feature_store:write", n_rows=len(df)):
            self.base_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.table_path, index=False)

            feature_columns = [c for c in df.columns if c not in ENTITY_KEYS + [TARGET_COLUMN]]
            manifest = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "entity_keys": ENTITY_KEYS,
                "target_column": TARGET_COLUMN,
                "feature_columns": feature_columns,
                "dtypes": {c: str(df[c].dtype) for c in df.columns},
                "n_rows": len(df),
                "config": {
                    "lag_steps": list(config.lag_steps),
                    "rolling_windows": list(config.rolling_windows),
                    "random_seed": config.random_seed,
                },
                "content_sha256": None,  # filled after write below
            }
            with open(self.manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            manifest["content_sha256"] = file_sha256(self.table_path)
            with open(self.manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"feature store written: {len(feature_columns)} features, {len(df)} rows")

    def read_features(self, columns: list[str] | None = None) -> pd.DataFrame:
        with stage(logger, "feature_store:read"):
            usecols = None
            if columns is not None:
                usecols = ENTITY_KEYS + [TARGET_COLUMN] + [c for c in columns if c not in ENTITY_KEYS + [TARGET_COLUMN]]
            df = pd.read_csv(self.table_path, usecols=usecols, parse_dates=["timestamp"])
            return df

    def get_feature_names(self) -> list[str]:
        with open(self.manifest_path) as f:
            manifest = json.load(f)
        return manifest["feature_columns"]

    def get_manifest(self) -> dict:
        with open(self.manifest_path) as f:
            return json.load(f)
