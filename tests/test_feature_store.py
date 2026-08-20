"""
Sprint 3 — Feature Store tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import PipelineConfig
from src.feature_store.store import FeatureStore


def test_feature_store_write_and_read_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "household_id": ["HH000", "HH000"],
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="30min"),
            "consumption_kwh": [1.0, 2.0],
            "lag_1": [np.nan, 1.0],
        }
    )
    store = FeatureStore(base_dir=tmp_path)
    store.write_features(df, PipelineConfig())

    assert store.table_path.exists()
    assert store.manifest_path.exists()

    read_back = store.read_features()
    assert len(read_back) == 2
    assert "lag_1" in store.get_feature_names()


def test_feature_store_manifest_has_content_hash(tmp_path):
    df = pd.DataFrame(
        {
            "household_id": ["HH000"],
            "timestamp": pd.date_range("2024-01-01", periods=1, freq="30min"),
            "consumption_kwh": [1.0],
        }
    )
    store = FeatureStore(base_dir=tmp_path)
    store.write_features(df, PipelineConfig())
    manifest = store.get_manifest()
    assert manifest["content_sha256"] is not None
    assert manifest["n_rows"] == 1
