"""
Sprint 1 - Data Collection tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.config import PipelineConfig
from src.data_engineering.ingest import DataSource, RealOrFallbackSource


class _AlwaysFailsSource(DataSource):
    name = "fails"

    def fetch(self, config):
        raise RuntimeError("simulated network failure")


class _AlwaysWorksSource(DataSource):
    name = "works"

    def fetch(self, config):
        return pd.DataFrame({"x": [1, 2, 3]})


def test_real_or_fallback_uses_fallback_on_failure():
    src = RealOrFallbackSource("test", real=_AlwaysFailsSource(), fallback=_AlwaysWorksSource())
    df = src.fetch(PipelineConfig())
    assert df["is_real_data"].eq(False).all()
    assert len(df) == 3


def test_real_or_fallback_uses_real_on_success():
    src = RealOrFallbackSource("test", real=_AlwaysWorksSource(), fallback=_AlwaysFailsSource())
    df = src.fetch(PipelineConfig())
    assert df["is_real_data"].eq(True).all()


def test_real_or_fallback_respects_use_real_data_flag():
    config = PipelineConfig(use_real_data=False)
    src = RealOrFallbackSource("test", real=_AlwaysWorksSource(), fallback=_AlwaysWorksSource())
    df = src.fetch(config)
    # Even though 'real' would succeed, config says don't try it
    assert df["is_real_data"].eq(False).all()
