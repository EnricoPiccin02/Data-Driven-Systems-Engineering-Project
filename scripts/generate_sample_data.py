"""
Sprint 1 - Convenience CLI: generate raw data only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_engineering.ingest import ingest_all

if __name__ == "__main__":
    ingest_all()
