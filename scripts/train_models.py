"""
Sprints 4,5 - Runs the ML kernel tranining, comparisong and promotion:

1. Baseline
2. LinearRegression
3. RandomForest
4. GradientBoosting (XGBoost)
5. GradientBoosting (LightGBM)

Everything is logged to the local experiment tracker and registered
in the local model registry, with the best model promoted to Staging.

Usage:
    python3 scripts/train_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from src.ml.train import train_and_compare

if __name__ == "__main__":
    results = train_and_compare()
    print(json.dumps(results, indent=2, default=str))
