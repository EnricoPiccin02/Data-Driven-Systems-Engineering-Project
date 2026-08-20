"""
Explicit `Pickle`-based model persistence.

Although `MLflow` is used (which already serialises scikit-learn/XGBoost/LightGBM
estimators internally), `Pickle`-based model persistence was implemented as
a directly inspectable, dependency-free artifact.

Every model registered via `ModelRegistry.register_model()` is additionally
pickled here and the resulting `.pkl` file is logged as an MLflow artifact
(`mlflow.log_artifact`), so both persistence mechanisms exist side by side.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from src.common.config import PROJECT_ROOT
from src.common.logging_config import get_logger
from src.common.versioning import file_sha256

logger = get_logger(__name__)

PICKLE_STORE_DIR = PROJECT_ROOT / "model_registry_pickles"
PICKLE_STORE_DIR.mkdir(parents=True, exist_ok=True)


def pickle_path_for(model_name: str, version: int) -> Path:
    d = PICKLE_STORE_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"v{version}.pkl"


def save_model_pickle(model, model_name: str, version: int) -> Path:
    """Serialise `model` with the standard library `pickle` module
    to `model_registry_pickles/<model_name>/v<N>.pkl`."""
    path = pickle_path_for(model_name, version)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"model_io: pickled {model_name} v{version} -> {path} "
                f"(sha256={file_sha256(path)[:16]}...)")
    return path


def load_model_pickle(model_name: str, version: int):
    path = pickle_path_for(model_name, version)
    with open(path, "rb") as f:
        return pickle.load(f)
