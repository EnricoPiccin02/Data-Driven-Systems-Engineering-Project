"""
Sprint 4 — Experiment Tracking.

This module wraps `MLflow` to preserve the exact call-site API (`start_run`,
`log_param`, `log_metric`, `log_artifact`) that `train.py` already use.

Tracking store: local SQLite file `mlflow.db` backend store.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import mlflow

from src.common.config import PROJECT_ROOT
from src.common.logging_config import get_logger

logger = get_logger(__name__)

# Using a local SQLite file as the default tracking store
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))


class MLflowExperimentTracker:
    """Real-MLflow-backed tracker with the same method surface
    `train.py` invokes."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self._active_run = None

    @contextmanager
    def start_run(self, run_name: str | None = None):
        with mlflow.start_run(run_name=run_name) as run:
            self._active_run = run
            logger.info(f"experiment_tracking(mlflow): START run {run.info.run_id} "
                        f"({run_name or 'unnamed'})")
            try:
                yield self
            finally:
                logger.info(f"experiment_tracking(mlflow): END run {run.info.run_id}")
                self._active_run = None

    def log_param(self, key: str, value) -> None:
        mlflow.log_param(key, value)

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float) -> None:
        mlflow.log_metric(key, value)

    def log_metrics(self, metrics: dict) -> None:
        # mlflow.log_metrics requires numeric values only.
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(numeric)

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def all_runs(self) -> list[dict]:
        """Return every run in this experiment as a flat dict
        (params + metrics + status)."""
        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:
            return []
        df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        return df.to_dict(orient="records")

    def best_run(self, metric: str = "mae", minimize: bool = True) -> dict | None:
        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:
            return None
        order = "ASC" if minimize else "DESC"
        df = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )
        if df.empty:
            return None
        return df.iloc[0].to_dict()
