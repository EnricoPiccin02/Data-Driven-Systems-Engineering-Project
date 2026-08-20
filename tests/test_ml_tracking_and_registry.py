"""
Sprint 4 — Experiment Tracking tests.

Each test points `MLflow` at an isolated temporary tracking directory so
tests never touch the repository's real store and can run in parallel safely.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("mlflow")

import mlflow

from src.ml.experiment_tracking import MLflowExperimentTracker
from src.ml.registry import ModelRegistry


@pytest.fixture()
def isolated_mlflow(tmp_path):
    """Point MLflow at an isolated temporary SQLite database."""
    old_tracking_uri = mlflow.get_tracking_uri()
    old_registry_uri = mlflow.get_registry_uri()

    db_path = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    yield tmp_path

    mlflow.set_tracking_uri(old_tracking_uri)
    mlflow.set_registry_uri(old_registry_uri)


def _fitted_dummy_model():
    return LinearRegression().fit(np.array([[1], [2], [3]]), np.array([1.0, 2.0, 3.0]))


def test_experiment_tracker_records_runs(isolated_mlflow):
    tracker = MLflowExperimentTracker("test_experiment")
    with tracker.start_run(run_name="run1"):
        tracker.log_param("model_type", "dummy")
        tracker.log_metric("mae", 0.1)
    with tracker.start_run(run_name="run2"):
        tracker.log_metric("mae", 0.05)

    runs = tracker.all_runs()
    assert len(runs) == 2
    best = tracker.best_run("mae")
    assert best["metrics.mae"] == 0.05


def test_experiment_tracker_marks_failed_run(isolated_mlflow):
    tracker = MLflowExperimentTracker("test_experiment")
    try:
        with tracker.start_run(run_name="will_fail"):
            raise ValueError("boom")
    except ValueError:
        pass
    runs = tracker.all_runs()
    assert runs[0]["status"] == "FAILED"


def test_model_registry_register_and_promote(isolated_mlflow):
    tracker = MLflowExperimentTracker("test_experiment_registry")
    registry = ModelRegistry()

    with tracker.start_run(run_name="dummy_v1"):
        tracker.log_metric("mae", 0.5)
        meta1 = registry.register_model(
            "dummy_model", _fitted_dummy_model(), metrics={"mae": 0.5}, params={},
            feature_columns=["a", "b"],
        )
    with tracker.start_run(run_name="dummy_v2"):
        tracker.log_metric("mae", 0.3)
        meta2 = registry.register_model(
            "dummy_model", _fitted_dummy_model(), metrics={"mae": 0.3}, params={},
            feature_columns=["a", "b"],
        )

    assert meta1.version == 1
    assert meta2.version == 2

    registry.transition_stage("dummy_model", 2, "Production")
    _model, meta = registry.get_production_model("dummy_model")
    assert meta.version == 2

    # Promoting v1 to Production should auto-archive v2 (`MLflow`'s
    # native archive_existing_versions=True behaviour)
    registry.transition_stage("dummy_model", 1, "Production")
    versions = {m.version: m.stage for m in registry.list_versions("dummy_model")}
    assert versions[1] == "Production"
    assert versions[2] == "Archived"


def test_model_registry_pickle_artifact_matches_mlflow_model(isolated_mlflow):
    """The explicit `Pickle` artifact should round-trip to a model
    that predicts identically to the one `MLflow` itself serialised."""
    from src.ml.model_io import load_model_pickle

    tracker = MLflowExperimentTracker("test_experiment_pickle")
    registry = ModelRegistry()
    original = _fitted_dummy_model()

    with tracker.start_run(run_name="pickle_check"):
        meta = registry.register_model(
            "pickle_check_model", original, metrics={"mae": 0.1}, params={},
            feature_columns=["a"],
        )

    pickled_model = load_model_pickle("pickle_check_model", meta.version)
    mlflow_model, _ = registry.load_version("pickle_check_model", meta.version)

    X = np.array([[10.0]])
    assert pickled_model.predict(X)[0] == pytest.approx(mlflow_model.predict(X)[0])
