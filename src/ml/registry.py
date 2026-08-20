"""
Sprints 4,5 — Model Registry through `Pickle` and `MLflow`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import mlflow
from mlflow.tracking import MlflowClient

from src.common.logging_config import get_logger
from src.common.versioning import file_sha256
from src.ml.model_io import save_model_pickle

logger = get_logger(__name__)

VALID_STAGES = ("None", "Staging", "Production", "Archived")

# LinearRegression/RandomForestRegressor are plain sklearn and already
# trusted by skops. XGBoost and LightGBM wrap non-sklearn compiled
# objects (Booster) that skops' audit doesn't recognize by default.
# Since these are the project own trained models, not untrusted
# external files, it's safe to trust them explicitly.
SKOPS_TRUSTED_TYPES = [
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBRegressor",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMRegressor",
    "collections.OrderedDict",
]


@dataclass
class ModelVersionMeta:
    model_name: str
    version: int
    stage: str
    created_at: str
    metrics: dict
    params: dict
    feature_columns: list[str]
    training_data_manifest_sha256: str | None
    artifact_path: str      # MLflow model URI
    artifact_sha256: str    # SHA-256 of the explicit pickle artifact
    run_id: str


class ModelRegistry:
    """Thin, call-site-compatible wrapper around `MLflow` Model Registry."""

    def __init__(self):
        self.client = MlflowClient()

    def register_model(
        self,
        model_name: str,
        model_obj,
        metrics: dict,
        params: dict,
        feature_columns: list[str],
        training_data_manifest_sha256: str | None = None,
        stage: str = "None",
    ) -> ModelVersionMeta:
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError(
                "register_model() must be called inside an "
                "MLflowExperimentTracker.start_run(...) block so the model "
                "logs against the run that produced it."
            )

        # `mlflow.sklearn.log_model` works for any picklable estimator with a
        # scikit-learn-compatible .predict(), so LinearRegression,
        # RandomForestRegressor, XGBRegressor, and LGBMRegressor are supported.
        mlflow.sklearn.log_model(
            model_obj,
            artifact_path="model",
            registered_model_name=model_name,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        # Read the registered version number from the registry directly.
        versions = self.client.search_model_versions(f"name='{model_name}'")
        version = max(int(v.version) for v in versions)

        # Explicit `Pickle`` artifact alongside `MLflow`'s own serialisation.
        pickle_path = save_model_pickle(model_obj, model_name, version)
        mlflow.log_artifact(str(pickle_path), artifact_path="pickle")
        artifact_sha256 = file_sha256(pickle_path)

        self.client.set_model_version_tag(model_name, version, "feature_columns", json.dumps(feature_columns))
        self.client.set_model_version_tag(model_name, version, "artifact_sha256", artifact_sha256)
        if training_data_manifest_sha256:
            self.client.set_model_version_tag(
                model_name, version, "training_data_manifest_sha256", training_data_manifest_sha256
            )

        if stage != "None":
            self.client.transition_model_version_stage(model_name, str(version), stage)

        meta = self._build_meta(model_name, version)
        logger.info(f"registry(mlflow): registered {model_name} v{version} (stage={meta.stage})")
        return meta

    def list_versions(self, model_name: str) -> list[ModelVersionMeta]:
        versions = self.client.search_model_versions(f"name='{model_name}'")
        return [self._build_meta(model_name, int(v.version)) for v in versions]

    def transition_stage(self, model_name: str, version: int, stage: str) -> ModelVersionMeta:
        if stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}, got {stage}")
        # Enforcement of models single-Production invariant. The promotion of
        # a model+version to Production auto-archives any other version currently
        #  in Production.
        self.client.transition_model_version_stage(
            model_name, str(version), stage, archive_existing_versions=(stage == "Production")
        )
        meta = self._build_meta(model_name, version)
        logger.info(f"registry(mlflow): {model_name} v{version} -> stage={stage}")
        return meta

    def get_production_model(self, model_name: str):
        versions = self.client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            raise LookupError(f"No Production-stage version found for model '{model_name}'")
        version = int(versions[0].version)
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
        return model, self._build_meta(model_name, version)

    def find_production_model(self, candidate_model_names: list[str]):
        """Search across multiple registered model names (one per model
        family) for whichever currently holds a Production-stage version."""
        found = []
        for name in candidate_model_names:
            versions = self.client.get_latest_versions(name, stages=["Production"])
            if versions:
                found.append((name, int(versions[0].version)))

        if not found:
            raise LookupError(f"No Production-stage version found among {candidate_model_names}")
        if len(found) > 1:
            raise RuntimeError(
                f"Multiple model families are simultaneously in Production: {found}. "
                "This should never happen via promote_champion.py — check for a manual "
                "stage transition outside that script."
            )

        name, version = found[0]
        model = mlflow.sklearn.load_model(f"models:/{name}/{version}")
        return model, self._build_meta(name, version)

    def load_version(self, model_name: str, version: int):
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
        return model, self._build_meta(model_name, version)

    def _build_meta(self, model_name: str, version: int) -> ModelVersionMeta:
        v = self.client.get_model_version(model_name, str(version))
        run = self.client.get_run(v.run_id)
        feature_columns = json.loads(v.tags.get("feature_columns", "[]"))
        return ModelVersionMeta(
            model_name=model_name,
            version=int(v.version),
            stage=v.current_stage,
            created_at=datetime.fromtimestamp(v.creation_timestamp / 1000, tz=timezone.utc).isoformat(),
            metrics=dict(run.data.metrics),
            params=dict(run.data.params),
            feature_columns=feature_columns,
            training_data_manifest_sha256=v.tags.get("training_data_manifest_sha256"),
            artifact_path=f"models:/{model_name}/{v.version}",
            artifact_sha256=v.tags.get("artifact_sha256", ""),
            run_id=v.run_id,
        )
