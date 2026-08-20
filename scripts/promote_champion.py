"""
Promote the champion model to Production.

Selection rule: the model with the lowest validation MAE among the most recent
version of each of the four candidate families is promoted.

Usage:
    python3 scripts/promote_champion.py                              # auto-select + promote
    python3 scripts/promote_champion.py --model xgboost --version 2  # explicit
    python3 scripts/promote_champion.py --dry-run                    # report only, no promotion
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logging_config import get_logger
from src.ml.registry import ModelRegistry
from src.ml.train import MODEL_FACTORIES, MODEL_NAME

logger = get_logger("promote_champion")


def find_champion(registry: ModelRegistry, candidate_keys: list[str] | None = None) -> tuple[str, int, dict]:
    """Across the *latest* registered version of each candidate model
    family, return (model_key, version, metrics) for the one with the
    lowest validation MAE."""
    candidate_keys = candidate_keys or list(MODEL_FACTORIES.keys())
    best = None
    for key in candidate_keys:
        full_name = f"{MODEL_NAME}__{key}"
        try:
            versions = registry.list_versions(full_name)
        except Exception as exc:  # noqa: BLE001 — model family may not be registered yet
            logger.info(f"promote_champion: no registered versions for '{full_name}' ({exc}); skipping")
            continue
        if not versions:
            continue
        latest = max(versions, key=lambda v: v.version)
        mae = latest.metrics.get("mae")
        if mae is None:
            continue
        logger.info(f"promote_champion: {full_name} v{latest.version} val MAE={mae:.4f}")
        if best is None or mae < best[2]["mae"]:
            best = (key, latest.version, latest.metrics)

    if best is None:
        raise LookupError(
            "No registered model versions with an 'mae' metric were found for any "
            f"candidate in {candidate_keys}. Run scripts/train_models.py first."
        )
    return best


def promote_champion(model_key: str | None = None, version: int | None = None, dry_run: bool = False) -> dict:
    registry = ModelRegistry()

    if model_key is None:
        model_key, version, metrics = find_champion(registry)
    else:
        full_name = f"{MODEL_NAME}__{model_key}"
        versions = registry.list_versions(full_name)
        if version is None:
            version = max(v.version for v in versions)
        metrics = next(v.metrics for v in versions if v.version == version)

    full_name = f"{MODEL_NAME}__{model_key}"
    logger.info(f"promote_champion: selected {full_name} v{version} (MAE={metrics.get('mae')}) "
                f"{'[DRY RUN]' if dry_run else '-> promoting to Production'}")

    if dry_run:
        return {"model_name": full_name, "version": version, "metrics": metrics, "promoted": False}

    # MLflow's archive_existing_versions=True only protects the same registered model name.
    # Demote any other family's Production version here, so only one family total is ever in Production at once.
    for key in MODEL_FACTORIES:
        other_name = f"{MODEL_NAME}__{key}"
        if other_name == full_name:
            continue
        prod_versions = registry.client.get_latest_versions(other_name, stages=["Production"])
        for v in prod_versions:
            registry.transition_stage(other_name, int(v.version), "Archived")
            logger.info(f"promote_champion: demoted {other_name} v{v.version} (Production -> Archived)")

    meta = registry.transition_stage(full_name, version, "Production")
    logger.info(f"promote_champion: {full_name} v{version} is now in stage='{meta.stage}'")
    return {"model_name": full_name, "version": version, "metrics": meta.metrics, "promoted": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODEL_FACTORIES.keys()), default=None,
                         help="Model family to promote (default: auto-select lowest-MAE champion)")
    parser.add_argument("--version", type=int, default=None,
                         help="Specific registered version to promote (default: latest of --model, "
                              "or the champion's version if --model is also omitted)")
    parser.add_argument("--dry-run", action="store_true", help="Report the selection without promoting")
    args = parser.parse_args()

    result = promote_champion(model_key=args.model, version=args.version, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))
