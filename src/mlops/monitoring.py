"""
Sprint 8 — Monitoring & Maintenance: Batch (windowed) Drift Detection.

This module uses 'Evidently''s 'DataDriftPreset' for batch/windowed drift detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.common.logging_config import get_logger, stage

logger = get_logger(__name__)


@dataclass
class DriftResult:
    feature: str
    drift_detected: bool
    drift_score: float
    stattest: str
    severity: str  # "CRITICAL" | "OK"


def _extract_drift_table(result_dict: dict) -> dict:
    """Extract Evidently's per-column drift results."""
    for metric in result_dict["metrics"]:
        if "drift_by_columns" in metric.get("result", {}):
            return metric["result"]["drift_by_columns"]

    raise KeyError(
        "Evidently report did not contain a per-column drift table "
        "(unexpected report structure — check the installed evidently version)"
    )


def _prepare_numeric_data(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Prepare reference/current data for Evidently.

    - Keeps only columns present in both datasets.
    - Explicitly converts monitored columns to numeric.
    - Converts invalid values to NaN.
    - Removes columns that contain no valid numeric observations in either dataset.
    """
    usable_columns = [
        c
        for c in numeric_columns
        if c in reference_df.columns and c in current_df.columns
    ]

    if not usable_columns:
        return pd.DataFrame(), pd.DataFrame(), []

    reference = reference_df[usable_columns].copy()
    current = current_df[usable_columns].copy()

    for col in usable_columns:
        reference[col] = pd.to_numeric(reference[col], errors="coerce")
        current[col] = pd.to_numeric(current[col], errors="coerce")

    # Keep only columns with at least one usable value in BOTH datasets.
    usable_columns = [
        c
        for c in usable_columns
        if reference[c].notna().any() and current[c].notna().any()
    ]

    if not usable_columns:
        return pd.DataFrame(), pd.DataFrame(), []

    reference = reference[usable_columns].copy()
    current = current[usable_columns].copy()

    return reference, current, usable_columns


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_columns: list[str],
    save_html_path: Path | None = None,
) -> list[DriftResult]:

    with stage(
        logger,
        "monitoring(evidently):detect_feature_drift",
        n_features=len(numeric_columns),
    ):
        reference, current, usable_columns = _prepare_numeric_data(
            reference_df,
            current_df,
            numeric_columns,
        )

        if not usable_columns:
            logger.warning(
                "drift check: no usable numeric columns found"
            )
            return []

        logger.debug(
            "Evidently reference dtypes:\n%s",
            reference.dtypes.to_string(),
        )
        logger.debug(
            "Evidently current dtypes:\n%s",
            current.dtypes.to_string(),
        )

        column_mapping = ColumnMapping(
            numerical_features=usable_columns
        )

        report = Report(
            metrics=[
                DataDriftPreset(columns=usable_columns)
            ]
        )

        # Run Evidently only on the prepared numeric DataFrames.
        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=column_mapping,
        )

        if save_html_path:
            save_html_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            report.save_html(str(save_html_path))
            logger.info(
                "monitoring: Evidently HTML report saved -> %s",
                save_html_path,
            )

        per_column = _extract_drift_table(report.as_dict())

        results: list[DriftResult] = []

        for col in usable_columns:
            info = per_column.get(col)

            if info is None:
                continue

            drift_detected = bool(
                info.get("drift_detected", False)
            )

            results.append(
                DriftResult(
                    feature=col,
                    drift_detected=drift_detected,
                    drift_score=float(
                        info.get("drift_score", 0.0)
                    ),
                    stattest=info.get(
                        "stattest_name",
                        "unknown",
                    ),
                    severity=(
                        "CRITICAL"
                        if drift_detected
                        else "OK"
                    ),
                )
            )

        n_drifted = sum(
            r.drift_detected
            for r in results
        )

        logger.info(
            "drift check: %d/%d features flagged by Evidently",
            n_drifted,
            len(results),
        )

        return results


def dataset_drift_summary(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_columns: list[str],
) -> dict:
    """
    Dataset-level share-of-drifted-columns summary.
    """

    reference, current, usable_columns = _prepare_numeric_data(
        reference_df,
        current_df,
        numeric_columns,
    )

    if not usable_columns:
        return {
            "dataset_drift": False,
            "n_drifted_columns": 0,
            "share_drifted_columns": 0.0,
        }

    column_mapping = ColumnMapping(
        numerical_features=usable_columns
    )

    report = Report(
        metrics=[
            DataDriftPreset(columns=usable_columns)
        ]
    )

    # Run Evidently only on the prepared numeric DataFrames.
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    result_dict = report.as_dict()

    for metric in result_dict["metrics"]:
        result = metric.get("result", {})

        if "dataset_drift" in result:
            return {
                "dataset_drift": bool(
                    result.get("dataset_drift", False)
                ),
                "n_drifted_columns": int(
                    result.get(
                        "number_of_drifted_columns",
                        0,
                    )
                ),
                "share_drifted_columns": float(
                    result.get(
                        "share_of_drifted_columns",
                        0.0,
                    )
                ),
            }

    return {
        "dataset_drift": False,
        "n_drifted_columns": 0,
        "share_drifted_columns": 0.0,
    }


def detect_prediction_drift(
    reference_errors: pd.Series,
    current_errors: pd.Series,
) -> DriftResult:
    """
    Detect whether model residuals are drifting.
    """

    df_ref = reference_errors.to_frame(name="residual")
    df_cur = current_errors.to_frame(name="residual")

    results = detect_feature_drift(
        df_ref,
        df_cur,
        ["residual"],
    )

    return (
        results[0]
        if results
        else DriftResult(
            "residual",
            False,
            0.0,
            "unknown",
            "OK",
        )
    )