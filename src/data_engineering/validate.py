"""
Sprint 1 — Data Validation.

This uses `great_expectations` (GE)'s Runtime-Batch-Request / ephemeral-context
pattern. In particular
  
  1. there is one Expectation Suite per source (smart_meter / weather / calendar),
  2. every expectation carries `meta={"severity": "CRITICAL"|"WARN"}`, since
     GE itself doesn't have a built-in blocking-vs-advisory concept
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import great_expectations as gx
import pandas as pd
from great_expectations.core.batch import RuntimeBatchRequest

from src.common.config import DEFAULT_CONFIG, REPORTS_DIR, PipelineConfig
from src.common.logging_config import get_logger, stage

logger = get_logger(__name__)


@dataclass
class ExpectationResult:
    expectation: str
    column: str | None
    severity: str  # "CRITICAL" | "WARN"
    success: bool
    observed: str
    details: str = ""


def _get_context():
    """Ephemeral (in-memory) Data Context. Validation is
    therefore self-contained and CI-friendly."""
    context = gx.get_context(mode="ephemeral")
    context.add_datasource(
        name="pandas_runtime_datasource",
        class_name="Datasource",
        execution_engine={"class_name": "PandasExecutionEngine"},
        data_connectors={
            "runtime_connector": {
                "class_name": "RuntimeDataConnector",
                "batch_identifiers": ["batch_id"],
            }
        },
    )
    return context


def _validator_for(context, df: pd.DataFrame, asset_name: str, suite_name: str):
    batch_request = RuntimeBatchRequest(
        datasource_name="pandas_runtime_datasource",
        data_connector_name="runtime_connector",
        data_asset_name=asset_name,
        runtime_parameters={"batch_data": df},
        batch_identifiers={"batch_id": f"{asset_name}_batch"},
    )
    context.add_or_update_expectation_suite(suite_name)
    return context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)


def _build_smart_meter_suite(validator) -> None:
    validator.expect_column_to_exist("timestamp", meta={"severity": "CRITICAL"})
    validator.expect_column_to_exist("household_id", meta={"severity": "CRITICAL"})
    validator.expect_column_to_exist("consumption_kwh", meta={"severity": "CRITICAL"})
    validator.expect_column_values_to_not_be_null(
        "consumption_kwh", mostly=0.99, meta={"severity": "WARN"}
    )
    validator.expect_column_values_to_be_between(
        "consumption_kwh", min_value=0, max_value=25, meta={"severity": "CRITICAL"},
        mostly=0.9999,   # tolerate up to ~0.01% outliers
    )
    validator.expect_compound_columns_to_be_unique(
        ["timestamp", "household_id"], meta={"severity": "CRITICAL"},
        mostly=0.999,    # tolerate up to ~0.1% duplication
    )


def _build_weather_suite(validator) -> None:
    validator.expect_column_to_exist("timestamp", meta={"severity": "CRITICAL"})
    validator.expect_column_values_to_be_between(
        "temperature_c", min_value=-25, max_value=45, meta={"severity": "CRITICAL"}
    )
    validator.expect_column_values_to_be_between(
        "humidity_pct", min_value=0, max_value=100, meta={"severity": "CRITICAL"}
    )
    validator.expect_column_values_to_be_between(
        "wind_speed_ms", min_value=0, max_value=60, meta={"severity": "CRITICAL"}
    )
    validator.expect_column_values_to_not_be_null(
        "temperature_c", mostly=1.0, meta={"severity": "WARN"}
    )


def _build_calendar_suite(validator) -> None:
    validator.expect_column_to_exist("date", meta={"severity": "CRITICAL"})
    validator.expect_column_to_exist("is_weekend", meta={"severity": "CRITICAL"})
    validator.expect_column_to_exist("is_holiday", meta={"severity": "CRITICAL"})
    validator.expect_column_values_to_be_unique("date", meta={"severity": "CRITICAL"})


SUITE_BUILDERS = {
    "smart_meter": _build_smart_meter_suite,
    "weather": _build_weather_suite,
    "calendar": _build_calendar_suite,
}


def _check_timestamp_regularity(df: pd.DataFrame, freq_minutes: int = 30) -> ExpectationResult:
    """GE's per-row expectation model has no natural "are timestamps
    regular within each group" check, so it is custom built."""
    d = df.dropna(subset=["timestamp"]).copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    expected = pd.Timedelta(minutes=freq_minutes)
    n_gaps = 0
    for _, g in d.groupby("household_id", sort=False):
        deltas = g.sort_values("timestamp")["timestamp"].diff().dropna()
        n_gaps += int((deltas > expected).sum())
    return ExpectationResult(
        expectation="expect_timestamps_regular", column="timestamp", severity="WARN",
        success=(n_gaps == 0), observed=f"n_gaps={n_gaps}",
        details=f"expected_freq={freq_minutes}min (plain-pandas check, not a GE expectation)",
    )


def run_suite(df: pd.DataFrame, suite_name: str) -> list[ExpectationResult]:
    with stage(logger, f"validate(ge):{suite_name}", n_rows=len(df)):
        context = _get_context()
        validator = _validator_for(context, df, asset_name=suite_name, suite_name=f"{suite_name}_suite")
        SUITE_BUILDERS[suite_name](validator)

        ge_result = validator.validate(result_format="SUMMARY")

        results = []
        for r in ge_result.results:
            cfg = r.expectation_config
            severity = cfg.meta.get("severity", "WARN") if cfg.meta else "WARN"
            column = cfg.kwargs.get("column") or ",".join(cfg.kwargs.get("column_list", []) or [])
            observed = str(r.result.get("unexpected_count", r.result.get("observed_value", r.success)))
            results.append(
                ExpectationResult(
                    expectation=cfg.expectation_type,
                    column=column or None,
                    severity=severity,
                    success=bool(r.success),
                    observed=observed,
                )
            )

        n_fail_critical = sum(1 for r in results if not r.success and r.severity == "CRITICAL")
        n_fail_warn = sum(1 for r in results if not r.success and r.severity == "WARN")

        if suite_name == "smart_meter":
            results.append(_check_timestamp_regularity(df))
            n_fail_critical = sum(1 for r in results if not r.success and r.severity == "CRITICAL")
            n_fail_warn = sum(1 for r in results if not r.success and r.severity == "WARN")

        logger.info(
            f"{suite_name}: {len(results)} expectations run "
            f"({len(results) - (1 if suite_name == 'smart_meter' else 0)} via Great Expectations), "
            f"{n_fail_critical} CRITICAL failures, {n_fail_warn} WARN failures"
        )
        return results


def has_critical_failure(results: list[ExpectationResult]) -> bool:
    return any((not r.success) and r.severity == "CRITICAL" for r in results)


def write_report(all_results: dict[str, list[ExpectationResult]], out_path: Path) -> dict:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_engine": "great_expectations",
        "suites": {
            suite: [asdict(r) for r in results] for suite, results in all_results.items()
        },
        "summary": {
            suite: {
                "n_expectations": len(results),
                "n_critical_failures": sum(1 for r in results if not r.success and r.severity == "CRITICAL"),
                "n_warn_failures": sum(1 for r in results if not r.success and r.severity == "WARN"),
            }
            for suite, results in all_results.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def validate_raw(config: PipelineConfig = DEFAULT_CONFIG) -> dict:
    meter = pd.read_csv(config.raw_meter_path)
    weather = pd.read_csv(config.raw_weather_path)
    calendar = pd.read_csv(config.raw_calendar_path)

    all_results = {
        "smart_meter": run_suite(meter, "smart_meter"),
        "weather": run_suite(weather, "weather"),
        "calendar": run_suite(calendar, "calendar"),
    }
    report = write_report(all_results, REPORTS_DIR / "data_quality_report.json")

    for suite, results in all_results.items():
        if has_critical_failure(results):
            logger.warning(f"Suite '{suite}' has CRITICAL failures — see report for detail")

    return report


if __name__ == "__main__":
    import sys
    report = validate_raw()
    print(json.dumps(report["summary"], indent=2))
    any_critical = any(s["n_critical_failures"] > 0 for s in report["summary"].values())
    sys.exit(1 if any_critical else 0)
