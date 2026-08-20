"""
Sprint 8 — Automated Retraining Trigger.

Encodes the decision policy in code so "when do we retrain" is enforced
consistently. This module makes the just the decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class HasSeverity(Protocol):
    """Entity modelling with a `.severity` field."""
    severity: str

# Policy thresholds
MAX_CRITICAL_DRIFTED_FEATURES = 0     # Any CRITICAL feature drift -> retrain
MAX_WARNING_DRIFTED_FEATURES = 3      # Tolerate a few WARNING features before acting
MAX_DAYS_SINCE_LAST_TRAIN = 30        # Scheduled retrain cadence, independent of drift
PERFORMANCE_DEGRADATION_MAE_PCT = 15  # Retrain if live MAE is >15% worse than at training time


@dataclass
class RetrainDecision:
    should_retrain: bool
    reasons: list[str]
    urgency: str  # "NONE" | "SCHEDULED" | "URGENT"


def decide_retrain(
    drift_results: list[HasSeverity],
    days_since_last_train: int,
    live_mae: float | None = None,
    training_time_mae: float | None = None,
) -> RetrainDecision:
    reasons = []
    urgency = "NONE"

    n_critical = sum(1 for r in drift_results if r.severity == "CRITICAL")
    n_warning = sum(1 for r in drift_results if r.severity == "WARNING")

    if n_critical > MAX_CRITICAL_DRIFTED_FEATURES:
        reasons.append(f"{n_critical} feature(s) show CRITICAL drift")
        urgency = "URGENT"

    if n_warning > MAX_WARNING_DRIFTED_FEATURES:
        reasons.append(f"{n_warning} feature(s) show WARNING-level drift (threshold {MAX_WARNING_DRIFTED_FEATURES})")
        if urgency == "NONE":
            urgency = "SCHEDULED"

    if live_mae is not None and training_time_mae is not None and training_time_mae > 0:
        degradation_pct = (live_mae - training_time_mae) / training_time_mae * 100
        if degradation_pct > PERFORMANCE_DEGRADATION_MAE_PCT:
            reasons.append(
                f"live MAE is {degradation_pct:.1f}% worse than at training time "
                f"(threshold {PERFORMANCE_DEGRADATION_MAE_PCT}%)"
            )
            urgency = "URGENT"

    if days_since_last_train >= MAX_DAYS_SINCE_LAST_TRAIN:
        reasons.append(f"{days_since_last_train} days since last training run (cadence: {MAX_DAYS_SINCE_LAST_TRAIN})")
        if urgency == "NONE":
            urgency = "SCHEDULED"

    return RetrainDecision(should_retrain=bool(reasons), reasons=reasons, urgency=urgency)
