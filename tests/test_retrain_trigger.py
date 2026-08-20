"""
Sprint 8 — Automated Retraining Trigger tests.

It uses a minimal local stand-in for the `.severity` field that `decide_retrain`
actually reads, thus decoupling these tests from the `Evidently` dependency.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mlops.retrain_trigger import decide_retrain


@dataclass
class _FakeDriftResult:
    """Duck-types the one attribute (`severity`) decide_retrain() reads."""
    severity: str


def test_decide_retrain_no_action_when_healthy():
    decision = decide_retrain([], days_since_last_train=2)
    assert not decision.should_retrain
    assert decision.urgency == "NONE"


def test_decide_retrain_urgent_on_critical_drift():
    drift = [_FakeDriftResult(severity="CRITICAL")]
    decision = decide_retrain(drift, days_since_last_train=2)
    assert decision.should_retrain
    assert decision.urgency == "URGENT"


def test_decide_retrain_scheduled_on_warning_drift():
    drift = [_FakeDriftResult(severity="WARNING") for _ in range(4)]
    decision = decide_retrain(drift, days_since_last_train=2)
    assert decision.should_retrain
    assert decision.urgency == "SCHEDULED"


def test_decide_retrain_scheduled_on_cadence():
    decision = decide_retrain([], days_since_last_train=31)
    assert decision.should_retrain
    assert decision.urgency == "SCHEDULED"


def test_decide_retrain_urgent_on_performance_degradation():
    decision = decide_retrain([], days_since_last_train=1, live_mae=0.2, training_time_mae=0.1)
    assert decision.should_retrain
    assert decision.urgency == "URGENT"


def test_decide_retrain_ignores_ok_severity():
    drift = [_FakeDriftResult(severity="OK") for _ in range(10)]
    decision = decide_retrain(drift, days_since_last_train=2)
    assert not decision.should_retrain
