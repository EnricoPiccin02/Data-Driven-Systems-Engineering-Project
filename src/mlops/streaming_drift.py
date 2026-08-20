"""
Sprint 8 - Online Drift Detection with `River`'s `ADWIN`.

Complements `Evidently` batch/windowed drift.
`ADWIN` (ADaptive WINdowing) processes one value at a time and maintains an
adaptive-size sliding window, flagging a change the moment the statistics
of the recent sub-window diverge significantly from the older sub-window.

One `ADWIN` instance is kept per monitored feature, because drift in
one feature (e.g. a sudden weather-sensor fault) shouldn't be masked by
stability in another.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from river import drift

from src.common.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MONITORED_FEATURES = [
    "consumption_kwh", "temperature_c", "heating_degree", "lag_1", "lag_48",
]


@dataclass
class DriftEvent:
    feature: str
    timestamp: str
    detector_width: int  # `ADWIN`'s current adaptive window size at the moment of detection


class OnlineDriftMonitor:
    """One ADWIN detector per feature; call `update(readings)` once per new
    live data point (e.g. once per streamed household reading in the
    dashboard's real-time loop)."""

    def __init__(self, feature_names: list[str] | None = None, delta: float = 0.002):
        self.feature_names = feature_names or DEFAULT_MONITORED_FEATURES
        # `delta` is `ADWIN`'s confidence parameter: smaller delta = more
        # conservative (fewer false positives, slower to detect real drift).
        self.delta = delta
        self.detectors = {f: drift.ADWIN(delta=delta) for f in self.feature_names}
        self.n_updates = 0
        self.drift_log: list[DriftEvent] = []

    def update(self, readings: dict[str, float]) -> list[str]:
        """Feed one new live observation. Returns the list of feature names
        that just triggered a drift signal on this update."""
        self.n_updates += 1
        newly_drifted = []
        for feature, detector in self.detectors.items():
            if feature not in readings or readings[feature] is None:
                continue
            detector.update(float(readings[feature]))
            if detector.drift_detected:
                newly_drifted.append(feature)
                event = DriftEvent(
                    feature=feature,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    detector_width=detector.width,
                )
                self.drift_log.append(event)
                logger.warning(f"streaming_drift(ADWIN): drift detected on '{feature}' "
                                f"after {self.n_updates} updates (window width={detector.width})")
        return newly_drifted

    def status(self) -> dict:
        """Snapshot for the dashboard's live status panel."""
        return {
            "n_updates": self.n_updates,
            "n_drift_events_total": len(self.drift_log),
            "features_with_drift_ever": sorted({e.feature for e in self.drift_log}),
            "recent_events": [
                {"feature": e.feature, "timestamp": e.timestamp} for e in self.drift_log[-10:]
            ],
            "detector_windows": {f: d.width for f, d in self.detectors.items()},
            "delta": self.delta,
        }

    def reset(self, feature: str | None = None) -> None:
        """Reset one detector or all of them."""
        targets = [feature] if feature else list(self.detectors.keys())
        for f in targets:
            self.detectors[f] = drift.ADWIN(delta=self.delta)
        logger.info(f"streaming_drift(ADWIN): reset detector(s) for {targets} (delta={self.delta})")

    def set_delta(self, delta: float) -> None:
        """Change sensitivity live and rebuild every detector with it. `River`'s
        `ADWIN` has no in-place way to change delta on a built detector, so
        this necessarily discards accumulated window state."""
        self.delta = delta
        for f in self.detectors:
            self.detectors[f] = drift.ADWIN(delta=delta)
        logger.info(f"streaming_drift(ADWIN): delta set to {delta} (all detectors rebuilt)")