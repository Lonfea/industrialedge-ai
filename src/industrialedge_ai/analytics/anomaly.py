from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

FEATURES = ("temperature", "vibration", "pressure", "rpm", "power")
ALARM_THRESHOLD = 0.72
# Calibration: scores are rescaled so the median healthy sample maps to 0 and the
# 99.5th-percentile healthy sample maps to 1.
CALIBRATION_QUANTILES = (0.50, 0.995)


@dataclass
class Detection:
    score: float
    is_anomaly: bool
    top_signals: list[str]


class MachineAnomalyDetector:
    """Isolation Forest fitted on a machine's healthy operating data.

    Pass ``baseline`` (rows of healthy telemetry, columns in ``features`` order)
    to fit on real history. Without it, a reproducible synthetic baseline is
    generated from the machine's configured nominal values and standard
    deviations, which keeps the demo self-contained.

    The alarm threshold is calibrated on the baseline's own score distribution
    rather than set through a fixed contamination rate. On the SKAB benchmark
    that is the difference between F1 0.29 and 0.67 (see ``scripts/benchmark_skab.py``).
    """

    def __init__(
        self,
        machine: dict,
        baseline: np.ndarray | None = None,
        features: Sequence[str] = FEATURES,
        baseline_samples: int = 1200,
        random_state: int = 42,
    ):
        self.machine = machine
        self.features = tuple(features)
        if baseline is None:
            rng = np.random.default_rng(random_state + sum(map(ord, machine["machine_id"])))
            baseline = np.column_stack(
                [
                    rng.normal(machine["nominal"][f], machine["std"][f], baseline_samples)
                    for f in self.features
                ]
            )
        baseline = np.asarray(baseline, dtype=float)
        if baseline.ndim != 2 or baseline.shape[1] != len(self.features):
            raise ValueError(
                f"baseline must have shape (n, {len(self.features)}), got {baseline.shape}"
            )
        if len(baseline) < 50:
            raise ValueError("baseline needs at least 50 healthy samples to calibrate")

        self.nominal = dict(zip(self.features, baseline.mean(axis=0), strict=True))
        self.std = dict(zip(self.features, baseline.std(axis=0), strict=True))
        if "nominal" in machine and all(f in machine["nominal"] for f in self.features):
            # Engineering specs, when present, stay the reference for explanations.
            self.nominal = {f: machine["nominal"][f] for f in self.features}
            self.std = {f: machine["std"][f] for f in self.features}

        self.model = IsolationForest(
            n_estimators=180,
            contamination=0.025,
            random_state=random_state,
            n_jobs=-1,
        ).fit(baseline)
        train_raw = -self.model.score_samples(baseline)
        self.low, self.high = (float(q) for q in np.quantile(train_raw, CALIBRATION_QUANTILES))

    def score_many(self, samples: np.ndarray) -> np.ndarray:
        """Calibrated anomaly scores in [0, 1] for rows in ``features`` order."""
        raw = -self.model.score_samples(np.asarray(samples, dtype=float))
        return np.clip((raw - self.low) / max(self.high - self.low, 1e-6), 0.0, 1.0)

    def detect(self, sample: dict) -> Detection:
        vector = np.array([[float(sample[f]) for f in self.features]])
        score = float(self.score_many(vector)[0])
        z = {
            f: abs((float(sample[f]) - self.nominal[f]) / max(self.std[f], 1e-6))
            for f in self.features
        }
        ranked = sorted(z.items(), key=lambda item: item[1], reverse=True)[:3]
        top = [name for name, value in ranked if value >= 1.5]
        return Detection(score=score, is_anomaly=score >= ALARM_THRESHOLD, top_signals=top)


class AlarmDebouncer:
    """Confirm an alarm only when ``required`` of the last ``window`` readings are anomalous.

    Single noisy readings stop paging anyone. On SKAB, 3-of-5 cut the false-alarm
    rate from 27.7% to 21.3% without missing more anomalies.
    """

    def __init__(self, required: int = 3, window: int = 5):
        if not 1 <= required <= window:
            raise ValueError("required must be between 1 and window")
        self.required = required
        self.recent: deque[bool] = deque(maxlen=window)

    def update(self, is_anomaly: bool) -> bool:
        self.recent.append(is_anomaly)
        return sum(self.recent) >= self.required
