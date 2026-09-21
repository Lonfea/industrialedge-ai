from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

FEATURES = ("temperature", "vibration", "pressure", "rpm", "power")


@dataclass
class Detection:
    score: float
    is_anomaly: bool
    top_signals: list[str]


class MachineAnomalyDetector:
    """Isolation Forest trained on a reproducible synthetic healthy baseline.

    The baseline is generated from each machine's engineering nominal values and
    standard deviations. This makes the demo self-contained while keeping the
    model/data boundary explicit for replacement with real historical telemetry.
    """

    def __init__(self, machine: dict, baseline_samples: int = 1200, random_state: int = 42):
        self.machine = machine
        self.nominal = machine["nominal"]
        self.std = machine["std"]
        rng = np.random.default_rng(random_state + sum(map(ord, machine["machine_id"])))
        baseline = np.column_stack([
            rng.normal(self.nominal[f], self.std[f], baseline_samples) for f in FEATURES
        ])
        self.model = IsolationForest(
            n_estimators=180,
            contamination=0.025,
            random_state=random_state,
            n_jobs=-1,
        ).fit(baseline)
        train_raw = -self.model.score_samples(baseline)
        self.low = float(np.quantile(train_raw, 0.50))
        self.high = float(np.quantile(train_raw, 0.995))

    def detect(self, sample: dict) -> Detection:
        vector = np.array([[float(sample[f]) for f in FEATURES]])
        raw = float(-self.model.score_samples(vector)[0])
        score = float(np.clip((raw - self.low) / max(self.high - self.low, 1e-6), 0.0, 1.0))
        z = {
            f: abs((float(sample[f]) - self.nominal[f]) / max(self.std[f], 1e-6))
            for f in FEATURES
        }
        top = [name for name, value in sorted(z.items(), key=lambda x: x[1], reverse=True)[:3] if value >= 1.5]
        return Detection(score=score, is_anomaly=score >= 0.72, top_signals=top)
