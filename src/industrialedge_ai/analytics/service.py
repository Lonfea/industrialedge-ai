from __future__ import annotations

from industrialedge_ai.analytics.anomaly import AlarmDebouncer, MachineAnomalyDetector
from industrialedge_ai.analytics.health import classify, health_from_anomaly
from industrialedge_ai.config import load_machines
from industrialedge_ai.models import AnalysisResult, Telemetry


class AnalyticsService:
    def __init__(self):
        machines = load_machines()
        self.configs = {m["machine_id"]: m for m in machines}
        self.detectors = {m["machine_id"]: MachineAnomalyDetector(m) for m in machines}
        self.debouncers = {m["machine_id"]: AlarmDebouncer() for m in machines}

    def analyze(self, telemetry: Telemetry) -> AnalysisResult:
        config = self.configs[telemetry.machine_id]
        sample = telemetry.model_dump()
        detection = self.detectors[telemetry.machine_id].detect(sample)
        alarm = self.debouncers[telemetry.machine_id].update(detection.is_anomaly)
        health = health_from_anomaly(detection.score, sample, config["nominal"], config["std"])
        severity, cause, recommendation = classify(health, detection.top_signals)
        return AnalysisResult(
            machine_id=telemetry.machine_id,
            timestamp=telemetry.timestamp,
            anomaly_score=round(detection.score, 3),
            is_anomaly=detection.is_anomaly,
            alarm=alarm,
            health_score=health,
            severity=severity,
            likely_cause=cause,
            recommendation=recommendation,
            top_signals=detection.top_signals,
        )
