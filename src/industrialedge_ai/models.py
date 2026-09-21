from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Telemetry(BaseModel):
    machine_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    temperature: float
    vibration: float
    pressure: float
    rpm: float
    power: float


class AnalysisResult(BaseModel):
    machine_id: str
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool
    health_score: float
    severity: str
    likely_cause: str
    recommendation: str
    top_signals: list[str]


class MachineSnapshot(BaseModel):
    telemetry: Telemetry
    analysis: AnalysisResult
