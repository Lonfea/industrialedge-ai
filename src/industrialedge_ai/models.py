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
    is_anomaly: bool  # this reading alone crossed the threshold
    alarm: bool = False  # confirmed: enough recent readings crossed it
    health_score: float
    severity: str
    likely_cause: str
    recommendation: str
    top_signals: list[str]


class MachineSnapshot(BaseModel):
    telemetry: Telemetry
    analysis: AnalysisResult


class CopilotQuestion(BaseModel):
    machine_id: str
    question: str = Field(min_length=2, max_length=500)


class CopilotResponse(BaseModel):
    machine_id: str
    answer: str
    grounded_in_points: int
    provider: str
