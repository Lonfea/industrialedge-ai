from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from industrialedge_ai.models import AnalysisResult, MachineSnapshot, Telemetry


class Base(DeclarativeBase):
    pass


class TelemetryRow(Base):
    __tablename__ = "telemetry"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    temperature: Mapped[float] = mapped_column(Float)
    vibration: Mapped[float] = mapped_column(Float)
    pressure: Mapped[float] = mapped_column(Float)
    rpm: Mapped[float] = mapped_column(Float)
    power: Mapped[float] = mapped_column(Float)
    anomaly_score: Mapped[float] = mapped_column(Float)
    is_anomaly: Mapped[bool] = mapped_column(Boolean)
    health_score: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(24))
    likely_cause: Mapped[str] = mapped_column(Text)
    recommendation: Mapped[str] = mapped_column(Text)
    top_signals: Mapped[str] = mapped_column(Text)


class TelemetryRepository:
    def __init__(self, database_url: str):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
        Base.metadata.create_all(self.engine)

    def save(self, telemetry: Telemetry, analysis: AnalysisResult) -> None:
        row = TelemetryRow(
            **telemetry.model_dump(),
            anomaly_score=analysis.anomaly_score,
            is_anomaly=analysis.is_anomaly,
            health_score=analysis.health_score,
            severity=analysis.severity,
            likely_cause=analysis.likely_cause,
            recommendation=analysis.recommendation,
            top_signals=json.dumps(analysis.top_signals),
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()

    def latest_all(self) -> list[MachineSnapshot]:
        with Session(self.engine) as session:
            ids = session.scalars(select(TelemetryRow.machine_id).distinct()).all()
            rows = []
            for machine_id in ids:
                row = session.scalar(
                    select(TelemetryRow)
                    .where(TelemetryRow.machine_id == machine_id)
                    .order_by(TelemetryRow.timestamp.desc())
                    .limit(1)
                )
                if row:
                    rows.append(self._snapshot(row))
            return sorted(rows, key=lambda x: x.telemetry.machine_id)

    def history(self, machine_id: str, limit: int = 120) -> list[MachineSnapshot]:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(TelemetryRow)
                .where(TelemetryRow.machine_id == machine_id)
                .order_by(TelemetryRow.timestamp.desc())
                .limit(limit)
            ).all()
            return [self._snapshot(r) for r in reversed(rows)]

    @staticmethod
    def _snapshot(row: TelemetryRow) -> MachineSnapshot:
        telemetry = Telemetry(
            machine_id=row.machine_id,
            timestamp=row.timestamp,
            temperature=row.temperature,
            vibration=row.vibration,
            pressure=row.pressure,
            rpm=row.rpm,
            power=row.power,
        )
        analysis = AnalysisResult(
            machine_id=row.machine_id,
            timestamp=row.timestamp,
            anomaly_score=row.anomaly_score,
            is_anomaly=row.is_anomaly,
            health_score=row.health_score,
            severity=row.severity,
            likely_cause=row.likely_cause,
            recommendation=row.recommendation,
            top_signals=json.loads(row.top_signals),
        )
        return MachineSnapshot(telemetry=telemetry, analysis=analysis)
