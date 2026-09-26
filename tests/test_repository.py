from datetime import datetime, timezone

from sqlalchemy import create_engine, text

from industrialedge_ai.models import AnalysisResult, Telemetry
from industrialedge_ai.storage.repository import TelemetryRepository


def reading(alarm: bool) -> tuple[Telemetry, AnalysisResult]:
    now = datetime.now(timezone.utc)
    telemetry = Telemetry(
        machine_id="MACHINE-04", timestamp=now, temperature=90, vibration=7, pressure=6, rpm=3200, power=5
    )
    analysis = AnalysisResult(
        machine_id="MACHINE-04",
        timestamp=now,
        anomaly_score=0.9,
        is_anomaly=True,
        alarm=alarm,
        health_score=40,
        severity="critical",
        likely_cause="Possible bearing degradation",
        recommendation="Inspect bearing",
        top_signals=["vibration"],
    )
    return telemetry, analysis


def test_alarm_round_trips(tmp_path):
    repository = TelemetryRepository(f"sqlite:///{tmp_path / 'edge.db'}")
    repository.save(*reading(alarm=True))
    assert repository.history("MACHINE-04")[0].analysis.alarm is True


def test_databases_from_before_the_alarm_column_are_upgraded(tmp_path):
    url = f"sqlite:///{tmp_path / 'old.db'}"
    with create_engine(url).begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE telemetry (id INTEGER PRIMARY KEY, machine_id VARCHAR(64), "
                "timestamp DATETIME, temperature FLOAT, vibration FLOAT, pressure FLOAT, rpm FLOAT, "
                "power FLOAT, anomaly_score FLOAT, is_anomaly BOOLEAN, health_score FLOAT, "
                "severity VARCHAR(24), likely_cause TEXT, recommendation TEXT, top_signals TEXT)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO telemetry VALUES (1, 'MACHINE-04', '2026-01-01 00:00:00', 60, 2, 6, "
                "3000, 4, 0.1, 0, 95, 'normal', 'none', 'none', '[]')"
            )
        )
    repository = TelemetryRepository(url)
    assert repository.history("MACHINE-04")[0].analysis.alarm is False
    repository.save(*reading(alarm=True))
    assert [s.analysis.alarm for s in repository.history("MACHINE-04")][-1] is True
