from datetime import datetime, timezone

from industrialedge_ai.copilot.service import MaintenanceCopilot
from industrialedge_ai.models import AnalysisResult, MachineSnapshot, Telemetry


def snapshot(temp: float, vibration: float, health: float, anomaly: float) -> MachineSnapshot:
    telemetry = Telemetry(
        machine_id="MACHINE-04",
        timestamp=datetime.now(timezone.utc),
        temperature=temp,
        vibration=vibration,
        pressure=5.8,
        rpm=3200,
        power=5.0,
    )
    analysis = AnalysisResult(
        machine_id="MACHINE-04",
        timestamp=telemetry.timestamp,
        anomaly_score=anomaly,
        is_anomaly=True,
        health_score=health,
        severity="critical",
        likely_cause="Possible bearing degradation",
        recommendation="Inspect bearing assembly and lubrication.",
        top_signals=["vibration", "temperature", "power"],
    )
    return MachineSnapshot(telemetry=telemetry, analysis=analysis)


def test_copilot_is_grounded_in_machine_state():
    history = [snapshot(72 + i, 3 + 0.4 * i, 65 - 3 * i, 0.5 + 0.07 * i) for i in range(6)]
    result = MaintenanceCopilot().answer("MACHINE-04", "Why is it abnormal?", history)
    assert result.provider == "deterministic-grounded"
    assert "bearing" in result.answer.lower()
    assert "vibration" in result.answer.lower()
    assert result.grounded_in_points == 6
