from industrialedge_ai.analytics.service import AnalyticsService
from industrialedge_ai.models import Telemetry


def test_end_to_end_analysis_returns_explanation():
    service = AnalyticsService()
    telemetry = Telemetry(
        machine_id="MACHINE-04",
        temperature=93.0,
        vibration=7.9,
        pressure=5.8,
        rpm=3200,
        power=5.1,
    )
    result = service.analyze(telemetry)
    assert result.is_anomaly
    assert result.health_score < 60
    assert result.likely_cause
    assert result.recommendation
