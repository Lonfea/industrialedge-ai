from industrialedge_ai.analytics.health import classify, health_from_anomaly


def test_health_decreases_with_stress():
    nominal = {"temperature":60.0,"vibration":2.0,"pressure":6.0,"rpm":3000.0,"power":4.0}
    std = {"temperature":2.0,"vibration":0.2,"pressure":0.2,"rpm":50.0,"power":0.2}
    healthy = health_from_anomaly(0.05, nominal, nominal, std)
    stressed = {**nominal, "temperature":82.0, "vibration":6.0, "power":5.0}
    bad = health_from_anomaly(0.95, stressed, nominal, std)
    assert healthy > bad
    assert bad < 60


def test_bearing_pattern_classification():
    severity, cause, _ = classify(45.0, ["vibration", "temperature"])
    assert severity == "critical"
    assert "bearing" in cause.lower()
