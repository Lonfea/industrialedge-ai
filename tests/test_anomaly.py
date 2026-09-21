from industrialedge_ai.analytics.anomaly import MachineAnomalyDetector
from industrialedge_ai.config import load_machines


def test_obvious_fault_scores_higher_than_nominal():
    machine = load_machines()[3]
    detector = MachineAnomalyDetector(machine)
    normal = {**machine["nominal"]}
    fault = {**machine["nominal"], "temperature": 92.0, "vibration": 7.5, "power": 5.2}
    assert detector.detect(fault).score > detector.detect(normal).score
    assert detector.detect(fault).is_anomaly
