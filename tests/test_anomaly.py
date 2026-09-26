import numpy as np
import pytest

from industrialedge_ai.analytics.anomaly import AlarmDebouncer, MachineAnomalyDetector
from industrialedge_ai.config import load_machines


def test_obvious_fault_scores_higher_than_nominal():
    machine = load_machines()[3]
    detector = MachineAnomalyDetector(machine)
    normal = {**machine["nominal"]}
    fault = {**machine["nominal"], "temperature": 92.0, "vibration": 7.5, "power": 5.2}
    assert detector.detect(fault).score > detector.detect(normal).score
    assert detector.detect(fault).is_anomaly


def healthy_history(rows: int = 400) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.column_stack([rng.normal(50, 2, rows), rng.normal(1.0, 0.1, rows), rng.normal(6, 0.2, rows)])


def test_detector_fits_on_real_history_with_custom_features():
    detector = MachineAnomalyDetector(
        {"machine_id": "pump"}, baseline=healthy_history(), features=["temp", "vib", "press"]
    )
    scores = detector.score_many(np.array([[50, 1.0, 6.0], [70, 3.0, 6.0]]))
    assert 0.0 <= scores[0] < scores[1] <= 1.0
    fault = detector.detect({"temp": 70, "vib": 3.0, "press": 6.0})
    assert fault.is_anomaly
    assert fault.top_signals[:2] == ["vib", "temp"]


def test_detector_rejects_unusable_baselines():
    with pytest.raises(ValueError, match="shape"):
        MachineAnomalyDetector({"machine_id": "pump"}, baseline=healthy_history(), features=["a", "b"])
    with pytest.raises(ValueError, match="at least 50"):
        MachineAnomalyDetector({"machine_id": "pump"}, baseline=healthy_history(10), features=["a", "b", "c"])


def test_single_spike_does_not_confirm_an_alarm():
    debouncer = AlarmDebouncer(required=3, window=5)
    assert [debouncer.update(flag) for flag in [True, False, False, False, False]] == [False] * 5


def test_sustained_anomaly_confirms_then_clears():
    debouncer = AlarmDebouncer(required=3, window=5)
    states = [debouncer.update(flag) for flag in [True, True, True, False, False, False]]
    assert states == [False, False, True, True, True, False]


def test_debouncer_validates_arguments():
    with pytest.raises(ValueError):
        AlarmDebouncer(required=6, window=5)
