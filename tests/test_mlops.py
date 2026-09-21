from industrialedge_ai.mlops.tracking import ExperimentTracker


def test_mlflow_tracker_is_noop_when_disabled():
    tracker = ExperimentTracker("http://localhost:5000", enabled=False)
    assert tracker.register_baselines({}) is False
