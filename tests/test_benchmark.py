import benchmark_skab
import numpy as np
import pandas as pd


def test_scores_match_skab_definitions():
    labels = np.array([True, True, False, False])
    alarms = np.array([True, False, True, False])
    # TP=1, FN=1, FP=1, TN=1 -> F1 = 1 / (1 + 1) = 0.5, FAR = 50%, MAR = 50%
    assert benchmark_skab.summarize([(labels, alarms)]) == {
        "f1": 0.5,
        "far_percent": 50.0,
        "mar_percent": 50.0,
    }


def test_shipped_detector_flags_a_clear_shift():
    rng = np.random.default_rng(1)
    train = pd.DataFrame(rng.normal(0, 1, (400, 4)), columns=list("abcd"))
    test = pd.DataFrame(np.vstack([rng.normal(0, 1, (50, 4)), rng.normal(8, 1, (50, 4))]), columns=list("abcd"))
    alarms = benchmark_skab.industrialedge(train, test, debounce=True)
    assert alarms[60:].all()
    assert alarms[:50].mean() < 0.2
