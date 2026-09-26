"""Benchmark the anomaly detector on real pump telemetry from SKAB.

SKAB (Skoltech Anomaly Benchmark, github.com/waico/SKAB) records a water-pump
testbed: two vibration accelerometers, motor current and voltage, pressure,
two temperatures and flow rate, with labelled anomalies from valve closures,
leaks and rotor imbalance.

The protocol follows SKAB's published outlier-detection leaderboard: for
each of the 34 experiments, fit on the first 400 rows and score the rest,
then pool the confusion counts. F1 = TP / (TP + (FP + FN) / 2),
FAR = false-alarm rate, MAR = missed-alarm rate.

The script reproduces the leaderboard's Isolation Forest entry first. That
checks the protocol is implemented the same way before comparing anything.

The dataset is GPL-3.0 licensed and is downloaded at run time, not committed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrialedge_ai.analytics.anomaly import (
    ALARM_THRESHOLD,
    AlarmDebouncer,
    MachineAnomalyDetector,
)

BASE_URL = "https://raw.githubusercontent.com/waico/SKAB/master/data"
EXPERIMENTS = {"valve1": range(16), "valve2": range(4), "other": range(1, 15)}
TRAIN_ROWS = 400
LEADERBOARD_ISOLATION_FOREST = {"f1": 0.29, "far_percent": 2.56, "mar_percent": 82.89}


def download(data_dir: Path) -> list[Path]:
    paths = []
    for group, numbers in EXPERIMENTS.items():
        for number in numbers:
            path = data_dir / group / f"{number}.csv"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                response = requests.get(f"{BASE_URL}/{group}/{number}.csv", timeout=60)
                response.raise_for_status()
                path.write_bytes(response.content)
            paths.append(path)
    return paths


def load(paths: list[Path]) -> list[tuple[str, pd.DataFrame, pd.DataFrame, np.ndarray]]:
    experiments = []
    for path in paths:
        frame = pd.read_csv(path, sep=";", index_col="datetime", parse_dates=True)
        features = frame.drop(columns=["anomaly", "changepoint"])
        labels = frame["anomaly"].to_numpy()[TRAIN_ROWS:] == 1
        experiments.append(
            (path.parent.name, features.iloc[:TRAIN_ROWS], features.iloc[TRAIN_ROWS:], labels)
        )
    return experiments


def summarize(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    tp = tn = fp = fn = 0
    for labels, alarms in pairs:
        alarms = np.asarray(alarms, dtype=bool)
        tp += int((labels & alarms).sum())
        tn += int((~labels & ~alarms).sum())
        fp += int((~labels & alarms).sum())
        fn += int((labels & ~alarms).sum())
    return {
        "f1": round(tp / (tp + (fn + fp) / 2), 2),
        "far_percent": round(fp / (fp + tn) * 100, 2),
        "mar_percent": round(fn / (fn + tp) * 100, 2),
    }


def leaderboard_isolation_forest(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """SKAB's reference configuration: fixed contamination and a 3-point rolling median."""
    model = IsolationForest(random_state=0, n_jobs=-1, contamination=0.0005).fit(train)
    flags = pd.Series(model.predict(test) * -1, index=test.index)
    return flags.rolling(3).median().fillna(0).replace(-1, 0).to_numpy() == 1


def industrialedge(train: pd.DataFrame, test: pd.DataFrame, debounce: bool) -> np.ndarray:
    detector = MachineAnomalyDetector(
        {"machine_id": "skab-pump"}, baseline=train.to_numpy(), features=list(train.columns)
    )
    flags = detector.score_many(test.to_numpy()) >= ALARM_THRESHOLD
    if not debounce:
        return flags
    debouncer = AlarmDebouncer()
    return np.array([debouncer.update(bool(flag)) for flag in flags])


def three_sigma(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    z = (test - train.mean()) / train.std().replace(0, 1e-6)
    return (z.abs().max(axis=1) >= 3).to_numpy()


def hotelling_t2(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    mean = train.mean().to_numpy()
    inverse = np.linalg.pinv(np.cov(train.to_numpy(), rowvar=False))

    def t2(values: np.ndarray) -> np.ndarray:
        centred = values - mean
        return np.einsum("ij,jk,ik->i", centred, inverse, centred)

    limit = np.quantile(t2(train.to_numpy()), 0.99)
    return t2(test.to_numpy()) > limit


DETECTORS = {
    "SKAB leaderboard Isolation Forest (reproduction)": leaderboard_isolation_forest,
    "IndustrialEdge detector": lambda tr, te: industrialedge(tr, te, debounce=False),
    "IndustrialEdge detector + 3-of-5 alarm confirmation": lambda tr, te: industrialedge(
        tr, te, debounce=True
    ),
    "3-sigma on any signal": three_sigma,
    "Hotelling T-squared (99th percentile limit)": hotelling_t2,
}


def benchmark(data_dir: Path) -> dict:
    experiments = load(download(data_dir))
    results = {}
    for name, detect in DETECTORS.items():
        pairs = [(labels, detect(train, test)) for _, train, test, labels in experiments]
        by_group = {
            group: summarize([p for (g, *_), p in zip(experiments, pairs, strict=True) if g == group])
            for group in EXPERIMENTS
        }
        results[name] = {**summarize(pairs), "by_experiment_group": by_group}

    reproduction = results["SKAB leaderboard Isolation Forest (reproduction)"]
    return {
        "dataset": "SKAB v0.9 (waico/SKAB), 34 labelled experiments",
        "protocol": f"per experiment: fit on first {TRAIN_ROWS} rows, score the rest; pooled counts",
        "test_rows": int(sum(len(labels) for *_, labels in experiments)),
        "anomalous_test_rows": int(sum(labels.sum() for *_, labels in experiments)),
        "leaderboard_reference": LEADERBOARD_ISOLATION_FOREST,
        "protocol_reproduced": all(
            reproduction[key] == value for key, value in LEADERBOARD_ISOLATION_FOREST.items()
        ),
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark anomaly detection on SKAB.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "skab")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = benchmark(args.data_dir)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
