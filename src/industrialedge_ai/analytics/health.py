from __future__ import annotations


def health_from_anomaly(anomaly_score: float, sample: dict, nominal: dict, std: dict) -> float:
    stress = 0.0
    weights = {"temperature": 0.32, "vibration": 0.42, "power": 0.16, "pressure": 0.10}
    for feature, weight in weights.items():
        deviation = abs(sample[feature] - nominal[feature]) / max(std[feature], 1e-6)
        stress += weight * min(deviation / 6.0, 1.0)
    penalty = 0.68 * anomaly_score + 0.32 * stress
    return round(max(0.0, min(100.0, 100.0 * (1.0 - penalty))), 1)


def classify(health: float, top_signals: list[str]) -> tuple[str, str, str]:
    if health >= 80:
        return "normal", "No abnormal pattern detected", "Continue normal operation and monitoring."
    if health >= 60:
        return "warning", "Developing mechanical or process deviation", "Inspect during the next maintenance window."
    if "vibration" in top_signals and "temperature" in top_signals:
        return "critical", "Possible bearing degradation", "Inspect bearing assembly and lubrication as soon as operationally safe."
    if "temperature" in top_signals:
        return "critical", "Possible overheating or cooling issue", "Check cooling path, load and thermal condition."
    return "critical", "Significant operating deviation", "Schedule immediate diagnostic inspection."
