from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(self, tracking_uri: str, enabled: bool = False):
        self.tracking_uri = tracking_uri
        self.enabled = enabled

    def register_baselines(self, detectors: dict, retries: int = 5) -> bool:
        if not self.enabled:
            return False

        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment("industrialedge-baselines")

        for attempt in range(retries):
            try:
                for machine_id, detector in detectors.items():
                    with mlflow.start_run(run_name=f"baseline-{machine_id}"):
                        mlflow.set_tags(
                            {
                                "project": "industrialedge-ai",
                                "machine_id": machine_id,
                                "model_type": "IsolationForest",
                            }
                        )
                        mlflow.log_params(
                            {
                                "n_estimators": detector.model.n_estimators,
                                "contamination": detector.model.contamination,
                                "features": 5,
                            }
                        )
                        mlflow.log_metrics(
                            {
                                "baseline_score_low": detector.low,
                                "baseline_score_high": detector.high,
                            }
                        )
                        mlflow.sklearn.log_model(detector.model, artifact_path="model")
                log.info("Registered %d machine baselines in MLflow", len(detectors))
                return True
            except Exception as exc:
                if attempt == retries - 1:
                    log.warning("MLflow registration skipped after retries: %s", exc)
                    return False
                time.sleep(2)
        return False
