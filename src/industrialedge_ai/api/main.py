from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query

from industrialedge_ai.analytics.service import AnalyticsService
from industrialedge_ai.config import Settings, load_machines
from industrialedge_ai.ingestion.mqtt_consumer import MQTTConsumer
from industrialedge_ai.models import AnalysisResult, MachineSnapshot, Telemetry
from industrialedge_ai.storage.repository import TelemetryRepository

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
settings = Settings()
analytics = AnalyticsService()
repository = TelemetryRepository(settings.database_url)


def process(telemetry: Telemetry) -> AnalysisResult:
    if telemetry.machine_id not in analytics.configs:
        raise ValueError(f"Unknown machine_id: {telemetry.machine_id}")
    result = analytics.analyze(telemetry)
    repository.save(telemetry, result)
    if result.is_anomaly:
        log.warning("Anomaly %s score=%s health=%s", telemetry.machine_id, result.anomaly_score, result.health_score)
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.mqtt_enabled:
        consumer = MQTTConsumer(settings.mqtt_host, settings.mqtt_port, settings.mqtt_topic, process)
        consumer.start()
    yield


app = FastAPI(
    title="IndustrialEdge AI API",
    version="0.1.0",
    description="Real-time factory telemetry, anomaly detection and predictive-maintenance demo.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "industrialedge-ai"}


@app.get("/machine-config")
def machine_config():
    return load_machines()


@app.post("/telemetry", response_model=AnalysisResult)
def ingest(telemetry: Telemetry):
    try:
        return process(telemetry)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/machines", response_model=list[MachineSnapshot])
def machines():
    return repository.latest_all()


@app.get("/machines/{machine_id}/history", response_model=list[MachineSnapshot])
def history(machine_id: str, limit: int = Query(default=120, ge=1, le=2000)):
    if machine_id not in analytics.configs:
        raise HTTPException(status_code=404, detail="Machine not found")
    return repository.history(machine_id, limit=limit)
