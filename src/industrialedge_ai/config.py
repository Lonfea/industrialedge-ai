from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./industrialedge.db")
    mqtt_host: str = os.getenv("MQTT_HOST", "localhost")
    mqtt_port: int = int(os.getenv("MQTT_PORT", "1883"))
    mqtt_topic: str = os.getenv("MQTT_TOPIC", "factory/telemetry/#")
    mqtt_enabled: bool = os.getenv("MQTT_ENABLED", "true").lower() in {"1", "true", "yes"}
    simulator_interval_seconds: float = float(os.getenv("SIMULATOR_INTERVAL_SECONDS", "1.0"))
    fault_machine_id: str = os.getenv("FAULT_MACHINE_ID", "MACHINE-04")


def machine_config_path() -> Path:
    configured = os.getenv("MACHINE_CONFIG")
    if configured:
        return Path(configured)
    cwd_path = Path.cwd() / "config" / "machines.json"
    if cwd_path.exists():
        return cwd_path
    return Path(__file__).resolve().parents[2] / "config" / "machines.json"


def load_machines() -> list[dict]:
    with machine_config_path().open("r", encoding="utf-8") as f:
        return json.load(f)["machines"]
