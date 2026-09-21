from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
import numpy as np
import paho.mqtt.client as mqtt

from industrialedge_ai.config import Settings, load_machines

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
FEATURES = ("temperature", "vibration", "pressure", "rpm", "power")


class FactorySimulator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.machines = load_machines()
        self.rng = np.random.default_rng(2026)
        self.step = 0
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="factory-simulator")

    def connect(self) -> None:
        while True:
            try:
                self.client.connect(self.settings.mqtt_host, self.settings.mqtt_port, keepalive=60)
                self.client.loop_start()
                log.info("Connected to MQTT broker at %s:%s", self.settings.mqtt_host, self.settings.mqtt_port)
                return
            except OSError:
                log.warning("MQTT not ready; retrying...")
                time.sleep(2)

    def sample(self, machine: dict) -> dict:
        values = {
            f: float(self.rng.normal(machine["nominal"][f], machine["std"][f])) for f in FEATURES
        }
        if machine["machine_id"] == self.settings.fault_machine_id:
            phase = self.step % 180
            # Healthy first, then progressive bearing-style temperature/vibration drift.
            if phase >= 45:
                severity = min((phase - 45) / 90.0, 1.0)
                values["vibration"] += severity * 5.2
                values["temperature"] += severity * 24.0
                values["power"] += severity * 1.0
                values["rpm"] += severity * 170.0
        return {
            "machine_id": machine["machine_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **{k: round(v, 3) for k, v in values.items()},
        }

    def run(self) -> None:
        self.connect()
        while True:
            for machine in self.machines:
                data = self.sample(machine)
                topic = f"factory/telemetry/{machine['machine_id']}"
                self.client.publish(topic, json.dumps(data), qos=0)
            if self.step % 10 == 0:
                log.info("Published telemetry cycle %d for %d machines", self.step, len(self.machines))
            self.step += 1
            time.sleep(self.settings.simulator_interval_seconds)


if __name__ == "__main__":
    FactorySimulator(Settings()).run()
