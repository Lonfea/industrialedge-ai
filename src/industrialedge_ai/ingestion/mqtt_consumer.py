from __future__ import annotations

import json
import logging
import time
from threading import Thread

import paho.mqtt.client as mqtt

from industrialedge_ai.models import Telemetry

log = logging.getLogger(__name__)


class MQTTConsumer:
    def __init__(self, host: str, port: int, topic: str, on_telemetry):
        self.host, self.port, self.topic = host, port, topic
        self.on_telemetry = on_telemetry
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="industrialedge-api")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        log.info("MQTT connected with reason=%s", reason_code)
        client.subscribe(self.topic)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            self.on_telemetry(Telemetry.model_validate(payload))
        except Exception:
            log.exception("Failed to process MQTT message on %s", msg.topic)

    def run(self) -> None:
        while True:
            try:
                self.client.connect(self.host, self.port, keepalive=60)
                self.client.loop_forever(retry_first_connection=True)
            except OSError:
                log.warning("MQTT broker unavailable; retrying in 2 seconds")
                time.sleep(2)

    def start(self) -> Thread:
        thread = Thread(target=self.run, daemon=True, name="mqtt-consumer")
        thread.start()
        return thread
