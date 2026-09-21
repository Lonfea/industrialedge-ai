from __future__ import annotations

import requests

from industrialedge_ai.models import CopilotResponse, MachineSnapshot


class MaintenanceCopilot:
    """Telemetry-grounded maintenance assistant with an optional external LLM endpoint."""

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_url = api_url.rstrip("/") if api_url else None
        self.api_key = api_key
        self.model = model

    def answer(
        self,
        machine_id: str,
        question: str,
        history: list[MachineSnapshot],
    ) -> CopilotResponse:
        if not history:
            return CopilotResponse(
                machine_id=machine_id,
                answer="I do not have telemetry for this machine yet.",
                grounded_in_points=0,
                provider="deterministic",
            )

        context = self._context(history)
        if self.api_url and self.model:
            try:
                answer = self._external_answer(question, context)
                return CopilotResponse(
                    machine_id=machine_id,
                    answer=answer,
                    grounded_in_points=len(history),
                    provider=self.model,
                )
            except requests.RequestException:
                pass

        return CopilotResponse(
            machine_id=machine_id,
            answer=self._deterministic_answer(question, history),
            grounded_in_points=len(history),
            provider="deterministic-grounded",
        )

    @staticmethod
    def _context(history: list[MachineSnapshot]) -> str:
        latest = history[-1]
        telemetry = latest.telemetry
        analysis = latest.analysis
        return (
            f"Machine {telemetry.machine_id}; health={analysis.health_score:.1f}; "
            f"anomaly={analysis.anomaly_score:.3f}; severity={analysis.severity}; "
            f"temperature={telemetry.temperature:.2f} C; "
            f"vibration={telemetry.vibration:.2f} mm/s; pressure={telemetry.pressure:.2f} bar; "
            f"rpm={telemetry.rpm:.0f}; power={telemetry.power:.2f} kW; "
            f"top signals={', '.join(analysis.top_signals) or 'none'}; "
            f"diagnosis={analysis.likely_cause}; recommendation={analysis.recommendation}."
        )

    def _external_answer(self, question: str, context: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "temperature": 0.1,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an industrial maintenance assistant. Use only the supplied telemetry "
                            "context. Do not invent measurements or failure certainty. Keep advice concise "
                            "and state that safety-critical action needs qualified personnel."
                        ),
                    },
                    {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
                ],
            },
            timeout=12,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _deterministic_answer(question: str, history: list[MachineSnapshot]) -> str:
        latest = history[-1]
        telemetry = latest.telemetry
        analysis = latest.analysis
        query = question.lower()
        signals = ", ".join(analysis.top_signals) or "no strongly deviating signals"

        if len(history) >= 6:
            recent = history[-6:]
            temp_delta = telemetry.temperature - recent[0].telemetry.temperature
            vib_delta = telemetry.vibration - recent[0].telemetry.vibration
            trend = (
                f"Over the recent window, temperature changed {temp_delta:+.1f} °C "
                f"and vibration {vib_delta:+.2f} mm/s."
            )
        else:
            trend = "There is not enough history yet for a reliable short-term trend."

        if any(word in query for word in ("why", "cause", "abnormal", "anomaly")):
            return (
                f"{telemetry.machine_id} is {analysis.severity} with health "
                f"{analysis.health_score:.0f}% and anomaly score {analysis.anomaly_score:.2f}. "
                f"The strongest drivers are {signals}. The current diagnostic pattern is: "
                f"{analysis.likely_cause}. {trend}"
            )
        if any(word in query for word in ("do", "action", "maintenance", "recommend")):
            return (
                f"Recommended next step: {analysis.recommendation} Current health is "
                f"{analysis.health_score:.0f}% with the largest deviations in {signals}. "
                "Treat this as decision support, not a safety-critical maintenance authorization."
            )
        if any(word in query for word in ("trend", "changing", "worse", "improve")):
            return (
                f"{trend} Current health is {analysis.health_score:.0f}% and anomaly score is "
                f"{analysis.anomaly_score:.2f}."
            )

        return (
            f"{telemetry.machine_id}: health {analysis.health_score:.0f}%, anomaly "
            f"{analysis.anomaly_score:.2f}, severity {analysis.severity}. "
            f"{analysis.likely_cause}. {analysis.recommendation} {trend}"
        )
