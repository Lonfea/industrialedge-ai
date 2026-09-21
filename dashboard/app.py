from __future__ import annotations

import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="IndustrialEdge AI", page_icon="🏭", layout="wide")
st.title("🏭 IndustrialEdge AI")
st.caption(
    "Real-time machine health · anomaly detection · predictive maintenance · maintenance copilot"
)


def api(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=4)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


try:
    snapshots = api("/machines")
except Exception as exc:
    st.error(f"API unavailable: {exc}")
    st.stop()

if not snapshots:
    st.info(
        "Waiting for factory telemetry… The simulator will begin publishing automatically "
        "in Docker Compose."
    )
    time.sleep(2)
    st.rerun()

cols = st.columns(len(snapshots))
for col, snap in zip(cols, snapshots):
    analysis = snap["analysis"]
    telemetry = snap["telemetry"]
    icon = (
        "🟢"
        if analysis["severity"] == "normal"
        else "🟠"
        if analysis["severity"] == "warning"
        else "🔴"
    )
    with col:
        st.subheader(f"{icon} {telemetry['machine_id']}")
        st.metric("Health", f"{analysis['health_score']:.0f}%")
        st.metric("Anomaly", f"{analysis['anomaly_score']:.2f}")

machine_ids = [x["telemetry"]["machine_id"] for x in snapshots]
selected = st.selectbox("Inspect machine", machine_ids, index=len(machine_ids) - 1)
history = api(f"/machines/{selected}/history?limit=120")
latest = history[-1]
telemetry = latest["telemetry"]
analysis = latest["analysis"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Temperature", f"{telemetry['temperature']:.1f} °C")
m2.metric("Vibration", f"{telemetry['vibration']:.2f} mm/s")
m3.metric("Pressure", f"{telemetry['pressure']:.2f} bar")
m4.metric("RPM", f"{telemetry['rpm']:.0f}")
m5.metric("Power", f"{telemetry['power']:.2f} kW")

if analysis["severity"] == "critical":
    st.error(f"{analysis['likely_cause']} — {analysis['recommendation']}")
elif analysis["severity"] == "warning":
    st.warning(f"{analysis['likely_cause']} — {analysis['recommendation']}")
else:
    st.success(analysis["likely_cause"])

st.write("**Signals driving the anomaly:**", ", ".join(analysis["top_signals"]) or "none")
rows = []
for snap in history:
    row = dict(snap["telemetry"])
    row.update(
        {
            "anomaly_score": snap["analysis"]["anomaly_score"],
            "health_score": snap["analysis"]["health_score"],
        }
    )
    rows.append(row)
df = pd.DataFrame(rows).set_index("timestamp")
st.line_chart(df[["temperature", "vibration", "power"]])
st.line_chart(df[["health_score", "anomaly_score"]])

st.divider()
st.subheader("Maintenance Copilot")
question = st.text_input(
    "Ask about this machine",
    value=f"Why is {selected} abnormal and what should maintenance inspect?",
)
if st.button("Ask copilot", type="primary"):
    try:
        result = api_post("/copilot", {"machine_id": selected, "question": question})
        st.info(result["answer"])
        st.caption(
            f"Grounded in {result['grounded_in_points']} telemetry points · "
            f"provider: {result['provider']}"
        )
    except Exception as exc:
        st.error(f"Copilot unavailable: {exc}")

st.caption("Auto-refreshes every 2 seconds")
time.sleep(2)
st.rerun()
