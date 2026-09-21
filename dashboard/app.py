from __future__ import annotations

import os
import time
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="IndustrialEdge AI", page_icon="🏭", layout="wide")
st.title("🏭 IndustrialEdge AI")
st.caption("Real-time machine health · anomaly detection · predictive maintenance")


def api(path: str):
    r = requests.get(f"{API_URL}{path}", timeout=4)
    r.raise_for_status()
    return r.json()


try:
    snapshots = api("/machines")
except Exception as exc:
    st.error(f"API unavailable: {exc}")
    st.stop()

if not snapshots:
    st.info("Waiting for factory telemetry… The simulator will begin publishing automatically in Docker Compose.")
    time.sleep(2)
    st.rerun()

cols = st.columns(len(snapshots))
for col, snap in zip(cols, snapshots):
    a = snap["analysis"]
    t = snap["telemetry"]
    icon = "🟢" if a["severity"] == "normal" else "🟠" if a["severity"] == "warning" else "🔴"
    with col:
        st.subheader(f"{icon} {t['machine_id']}")
        st.metric("Health", f"{a['health_score']:.0f}%")
        st.metric("Anomaly", f"{a['anomaly_score']:.2f}")

machine_ids = [x["telemetry"]["machine_id"] for x in snapshots]
selected = st.selectbox("Inspect machine", machine_ids, index=len(machine_ids)-1)
history = api(f"/machines/{selected}/history?limit=120")
latest = history[-1]
t = latest["telemetry"]
a = latest["analysis"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Temperature", f"{t['temperature']:.1f} °C")
m2.metric("Vibration", f"{t['vibration']:.2f} mm/s")
m3.metric("Pressure", f"{t['pressure']:.2f} bar")
m4.metric("RPM", f"{t['rpm']:.0f}")
m5.metric("Power", f"{t['power']:.2f} kW")

if a["severity"] == "critical":
    st.error(f"{a['likely_cause']} — {a['recommendation']}")
elif a["severity"] == "warning":
    st.warning(f"{a['likely_cause']} — {a['recommendation']}")
else:
    st.success(a["likely_cause"])

st.write("**Signals driving the anomaly:**", ", ".join(a["top_signals"]) or "none")
rows = []
for snap in history:
    row = dict(snap["telemetry"])
    row.update({"anomaly_score": snap["analysis"]["anomaly_score"], "health_score": snap["analysis"]["health_score"]})
    rows.append(row)
df = pd.DataFrame(rows).set_index("timestamp")
st.line_chart(df[["temperature", "vibration", "power"]])
st.line_chart(df[["health_score", "anomaly_score"]])
st.caption("Auto-refreshes every 2 seconds")
time.sleep(2)
st.rerun()
