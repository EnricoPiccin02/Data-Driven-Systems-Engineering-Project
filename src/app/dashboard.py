"""
Sprint 7 — Real-time dashboard, with

1. An "Household & Training Profile" panel
2. An optional location-drift simulator
3. ADWIN delta slider

As an offline fallback, a CSV upload functionality is implemented.
In the default real-time mode:

1. `streamlit_autorefresh` reruns the page on a timer; each rerun polls one
   new live readingand appends it to a session-held buffer.
2. Every poll is fed through the streaming ADWIN detector.
3. A button triggers a batch Evidently drift comparison against a
   training-time reference sample.
4. Forecast + recommendations are fetched over the REST API

Usage:
    streamlit run src/app/dashboard.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import DEFAULT_CONFIG
from src.mlops.streaming_drift import OnlineDriftMonitor
from src.streaming.live_data import LiveDataStreamer, is_online

# The household's local timezone — used for display only. Internally,
# all timestamps stay UTC (LiveMeterSimulator/LiveWeatherClient produce
# UTC; API payloads use UTC)..
HOUSEHOLD_TZ = ZoneInfo("Europe/Rome")

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REFRESH_INTERVAL_MS = 30000  # 30 seconds
BATCH_MONITORED_COLUMNS = ["temperature_c", "heating_degree", "lag_1", "lag_48", "roll_mean_48", "roll_std_48"]

# Of the full monitored set `BATCH_MONITORED_COLUMNS`, only
# `temperature_c` and `heating_degree` are computable client-side
# from the raw live buffer without running the full offline
# feature-engineering pipeline.
BATCH_COMPARABLE_COLUMNS = [c for c in BATCH_MONITORED_COLUMNS if c in ("temperature_c", "heating_degree")]

# River's own ADWIN default (0.002) is tuned for continuous, high-throughput
# production monitoring (i.e., thousands of observations). This dashboard
# polls once per `REFRESH_INTERVAL_MS`, so a few-minute interactive test
# session yields comparably too few samples for that conservative delta
# to have statistical power, regardless of how large the real distribution
# shift is.
DEFAULT_ADWIN_DELTA_DEMO = 0.05

# Preset locations for the drift-simulation control. The first
# entry always mirrors `DEFAULT_CONFIG``, so "training location" can never
# silently drift out of sync.
PRESET_LOCATIONS = {
    f"Training location — {DEFAULT_CONFIG.location_label}": (
        DEFAULT_CONFIG.weather_latitude, DEFAULT_CONFIG.weather_longitude
    ),
    "Helsinki, Finland (cold climate)": (60.1699, 24.9384),
    "Dubai, UAE (hot climate)": (25.2048, 55.2708),
    "Custom coordinates...": None,
}
TRAINING_LOCATION_KEY = next(iter(PRESET_LOCATIONS))  # First key, by construction above


def _reset_location_to_training() -> None:
    st.session_state.location_choice = TRAINING_LOCATION_KEY


st.set_page_config(page_title="Smart Energy Consumption Forecasting", layout="wide")
st.title("⚡ Smart Energy Consumption Forecasting — Live Dashboard")


# --------------------------------------------------------------------------- #
# Connectivity gate — Real-time mode vs. CSV fallback
# --------------------------------------------------------------------------- #
if "online_checked" not in st.session_state:
    st.session_state.online_checked = is_online()
online = st.session_state.online_checked

if online:
    st.success("Live network connection detected — showing real-time data.", icon="🟢")
else:
    st.warning(
        "No network connection detected — falling back to CSV upload for historical "
        "data. Real-time monitoring, drift detection, and live plots require network "
        "access (Open-Meteo). Recheck connectivity and refresh the page to retry.",
        icon="🟡",
    )


# ============================================================================ #
# REAL-TIME MODE
# ============================================================================ #
if online:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=REFRESH_INTERVAL_MS, key="live_refresh")
    except ImportError:
        st.info("Install `streamlit-autorefresh` for automatic live updates; "
                "showing a manual refresh button instead.")
        st.button("🔄 Refresh now")

    if "streamer" not in st.session_state:
        st.session_state.streamer = LiveDataStreamer()
    if "adwin_delta" not in st.session_state:
        st.session_state.adwin_delta = DEFAULT_ADWIN_DELTA_DEMO
    if "online_monitor" not in st.session_state:
        st.session_state.online_monitor = OnlineDriftMonitor(delta=st.session_state.adwin_delta)
    if "location_change_log" not in st.session_state:
        st.session_state.location_change_log = []  # List of {time, from, to}
    if "location_choice" not in st.session_state:
        st.session_state.location_choice = TRAINING_LOCATION_KEY
    if "batch_drift_result" not in st.session_state:
        st.session_state.batch_drift_result = None  # Persists across autorefresh reruns

    streamer = st.session_state.streamer
    weather_client = streamer.weather_client

    # Poll one new live reading this tick
    try:
        reading = st.session_state.streamer.poll()
        heating_degree = max(18.0 - reading["temperature_c"], 0.0)
        drifted_now = st.session_state.online_monitor.update(
            {
                "consumption_kwh": reading["consumption_kwh"],
                "temperature_c": reading["temperature_c"],
                "heating_degree": heating_degree,
            }
        )
        if drifted_now:
            st.toast(f"ADWIN flagged live drift on: {', '.join(drifted_now)}", icon="⚠️")
    except requests.RequestException as exc:
        st.error(f"Live weather fetch failed this tick ({exc}); will retry next refresh.")

    # ----------------------------------------------------------------------- #
    # Sidebar
    # ----------------------------------------------------------------------- #
    with st.sidebar:
        st.header("Live Stream Controls")

        st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                min-width: 450px;
                max-width: 450px;
            }
        </style>
        """, unsafe_allow_html=True)

        st.subheader(f"🏠 `{streamer.household_id}`")
        st.caption("Wall-clock-synced simulated meter + real live weather.")
        st.caption(f"Calling API at `{API_BASE_URL}`.")

        # --- Household & Training Profile ---
        with st.expander("📋 Household & Training Profile", expanded=False):
            st.markdown("**What the model was trained on** (`DEFAULT_CONFIG`):")
            st.markdown(
                f"- **Location:** {DEFAULT_CONFIG.location_label}  \n"
                f"  ({DEFAULT_CONFIG.weather_latitude:.4f}°N, {DEFAULT_CONFIG.weather_longitude:.4f}°E)\n"
                f"- **Households:** {DEFAULT_CONFIG.n_households} "
                f"(1 real/fallback-synthetic `{DEFAULT_CONFIG.real_household_id}` "
                f"+ {DEFAULT_CONFIG.n_households - 1} synthetic peers)\n"
                f"- **Date range:** {DEFAULT_CONFIG.start_date} → {DEFAULT_CONFIG.end_date} "
                f"(exclusive), {DEFAULT_CONFIG.freq_minutes}-min resolution\n"
                f"- **Holiday calendar:** `{DEFAULT_CONFIG.country_holidays}`\n"
                f"- **Random seed:** {DEFAULT_CONFIG.random_seed}"
            )

            st.markdown("**Feature engineering:**")
            lag_names = [f"lag_{n}" for n in DEFAULT_CONFIG.lag_steps]
            roll_names = [
                f"{stat}_{n}" for n in DEFAULT_CONFIG.rolling_windows for stat in ("roll_mean", "roll_std")
            ]
            st.markdown(
                f"- **Lag steps:** {DEFAULT_CONFIG.lag_steps} → `{', '.join(lag_names)}`\n"
                f"- **Rolling windows:** {DEFAULT_CONFIG.rolling_windows} → `{', '.join(roll_names)}`\n"
                f"- plus calendar features (hour, day-of-week, is_weekend, is_holiday) "
                f"and weather features (temperature, humidity, wind, heating-degree)."
            )
            st.caption(
                "⚠️ Live sessions rarely accumulate enough half-hourly history for "
                "`lag_336`/`roll_*_336` (1 week) or even `roll_*_48` (1 day) — expect "
                "those to show as missing (imputed to 0) during short interactive tests. "
            )

        # Location-drift simulation
        with st.expander("🧪 Simulate Location Drift", expanded=False):
            st.caption(
                "For demonstrating the drift-detection panels only. This changes "
                "where *live weather* is fetched from — it never touches the "
                "trained model or `DEFAULT_CONFIG`. The live meter's consumption "
                "pattern is wall-clock-based and location-independent (not derived "
                "from latitude/longitude at all), so the **forecast kWh value itself "
                "is not expected to shift much** — watch the drift-detection panels "
                "below instead, which compare feature *distributions*, not the "
                "prediction. Note: this tick's reading was already polled before "
                "this control rendered, so a location change made now takes effect "
                "starting the *next* refresh, not this one."
            )
            choice = st.selectbox(
                "Live weather location",
                options=list(PRESET_LOCATIONS.keys()),
                key="location_choice",
            )
            if choice == "Custom coordinates...":
                custom_lat = st.number_input("Latitude", value=weather_client.latitude, format="%.4f")
                custom_lon = st.number_input("Longitude", value=weather_client.longitude, format="%.4f")
                new_coords = (custom_lat, custom_lon)
            else:
                new_coords = PRESET_LOCATIONS[choice]

            current_coords = (weather_client.latitude, weather_client.longitude)
            if new_coords != current_coords:
                st.session_state.location_change_log.append({
                    "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "from": current_coords,
                    "to": new_coords,
                    "label": choice,
                })
                weather_client.latitude, weather_client.longitude = new_coords
                st.toast(f"Live weather location changed to: {choice}", icon="🧪")

            if choice != TRAINING_LOCATION_KEY:
                st.button("↩ Reset to training location", on_click=_reset_location_to_training)
                if st.button("🗑 Clear buffered readings"):
                    streamer.buffer.clear()
                    st.session_state.online_monitor.reset()
                    st.session_state.batch_drift_result = None
                    st.success("Buffer, ADWIN detectors, and batch results cleared — starting fresh.")

            if st.session_state.location_change_log:
                st.caption("Change log (this session):")
                st.dataframe(pd.DataFrame(st.session_state.location_change_log), hide_index=True)

        # ADWIN sensitivity tuning
        with st.expander("⚙️ Drift Sensitivity (ADWIN)", expanded=False):
            st.caption(
                "ADWIN's `delta` controls how much evidence it needs before "
                "flagging drift: smaller = more conservative (needs stronger, "
                "more sustained evidence; fewer false positives) but slower to "
                "fire. River's own library default is 0.002, tuned for "
                f"continuous, high-throughput monitoring. This dashboard polls "
                f"only once every {REFRESH_INTERVAL_MS // 1000}s, so a short "
                f"interactive session may never accumulate enough samples for "
                f"that default to fire — the default here "
                f"({DEFAULT_ADWIN_DELTA_DEMO}) is tuned looser for demos. "
                "Changing this rebuilds all detectors and clears their "
                "accumulated window state (same trade-off as a manual reset)."
            )
            new_delta = st.slider(
                "ADWIN delta (lower = more sensitive)",
                min_value=0.001, max_value=0.5,
                value=st.session_state.adwin_delta, step=0.001, format="%.3f",
            )
            if new_delta != st.session_state.adwin_delta:
                st.session_state.adwin_delta = new_delta
                st.session_state.online_monitor.set_delta(new_delta)
                st.toast(f"ADWIN delta set to {new_delta:.3f} — detectors rebuilt.", icon="⚙️")
            st.caption(
                "Even at a low delta, ADWIN still needs *some* new samples after "
                "a location switch before it can detect anything — clearing the "
                "buffer (above) first gives it a clean signal to work with. Note "
                "consumption_kwh drift events may fire just from crossing a normal "
                "diurnal peak/trough — that's not location drift, just the expected "
                "daily cycle being detected as *a* change."
            )


        # Session statistics
        sidebar_history = streamer.history_dataframe()
        if not sidebar_history.empty:
            st.divider()
            st.markdown("**This session's stream**")
            st.metric("Readings buffered", len(sidebar_history))
            started_local = sidebar_history["timestamp"].iloc[0].astimezone(HOUSEHOLD_TZ)
            st.caption(f"Stream started: {started_local:%H:%M:%S %Z}")
            st.caption(f"Mean consumption so far: {sidebar_history['consumption_kwh'].mean():.3f} kWh")
            latest = sidebar_history.iloc[-1]
            badges = [
                "🟢 real weather" if latest.get("is_real_weather") else "🟠 fallback weather",
                "🟢 real meter data" if latest.get("is_real_data") else "🟠 simulated meter",
            ]
            st.caption(" · ".join(badges))


            # Handle the case the buffer containes readings of different locations
            if {"source_latitude", "source_longitude"}.issubset(sidebar_history.columns):
                at_current = (
                    (sidebar_history["source_latitude"].round(2) == round(weather_client.latitude, 2))
                    & (sidebar_history["source_longitude"].round(2) == round(weather_client.longitude, 2))
                ).sum()
                n_total = len(sidebar_history)
                if 0 < at_current < n_total:
                    st.warning(
                        f"Buffer is mixed: only {at_current}/{n_total} readings match the "
                        f"current location. Drift signal is diluted — use the '🗑 Clear buffered "
                        f"readings' button below for a clean before/after comparison.",
                        icon="⚠️",
                    )
                    if st.button("🗑 Clear buffered readings", key="clear_buffer_mixed"):
                        streamer.buffer.clear()
                        st.session_state.online_monitor.reset()
                        st.session_state.batch_drift_result = None
                        st.success("Buffer, ADWIN detectors, and batch results cleared — starting fresh.")
            else:
                st.caption(
                    "ℹ️ Location-mix diagnostic unavailable — `LiveDataStreamer.poll()` needs "
                    "to record `source_latitude`/`source_longitude` per reading."
                )

        st.divider()
        if st.button("↺ Reset drift detectors"):
            st.session_state.online_monitor.reset()
            st.success("ADWIN detectors reset.")

    # Persistent, unmistakable banner while simulating drift
    if (weather_client.latitude, weather_client.longitude) != PRESET_LOCATIONS[TRAINING_LOCATION_KEY]:
        st.error(
            f"**SIMULATION MODE** — live weather is being fetched from "
            f"`{st.session_state.location_choice}` instead of the training location "
            f"`{DEFAULT_CONFIG.location_label}`. Forecasts and drift readings below "
            f"reflect this deliberate mismatch. Reset from the sidebar when done.",
            icon="🧪",
        )

    history_df = st.session_state.streamer.history_dataframe()
    household_id = streamer.household_id

    col1, col2, col3 = st.columns(3)
    if not history_df.empty:
        latest = history_df.iloc[-1]
        col1.metric("Live consumption", f"{latest['consumption_kwh']:.3f} kWh")
        col2.metric("Live temperature", f"{latest['temperature_c']:.1f} °C")
        col3.metric("Readings buffered", f"{len(history_df)}")

    st.subheader("Live consumption stream")

    if len(history_df) >= 2:
        chart_df = history_df.copy()
        chart_df["timestamp"] = (
            chart_df["timestamp"]
            .dt.tz_convert(HOUSEHOLD_TZ)
        )

        # Consumption real-time plot
        chart = (
            alt.Chart(chart_df)
            .mark_line()
            .encode(
                x=alt.X(
                    "timestamp:T",
                    axis=alt.Axis(format="%H:%M:%S", title=None)
                ),
                y=alt.Y(
                    "consumption_kwh:Q",
                    title="Consumption (kWh)"
                )
            )
            .properties(height=350)
        )

        st.altair_chart(chart, use_container_width=True)

    else:
        st.info("Buffering — need at least 2 live readings to plot.")

    # Forecast + recommendations + explanation
    st.subheader("Forecast + optimisation suggestions")
    if len(history_df) >= 5:
        last_ts = history_df["timestamp"].iloc[-1]  # tz-aware UTC
        forecast_ts_utc = last_ts + timedelta(hours=1)
        payload = {
            "household_id": household_id,
            "forecast_timestamp": forecast_ts_utc.isoformat(),
            "temperature_c": float(history_df["temperature_c"].iloc[-1]),
            "recent_readings": [
                {"timestamp": row.timestamp.isoformat(), "consumption_kwh": float(row.consumption_kwh)}
                for row in history_df.itertuples()
            ],
        }
        try:
            resp = requests.post(f"{API_BASE_URL}/recommendations", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            forecast_resp = requests.post(f"{API_BASE_URL}/forecast", json=payload, timeout=10).json()
            forecast_ts_local = forecast_ts_utc.astimezone(HOUSEHOLD_TZ)
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                st.metric(f"Forecast for {forecast_ts_local:%H:%M %Z}",
                          f"{forecast_resp['predicted_consumption_kwh']:.2f} kWh")
                st.caption(f"Model: {forecast_resp['model_name']} v{forecast_resp['model_version']} "
                           f"({forecast_resp['model_stage']})")
                st.caption(f"Live temperature used: {payload['temperature_c']:.1f} °C "
                           f"(heating_degree={max(18.0 - payload['temperature_c'], 0.0):.1f})")
            with fcol2:
                for rec in result["recommendations"]:
                    st.info(f"**{rec['title']}**\n\n{rec['description']}\n\n"
                            f"Estimated saving: {rec['estimated_saving_kwh']:.2f} kWh")

            # SHAP local explanation chart about "Why this forecast?"
            st.markdown("**🔍 Why this forecast?**")
            try:
                explain_resp = requests.post(f"{API_BASE_URL}/forecast/explain", json=payload, timeout=15)
                explain_resp.raise_for_status()
                contrib_df = pd.DataFrame(explain_resp.json()["top_contributions"])
                if not contrib_df.empty:
                    contrib_chart = (
                        alt.Chart(contrib_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("shap_value:Q", title="Contribution to forecast (kWh)"),
                            y=alt.Y("feature:N", sort="-x", title=None),
                            color=alt.condition(
                                alt.datum.shap_value > 0, alt.value("#d62728"), alt.value("#1f77b4")
                            ),
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(contrib_chart, use_container_width=True)
                    st.caption(
                        "Red bars push the forecast up, blue bars pull it down — this "
                        "specific prediction's top feature contributions (SHAP). Features "
                        "showing zero contribution are commonly `lag_336`/`roll_*_336` (or "
                        "even `roll_*_48`) when the live buffer hasn't accumulated enough "
                        "history yet — see the caveat in '📋 Household & Training Profile'."
                    )
            except requests.RequestException as exc:
                st.caption(f"Explanation unavailable: {exc}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the prediction API at {API_BASE_URL}: {exc}")
    else:
        st.info(f"Need at least 5 buffered readings for a forecast (have {len(history_df)}).")

    # Streaming drift panel (River ADWIN)
    st.subheader("Online drift detection (River ADWIN)")
    status = st.session_state.online_monitor.status()
    dcol1, dcol2, dcol3 = st.columns(3)
    dcol1.metric("Live updates processed", status["n_updates"])
    dcol2.metric("Drift events (session total)", status["n_drift_events_total"])
    dcol3.metric("Features ever flagged", ", ".join(status["features_with_drift_ever"]) or "none")
    st.caption(f"Current ADWIN delta: {status['delta']:.3f} (tune in sidebar → ⚙️ Drift Sensitivity)")
    if status["recent_events"]:
        st.dataframe(pd.DataFrame(status["recent_events"]))

    # Batch drift panel (Evidently), button-triggered
    st.subheader("Batch drift report vs. training data (Evidently)")
    st.caption(
        f"Compares `{'`, `'.join(BATCH_COMPARABLE_COLUMNS)}` (the subset of the full "
        f"monitored feature set computable from the raw live buffer without running "
        f"the offline feature-engineering pipeline) against a training-time reference "
        f"sample."
    )
    if st.button("▶ Run batch drift comparison now"):
        if len(history_df) < 30:
            st.warning(f"Need at least 30 buffered readings for a meaningful batch "
                       f"comparison (have {len(history_df)}). Keep the dashboard open "
                       f"a little longer and try again.")
        else:
            try:
                ref_resp = requests.get(
                    f"{API_BASE_URL}/monitoring/reference-sample",
                    params={"columns": ",".join(BATCH_COMPARABLE_COLUMNS)}, timeout=15,
                )
                ref_resp.raise_for_status()
                reference_df = pd.DataFrame(ref_resp.json()["records"])

                current_df = history_df[["temperature_c"]].dropna().copy()
                current_df["heating_degree"] = (18 - current_df["temperature_c"]).clip(lower=0)

                from src.mlops.monitoring import detect_feature_drift
                results = detect_feature_drift(reference_df, current_df, BATCH_COMPARABLE_COLUMNS)

                st.session_state.batch_drift_result = {
                    "ran_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "n_current_rows": len(current_df),
                    "results": [
                        {"feature": r.feature, "drift_detected": r.drift_detected,
                         "score": r.drift_score, "test": r.stattest}
                        for r in results
                    ],
                }
            except requests.RequestException as exc:
                st.error(f"Could not fetch the reference sample from the API: {exc}")

    if st.session_state.batch_drift_result is not None:
        res = st.session_state.batch_drift_result
        ran_at_local = pd.Timestamp(res["ran_at_utc"]).tz_convert(HOUSEHOLD_TZ)
        st.caption(f"Last run: {ran_at_local:%H:%M:%S %Z} · {res['n_current_rows']} buffered rows compared")
        for r in res["results"]:
            icon = "🔴" if r["drift_detected"] else "🟢"
            st.write(f"{icon} **{r['feature']}** — drift_detected={r['drift_detected']}, "
                     f"score={r['score']:.4f}, test={r['test']}")


# ============================================================================ #
# OFFLINE FALLBACK — CSV upload
# ============================================================================ #
else:
    st.subheader("Offline mode: upload historical data")
    household_id = st.text_input("Household ID", value="HH000")
    temperature_c = st.slider("Forecast temperature (°C)", -10.0, 35.0, 10.0)
    horizon_hours = st.slider("Forecast horizon (hours ahead)", 1, 48, 24)

    uploaded = st.file_uploader(
        "Upload recent half-hourly readings (CSV: timestamp,consumption_kwh)", type="csv"
    )
    if uploaded is not None:
        history_df = pd.read_csv(uploaded, parse_dates=["timestamp"])
        st.line_chart(history_df.set_index("timestamp")["consumption_kwh"])
        forecast_ts = history_df["timestamp"].max() + timedelta(hours=horizon_hours)

        if st.button("Get forecast + recommendations"):
            payload = {
                "household_id": household_id,
                "forecast_timestamp": forecast_ts.isoformat(),
                "temperature_c": temperature_c,
                "recent_readings": [
                    {"timestamp": row.timestamp.isoformat(), "consumption_kwh": row.consumption_kwh}
                    for row in history_df.itertuples()
                ],
            }
            try:
                resp = requests.post(f"{API_BASE_URL}/recommendations", json=payload, timeout=10)
                resp.raise_for_status()
                result = resp.json()
                forecast_resp = requests.post(f"{API_BASE_URL}/forecast", json=payload, timeout=10).json()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"Forecast for {forecast_ts:%Y-%m-%d %H:%M}",
                              f"{forecast_resp['predicted_consumption_kwh']:.2f} kWh")
                    st.caption(f"Model: {forecast_resp['model_name']} v{forecast_resp['model_version']} "
                               f"({forecast_resp['model_stage']})")
                with col2:
                    st.subheader("Optimisation suggestions")
                    for rec in result["recommendations"]:
                        st.info(f"**{rec['title']}**\n\n{rec['description']}\n\n"
                                f"Estimated saving: {rec['estimated_saving_kwh']:.2f} kWh")
            except requests.RequestException as exc:
                st.error(f"Could not reach the prediction API at {API_BASE_URL}: {exc}")
    else:
        st.info("Upload a CSV of recent half-hourly readings to get a forecast.")


st.divider()
st.caption(
    "Real-time mode uses real live weather (Open-Meteo) + a wall-clock-synced "
    "live meter simulator and River's ADWIN + Evidently for drift detection."
)