"""
app.py — Streamlit interface for Dynamic Train ETA Prediction.

HISTORICAL REPLAY of September 2024 Indian Railway data.
This is a college micro-project prototype, NOT a live tracking service.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time

from data_loader import load_all, get_journey, get_available_trains, get_dates_for_train
from model import (
    DelayPredictor,
    timetable_baseline,
    persistence_baseline,
)


# ============================================================================
#  PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="🚂 Dynamic Train ETA Prediction",
    page_icon="🚂",
    layout="wide",
)


# ============================================================================
#  HELPER FUNCTIONS
# ============================================================================

def time_to_minutes(t) -> float:
    """Convert datetime.time → minutes since midnight."""
    if pd.isna(t) or t is None:
        return np.nan
    return t.hour * 60 + t.minute


def minutes_to_time_str(minutes: float) -> str:
    """Convert minutes since midnight → '08:32 AM' style string."""
    if pd.isna(minutes) or minutes is None:
        return "—"
    minutes = float(minutes)
    h = int(minutes // 60) % 24
    m = int(minutes % 60)
    period = "AM" if h < 12 else "PM"
    display_h = h % 12
    if display_h == 0:
        display_h = 12
    return f"{display_h}:{m:02d} {period}"


def format_delay(delay_min) -> str:
    """Format delay as a readable string with color hint."""
    if pd.isna(delay_min) or delay_min is None:
        return "—"
    delay_min = float(delay_min)
    if delay_min <= 0:
        return "On time ✅"
    elif delay_min <= 10:
        return f"+{delay_min:.0f} min 🟡"
    else:
        return f"+{delay_min:.0f} min 🔴"


def compute_eta_str(sch_arr, delay_minutes) -> str:
    """
    Compute predicted ETA = scheduled arrival + predicted delay.
    Returns a formatted time string.
    """
    sch_min = time_to_minutes(sch_arr)
    if pd.isna(sch_min) or pd.isna(delay_minutes):
        return "—"
    eta_min = sch_min + float(delay_minutes)
    return minutes_to_time_str(eta_min)


# ============================================================================
#  DATA LOADING  (cached so it loads only once)
# ============================================================================

@st.cache_data(show_spinner="📦 Loading railway data …  (first time takes ~30 s)")
def cached_load_all():
    return load_all()


@st.cache_resource(show_spinner="🤖 Training Random Forest model …")
def cached_train_model(_merged_df):
    """Train the RF model once and cache it."""
    predictor = DelayPredictor(n_estimators=100, max_depth=12)
    predictor.train(_merged_df)
    return predictor


# ============================================================================
#  MAIN APP
# ============================================================================

def main():
    # ---- Header ----
    st.title("🚂 Dynamic Train ETA Prediction")
    st.caption(
        "⚠️ **Historical Demonstration** — replays recorded journeys from "
        "September 2024 Indian Railway data. This is **not** a live service."
    )

    # ---- Load data ----
    merged_df, routes_df, zones = cached_load_all()

    # ---- Train model ----
    predictor = cached_train_model(merged_df)

    # ---- Sidebar: Train & Date selectors ----
    st.sidebar.header("🔍 Select Journey")

    trains_summary = get_available_trains(merged_df)

    # Build display labels: "12303 — Poorva Express (28 journeys)"
    train_labels = {
        row.train: f"{row.train} — {row.trainName}  ({row.journey_count} days)"
        for row in trains_summary.itertuples()
    }
    selected_train = st.sidebar.selectbox(
        "Train",
        options=list(train_labels.keys()),
        format_func=lambda t: train_labels.get(t, t),
    )

    # Dates available for the selected train
    dates = get_dates_for_train(merged_df, selected_train)
    if not dates:
        st.warning("No journey dates found for this train.")
        return

    selected_date = st.sidebar.selectbox(
        "Journey Date",
        options=dates,
        format_func=lambda d: d.strftime("%d %b %Y (%A)"),
    )

    # ---- Retrieve journey ----
    journey = get_journey(merged_df, selected_train, str(selected_date))

    if journey.empty or len(journey) < 2:
        st.error(
            f"No usable journey data for train {selected_train} "
            f"on {selected_date}. Try another date."
        )
        return

    n_stations = len(journey)

    # ---- Model status in sidebar ----
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Model Info")
    if predictor.is_trained:
        st.sidebar.success("Random Forest trained ✅")
        st.sidebar.metric("Training samples", f"{predictor.n_train:,}")
        if predictor.test_mae is not None:
            st.sidebar.metric("Test MAE", f"{predictor.test_mae:.1f} min")
    else:
        st.sidebar.warning("Model not trained — insufficient data. "
                           "Baselines still work.")

    # ---- Journey header ----
    train_name = journey["trainName"].iloc[0]
    if pd.isna(train_name):
        train_name = f"Train {selected_train}"
    st.subheader(f"🛤️ {train_name}  ({selected_train})  —  "
                 f"{selected_date.strftime('%d %b %Y')}")

    origin_stn  = journey["station"].iloc[0]
    dest_stn    = journey["station"].iloc[-1]
    origin_name = journey["station_name"].iloc[0] if "station_name" in journey.columns else origin_stn
    dest_name   = journey["station_name"].iloc[-1] if "station_name" in journey.columns else dest_stn
    if pd.isna(origin_name): origin_name = origin_stn
    if pd.isna(dest_name):   dest_name = dest_stn
    st.write(f"**Route:** {origin_name} → {dest_name}  |  "
             f"**Stops:** {n_stations}  |  "
             f"**Total distance:** {journey['distance'].max():.0f} km")

    # ---- Replay controls ----
    st.markdown("---")
    st.markdown("### ▶️ Journey Replay")
    st.info(
        "Use the **Next Station** button to reveal one station at a time, "
        "simulating the journey as it unfolds.  Predictions update at each step."
    )

    # Session-state key unique to this journey
    journey_key = f"step_{selected_train}_{selected_date}"
    if journey_key not in st.session_state:
        st.session_state[journey_key] = 1  # start with origin revealed

    # Control buttons
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        if st.button("⏮️ Reset", use_container_width=True):
            st.session_state[journey_key] = 1
            st.rerun()
    with col_b2:
        if st.button("⏪ Back", use_container_width=True):
            if st.session_state[journey_key] > 1:
                st.session_state[journey_key] -= 1
                st.rerun()
    with col_b3:
        if st.button("▶️ Next Station", type="primary", use_container_width=True):
            if st.session_state[journey_key] < n_stations:
                st.session_state[journey_key] += 1
                st.rerun()
    with col_b4:
        if st.button("⏭️ Show All", use_container_width=True):
            st.session_state[journey_key] = n_stations
            st.rerun()

    revealed = st.session_state[journey_key]

    # Progress bar
    st.progress(revealed / n_stations,
                text=f"Stations revealed: {revealed} / {n_stations}")

    # ---- Compute predictions for upcoming stations ----
    n_upcoming = n_stations - revealed

    # Last known delay
    last_known_delay = 0.0
    if revealed > 0:
        d = journey.iloc[revealed - 1]["arr_delay"]
        if not pd.isna(d):
            last_known_delay = float(d)

    # Baselines
    tt_preds   = timetable_baseline(n_upcoming)
    pers_preds = persistence_baseline(last_known_delay, n_upcoming)

    # Random Forest
    rf_preds   = predictor.predict_upcoming(journey, revealed)

    # ---- Build display table ----
    rows = []
    for i in range(n_stations):
        stn      = journey.iloc[i]
        stn_code = stn["station"]
        stn_name = stn.get("station_name", stn_code)
        if pd.isna(stn_name):
            stn_name = stn_code
        distance = stn["distance"]
        zone     = zones.get(stn_code, "—")

        sch_arr_str = (
            stn["sch_arr"].strftime("%I:%M %p") if isinstance(stn["sch_arr"], dt_time) else "—"
        )

        if i < revealed:
            # ---- REVEALED station ----
            act_arr_str = (
                stn["act_arr"].strftime("%I:%M %p") if isinstance(stn["act_arr"], dt_time) else "—"
            )
            delay_str = format_delay(stn["arr_delay"])
            status = "✅ Passed"

            rows.append({
                "🚏 #":          stn.get("stnSerialNumber", i + 1),
                "Station":       f"{stn_name} ({stn_code})",
                "Zone":          zone,
                "Dist (km)":     f"{distance:.0f}",
                "Scheduled":     sch_arr_str,
                "Actual":        act_arr_str,
                "Delay":         delay_str,
                "📅 Timetable ETA": sch_arr_str,
                "🔄 Persist ETA":   act_arr_str,
                "🤖 RF ETA":        act_arr_str,
                "Status":        status,
            })
        else:
            # ---- UPCOMING station (predictions) ----
            idx = i - revealed
            tt_eta   = compute_eta_str(stn["sch_arr"], tt_preds[idx])
            pers_eta = compute_eta_str(stn["sch_arr"], pers_preds[idx])
            rf_eta   = compute_eta_str(stn["sch_arr"], rf_preds[idx])

            rows.append({
                "🚏 #":          stn.get("stnSerialNumber", i + 1),
                "Station":       f"{stn_name} ({stn_code})",
                "Zone":          zone,
                "Dist (km)":     f"{distance:.0f}",
                "Scheduled":     sch_arr_str,
                "Actual":        "⏳",
                "Delay":         "⏳",
                "📅 Timetable ETA": tt_eta,
                "🔄 Persist ETA":   pers_eta,
                "🤖 RF ETA":        rf_eta,
                "Status":        "🔮 Upcoming",
            })

    display_df = pd.DataFrame(rows)
    st.dataframe(display_df, use_container_width=True, hide_index=True,
                 height=min(45 * n_stations + 40, 600))

    # ---- Prediction accuracy (once journey is fully revealed) ----
    if revealed == n_stations:
        st.markdown("---")
        st.markdown("### 📊 Prediction Accuracy Summary")

        # Recompute predictions at each step and collect errors
        # For a fair comparison, we do step-by-step replay
        tt_errors   = []
        pers_errors = []
        rf_errors   = []

        for step in range(1, n_stations):
            actual_delay = journey.iloc[step]["arr_delay"]
            if pd.isna(actual_delay):
                continue

            # What was the last known delay at that step?
            prev_d = journey.iloc[step - 1]["arr_delay"]
            prev_d = float(prev_d) if not pd.isna(prev_d) else 0.0

            # Timetable: always predicts 0
            tt_errors.append(abs(0 - actual_delay))

            # Persistence: predicts prev_d
            pers_errors.append(abs(prev_d - actual_delay))

            # Random Forest: re-predict from that step
            rf_pred = predictor.predict_upcoming(journey, step)
            if len(rf_pred) > 0:
                rf_errors.append(abs(rf_pred[0] - actual_delay))

        col1, col2, col3 = st.columns(3)
        with col1:
            mae_tt = np.mean(tt_errors) if tt_errors else float("nan")
            st.metric("📅 Timetable MAE", f"{mae_tt:.1f} min")
        with col2:
            mae_pers = np.mean(pers_errors) if pers_errors else float("nan")
            st.metric("🔄 Persistence MAE", f"{mae_pers:.1f} min")
        with col3:
            mae_rf = np.mean(rf_errors) if rf_errors else float("nan")
            st.metric("🤖 Random Forest MAE", f"{mae_rf:.1f} min")

        # Visual comparison bar chart
        chart_data = pd.DataFrame({
            "Method": ["Timetable", "Persistence", "Random Forest"],
            "MAE (min)": [mae_tt, mae_pers, mae_rf],
        })
        st.bar_chart(chart_data.set_index("Method"), height=300)

        st.caption(
            "**MAE** = Mean Absolute Error — average prediction miss in minutes. "
            "Lower is better."
        )

    # ---- Footer ----
    st.markdown("---")
    st.caption(
        "🎓 College Micro-Project  •  Dynamic Train ETA Prediction  •  "
        "Data: Indian Railways, September 2024  •  "
        "This is a **historical demonstration**, not a live service."
    )


if __name__ == "__main__":
    main()
