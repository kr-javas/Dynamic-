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
#  PAGE CONFIG & CUSTOM CSS
# ============================================================================
st.set_page_config(
    page_title="🚂 Dynamic Train ETA Prediction",
    page_icon="🚂",
    layout="wide",
)

# --- Colour scheme: green (on-time), amber (≤15 min late), red (>15 min) ---
st.markdown("""
<style>
/* Metric cards row */
.metric-card {
    background: #f8f9fa; border-radius: 10px; padding: 14px 18px;
    border-left: 4px solid #4361ee; text-align: center;
}
.metric-card h4 { margin:0; font-size:0.78rem; color:#666; text-transform:uppercase; }
.metric-card p  { margin:4px 0 0; font-size:1.25rem; font-weight:700; color:#1a1a2e; }

/* Delay colours */
.delay-green  { color: #0a8754; font-weight: 600; }
.delay-amber  { color: #d4850a; font-weight: 600; }
.delay-red    { color: #c0392b; font-weight: 600; }
.delay-pend   { color: #888; }

/* Row highlights — dark text forced for readability in any theme */
.row-current td { background: #c8e6c9 !important; color: #1b5e20 !important; }
.row-next    td { background: #fff3cd !important; color: #664d03 !important; }

/* Info banner */
.info-banner {
    background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
    border-radius: 8px; padding: 10px 16px; margin-bottom: 16px;
    font-size: 0.85rem; color: #37474f; border: 1px solid #bbdefb;
}

/* Smaller station zone text */
.zone-text { font-size: 0.72rem; color: #999; }

/* Table tweaks — no wrap, dark readable text, compact */
table.eta-table { font-size: 0.84rem !important; width: 100%;
                  border-collapse: collapse; }
table.eta-table th, table.eta-table td {
    white-space: nowrap; padding: 5px 7px; text-align: left;
}
table.eta-table thead th {
    background: #343a40 !important; color: #f8f9fa !important;
    font-weight: 600; font-size: 0.8rem;
}
table.eta-table tbody tr:nth-child(even) { background: #fafafa; }
table.eta-table tbody tr:hover { background: #e9ecef; }
table.eta-table tbody td { color: #212529; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
#  HELPER FUNCTIONS
# ============================================================================

def time_to_minutes(t) -> float:
    """Convert datetime.time → minutes since midnight."""
    if pd.isna(t) or t is None:
        return np.nan
    return t.hour * 60 + t.minute


def minutes_to_time_str(minutes: float) -> str:
    """Convert minutes since midnight → '09:32' (24-hour HH:MM) string.
    Rounds to the nearest minute so fractional delays display correctly."""
    if pd.isna(minutes) or minutes is None:
        return "—"
    total = round(float(minutes))          # ← round, not truncate
    h = (total // 60) % 24
    m = total % 60
    return f"{h:02d}:{m:02d}"


def format_delay_short(delay_min) -> str:
    """Short delay text: 'On time', '+12 min', '-2 min'."""
    if pd.isna(delay_min) or delay_min is None:
        return "—"
    delay_min = float(delay_min)
    if abs(delay_min) < 0.5:
        return "On time"
    sign = "+" if delay_min > 0 else ""
    return f"{sign}{delay_min:.0f} min"


def delay_css_class(delay_min) -> str:
    """Return CSS class name for the delay value."""
    if pd.isna(delay_min) or delay_min is None:
        return "delay-pend"
    delay_min = float(delay_min)
    if delay_min <= 0:
        return "delay-green"
    elif delay_min <= 15:
        return "delay-amber"
    else:
        return "delay-red"


def compute_eta_str(sch_arr, delay_minutes) -> str:
    """predicted ETA = scheduled arrival + predicted delay."""
    sch_min = time_to_minutes(sch_arr)
    if pd.isna(sch_min) or pd.isna(delay_minutes):
        return "—"
    return minutes_to_time_str(sch_min + float(delay_minutes))


def compute_eta_range_str(sch_arr, p10, p90) -> str:
    """Format 10th–90th percentile ETA range."""
    sch_min = time_to_minutes(sch_arr)
    if pd.isna(sch_min) or pd.isna(p10) or pd.isna(p90):
        return "—"
    lo = minutes_to_time_str(sch_min + float(p10))
    hi = minutes_to_time_str(sch_min + float(p90))
    if lo == hi:
        return lo
    return f"{lo} – {hi}"


def format_error(predicted_delay, actual_delay) -> str:
    """Error = predicted - actual, shown as signed minutes."""
    if pd.isna(predicted_delay) or pd.isna(actual_delay):
        return "—"
    err = float(predicted_delay) - float(actual_delay)
    if abs(err) < 0.5:
        return "0"
    sign = "+" if err > 0 else ""
    return f"{sign}{err:.0f}"


# ============================================================================
#  DATA / MODEL CACHING
# ============================================================================

@st.cache_data(show_spinner="📦 Loading railway data …  (first time ≈ 30 s)")
def cached_load_all():
    return load_all()


@st.cache_resource(show_spinner="🤖 Training Random Forest model …")
def cached_train_model(_merged_df):
    predictor = DelayPredictor(n_estimators=100, max_depth=12)
    predictor.train(_merged_df)
    return predictor


# ============================================================================
#  MAIN APP
# ============================================================================

def main():
    # ---- Title + info banner (req 1) ----
    st.title("🚂 Dynamic Train ETA Prediction")
    st.markdown(
        '<div class="info-banner">'
        '📡 Replay of historical data <b>(Sept 2024)</b>, not a live feed.&ensp;'
        'ETA = scheduled arrival + predicted delay, recomputed after each station.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Load data & train model ----
    merged_df, routes_df, zones = cached_load_all()
    predictor = cached_train_model(merged_df)

    # ==================================================================
    #  SIDEBAR — Train/Date selectors + Model Info (req 7)
    # ==================================================================
    st.sidebar.header("🔍 Select Journey")

    trains_summary = get_available_trains(merged_df)
    train_labels = {
        row.train: f"{row.train} — {row.trainName}  ({row.journey_count} days)"
        for row in trains_summary.itertuples()
    }
    selected_train = st.sidebar.selectbox(
        "Train",
        options=list(train_labels.keys()),
        format_func=lambda t: train_labels.get(t, t),
    )

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
        st.error(f"No usable journey data for train {selected_train} "
                 f"on {selected_date}. Try another date.")
        return

    n_stations = len(journey)

    # ---- Sidebar: Model Info (req 7) ----
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Model Info")
    if predictor.is_trained:
        st.sidebar.success("Random Forest trained ✅")
        st.sidebar.metric("Training samples", f"{predictor.n_train:,}")

        # Train / test split description
        if predictor.train_date_range and predictor.test_date_range:
            st.sidebar.caption(
                f"**Train split:** {predictor.train_date_range}  \n"
                f"**Test split:**  {predictor.test_date_range}"
            )

        # Comparison table of test MAEs
        if predictor.test_mae is not None:
            st.sidebar.markdown("**Test MAE (min)**")
            mae_table = pd.DataFrame({
                "Method": ["Timetable", "Persistence", "Random Forest"],
                "MAE": [
                    f"{predictor.tt_test_mae:.1f}" if predictor.tt_test_mae else "—",
                    f"{predictor.pers_test_mae:.1f}" if predictor.pers_test_mae else "—",
                    f"{predictor.test_mae:.1f}",
                ],
            })
            st.sidebar.dataframe(mae_table, hide_index=True, use_container_width=True)

        # MAE by horizon bar chart
        if predictor.mae_by_horizon:
            st.sidebar.markdown("**RF MAE by horizon (stations ahead)**")
            horizon_df = pd.DataFrame({
                "Horizon": list(predictor.mae_by_horizon.keys()),
                "MAE (min)": list(predictor.mae_by_horizon.values()),
            })
            st.sidebar.bar_chart(horizon_df.set_index("Horizon"), height=180)
    else:
        st.sidebar.warning("Model not trained — insufficient data. "
                           "Baselines still work.")

    # ==================================================================
    #  JOURNEY HEADER
    # ==================================================================
    train_name = journey["trainName"].iloc[0]
    if pd.isna(train_name):
        train_name = f"Train {selected_train}"

    origin_stn  = journey["station"].iloc[0]
    dest_stn    = journey["station"].iloc[-1]
    origin_name = journey.get("station_name", pd.Series()).iloc[0] if "station_name" in journey.columns else origin_stn
    dest_name   = journey.get("station_name", pd.Series()).iloc[-1] if "station_name" in journey.columns else dest_stn
    if pd.isna(origin_name): origin_name = origin_stn
    if pd.isna(dest_name):   dest_name = dest_stn

    st.subheader(f"🛤️ {train_name} ({selected_train}) — "
                 f"{selected_date.strftime('%d %b %Y')}")
    st.write(f"**Route:** {origin_name} → {dest_name}  |  "
             f"**Stops:** {n_stations}  |  "
             f"**Distance:** {journey['distance'].max():.0f} km")

    # ==================================================================
    #  REPLAY CONTROLS (preserved as requested)
    # ==================================================================
    st.markdown("---")

    journey_key = f"step_{selected_train}_{selected_date}"
    if journey_key not in st.session_state:
        st.session_state[journey_key] = 1

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

    st.progress(revealed / n_stations,
                text=f"Stations revealed: {revealed} / {n_stations}")

    # ==================================================================
    #  COMPUTE PREDICTIONS
    # ==================================================================
    n_upcoming = n_stations - revealed

    # BUG FIX (req 11): Ensure last_known_delay is exactly the float
    # value from the last revealed station (not rounded to int).
    last_known_delay = 0.0
    if revealed > 0:
        d = journey.iloc[revealed - 1]["arr_delay"]
        if not pd.isna(d):
            last_known_delay = float(d)

    # Baselines
    tt_preds   = timetable_baseline(n_upcoming)
    pers_preds = persistence_baseline(last_known_delay, n_upcoming)

    # Random Forest + prediction intervals (req 10)
    rf_preds, rf_p10, rf_p90 = predictor.predict_upcoming_with_intervals(
        journey, revealed, max_stations=5
    )

    # ==================================================================
    #  METRIC CARDS (req 6) — Train, Date, Current delay, Next ETA
    # ==================================================================
    current_delay_str = format_delay_short(last_known_delay)
    current_delay_css = delay_css_class(last_known_delay)

    if n_upcoming > 0:
        next_rf_eta = compute_eta_str(journey.iloc[revealed]["sch_arr"], rf_preds[0])
    else:
        next_rf_eta = "Journey complete"

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f'<div class="metric-card"><h4>Train</h4>'
                    f'<p>{train_name}</p></div>', unsafe_allow_html=True)
    with mc2:
        st.markdown(f'<div class="metric-card"><h4>Journey Date</h4>'
                    f'<p>{selected_date.strftime("%d %b %Y")}</p></div>',
                    unsafe_allow_html=True)
    with mc3:
        st.markdown(f'<div class="metric-card"><h4>Current Delay</h4>'
                    f'<p class="{current_delay_css}">{current_delay_str}</p></div>',
                    unsafe_allow_html=True)
    with mc4:
        st.markdown(f'<div class="metric-card"><h4>Next-Stn ETA (RF)</h4>'
                    f'<p>{next_rf_eta}</p></div>', unsafe_allow_html=True)

    st.markdown("")  # spacing

    # ==================================================================
    #  PRECOMPUTE per-step RF predictions for the line chart (req 9)
    #  and for showing per-row errors (req 8)
    # ==================================================================
    # At each step s (1..revealed), we stored what the model WOULD have
    # predicted for station s.  Compute that now.
    rf_pred_at_step  = [np.nan] * n_stations   # RF prediction for station i
    pers_pred_at_step = [np.nan] * n_stations  # Persistence prediction
    tt_pred_at_step   = [np.nan] * n_stations  # Timetable prediction (always 0)

    for step in range(1, revealed):
        # What was the last known delay just before revealing station `step`?
        prev_d = journey.iloc[step - 1]["arr_delay"]
        prev_d = float(prev_d) if not pd.isna(prev_d) else 0.0

        tt_pred_at_step[step]   = 0.0
        pers_pred_at_step[step] = prev_d

        rf_p = predictor.predict_upcoming(journey, step)
        if len(rf_p) > 0:
            rf_pred_at_step[step] = float(rf_p[0])

    # ==================================================================
    #  BUILD HTML TABLE (req 3, 4, 5, 8, 10)
    # ==================================================================
    # Build rows as HTML for colour-coding and compact layout.
    header = (
        "<tr>"
        "<th>#</th>"
        "<th>Station</th>"
        "<th>Sched</th>"
        "<th>Actual</th>"
        "<th>Delay</th>"
        "<th>TT ETA</th>"
        "<th>Pers ETA</th>"
        "<th>RF ETA</th>"
        "<th>Range (p10–p90)</th>"
        "<th>Err: TT/Pers/RF</th>"
        "<th>Status</th>"
        "</tr>"
    )

    body_rows = []
    for i in range(n_stations):
        stn      = journey.iloc[i]
        stn_code = stn["station"]
        stn_name = stn.get("station_name", stn_code)
        if pd.isna(stn_name):
            stn_name = stn_code
        zone     = zones.get(stn_code, "")

        # 24-hour HH:MM format for schedule/actual times
        sch_arr_str = (
            stn["sch_arr"].strftime("%H:%M") if isinstance(stn["sch_arr"], dt_time) else "—"
        )
        serial = stn.get("stnSerialNumber", i + 1)

        # Station cell: name + zone underneath
        zone_html = f'<span class="zone-text">{zone}</span>' if zone else ""
        stn_cell = f"{stn_name} ({stn_code})<br>{zone_html}"

        # Row highlight class
        row_class = ""
        if i == revealed - 1:
            row_class = "row-current"
        elif i == revealed:
            row_class = "row-next"

        if i < revealed:
            # ---- REVEALED ----
            act_arr_str = (
                stn["act_arr"].strftime("%H:%M")
                if isinstance(stn["act_arr"], dt_time) else "—"
            )
            delay_val = stn["arr_delay"]
            delay_str = format_delay_short(delay_val)
            css = delay_css_class(delay_val)

            # Error column
            if i >= 1 and not pd.isna(delay_val):
                e_tt   = format_error(tt_pred_at_step[i], delay_val)
                e_pers = format_error(pers_pred_at_step[i], delay_val)
                e_rf   = format_error(rf_pred_at_step[i], delay_val)
                err_cell = f"{e_tt} / {e_pers} / {e_rf}"
            else:
                err_cell = "—"

            status = "Done"

            body_rows.append(
                f'<tr class="{row_class}">'
                f'<td>{serial}</td>'
                f'<td>{stn_cell}</td>'
                f'<td>{sch_arr_str}</td>'
                f'<td>{act_arr_str}</td>'
                f'<td class="{css}">{delay_str}</td>'
                f'<td>{sch_arr_str}</td>'
                f'<td>{act_arr_str}</td>'
                f'<td>{act_arr_str}</td>'
                f'<td>—</td>'
                f'<td>{err_cell}</td>'
                f'<td>{status}</td>'
                f'</tr>'
            )
        else:
            # ---- UPCOMING ----
            idx = i - revealed

            # Persistence ETA = sch_arr + last_known_delay (exact float,
            # rounded to nearest minute by minutes_to_time_str).
            tt_eta   = compute_eta_str(stn["sch_arr"], tt_preds[idx])
            pers_eta = compute_eta_str(stn["sch_arr"], pers_preds[idx])
            rf_eta   = compute_eta_str(stn["sch_arr"], rf_preds[idx])

            # Prediction range
            range_str = compute_eta_range_str(stn["sch_arr"],
                                               rf_p10[idx], rf_p90[idx])

            status_text = "Next" if idx == 0 else "Upcoming"

            body_rows.append(
                f'<tr class="{row_class}">'
                f'<td>{serial}</td>'
                f'<td>{stn_cell}</td>'
                f'<td>{sch_arr_str}</td>'
                f'<td class="delay-pend">⏳</td>'
                f'<td class="delay-pend">⏳</td>'
                f'<td>{tt_eta}</td>'
                f'<td>{pers_eta}</td>'
                f'<td><b>{rf_eta}</b></td>'
                f'<td style="font-size:0.78rem">{range_str}</td>'
                f'<td class="delay-pend">—</td>'
                f'<td>{status_text}</td>'
                f'</tr>'
            )

    table_html = (
        '<div style="overflow-x:auto;">'
        '<table class="eta-table">'
        f'<thead>{header}</thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    if n_upcoming > 0 and any(not np.isnan(v) for v in rf_p10[:5]):
        st.caption("**Range (p10-p90):** Approx. prediction range from "
                   "individual Random Forest trees (10th–90th percentile).")

    # ==================================================================
    #  LINE CHART — Actual vs Predicted delays (req 9)
    # ==================================================================
    st.markdown("---")
    st.markdown("### 📈 Delay Predictions vs Actuals")

    chart_df = pd.DataFrame({
        "Station #": list(range(1, n_stations + 1)),
    })

    # Actual delay (only for revealed)
    actual_delays = []
    for i in range(n_stations):
        if i < revealed:
            d = journey.iloc[i]["arr_delay"]
            actual_delays.append(float(d) if not pd.isna(d) else np.nan)
        else:
            actual_delays.append(np.nan)
    chart_df["Actual Delay"] = actual_delays

    # RF predicted delay (for revealed: what was predicted before reveal;
    # for upcoming: current prediction)
    rf_chart = []
    for i in range(n_stations):
        if i < revealed:
            rf_chart.append(rf_pred_at_step[i])
        else:
            idx = i - revealed
            rf_chart.append(float(rf_preds[idx]))
    chart_df["RF Predicted"] = rf_chart

    # Persistence predicted
    pers_chart = []
    for i in range(n_stations):
        if i < revealed:
            pers_chart.append(pers_pred_at_step[i])
        else:
            idx = i - revealed
            pers_chart.append(float(pers_preds[idx]))
    chart_df["Persistence"] = pers_chart

    chart_df = chart_df.set_index("Station #")
    st.line_chart(chart_df, height=320, use_container_width=True)

    st.caption("Predicted values update as stations are revealed. "
               "Values before station 2 are not shown (no prior info).")

    # ==================================================================
    #  ACCURACY SUMMARY (shown after full reveal)
    # ==================================================================
    if revealed == n_stations:
        st.markdown("---")
        st.markdown("### 📊 Journey Prediction Accuracy")

        tt_errors   = []
        pers_errors = []
        rf_errors   = []

        for step in range(1, n_stations):
            actual_delay = journey.iloc[step]["arr_delay"]
            if pd.isna(actual_delay):
                continue

            prev_d = journey.iloc[step - 1]["arr_delay"]
            prev_d = float(prev_d) if not pd.isna(prev_d) else 0.0

            tt_errors.append(abs(0 - actual_delay))
            pers_errors.append(abs(prev_d - actual_delay))

            rf_pred = predictor.predict_upcoming(journey, step)
            if len(rf_pred) > 0:
                rf_errors.append(abs(rf_pred[0] - actual_delay))

        c1, c2, c3 = st.columns(3)
        with c1:
            mae_tt = np.mean(tt_errors) if tt_errors else float("nan")
            st.metric("📅 Timetable MAE", f"{mae_tt:.1f} min")
        with c2:
            mae_pers = np.mean(pers_errors) if pers_errors else float("nan")
            st.metric("🔄 Persistence MAE", f"{mae_pers:.1f} min")
        with c3:
            mae_rf = np.mean(rf_errors) if rf_errors else float("nan")
            st.metric("🤖 Random Forest MAE", f"{mae_rf:.1f} min")

        bar_data = pd.DataFrame({
            "Method": ["Timetable", "Persistence", "Random Forest"],
            "MAE (min)": [mae_tt, mae_pers, mae_rf],
        })
        st.bar_chart(bar_data.set_index("Method"), height=260)
        st.caption("**MAE** = Mean Absolute Error — average prediction miss "
                   "in minutes. Lower is better.")

    # ---- Footer ----
    st.markdown("---")
    st.caption(
        "🎓 College Micro-Project  •  Dynamic Train ETA Prediction  •  "
        "Data: Indian Railways, September 2024  •  "
        "This is a **historical demonstration**, not a live service."
    )


if __name__ == "__main__":
    main()
