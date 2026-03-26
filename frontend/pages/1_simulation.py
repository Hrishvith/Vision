#!/usr/bin/env python3
# coding: utf-8
"""
frontend/pages/1_simulation.py
-------------------------------
BMTC YPR → Kengeri Bus Route Simulation

Simulates a bus travelling through 43 stops with:
  - ML-based demand prediction per stop & hour
  - Special day multipliers (festival, holiday, etc.)
  - Random realistic boarding/deboarding at each stop
  - Live bus position tracker
  - Trip summary (total boarded, deboarded, peak boarding)
  - Live ticket log (AM Peak / PM Peak / Off-peak coded)
  - Speed control (1x / 2x / 4x)
  - Bus frequency display
"""

import random
import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Route definition — all 43 stops YPR → Kengeri
# ─────────────────────────────────────────────────────────────────────────────
STOPS = [
    "Yeshwanthapura T.T.M.C.",
    "Govardhan Talkies",
    "R.M.C. Yard (Yashawanthapura New Railway Station)",
    "M.E.I. Factory",
    "Goraguntepalya",
    "Reliance Petrol Bunk Goraguntepalya",
    "Modern Food Goraguntepalya",
    "Kanteerava Studio",
    "Kanteerava Studio (Stop 2)",
    "Nandhini Layout",
    "Nandini Layout Ring Road",
    "Laggere Bridge",
    "Kempegowda Arch",
    "Lurdubayi Samudaya Bhavana",
    "Depot-31 Gate Summanahalli",
    "Sumanahalli",
    "Kottigepalya",
    "Vokkaliga School Kottigepalya",
    "B.D.A. 2nd Block Nagarabhavi",
    "B.D.A. Complex Nagarabhavi",
    "Aladamara Papareddypalya",
    "Papareddypalya",
    "Mallathahalli I.T.I. Layout",
    "Deepa Complex",
    "Dr. Ambedkar Institute Of Technology",
    "Kengunte Circle",
    "Kengunte Circle (Stop 2)",
    "Mallathahalli Cross",
    "Bengaluru University Quarters",
    "P.V.P. School",
    "Mariyappanapalya",
    "Kenchanapura Cross",
    "Nagadevanahalli",
    "Shirke K.H.B. Quarters",
    "Kengeri Church",
    "Kengeri Satellite Town",
    "Kengeri Railway Station",
    "Kengeri Post Office",
    "Kommaghatta Junction",
    "Kengeri Ganesha Temple",
    "Kengeri Police Station",
    "Kengeri Bus Station",
    "Kengeri T.T.M.C",
]

# Short labels for chart x-axis
LABELS = [
    "YPR T.T.M.C.", "Govardhan T.", "R.M.C. Yard", "M.E.I. Factory",
    "Goraguntepalya", "Reliance PB", "Modern Food", "Kanteerava St.",
    "Kanteerava St.2", "Nandhini Lay.", "Nandini RR", "Laggere Br.",
    "Kempegowda Arch", "Lurdubayi SB", "Depot-31 Gate", "Sumanahalli",
    "Kottigepalya", "Vokkaliga Sch.", "BDA 2nd Blk", "BDA Complex",
    "Aladamara", "Papareddypalya", "Mallath ITI", "Deepa Complex",
    "Dr. AIT", "Kengunte Cir.", "Kengunte Cir.2", "Mallath Cross",
    "BU Quarters", "P.V.P. School", "Mariyappa.", "Kenchanapura",
    "Nagadevanahalli", "Shirke KHB", "Kengeri Church", "KGI Sat.Town",
    "KGI Rly Stn", "KGI Post Off.", "Kommaghatta", "KGI Ganesha",
    "KGI Police", "KGI Bus Stn", "KGI T.T.M.C",
]

# Travel time (minutes) between consecutive stops
TRAVEL_MINS = [2, 2, 2, 3, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1,
               2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1,
               2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1]   # 42 gaps for 43 stops

BUS_CAPACITY   = 130  # total seats (BMTC standard full-size bus)
BACKEND_URL    = "http://127.0.0.1:8000/predict-demand"
SPEED_OPTIONS  = {"1×": 1.5, "2×": 0.75, "4×": 0.3}   # seconds per stop (wall clock)

# Special days with demand multipliers
SPECIAL_DAYS = {
    "Normal Weekday":   {"mult": 1.0,  "label": "", "color": "#8b949e"},
    "Normal Weekend":   {"mult": 1.2,  "label": "", "color": "#8b949e"},
    "Dasara":           {"mult": 1.75, "label": "SPECIAL DAY: DASARA",    "color": "#f39c12"},
    "Diwali":           {"mult": 1.60, "label": "SPECIAL DAY: DIWALI",    "color": "#f39c12"},
    "Independence Day": {"mult": 1.40, "label": "SPECIAL DAY: IND. DAY",  "color": "#27ae60"},
    "Republic Day":     {"mult": 1.35, "label": "SPECIAL DAY: REP. DAY",  "color": "#27ae60"},
    "Cricket Match":    {"mult": 1.50, "label": "EVENT: CRICKET MATCH",   "color": "#3498db"},
    "Rainy Day":        {"mult": 0.75, "label": "LOW DEMAND: HEAVY RAIN", "color": "#8b949e"},
}

PEAK_HOURS = {
    "AM Peak":  (6, 10),   # 6 AM – 10 AM
    "PM Peak":  (16, 20),  # 4 PM – 8 PM
    "Off-peak": None,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _peak_label(hour: int) -> str:
    if 6 <= hour < 10:
        return "AM Peak"
    if 16 <= hour < 20:
        return "PM Peak"
    return "Off-peak"


PEAK_COLOR = {"AM Peak": "#e07b39", "PM Peak": "#9b59b6", "Off-peak": "#3498db"}


def _demand_at_stop(stop: str, hour: int, day: str, date_str: str, multiplier: float) -> dict:
    """
    Fetch demand prediction from backend, or generate a realistic fallback.
    Returns dict: {boarding, deboarding}.
    """
    # Time-of-day base shape: higher in morning and evening peaks
    base_shape = {4: 0.3, 5: 0.5, 6: 0.9, 7: 1.4, 8: 1.6, 9: 1.3,
                  10: 0.8, 11: 0.7, 12: 0.8, 13: 0.8, 14: 0.7, 15: 0.8,
                  16: 1.1, 17: 1.4, 18: 1.5, 19: 1.2, 20: 0.9, 21: 0.6,
                  22: 0.4, 23: 0.2}
    shape = base_shape.get(hour, 0.5)

    # Stop position weight: early stops (near YPR) board more, later stops deboard more
    n = len(STOPS)
    idx = STOPS.index(stop) if stop in STOPS else 0
    board_weight = 1.5 - (idx / n)   # decreases as we move toward Kengeri
    deboard_weight = 0.2 + (idx / n)

    base_board  = int(random.gauss(max(3, 40 * shape * board_weight), 8) * multiplier)
    base_dboard = int(random.gauss(max(1, 15 * shape * deboard_weight), 5) * multiplier)

    # Try backend for a more accurate boarding count
    try:
        resp = requests.post(BACKEND_URL, json={
            "station_name":          stop,
            "boarding":              max(1, base_board // 3),
            "deboarding":            max(0, base_dboard // 3),
            "first_ticket_time":     f"{hour:02d}:00",
            "day":                   day,
            "date":                  date_str,
            "ticket_boarding_count": max(1, base_board // 4),
        }, timeout=2)
        if resp.status_code == 200:
            pred = resp.json().get("final_prediction", base_board)
            base_board = max(0, int(pred * multiplier))
    except Exception:
        pass

    return {
        "boarding":   max(0, base_board),
        "deboarding": max(0, base_dboard),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMTC Route Simulation",
    page_icon="🚌",
    layout="wide",
)
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
[data-testid="stSidebar"] { background-color: #161b22; }
div[data-testid="metric-container"] { background:#161b22; border-radius:10px; padding:12px; }
.sim-header {
    background: linear-gradient(135deg, #1a1f2e, #0e1117);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def _reset_sim():
    st.session_state.sim_running   = False
    st.session_state.sim_step      = 0
    st.session_state.sim_onboard   = 0
    st.session_state.sim_total_b   = 0
    st.session_state.sim_total_d   = 0
    st.session_state.sim_peak_b    = 0
    st.session_state.sim_peak_stop = ""
    st.session_state.sim_log       = []      # list of dicts per stop
    st.session_state.sim_done      = False

for k, v in {
    "sim_running": False, "sim_step": 0, "sim_onboard": 0,
    "sim_total_b": 0, "sim_total_d": 0, "sim_peak_b": 0,
    "sim_peak_stop": "", "sim_log": [], "sim_done": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚌 Simulation Controls")
    st.divider()

    # Date
    sim_date = st.date_input("Date", value=datetime(2024, 10, 13))
    date_str = sim_date.strftime("%Y-%m-%d")
    day_name = sim_date.strftime("%A")

    # Hour slider (4 AM – 10 PM)
    sim_hour = st.slider("Hour (4AM–10PM)", min_value=4, max_value=22, value=8,
                          format="%d:00")
    peak = _peak_label(sim_hour)
    st.markdown(
        f"<span style='color:{PEAK_COLOR[peak]}; font-weight:700;'>⏰ {peak}</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Special day
    special_day = st.selectbox("Day Type", list(SPECIAL_DAYS.keys()), index=0)
    sd_info = SPECIAL_DAYS[special_day]
    multiplier = sd_info["mult"]
    if multiplier != 1.0:
        st.markdown(
            f"<div style='background:#2b1f0a; border:1px solid #f39c12; "
            f"border-radius:6px; padding:6px 10px; font-size:0.82rem; color:#f39c12;'>"
            f"🎉 {sd_info['label']}<br>"
            f"Demand multiplier: <b>{multiplier}×</b></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Speed
    speed_label = st.radio("Speed", list(SPEED_OPTIONS.keys()), index=0, horizontal=True)
    step_delay  = SPEED_OPTIONS[speed_label]

    st.divider()

    # Buttons
    c1, c2 = st.columns(2)
    start_btn = c1.button("▶ START", type="primary",  use_container_width=True)
    reset_btn = c2.button("↺ RESET", use_container_width=True)

    if reset_btn:
        _reset_sim()
        st.rerun()

    if start_btn and not st.session_state.sim_running and not st.session_state.sim_done:
        st.session_state.sim_running = True
        st.session_state.sim_step    = 0
        st.session_state.sim_log     = []

    st.divider()
    st.markdown(
        "<p style='color:#8b949e; font-size:0.78rem;'>"
        f"Route: <b>YPR → Kengeri</b><br>"
        f"Stops: <b>{len(STOPS)}</b><br>"
        f"Distance: <b>14.1 km</b><br>"
        f"Capacity: <b>{BUS_CAPACITY} seats</b>"
        "</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Header bar
# ─────────────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([1, 4])
with h1:
    st.markdown("## 🚌")
with h2:
    st.markdown(
        "<div class='sim-header'>"
        "<span style='font-size:1.1rem; font-weight:800; letter-spacing:1px;'>"
        "BMTC ROUTE SIMULATION</span><br>"
        "<span style='color:#8b949e; font-size:0.82rem;'>"
        "YESHWANTHAPURA T.T.M.C. → KENGERI T.T.M.C. · 43 STOPS · 14.1 KM</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Special day banner
# ─────────────────────────────────────────────────────────────────────────────
if sd_info["label"]:
    st.markdown(
        f"<div style='background:#2b1f0a; border-left:4px solid {sd_info['color']}; "
        f"padding:8px 16px; border-radius:4px; margin-bottom:8px; font-size:0.9rem;'>"
        f"🎉 <b style='color:{sd_info['color']};'>{sd_info['label']}</b>"
        f" &nbsp;·&nbsp; Demand multiplier: <b>{multiplier}×</b>"
        f" &nbsp;·&nbsp; {day_name}, {sim_date.strftime('%d-%m-%Y')}"
        f"</div>",
        unsafe_allow_html=True,
    )

# Peak banner
st.markdown(
    f"<div style='background:#1a2233; border-left:4px solid {PEAK_COLOR[peak]}; "
    f"padding:6px 16px; border-radius:4px; margin-bottom:12px; font-size:0.85rem;'>"
    f"<b style='color:{PEAK_COLOR[peak]};'>⏰ {peak.upper()}</b>"
    f"{'&nbsp;·&nbsp; Heavy boarding at Yeshwanthapura side' if peak == 'AM Peak' else ''}"
    f"{'&nbsp;·&nbsp; Heavy boarding at Kengeri side' if peak == 'PM Peak' else ''}"
    f"</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Live KPI strip (always shown, updates each step)
# ─────────────────────────────────────────────────────────────────────────────
step      = st.session_state.sim_step
onboard   = st.session_state.sim_onboard
total_b   = st.session_state.sim_total_b
total_d   = st.session_state.sim_total_d
peak_b    = st.session_state.sim_peak_b
sim_log   = st.session_state.sim_log

kc1, kc2, kc3, kc4, kc5 = st.columns(5)
kc1.metric("STOP",        f"{step} / {len(STOPS)}")
kc2.metric("ONBOARD",     onboard)
kc3.metric("CAPACITY",    f"{min(100, round(onboard / BUS_CAPACITY * 100))}%")
travel_so_far = sum(TRAVEL_MINS[:step]) if step > 0 else 0
kc4.metric("TRAVEL TIME", f"{travel_so_far} min" if step > 0 else "—")
trip_status = ("WAITING" if not st.session_state.sim_running and step == 0
               else "COMPLETE" if st.session_state.sim_done
               else "RUNNING" if st.session_state.sim_running
               else "PAUSED")
kc5.metric("TRIP STATUS", trip_status)

# ─────────────────────────────────────────────────────────────────────────────
# Current stop panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
curr_stop_name = STOPS[step] if step < len(STOPS) else STOPS[-1]
last_log = sim_log[-1] if sim_log else None

c_left, c_right = st.columns([2, 1])
with c_left:
    # Bus icon with occupancy fill
    occ_pct  = min(100, round(onboard / BUS_CAPACITY * 100))
    bus_color = "#27ae60" if occ_pct < 60 else "#f39c12" if occ_pct < 85 else "#e74c3c"
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:16px; padding:12px 0;'>"
        f"<div style='font-size:2.8rem;'>🚌</div>"
        f"<div>"
        f"<div style='color:#8b949e; font-size:0.75rem; letter-spacing:2px;'>AT STOP</div>"
        f"<div style='font-size:1.5rem; font-weight:700;'>{curr_stop_name}</div>"
        f"<div style='margin-top:4px; font-size:0.9rem;'>"
        f"<span style='color:#27ae60;'>↑ {last_log['boarding'] if last_log else 0} boarding</span>"
        f"&nbsp;&nbsp;"
        f"<span style='color:#e74c3c;'>↓ {last_log['deboarding'] if last_log else 0} deboarding</span>"
        f"&nbsp;&nbsp;"
        f"<span style='color:#8b949e;'>{travel_so_far} min from origin</span>"
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with c_right:
    tick_time = f"{sim_hour:02d}:{(step * 2) % 60:02d} {'AM' if sim_hour < 12 else 'PM'}"
    st.markdown(
        f"<div style='text-align:right; padding-top:8px;'>"
        f"<div style='color:#8b949e; font-size:0.75rem; letter-spacing:2px;'>FIRST TICKET TIME</div>"
        f"<div style='font-size:1.6rem; font-weight:700; color:#f39c12;'>{tick_time}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Route progress bar
# ─────────────────────────────────────────────────────────────────────────────
progress_pct = step / (len(STOPS) - 1) if len(STOPS) > 1 else 0
pbar_w = int(progress_pct * 100)
st.markdown(
    f"<div style='display:flex; justify-content:space-between; color:#8b949e; "
    f"font-size:0.78rem; margin-bottom:4px;'>"
    f"<span>Yeshwanthapura T.T.M.C.</span><span>Kengeri T.T.M.C</span></div>"
    f"<div style='background:#21262d; border-radius:4px; height:8px; width:100%;'>"
    f"<div style='background:#00d4ff; height:8px; border-radius:4px; "
    f"width:{pbar_w}%; transition:width 0.4s;'></div></div>"
    f"<div style='color:#8b949e; font-size:0.75rem; text-align:center; margin-top:2px;'>"
    f"{pbar_w}% of route complete · {travel_so_far} min elapsed</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Main content: stop log + right panel (trip summary + chart)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col_log, col_summary = st.columns([3, 2])

with col_log:
    st.markdown("#### 🗒 Stop Log")
    if sim_log:
        log_df = pd.DataFrame(sim_log)
        log_df["Stop #"] = log_df.index + 1
        display_df = log_df[["Stop #", "stop", "time", "boarding", "deboarding", "onboard", "capacity_pct"]].copy()
        display_df.columns = ["#", "Stop", "Time", "↑ Board", "↓ Deboard", "Onboard", "Cap %"]
        st.dataframe(display_df, hide_index=True, use_container_width=True, height=min(400, 35 + len(display_df) * 35))
    else:
        st.markdown(
            "<div style='color:#8b949e; padding:40px; text-align:center; border:1px dashed #30363d; border-radius:8px;'>"
            "Press ▶ START to begin simulation</div>",
            unsafe_allow_html=True,
        )

with col_summary:
    st.markdown("#### 📊 Trip Summary")
    ts1, ts2 = st.columns(2)
    ts1.metric("Total Boarded",   total_b)
    ts2.metric("Total Deboarded", total_d)
    ts3, ts4 = st.columns(2)
    ts3.metric("Peak Boarding",   peak_b if peak_b else "—")
    ts4.metric("Bus Freq",        "6/hr")

    st.divider()

    # Live ticket log (last 8 stops)
    st.markdown("#### 🎟 Live Ticket Log")
    if sim_log:
        for entry in reversed(sim_log[-8:]):
            pk     = entry.get("peak", "Off-peak")
            pk_col = PEAK_COLOR.get(pk, "#3498db")
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; "
                f"padding:3px 0; border-bottom:1px solid #21262d; font-size:0.82rem;'>"
                f"<span>Stop {entry['idx']+1}: {entry['stop'][:25]}…</span>"
                f"<span style='color:#27ae60;'>+{entry['boarding']}</span>"
                f"<span style='color:#e74c3c;'>-{entry['deboarding']}</span>"
                f"<span style='color:{pk_col};'>●</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        # Legend
        am_col  = PEAK_COLOR["AM Peak"]
        pm_col  = PEAK_COLOR["PM Peak"]
        off_col = PEAK_COLOR["Off-peak"]
        st.markdown(
            "<div style='margin-top:8px; font-size:0.75rem; color:#8b949e;'>"
            f"<span style='color:{am_col};'>● AM Peak</span>&nbsp;"
            f"<span style='color:{pm_col};'>● PM Peak</span>&nbsp;"
            f"<span style='color:{off_col};'>● Off-peak</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<span style='color:#8b949e;'>Waiting for simulation to start…</span>", unsafe_allow_html=True)

    # Day / Date / Hour info box
    st.divider()
    info_c1, info_c2, info_c3 = st.columns(3)
    info_c1.metric("DAY",     day_name)
    info_c2.metric("DATE",    sim_date.strftime("%d-%m-%Y"))
    info_c3.metric("HOUR",    f"{sim_hour:02d}:00 {'AM' if sim_hour < 12 else 'PM'}")
    info_ca, info_cb = st.columns(2)
    info_ca.metric("MULT",    f"{multiplier}×")
    info_cb.metric("SPECIAL", sd_info["label"] if sd_info["label"] else "Normal")

# ─────────────────────────────────────────────────────────────────────────────
# Passengers onboard chart (bar chart, updates as simulation runs)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"#### 📈 Passengers Onboard  (0/{BUS_CAPACITY})")

chart_placeholder = st.empty()

def _render_chart(log):
    if not log:
        fig = go.Figure()
        fig.update_layout(
            height=200, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="Simulation not started", showarrow=False,
                              font=dict(color="#8b949e", size=14),
                              xref="paper", yref="paper", x=0.5, y=0.5)],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    xs = [e["stop"][:15] for e in log]
    ys = [e["onboard"] for e in log]
    boards = [e["boarding"] for e in log]
    colors = ["#27ae60" if y < BUS_CAPACITY * 0.6
              else "#f39c12" if y < BUS_CAPACITY * 0.85
              else "#e74c3c"
              for y in ys]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xs, y=ys,
        marker_color=colors,
        text=[str(v) for v in ys],
        textposition="outside",
        textfont=dict(color="#e6edf3", size=9),
        hovertemplate="<b>%{x}</b><br>Onboard: %{y}<extra></extra>",
    ))
    fig.add_hline(y=BUS_CAPACITY, line_dash="dash", line_color="#e74c3c",
                  line_width=1.5,
                  annotation_text=f"Capacity ({BUS_CAPACITY})",
                  annotation_position="top right",
                  annotation_font_color="#e74c3c")
    # Annotate current stop
    if len(log) > 0:
        fig.add_annotation(
            x=xs[-1], y=ys[-1],
            text="🚌 HERE", showarrow=True, arrowhead=2,
            arrowcolor="#00d4ff", font=dict(color="#00d4ff", size=10), ay=-35,
        )
    fig.update_layout(
        height=260,
        margin=dict(t=30, b=10, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
        xaxis=dict(gridcolor="#21262d", tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(gridcolor="#21262d", rangemode="tozero",
                   range=[0, max(BUS_CAPACITY * 1.15, max(ys) * 1.1) if ys else BUS_CAPACITY * 1.2]),
        showlegend=False,
    )
    return fig

chart_placeholder.plotly_chart(_render_chart(sim_log), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation loop — runs one stop per rerun when sim_running == True
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.sim_running and step < len(STOPS):
    current_stop = STOPS[step]
    demand = _demand_at_stop(current_stop, sim_hour, day_name, date_str, multiplier)

    boarding   = demand["boarding"]
    deboarding = min(demand["deboarding"], st.session_state.sim_onboard)

    # Don't exceed bus capacity for boarding
    space_left = BUS_CAPACITY - st.session_state.sim_onboard
    boarding   = min(boarding, max(0, space_left + deboarding))

    new_onboard = st.session_state.sim_onboard - deboarding + boarding
    new_onboard = max(0, min(BUS_CAPACITY, new_onboard))

    tick_min = (step * 2) % 60
    tick_am_pm = "AM" if sim_hour < 12 else "PM"
    tick_time_entry = f"{sim_hour:02d}:{tick_min:02d} {tick_am_pm}"

    log_entry = {
        "idx":          step,
        "stop":         current_stop,
        "time":         tick_time_entry,
        "boarding":     boarding,
        "deboarding":   deboarding,
        "onboard":      new_onboard,
        "capacity_pct": f"{min(100, round(new_onboard / BUS_CAPACITY * 100))}%",
        "peak":         _peak_label(sim_hour),
    }
    st.session_state.sim_log.append(log_entry)
    st.session_state.sim_onboard = new_onboard
    st.session_state.sim_total_b += boarding
    st.session_state.sim_total_d += deboarding
    if boarding > st.session_state.sim_peak_b:
        st.session_state.sim_peak_b    = boarding
        st.session_state.sim_peak_stop = current_stop

    st.session_state.sim_step += 1

    if st.session_state.sim_step >= len(STOPS):
        st.session_state.sim_running = False
        st.session_state.sim_done    = True
    else:
        time.sleep(step_delay)

    st.rerun()

# Completion banner
if st.session_state.sim_done:
    st.success(
        f"✅ **Trip Complete!**  "
        f"Total Boarded: **{st.session_state.sim_total_b}**  |  "
        f"Total Deboarded: **{st.session_state.sim_total_d}**  |  "
        f"Peak boarding: **{st.session_state.sim_peak_b} pax** at "
        f"**{st.session_state.sim_peak_stop}**"
    )
