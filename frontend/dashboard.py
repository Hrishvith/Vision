#!/usr/bin/env python3
# coding: utf-8
"""
frontend/dashboard.py
---------------------
BMTC Smart Bus Operations – Real-Time Monitoring Dashboard

Polls GET /routes-status every 10 seconds and renders:
  - Network-wide KPI summary
  - All-routes overview table (crowd-coded)
  - Per-route detail: stop flow chart + optimization recommendation

Run:
    streamlit run frontend/dashboard.py
"""

import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ROUTES_STATUS_URL = "http://127.0.0.1:8000/routes-status"
BUS_CAPACITY      = 40
REFRESH_INTERVAL  = 10  # seconds

CROWD_COLOR = {"LOW": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
CROWD_BG    = {"LOW": "#0d2b1a", "MEDIUM": "#2b1f0a", "HIGH": "#2b0a0a"}
CROWD_EMOJI = {"LOW": "🟢",      "MEDIUM": "🟡",      "HIGH": "🔴"}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_status():
    """Call GET /routes-status.  Returns (data_dict, error_str)."""
    try:
        resp = requests.get(ROUTES_STATUS_URL, timeout=6)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach backend at {ROUTES_STATUS_URL}. Is uvicorn running?"
    except Exception as exc:
        return None, f"API error: {exc}"


def _crowd_badge_html(level: str) -> str:
    col = CROWD_COLOR.get(level, "#888")
    bg  = CROWD_BG.get(level, "#111")
    em  = CROWD_EMOJI.get(level, "")
    return (
        f'<span style="background:{bg}; color:{col}; padding:3px 14px; '
        f'border-radius:20px; font-weight:700; font-size:0.85rem; '
        f'border:1px solid {col};">{em} {level}</span>'
    )


def _bar_color(pax: int) -> str:
    r = pax / BUS_CAPACITY
    if r < 0.60:
        return CROWD_COLOR["LOW"]
    if r < 0.85:
        return CROWD_COLOR["MEDIUM"]
    return CROWD_COLOR["HIGH"]


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMTC Ops Dashboard",
    page_icon="🚍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: tighter table cells, status pill colours
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
[data-testid="stSidebar"] { background-color: #161b22; }
div[data-testid="metric-container"] { background:#161b22; border-radius:10px; padding:12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
if "selected_route" not in st.session_state:
    st.session_state.selected_route = None
if "last_fetch_ts" not in st.session_state:
    st.session_state.last_fetch_ts = "—"

# ─────────────────────────────────────────────────────────────────────────────
# Fetch live data
# ─────────────────────────────────────────────────────────────────────────────
data, error = _fetch_status()
fetch_time  = datetime.now().strftime("%H:%M:%S")

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
col_title, col_sim_btn, col_cam_btn, col_status = st.columns([3, 1, 1, 1])
with col_title:
    st.markdown(
        "<h1 style='margin:0; font-size:1.8rem;'>🚍 BMTC Smart Bus Operations</h1>"
        "<p style='margin:2px 0 0; color:#8b949e; font-size:0.9rem;'>"
        "Real-time demand monitoring & fleet optimisation</p>",
        unsafe_allow_html=True,
    )
with col_sim_btn:
    st.markdown("<div style='padding-top:10px;'>", unsafe_allow_html=True)
    st.page_link("pages/1_simulation.py", label="🚌 View Simulation", icon=None)
    st.markdown("</div>", unsafe_allow_html=True)
with col_cam_btn:
    st.markdown("<div style='padding-top:10px;'>", unsafe_allow_html=True)
    st.page_link("pages/2_bus_camera.py", label="📷 View Inside Bus", icon=None)
    st.markdown("</div>", unsafe_allow_html=True)
with col_status:
    status_color = "#27ae60" if data else "#e74c3c"
    status_text  = "LIVE" if data else "OFFLINE"
    st.markdown(
        f"<div style='text-align:right; padding-top:8px;'>"
        f"<span style='color:{status_color}; font-size:0.8rem; font-weight:700;'>"
        f"● {status_text}</span><br>"
        f"<span style='color:#8b949e; font-size:0.75rem;'>Updated {fetch_time}</span><br>"
        f"<span style='color:#8b949e; font-size:0.72rem;'>Auto-refresh every {REFRESH_INTERVAL}s</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Error state
# ─────────────────────────────────────────────────────────────────────────────
if error:
    st.error(f"❌ {error}")
    st.markdown(
        """
        <div style="text-align:center; padding:60px 40px; border:2px dashed #333;
                    border-radius:16px; margin-top:20px;">
            <div style="font-size:3rem">⚠️</div>
            <h2>Backend Offline</h2>
            <p style="color:#8b949e;">
                Start the FastAPI backend:<br>
                <code>uvicorn backend.api.app:app --host 0.0.0.0 --port 8000</code>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – route selector
# ─────────────────────────────────────────────────────────────────────────────
routes = data.get("routes", [])
route_names = [r["route"] for r in routes]
alerts = [r for r in routes if r["crowd_level"] == "HIGH"]

with st.sidebar:
    st.markdown("## 🗂 Route Monitor")

    # Alert badge
    if alerts:
        st.markdown(
            f'<div style="background:#2b0a0a; border:1px solid #e74c3c; '
            f'border-radius:8px; padding:8px 12px; margin-bottom:8px;">'
            f'<span style="color:#e74c3c; font-weight:700;">⚠ {len(alerts)} route(s) need attention</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#0d2b1a; border:1px solid #27ae60; '
            'border-radius:8px; padding:8px 12px; margin-bottom:8px;">'
            '<span style="color:#27ae60; font-weight:700;">✓ All routes operating normally</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Select Route for Detail")
    options = ["— Overview (all routes) —"] + route_names
    # Preserve selection across auto-reruns
    current_sel = st.session_state.selected_route
    default_idx = (options.index(current_sel) if current_sel in options else 0)
    selected = st.selectbox("Route", options, index=default_idx, label_visibility="collapsed")
    st.session_state.selected_route = selected if selected != options[0] else None

    st.divider()
    st.markdown(
        "<p style='color:#8b949e; font-size:0.78rem;'>"
        f"Bus capacity: <b>{BUS_CAPACITY} seats</b><br>"
        f"Timestamp: <b>{data.get('timestamp', '—')}</b>"
        "</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Network KPI cards
# ─────────────────────────────────────────────────────────────────────────────
total_demand   = sum(r["predicted_demand_next_hour"] for r in routes)
total_buses_c  = sum(r["current_buses"]  for r in routes)
total_buses_r  = sum(r["required_buses"] for r in routes)
bus_gap        = max(0, total_buses_r - total_buses_c)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Routes Monitored",        len(routes))
k2.metric("High Demand Alerts",      len(alerts),       delta=f"{len(alerts)} route(s)" if alerts else "None", delta_color="inverse" if alerts else "off")
k3.metric("Predicted Demand / hr",   f"{total_demand} pax")
k4.metric("Buses Deployed",          total_buses_c)
k5.metric("Buses Short",             bus_gap,            delta=f"+{bus_gap} needed" if bus_gap else "Fleet OK", delta_color="inverse" if bus_gap else "off")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# All-routes overview table
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Network Overview")

overview_rows = []
for r in routes:
    lvl      = r["crowd_level"]
    em       = CROWD_EMOJI.get(lvl, "")
    buses_ok = r["required_buses"] <= r["current_buses"]
    overview_rows.append({
        "Route":            r["route"],
        "Current Stop":     r["current_stop"],
        "Occupancy %":      r["occupancy_percent"],
        "Demand (Next Hr)": r["predicted_demand_next_hour"],
        "Buses Now":        r["current_buses"],
        "Buses Needed":     r["required_buses"],
        "Crowd Level":      f"{em} {lvl}",
        "Action":           "Fleet OK" if buses_ok else "Deploy buses",
    })

st.dataframe(pd.DataFrame(overview_rows), hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Route detail
# ─────────────────────────────────────────────────────────────────────────────
sel = st.session_state.selected_route

if sel:
    route_obj = next((r for r in routes if r["route"] == sel), None)
    if route_obj is None:
        st.warning(f"Route '{sel}' not found in latest data.")
    else:
        st.divider()

        # ── Header ────────────────────────────────────────────────────────────
        lvl   = route_obj["crowd_level"]
        col   = CROWD_COLOR[lvl]
        bg    = CROWD_BG[lvl]

        hcol1, hcol2 = st.columns([2, 1])
        with hcol1:
            st.subheader(f"Route: {route_obj['route']}")
            st.markdown(
                f"Current stop: **{route_obj['current_stop']}**"
                f"  |  Occupancy: **{route_obj['occupancy_percent']}%**"
            )
        with hcol2:
            st.markdown(
                f"<div style='text-align:right; padding-top:8px;'>"
                f"<div style='display:inline-block; background:{bg}; border:2px solid {col}; "
                f"border-radius:50px; padding:8px 24px;'>"
                f"<span style='color:{col}; font-size:1.2rem; font-weight:800; letter-spacing:2px;'>"
                f"{CROWD_EMOJI[lvl]} {lvl} DEMAND</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        # ── Detail KPIs ───────────────────────────────────────────────────────
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Next Hour Demand",  f"{route_obj['predicted_demand_next_hour']} pax")
        d2.metric("Buses Deployed",    route_obj["current_buses"])
        d3.metric("Buses Required",    route_obj["required_buses"],
                  delta=f"+{route_obj['required_buses'] - route_obj['current_buses']} short"
                        if route_obj["required_buses"] > route_obj["current_buses"]
                        else "Adequate",
                  delta_color="inverse" if route_obj["required_buses"] > route_obj["current_buses"]
                              else "off")
        d4.metric("Stops on Route",    len(route_obj["stops"]))

        # ── Route flow chart ─────────────────────────────────────────────────
        st.subheader("📊 Stop-by-Stop Demand Forecast")

        stop_data  = route_obj["stops"]
        labels     = [s["label"] for s in stop_data]
        pax_vals   = [s["predicted_pax"] for s in stop_data]
        bar_colors = [_bar_color(p) for p in pax_vals]
        curr_idx   = route_obj["current_stop_idx"]

        # Highlight current stop with a distinct colour (white border)
        bar_colors[curr_idx] = "#00d4ff"

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels,
            y=pax_vals,
            marker_color=bar_colors,
            marker_line_color=["#ffffff" if i == curr_idx else "rgba(0,0,0,0)"
                               for i in range(len(labels))],
            marker_line_width=[2 if i == curr_idx else 0 for i in range(len(labels))],
            text=[f"{v}" for v in pax_vals],
            textposition="outside",
            textfont=dict(color="#e6edf3"),
            hovertemplate="<b>%{x}</b><br>Predicted: %{y} pax<extra></extra>",
        ))

        # Capacity threshold lines
        fig.add_hline(
            y=BUS_CAPACITY,
            line_dash="dash", line_color="#e74c3c", line_width=1.5,
            annotation_text=f"1 bus = {BUS_CAPACITY} seats",
            annotation_position="top right",
            annotation_font_color="#e74c3c",
        )
        fig.add_hline(
            y=BUS_CAPACITY * 2,
            line_dash="dot", line_color="#f39c12", line_width=1,
            annotation_text=f"2 buses = {BUS_CAPACITY * 2} seats",
            annotation_position="top right",
            annotation_font_color="#f39c12",
        )

        # Annotate current stop with bus icon
        fig.add_annotation(
            x=labels[curr_idx],
            y=pax_vals[curr_idx],
            text="🚍 HERE",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#00d4ff",
            font=dict(color="#00d4ff", size=11, family="monospace"),
            ay=-40,
        )

        fig.update_layout(
            xaxis_title="Stop",
            yaxis_title="Predicted Passengers",
            height=380,
            margin=dict(t=40, b=10, l=10, r=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", rangemode="tozero"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Stop-by-stop table ────────────────────────────────────────────────
        st.subheader("🗒 Stop-by-Stop Summary")
        table_rows = []
        for i, s in enumerate(stop_data):
            buses_n = -(-s["predicted_pax"] // BUS_CAPACITY)  # ceiling division
            table_rows.append({
                "Stop":            s["label"],
                "Station":         s["name"],
                "Predicted Pax":   s["predicted_pax"],
                "Buses Needed":    buses_n,
                "Demand Level":    f"{CROWD_EMOJI.get(s['crowd_level'], '')} {s['crowd_level']}",
                "Bus Position":    "🚍 HERE" if i == curr_idx else "",
            })
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

        # ── Optimization recommendation ───────────────────────────────────────
        st.subheader("💡 Optimization Recommendation")

        gap        = route_obj["required_buses"] - route_obj["current_buses"]
        peak_stop  = max(stop_data, key=lambda s: s["predicted_pax"])
        high_stops = [s for s in stop_data if s["crowd_level"] == "HIGH"]
        med_stops  = [s for s in stop_data if s["crowd_level"] == "MEDIUM"]

        if gap > 0:
            rec_title = f"Action Required: Deploy {gap} Additional Bus{'es' if gap > 1 else ''}"
            rec_body  = (
                f"**Peak demand** of **{peak_stop['predicted_pax']} pax** at "
                f"**{peak_stop['name']}** exceeds current fleet capacity.  \n"
                f"{len(high_stops)} stop(s) at HIGH demand, {len(med_stops)} at MEDIUM.\n\n"
                f"**Recommended actions:**\n"
                f"- Dispatch {gap} standby bus{'es' if gap > 1 else ''} to **{route_obj['route']}**\n"
                f"- Priority boarding at **{peak_stop['name']}**\n"
                f"- Alert depot control for real-time reallocation"
            )
            st.error(f"⚠️ **{rec_title}**")
            st.markdown(rec_body)
        elif med_stops:
            rec_title = "Monitor Closely: Medium Demand Detected"
            rec_body  = (
                f"{len(med_stops)} stop(s) approaching medium demand.  \n"
                f"Peak at **{peak_stop['name']}**: {peak_stop['predicted_pax']} pax.\n\n"
                f"**Recommended actions:**\n"
                f"- Keep one standby bus on alert for **{route_obj['route']}**\n"
                f"- Monitor occupancy at {', '.join(s['name'] for s in med_stops[:3])}"
            )
            st.warning(f"🔶 **{rec_title}**")
            st.markdown(rec_body)
        else:
            rec_title = "Fleet Sufficient: No Action Required"
            rec_body  = (
                f"All stops on **{route_obj['route']}** show LOW demand.  \n"
                f"Current {route_obj['current_buses']} bus(es) can handle predicted load.\n\n"
                f"**Recommended actions:**\n"
                f"- Consider reallocating spare buses to high-demand routes\n"
                f"- Continue routine monitoring"
            )
            st.success(f"✅ **{rec_title}**")
            st.markdown(rec_body)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"BMTC Smart Bus Operations Dashboard  |  "
    f"Data timestamp: {data.get('timestamp', '—')}  |  "
    f"Page fetched: {fetch_time}  |  "
    f"Auto-refreshes every {REFRESH_INTERVAL}s"
)

# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────────────────────────────────────────
time.sleep(REFRESH_INTERVAL)
st.rerun()
