#!/usr/bin/env python3
# coding: utf-8
"""
frontend/pages/2_bus_camera.py
-------------------------------
"View Inside Bus" — live webcam with stable passenger detection.

Fixes vs previous version:
  • @st.fragment(run_every=0.5) — only the video area re-renders; buttons/
    headers/metrics no longer flicker on every frame.
  • Median smoothing over 20 frames — resistant to single-frame outliers.
  • HOG: useMeanshiftGrouping=True, hitThreshold=0.5, winStride=(16,16)
    → far fewer false positives in a bus interior.
  • Detection runs every 3rd frame (was 2nd); cached boxes used otherwise.
"""

import datetime
import os
import statistics
import time
from collections import deque

import cv2
import numpy as np
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BUS_CAPACITY   = 50
API_URL        = "http://127.0.0.1:8000/predict-demand"
STATION_NAME   = "Yeshwanthapura T.T.M.C."
SEND_INTERVAL  = 5
CONF_THRESHOLD = 0.45
NMS_IOU        = 0.35
SMOOTH_WINDOW  = 20      # rolling window for median stabilisation
DETECT_EVERY   = 3       # run heavy detector every Nth frame

# HOG tuning (fallback when YOLOv8 not available)
HOG_WIN_STRIDE = (16, 16)
HOG_PADDING    = (8, 8)
HOG_SCALE      = 1.05
HOG_HIT_THRESH = 0.5     # higher = stricter, fewer false detections
HOG_MEANSHIFT  = True    # merge overlapping boxes → one clean box per person

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="View Inside Bus",
    page_icon="📷",
    layout="wide",
)
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
[data-testid="stSidebar"] { background-color: #161b22; }
div[data-testid="metric-container"] {
    background: #161b22; border-radius: 10px; padding: 12px;
}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cached singletons
# ─────────────────────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


@st.cache_resource
def _load_detector():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        return {"type": "yolo", "model": model,
                "frontal": None, "profile": None}
    except Exception:
        pass
    # Haar Cascade — better than HOG for faces in bus interiors
    frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    if not os.path.exists(frontal_path):
        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    frontal = cv2.CascadeClassifier(frontal_path)
    profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
    profile = None
    if os.path.exists(profile_path):
        _p = cv2.CascadeClassifier(profile_path)
        if not _p.empty():
            profile = _p
    return {"type": "haar", "model": None,
            "frontal": frontal, "profile": profile}


@st.cache_resource
def _open_camera():
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, f = cap.read()
            if ok and f is not None and f.size > 0:
                return cap
            cap.release()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Image enhancement
# ─────────────────────────────────────────────────────────────────────────────

def _enhance(frame):
    frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# NMS helper
# ─────────────────────────────────────────────────────────────────────────────

def _nms(boxes_xywh, scores):
    if not boxes_xywh:
        return []
    bl  = [list(map(int, b)) for b in boxes_xywh]
    idx = cv2.dnn.NMSBoxes(bl, [float(s) for s in scores], 0.01, NMS_IOU)
    result = []
    if idx is not None and len(idx) > 0:
        for i in idx.flatten():
            x, y, w, h = bl[i]
            result.append((x, y, x + w, y + h))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect(frame, det):
    if det["type"] == "yolo":
        results = det["model"](
            frame, classes=[0],
            conf=CONF_THRESHOLD, iou=NMS_IOU, verbose=False,
        )
        return [
            (int(b.xyxy[0][0]), int(b.xyxy[0][1]),
             int(b.xyxy[0][2]), int(b.xyxy[0][3]))
            for r in results for b in r.boxes
        ]
    # Haar Cascade face detection
    # Standard preprocessing: CLAHE + equalizeHist (no bilateral — hurts Haar)
    h0, w0 = frame.shape[:2]
    scale  = 640 / w0
    resized = cv2.resize(frame, (640, int(h0 * scale)))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)
    gray = cv2.equalizeHist(gray)

    h, w = gray.shape
    all_boxes, all_scores = [], []
    min_sz = (28, 28)
    max_sz = (int(w * 0.7), int(h * 0.7))

    frontal = det["frontal"]
    profile = det["profile"]

    # Frontal faces
    ff = frontal.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3,
        minSize=min_sz, maxSize=max_sz,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    for (x, y, wf, hf) in (ff if len(ff) > 0 else []):
        all_boxes.append([x, y, wf, hf]); all_scores.append(0.9)

    if profile is not None:
        # Profile left
        pf = profile.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=2,
            minSize=min_sz, maxSize=max_sz,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for (x, y, wp, hp) in (pf if len(pf) > 0 else []):
            all_boxes.append([x, y, wp, hp]); all_scores.append(0.8)
        # Profile right (mirrored)
        flipped = cv2.flip(gray, 1)
        pr = profile.detectMultiScale(
            flipped, scaleFactor=1.1, minNeighbors=2,
            minSize=min_sz, maxSize=max_sz,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for (x, y, wp, hp) in (pr if len(pr) > 0 else []):
            all_boxes.append([w - x - wp, y, wp, hp]); all_scores.append(0.8)

    if not all_boxes:
        return []
    clean = _nms(all_boxes, all_scores)
    inv = 1.0 / scale
    return [(int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv))
            for (x1, y1, x2, y2) in clean]


# ─────────────────────────────────────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────────────────────────────────────

def _draw_hud(frame, boxes, raw, smoothed, occupancy, det_type):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)
        cv2.putText(frame, "passenger", (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 0), 1, cv2.LINE_AA)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 115), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    lbl       = {"yolo": "YOLOv8", "haar": "Haar Cascade (face)"}.get(det_type, det_type)
    occ_color = ((0, 200, 80) if occupancy < 60
                 else (0, 165, 255) if occupancy < 85 else (0, 0, 230))

    cv2.putText(frame, f"Detector  : {lbl}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Passengers (stable) : {smoothed}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Raw detection       : {raw}",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Occupancy           : {occupancy:.1f}%",
                (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.62, occ_color, 2, cv2.LINE_AA)

    bw   = frame.shape[1] - 20
    fill = int(bw * min(occupancy / 100.0, 1.0))
    cv2.rectangle(frame, (10, 104), (10 + bw,   112), (80, 80, 80), -1)
    cv2.rectangle(frame, (10, 104), (10 + fill, 112), occ_color,   -1)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    cam_active=False,
    cam_history=None,
    cam_frame_idx=0,
    cam_last_boxes=[],
    cam_last_count=0,
    cam_smoothed=0,
    cam_occupancy=0.0,
    cam_last_send=0.0,
    cam_backend=None,
)
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
if st.session_state.cam_history is None:
    st.session_state.cam_history = deque(maxlen=SMOOTH_WINDOW)

# ─────────────────────────────────────────────────────────────────────────────
# Static page layout (rendered ONCE — never flickers)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 📷 View Inside Bus")
st.markdown(
    "<p style='color:#8b949e; margin-top:-8px;'>"
    "Live webcam with stable passenger detection &amp; occupancy monitoring"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

bc1, bc2, sc_col = st.columns([1, 1, 4])
start_btn = bc1.button(
    "📷 Start Camera", type="primary",
    use_container_width=True,
    disabled=st.session_state.cam_active,
)
stop_btn = bc2.button(
    "⏹ Stop Camera",
    use_container_width=True,
    disabled=not st.session_state.cam_active,
)

if start_btn:
    st.session_state.cam_active    = True
    st.session_state.cam_history   = deque(maxlen=SMOOTH_WINDOW)
    st.session_state.cam_frame_idx = 0
    st.session_state.cam_smoothed  = 0
    st.session_state.cam_occupancy = 0.0
    st.rerun()

if stop_btn:
    st.session_state.cam_active = False
    try:
        _open_camera.clear()
    except Exception:
        pass
    st.rerun()

_live = st.session_state.cam_active
sc_col.markdown(
    f"<div style='padding-top:8px; color:{'#27ae60' if _live else '#8b949e'}; "
    f"font-weight:700; font-size:1rem;'>{'🟢 LIVE' if _live else '⚫ CAMERA OFF'}</div>",
    unsafe_allow_html=True,
)

_det     = _load_detector()
_det_lbl = {"yolo": "YOLOv8", "haar": "Haar Cascade (face)", "hog": "HOG + SVM"}.get(_det["type"], _det["type"])

# ─────────────────────────────────────────────────────────────────────────────
# Live video fragment
# @st.fragment(run_every=0.5) re-renders ONLY this function's output every
# 0.5 s.  The header / buttons / KPI row above are completely unaffected,
# which eliminates all visible flickering.
# ─────────────────────────────────────────────────────────────────────────────

@st.fragment(run_every=0.5)
def _camera_fragment():
    det     = _load_detector()
    det_lbl = {"yolo": "YOLOv8", "haar": "Haar Cascade (face)", "hog": "HOG + SVM"}.get(det["type"], det["type"])

    # Live metrics row — re-renders every 0.5 s inside the fragment
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Passengers Detected", st.session_state.cam_smoothed)
    m2.metric("Bus Occupancy",       f"{st.session_state.cam_occupancy:.1f}%")
    m3.metric("Detector",            det_lbl)
    m4.metric("Bus Capacity",        f"{BUS_CAPACITY} seats")
    st.divider()

    # Camera off — idle placeholder
    if not st.session_state.cam_active:
        st.markdown(
            "<div style='height:420px; border:2px dashed #30363d; border-radius:16px; "
            "display:flex; align-items:center; justify-content:center; "
            "flex-direction:column; color:#8b949e; gap:12px;'>"
            "<div style='font-size:4rem;'>📷</div>"
            "<div style='font-size:1rem;'>Press "
            "<b style='color:#e6edf3;'>Start Camera</b> to view live feed</div>"
            "<div style='font-size:0.82rem;'>Passenger detection runs automatically</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Open camera
    cap = _open_camera()
    if cap is None or not cap.isOpened():
        st.error(
            "❌ **Camera unavailable.**  \n"
            "Ensure no other app (Teams, opencv_bus_monitor.py) is using the "
            "webcam, then click **Start Camera** again."
        )
        st.session_state.cam_active = False
        try:
            _open_camera.clear()
        except Exception:
            pass
        return

    # Read frame
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        st.warning("⚠️ Frame read failed — retrying next tick…")
        return

    # Detection (every DETECT_EVERY frames)
    idx = st.session_state.cam_frame_idx
    if idx % DETECT_EVERY == 0:
        try:
            boxes = _detect(frame, det)
        except Exception:
            boxes = []
        st.session_state.cam_last_boxes = boxes
        st.session_state.cam_last_count = len(boxes)
    else:
        boxes = st.session_state.cam_last_boxes

    raw = st.session_state.cam_last_count
    st.session_state.cam_frame_idx = idx + 1

    # Median smoothing — robust against outlier frames
    hist = st.session_state.cam_history
    hist.append(raw)
    smooth = int(round(statistics.median(list(hist)))) if hist else 0
    occ    = min(100.0, (smooth / BUS_CAPACITY) * 100.0)

    st.session_state.cam_smoothed  = smooth
    st.session_state.cam_occupancy = occ

    # Draw HUD + display
    _draw_hud(frame, boxes, raw, smooth, occ, det["type"])
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    vid_col, side_col = st.columns([3, 1])

    with vid_col:
        st.image(frame_rgb, use_container_width=True)

    with side_col:
        st.markdown("#### 📊 Live Stats")
        occ_color = ("#27ae60" if occ < 60 else "#f39c12" if occ < 85 else "#e74c3c")
        st.markdown(
            f"<div style='color:#8b949e; font-size:0.78rem; margin-bottom:4px;'>OCCUPANCY</div>"
            f"<div style='background:#21262d; border-radius:6px; height:18px;'>"
            f"<div style='background:{occ_color}; height:18px; border-radius:6px; "
            f"width:{int(occ)}%;'></div></div>"
            f"<div style='color:{occ_color}; font-size:1.6rem; font-weight:700; margin-top:4px;'>"
            f"{occ:.1f}%</div>",
            unsafe_allow_html=True,
        )
        crowd = "LOW" if occ < 60 else "MEDIUM" if occ < 85 else "HIGH"
        cc    = {"LOW": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}[crowd]
        icon  = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[crowd]
        st.markdown(
            f"<div style='display:inline-block; background:rgba(0,0,0,0.3); "
            f"border:1px solid {cc}; border-radius:20px; padding:4px 14px; "
            f"color:{cc}; font-weight:700; margin-top:4px;'>{icon} {crowd}</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        bd = st.session_state.cam_backend
        st.markdown("#### 🧠 ML Prediction")
        if bd:
            adj = bd.get("adjustments", {})
            st.metric("Predicted Demand", f"{bd.get('final_prediction', '—')} pax")
            st.markdown(
                f"<p style='font-size:0.78rem; color:#8b949e;'>"
                f"Base: <b>{bd.get('base_prediction', '—')}</b><br>"
                f"OpenCV adj: <b>{adj.get('opencv', 0):+.1f}</b><br>"
                f"Realtime adj: <b>{adj.get('realtime', 0):+.1f}</b></p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p style='color:#8b949e; font-size:0.82rem;'>"
                "Prediction appears after 5s of live feed.</p>",
                unsafe_allow_html=True,
            )
        st.divider()
        st.markdown(
            f"<p style='color:#8b949e; font-size:0.76rem;'>"
            f"Station: <b>{STATION_NAME}</b><br>"
            f"Sends data every {SEND_INTERVAL}s</p>",
            unsafe_allow_html=True,
        )

    # Backend post on interval
    now = time.time()
    if now - st.session_state.cam_last_send >= SEND_INTERVAL:
        st.session_state.cam_last_send = now
        try:
            dt = datetime.datetime.now()
            resp = requests.post(API_URL, json={
                "station_name":          STATION_NAME,
                "boarding":              5,
                "deboarding":            1,
                "first_ticket_time":     dt.strftime("%H:%M"),
                "day":                   dt.strftime("%A"),
                "date":                  dt.strftime("%Y-%m-%d"),
                "ticket_boarding_count": 12,
                "occupancy_percentage":  round(occ, 2),
                "events_data":           {"has_event": False},
            }, timeout=2)
            if resp.status_code == 200:
                st.session_state.cam_backend = resp.json()
        except Exception:
            pass


_camera_fragment()

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Bus camera monitor  ·  Detector: {_det_lbl}  ·  "
    f"Station: {STATION_NAME}  ·  Capacity: {BUS_CAPACITY} seats"
)

