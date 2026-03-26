"""
opencv_bus_monitor.py
---------------------
Real-time passenger detection from a bus camera.

Detection method:
  Detects FULL BODY / WHOLE PERSON (not just face/head).
  Works for standing, seated, and partially visible passengers.

Detection pipeline (automatic best-available):
  Priority 1 → YOLOv8 (ultralytics)           — most accurate, full-body person
               pip install ultralytics
  Priority 2 → Haar Cascade face detector       — built into OpenCV, no extra files
               cv2.CascadeClassifier (haarcascade_frontalface_default.xml)
  Priority 3 → HOG + SVM pedestrian detector   — built into OpenCV, no install needed
               cv2.HOGDescriptor + SVM people detector

Accuracy features:
  • Face / full-body person detection
  • Non-Maximum Suppression — one box per person
  • CLAHE contrast enhancement — works in dim bus lighting
  • Bilateral filtering — reduces noise while preserving edges
  • Rolling-average smoothing — stable count, no frame-to-frame jitter
  • Skip-frame inference — smooth video at full camera FPS
  • Multi-scale detection for HOG — catches near and far passengers

Usage:
    python opencv_bus_monitor.py

Press 'q' in the video window to quit.
"""

import datetime
import os
import sys
import time
import threading

from collections import deque

import cv2
import numpy as np
import requests

# ─────────────────────────────────────────────────────────────────────────────
# User configuration — edit these to match your deployment
# ─────────────────────────────────────────────────────────────────────────────
API_URL          = "http://127.0.0.1:8000/predict-demand"
SEND_INTERVAL    = 5        # seconds between API calls
BUS_CAPACITY     = 50       # assumed max passengers on the bus
WEBCAM_INDEX     = 0        # 0 = default webcam / laptop camera
WINDOW_NAME      = "Bus Passenger Monitor  |  press Q to quit"

STATION_NAME     = "Yeshwanthapura T.T.M.C."
TICKET_BOARDING  = 12
BOARDING         = 5
DEBOARDING       = 1
HAS_EVENT        = False

# Detection tuning
CONF_THRESHOLD   = 0.40     # YOLOv8 / DNN minimum confidence
NMS_IOU          = 0.45     # overlap threshold for NMS
DETECT_EVERY_N   = 2        # run detector every Nth frame; reuse boxes on others
SMOOTH_WINDOW    = 10       # rolling-average window length for count stabilisation

# HOG tuning (Priority 3 fallback)
HOG_WIN_STRIDE   = (8, 8)
HOG_PADDING      = (8, 8)
HOG_SCALE        = 1.03     # lower = more thorough but slower
HOG_HIT_THRESH   = 0.3      # lower = more sensitive (more detections)

# ─────────────────────────────────────────────────────────────────────────────
# CLAHE (Contrast Limited Adaptive Histogram Equalisation)
# ─────────────────────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _enhance(frame):
    """
    Apply CLAHE to the luminance channel + bilateral filter for better
    low-light, noisy bus-camera detection.
    """
    frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# NMS helper
# ─────────────────────────────────────────────────────────────────────────────

def _nms(boxes_xywh, scores, iou_thresh=None):
    """
    Apply Non-Maximum Suppression.
    boxes_xywh : list of [x, y, w, h]
    scores     : list of float confidence values (same length)
    Returns list of (x1, y1, x2, y2).
    """
    if iou_thresh is None:
        iou_thresh = NMS_IOU
    if len(boxes_xywh) == 0:
        return []

    boxes_list  = [list(map(int, b)) for b in boxes_xywh]
    scores_list = [float(s) for s in scores]

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_list,
        scores=scores_list,
        score_threshold=0.01,
        nms_threshold=iou_thresh,
    )
    result = []
    if indices is not None and len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes_list[i]
            result.append((x, y, x + w, y + h))
    return result


def _nms_xyxy(boxes_xyxy, scores, iou_thresh=None):
    """NMS on (x1,y1,x2,y2) boxes. Converts to xywh internally."""
    if iou_thresh is None:
        iou_thresh = NMS_IOU
    if len(boxes_xyxy) == 0:
        return []
    boxes_xywh = []
    for (x1, y1, x2, y2) in boxes_xyxy:
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
    return _nms(boxes_xywh, scores, iou_thresh)


# ─────────────────────────────────────────────────────────────────────────────
# Detector setup — YOLOv8 → Haar Cascade face → HOG pedestrian
# ─────────────────────────────────────────────────────────────────────────────
_DETECTOR       = None
_yolo_model     = None
_face_cascade   = None
_profile_casc   = None   # side-profile face
_hog            = None

# ── Priority 1: YOLOv8 ─────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    _yolo_model = _YOLO("yolov8n.pt")
    _DETECTOR   = "yolo"
    print("[INFO] ✅ Detector: YOLOv8 — full-body person detection (ultralytics)")
except (ImportError, Exception) as e:
    print(f"[INFO] YOLOv8 not available: {e}")

# ── Priority 2: Haar Cascade face detector (built into OpenCV) ─────────────
if _DETECTOR is None:
    try:
        # Use alt2 - better accuracy than default
        _frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        if not os.path.exists(_frontal_path):
            _frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(_frontal_path)
        if _face_cascade.empty():
            raise RuntimeError("Frontal cascade is empty")

        # Profile (side) face — catches passengers looking sideways
        _profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        if os.path.exists(_profile_path):
            _profile_casc = cv2.CascadeClassifier(_profile_path)
            if _profile_casc.empty():
                _profile_casc = None

        _DETECTOR = "haar"
        print("[INFO] ✅ Detector: Haar Cascade face detector (built-in OpenCV)")
        print(f"       frontal : {_frontal_path}")
        print(f"       profile : {'loaded' if _profile_casc else 'not found'}")
    except Exception as e:
        print(f"[INFO] Haar Cascade face detector unavailable: {e}")

# ── Priority 3: HOG + SVM pedestrian detector (built into OpenCV) ──────────
if _DETECTOR is None:
    try:
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _DETECTOR = "hog"
        print("[INFO] Detector: HOG + SVM pedestrian detector (built-in OpenCV)")
    except Exception as e:
        print(f"[FATAL] No detector available: {e}")
        sys.exit(
            "ERROR: No person detector could be initialised.\n"
            "  Install ultralytics for best results:  pip install ultralytics\n"
            "  Or ensure OpenCV is installed:  pip install opencv-python"
        )

print("[INFO] For best accuracy:  pip install ultralytics")


# ─────────────────────────────────────────────────────────────────────────────
# Detection functions — ALL detect FULL BODY, not just face
# ─────────────────────────────────────────────────────────────────────────────

def _detect_yolo(frame):
    """YOLOv8 full-body person detection."""
    results = _yolo_model(
        frame,
        classes=[0],            # COCO class 0 = person
        conf=CONF_THRESHOLD,
        iou=NMS_IOU,
        verbose=False,
    )
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes, len(boxes)


def _detect_haar(frame):
    """
    Haar Cascade multi-angle face detection.
    - Skips bilateral filter (hurts Haar — it removes the texture features)
    - Standard preprocessing: gray + CLAHE + equalizeHist
    - Frontal + profile (left + right mirror) detection
    - NMS to remove duplicate boxes
    - minNeighbors=3 for better sensitivity in bus interiors
    """
    h0, w0 = frame.shape[:2]
    scale  = 640 / w0
    resized = cv2.resize(frame, (640, int(h0 * scale)))

    # Standard Haar preprocessing: CLAHE then equalize (no bilateral filter)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)
    gray = cv2.equalizeHist(gray)

    h, w       = gray.shape
    all_boxes  = []
    all_scores = []
    min_sz     = (28, 28)
    max_sz     = (int(w * 0.7), int(h * 0.7))

    # ── Frontal faces ──────────────────────────
    frontal = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3,
        minSize=min_sz, maxSize=max_sz,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    for (x, y, wf, hf) in (frontal if len(frontal) > 0 else []):
        all_boxes.append([x, y, wf, hf])
        all_scores.append(0.9)

    # ── Profile faces (left) ───────────────────
    if _profile_casc is not None:
        profile = _profile_casc.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=2,
            minSize=min_sz, maxSize=max_sz,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for (x, y, wp, hp) in (profile if len(profile) > 0 else []):
            all_boxes.append([x, y, wp, hp])
            all_scores.append(0.8)

        # ── Profile faces (right — mirrored) ───
        flipped = cv2.flip(gray, 1)
        profile_r = _profile_casc.detectMultiScale(
            flipped, scaleFactor=1.1, minNeighbors=2,
            minSize=min_sz, maxSize=max_sz,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for (x, y, wp, hp) in (profile_r if len(profile_r) > 0 else []):
            x_mirror = w - x - wp
            all_boxes.append([x_mirror, y, wp, hp])
            all_scores.append(0.8)

    if not all_boxes:
        return [], 0

    # NMS to remove overlapping duplicates
    clean = _nms(all_boxes, all_scores, iou_thresh=0.40)

    # Scale boxes back to original frame size
    inv = 1.0 / scale
    boxes = [(int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv))
             for (x1, y1, x2, y2) in clean]
    return boxes, len(boxes)


def _detect_hog(frame):
    """
    HOG + SVM full-body pedestrian detector (built into OpenCV).
    Detects whole person silhouettes — works for standing and seated.
    Runs at multiple scales for better coverage in bus cameras.
    """
    enhanced = _enhance(frame)

    # Resize for speed while keeping aspect ratio
    target_w = 640
    h0, w0   = enhanced.shape[:2]
    if w0 == 0:
        return [], 0
    scale     = target_w / w0
    resized   = cv2.resize(enhanced, (target_w, int(h0 * scale)))
    inv       = 1.0 / scale

    # Run HOG people detector
    (rects, weights) = _hog.detectMultiScale(
        resized,
        winStride=HOG_WIN_STRIDE,
        padding=HOG_PADDING,
        scale=HOG_SCALE,
        hitThreshold=HOG_HIT_THRESH,
        useMeanshiftGrouping=False,
    )

    if len(rects) == 0:
        return [], 0

    boxes_xywh = []
    scores     = []
    for i, (x, y, w, h) in enumerate(rects):
        # Scale coordinates back to original frame size
        boxes_xywh.append([int(x * inv), int(y * inv),
                           int(w * inv), int(h * inv)])
        score = float(weights[i]) if i < len(weights) else 0.5
        scores.append(score)

    clean = _nms(boxes_xywh, scores, iou_thresh=0.50)
    return clean, len(clean)


def detect_passengers(frame):
    """Dispatch to the best available detector."""
    if _DETECTOR == "yolo":
        return _detect_yolo(frame)
    elif _DETECTOR == "haar":
        return _detect_haar(frame)
    elif _DETECTOR == "hog":
        return _detect_hog(frame)
    return [], 0


# ─────────────────────────────────────────────────────────────────────────────
# Drawing / HUD
# ─────────────────────────────────────────────────────────────────────────────

def _draw(frame, boxes, raw_count, smoothed_count, occupancy):
    """Annotate frame with detection boxes and on-screen HUD."""
    # Draw a box + label for each detected person (full body)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)
        label_y = max(y1 - 8, 16)
        label = "face" if _DETECTOR == "haar" else "passenger"
        cv2.putText(frame, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 0), 2,
                    cv2.LINE_AA)

    # Semi-transparent HUD bar at the top
    bar_h   = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # Line 1 — detector type
    det_label = {
        "yolo": "YOLOv8 (full body)",
        "haar": "Haar Cascade (face)",
        "hog":  "HOG Pedestrian (full body)",
    }.get(_DETECTOR, _DETECTOR.upper())

    cv2.putText(frame,
                f"Detector : {det_label}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (100, 200, 255), 1, cv2.LINE_AA)
    # Line 2 — stable (smoothed) count
    cv2.putText(frame,
                f"Passengers (stable) : {smoothed_count}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (255, 255, 255), 2, cv2.LINE_AA)
    # Line 3 — raw frame count
    cv2.putText(frame,
                f"Raw detection       : {raw_count}",
                (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (160, 160, 160), 1, cv2.LINE_AA)
    # Line 4 — occupancy
    occ_pct  = f"{occupancy:.1f}%"
    color    = ((0, 200, 80) if occupancy < 70
                else (0, 165, 255) if occupancy < 90
                else (0, 0, 255))
    cv2.putText(frame,
                f"Occupancy           : {occ_pct}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                color, 2, cv2.LINE_AA)

    # Occupancy bar visual
    bar_x, bar_y, bar_w, bar_bh = 10, 108, frame.shape[1] - 20, 8
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_bh), (80, 80, 80), -1)
    fill_w = int(bar_w * min(occupancy / 100.0, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill_w, bar_y + bar_bh), color, -1)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Backend API call  (runs in daemon thread — never blocks the camera loop)
# ─────────────────────────────────────────────────────��───────────────────────

def _send_to_backend(passenger_count: int, occupancy: float):
    now      = datetime.datetime.now()
    payload  = {
        "station_name":          STATION_NAME,
        "boarding":              BOARDING,
        "deboarding":            DEBOARDING,
        "first_ticket_time":     now.strftime("%H:%M"),
        "day":                   now.strftime("%A"),
        "date":                  now.strftime("%Y-%m-%d"),
        "ticket_boarding_count": TICKET_BOARDING,
        "occupancy_percentage":  round(occupancy, 2),
        "events_data":           {"has_event": HAS_EVENT},
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        adj  = data.get("adjustments", {})

        print("\n" + "─" * 50)
        print(f"  Passengers detected  : {passenger_count}")
        print(f"  Occupancy            : {occupancy:.1f}%")
        print()
        print("  Backend Prediction")
        print(f"    Stop               : {data.get('stop_name')}")
        print(f"    Base Prediction    : {data.get('base_prediction')}")
        print(f"    Final Prediction   : {data.get('final_prediction')} passengers")
        print(f"    Adjustments        : "
              f"event={adj.get('event', 0):+.2f}  "
              f"realtime={adj.get('realtime', 0):+.2f}  "
              f"opencv={adj.get('opencv', 0):+.2f}")
        print("─" * 50)

    except requests.exceptions.ConnectionError:
        print(f"[WARNING] Backend unreachable at {API_URL}")
    except requests.exceptions.Timeout:
        print("[WARNING] Backend request timed out")
    except Exception as exc:
        print(f"[WARNING] Backend error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def _open_camera():
    """
    Try to open the camera using multiple backends and indices.
    Returns an opened VideoCapture or raises SystemExit.
    """
    candidates = []
    is_windows = sys.platform.startswith("win")
    is_linux   = sys.platform.startswith("linux")

    if is_windows:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif is_linux:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        candidates.append((WEBCAM_INDEX, backend))
    for idx in (1, 2):
        if idx != WEBCAM_INDEX:
            for backend in backends:
                candidates.append((idx, backend))

    for idx, backend in candidates:
        backend_names = {
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF:  "MSMF",
            cv2.CAP_ANY:   "Auto",
        }
        if hasattr(cv2, "CAP_V4L2"):
            backend_names[cv2.CAP_V4L2] = "V4L2"
        backend_name = backend_names.get(backend, str(backend))

        print(f"[INFO] Trying camera index={idx}  backend={backend_name} ...")
        try:
            cap = cv2.VideoCapture(idx, backend)
        except Exception:
            cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, test_frame = cap.read()
        if ok and test_frame is not None and test_frame.size > 0:
            print(f"[INFO] Camera opened: index={idx}  backend={backend_name}  "
                  f"resolution={test_frame.shape[1]}x{test_frame.shape[0]}")
            return cap
        cap.release()

    sys.exit(
        "ERROR: No camera could be opened.\n"
        "  • Make sure a webcam is connected and not in use by another app.\n"
        "  • Try changing WEBCAM_INDEX at the top of this file.\n"
        "  • On Windows, grant camera permission in Privacy settings.\n"
        "  • On Linux, check: ls /dev/video*"
    )


def main():
    cap = _open_camera()
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    det_label = {
        "yolo": "YOLOv8 (full body)",
        "haar": "Haar Cascade (face)",
        "hog":  "HOG Pedestrian (full body)",
    }.get(_DETECTOR, _DETECTOR.upper())

    print(f"\n{'='*54}")
    print(f"  Bus Passenger Monitor — FULL BODY detection")
    print(f"{'='*54}")
    print(f"  Detector     : {det_label}")
    print(f"  Camera index : {WEBCAM_INDEX}")
    print(f"  API          : {API_URL}")
    print(f"  Send interval: {SEND_INTERVAL}s")
    print(f"  Smoothing    : rolling avg over {SMOOTH_WINDOW} frames")
    print(f"  Confidence   : {CONF_THRESHOLD}")
    print(f"  NMS IoU      : {NMS_IOU}")
    print(f"{'='*54}\n")
    print("  [Detection: FULL BODY — works for standing & seated passengers]")
    print("  Press  Q  in the video window to quit.\n")

    last_send        = time.time() - SEND_INTERVAL
    frame_idx        = 0
    cached_boxes     = []
    cached_count     = 0
    history          = deque(maxlen=SMOOTH_WINDOW)
    consecutive_fail = 0
    MAX_FAILS        = 30

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_fail += 1
            if consecutive_fail >= MAX_FAILS:
                print(f"\n[ERROR] Camera stopped after {MAX_FAILS} failures. Exiting.")
                break
            time.sleep(0.05)
            continue

        consecutive_fail = 0
        frame_idx += 1

        # ── Run detection every DETECT_EVERY_N frames ─────────────────────
        if frame_idx % DETECT_EVERY_N == 0:
            try:
                cached_boxes, cached_count = detect_passengers(frame)
            except Exception as e:
                print(f"[WARNING] Detection error: {e}")
                cached_boxes, cached_count = [], 0

        # ── Temporal smoothing ─────────────────────────────────────────────
        history.append(cached_count)
        smoothed  = round(sum(history) / len(history))
        occupancy = min((smoothed / BUS_CAPACITY) * 100.0, 100.0)

        # ── Draw and show ──────────────────────────────────────────────────
        display = _draw(frame.copy(), cached_boxes,
                        cached_count, smoothed, occupancy)
        cv2.imshow(WINDOW_NAME, display)

        # ── Send to backend every SEND_INTERVAL seconds ───────────────────
        now = time.time()
        if now - last_send >= SEND_INTERVAL:
            last_send = now
            threading.Thread(
                target=_send_to_backend,
                args=(smoothed, occupancy),
                daemon=True,
            ).start()

        # ── Quit on Q ─────────────────────────────────────────────────────
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit — closing camera.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()