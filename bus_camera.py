"""
bus_camera.py
=============
Production-ready real-time AI passenger detection for smart public transport.

Features
--------
* YOLOv8n (primary)  — most accurate, full-body person detection
* DNN SSD (fallback) — OpenCV built-in, uses deploy.prototxt + .caffemodel
* HOG+SVM (last resort) — zero-install fallback, always available
* CLAHE + bilateral-filter preprocessing for varying bus lighting
* Non-Maximum Suppression — one clean box per person
* Temporal smoothing — rolling average eliminates frame-to-frame jitter
* REST API integration — POSTs sensor data to FastAPI backend every N seconds
* Graceful shutdown — SIGINT / SIGTERM / Q key all exit cleanly
* Structured logging — timestamped console output

Usage
-----
    python bus_camera.py
    python bus_camera.py --station "Majestic" --camera 1 --capacity 60
    python bus_camera.py --api http://192.168.1.10:8000/predict-demand

Press Q in the video window to quit.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bus_camera")


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectorConfig:
    """
    All tunable parameters in one place.
    Override at the CLI or instantiate directly in your own code.
    """
    # ── Camera ─────────────────────────────────────────────────────────────
    webcam_index:    int   = 0

    # ── Backend API ────────────────────────────────────────────────────────
    api_url:         str   = "http://127.0.0.1:8000/predict-demand"
    send_interval:   float = 5.0    # seconds between API calls
    api_timeout:     float = 5.0    # request timeout

    # ── Bus metadata (sent to backend with every API call) ─────────────────
    bus_capacity:    int   = 50
    station_name:    str   = "Yeshwanthapura T.T.M.C."
    ticket_boarding: int   = 12
    boarding:        int   = 5
    deboarding:      int   = 1
    has_event:       bool  = False

    # ── Detection thresholds ───────────────────────────────────────────────
    conf_threshold:  float = 0.35   # minimum detection confidence
    nms_iou:         float = 0.45   # NMS overlap threshold
    detect_every_n:  int   = 2      # run detector on every Nth frame
    smooth_window:   int   = 10     # rolling-average window (frames)

    # ── HOG tuning (Priority-3 fallback) ───────────────────────────────────
    hog_win_stride:  Tuple[int, int] = (8, 8)
    hog_padding:     Tuple[int, int] = (8, 8)
    hog_scale:       float = 1.03
    hog_hit_thresh:  float = 0.30

    # ── DNN model file names (must sit in the same directory) ──────────────
    dnn_proto: str = "deploy.prototxt"
    dnn_model: str = "res10_300x300_ssd_iter_140000.caffemodel"


# ══════════════════════════════════════════════════════════════════════════════
# Image preprocessing
# ══════════════════════════════════════════════════════════════════════════════

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE contrast enhancement and bilateral noise reduction.
    Significantly improves face/body detection under uneven bus lighting.
    """
    frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# NMS utilities
# ══════════════════════════════════════════════════════════════════════════════

def nms_xywh(
    boxes: List[List[int]],
    scores: List[float],
    iou_thresh: float = 0.45,
) -> List[Tuple[int, int, int, int]]:
    """NMS on [x, y, w, h] boxes → filtered list of (x1, y1, x2, y2)."""
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(
        bboxes=[list(map(int, b)) for b in boxes],
        scores=[float(s) for s in scores],
        score_threshold=0.01,
        nms_threshold=iou_thresh,
    )
    result: List[Tuple[int, int, int, int]] = []
    if indices is not None and len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = map(int, boxes[i])
            result.append((x, y, x + w, y + h))
    return result


def nms_xyxy(
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    iou_thresh: float = 0.45,
) -> List[Tuple[int, int, int, int]]:
    """NMS on (x1, y1, x2, y2) boxes."""
    return nms_xywh(
        [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in boxes],
        scores,
        iou_thresh,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Camera open helper
# ══════════════════════════════════════════════════════════════════════════════

def open_camera(index: int = 0) -> cv2.VideoCapture:
    """
    Open a webcam with automatic backend selection.

    Tries DirectShow → MSMF → Auto on Windows (MSMF often fails to grab
    frames even when isOpened() returns True).  Falls back to extra indices
    1 and 2 in case the requested index is wrong.

    Returns
    -------
    cv2.VideoCapture — confirmed open and returning frames.
    Raises SystemExit on complete failure.
    """
    is_win = sys.platform.startswith("win")
    is_lin = sys.platform.startswith("linux")

    if is_win:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif is_lin:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    candidates: List[Tuple[int, int]] = [(index, b) for b in backends]
    for alt in (1, 2):
        if alt != index:
            candidates += [(alt, backends[0]), (alt, cv2.CAP_ANY)]

    _names = {cv2.CAP_DSHOW: "DirectShow", cv2.CAP_MSMF: "MSMF", cv2.CAP_ANY: "Auto"}
    if hasattr(cv2, "CAP_V4L2"):
        _names[cv2.CAP_V4L2] = "V4L2"

    for idx, backend in candidates:
        bname = _names.get(backend, str(backend))
        log.info("Trying camera index=%d  backend=%s", idx, bname)
        try:
            cap = cv2.VideoCapture(idx, backend)
        except Exception:
            cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            log.info(
                "Camera opened  index=%d  backend=%s  %dx%d",
                idx, bname, frame.shape[1], frame.shape[0],
            )
            return cap
        cap.release()

    sys.exit(
        "ERROR: No camera could be opened.\n"
        "  • Connect a webcam and make sure no other app is using it.\n"
        "  • Change --camera index if using an external camera.\n"
        "  • Windows: grant camera permission in Privacy Settings."
    )


# ══════════════════════════════════════════════════════════════════════════════
# BusPassengerDetector — main class
# ══════════════════════════════════════════════════════════════════════════════

class BusPassengerDetector:
    """
    Real-time passenger counter for bus camera feeds.

    Detection priority (automatic selection at startup):
      1. YOLOv8n   — full-body person, most accurate  (pip install ultralytics)
      2. DNN SSD   — face+head, good accuracy, no extra pip needed
      3. HOG+SVM   — full-body silhouette, always available (built-in OpenCV)

    Example
    -------
    >>> cfg = DetectorConfig(station_name="Majestic", bus_capacity=60)
    >>> detector = BusPassengerDetector(cfg)
    >>> detector.run()
    """

    WINDOW_TITLE = "Bus Passenger Monitor  |  press Q to quit"

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.cfg = config or DetectorConfig()
        self._stop = threading.Event()
        self._history: deque = deque(maxlen=self.cfg.smooth_window)
        self._last_send: float = 0.0

        # Set by _load_detector
        self._detector_type: str = ""
        self._yolo = None
        self._dnn: Optional[cv2.dnn.Net] = None
        self._hog: Optional[cv2.HOGDescriptor] = None

        self._load_detector()

    # ── Detector loading ──────────────────────────────────────────────────────

    def _load_detector(self) -> None:
        """Try each detector in priority order; exit if all fail."""
        if self._try_yolo():
            return
        if self._try_dnn():
            return
        if self._try_hog():
            return
        sys.exit(
            "FATAL: No detector could be initialised.\n"
            "  pip install ultralytics     ← best option\n"
            "  OR ensure OpenCV is installed:  pip install opencv-python"
        )

    def _try_yolo(self) -> bool:
        try:
            from ultralytics import YOLO  # type: ignore
            log.info("Loading YOLOv8n model (may download ~6 MB on first run) …")
            self._yolo = YOLO("yolov8n.pt")
            self._detector_type = "yolo"
            log.info("Detector: YOLOv8n — full-body person detection")
            return True
        except Exception as exc:
            log.info("YOLOv8 not available (%s) — trying DNN SSD", exc)
            return False

    def _try_dnn(self) -> bool:
        try:
            script_dir = Path(__file__).parent
            proto = self._find_file(self.cfg.dnn_proto, [script_dir, Path(".")])
            model = self._find_file(self.cfg.dnn_model, [script_dir, Path(".")])
            if proto and model:
                self._dnn = cv2.dnn.readNetFromCaffe(str(proto), str(model))
                self._detector_type = "dnn"
                log.info("Detector: OpenCV DNN SSD face detector")
                log.info("  proto : %s", proto)
                log.info("  model : %s", model)
                return True
            log.info(
                "DNN model files not found (%s / %s) — trying HOG",
                self.cfg.dnn_proto, self.cfg.dnn_model,
            )
            return False
        except Exception as exc:
            log.info("DNN setup failed (%s) — trying HOG", exc)
            return False

    def _try_hog(self) -> bool:
        try:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._detector_type = "hog"
            log.info("Detector: HOG + SVM pedestrian (built-in OpenCV)")
            log.info("  Tip: pip install ultralytics  for much higher accuracy")
            return True
        except Exception as exc:
            log.error("HOG setup failed: %s", exc)
            return False

    @staticmethod
    def _find_file(name: str, dirs: List[Path]) -> Optional[Path]:
        for d in dirs:
            p = d / name
            if p.is_file():
                return p
        return None

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], int]:
        """
        Run passenger detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV VideoCapture.

        Returns
        -------
        boxes : list of (x1, y1, x2, y2) bounding boxes
        count : number of persons/faces detected
        """
        if self._detector_type == "yolo":
            return self._detect_yolo(frame)
        if self._detector_type == "dnn":
            return self._detect_dnn(frame)
        if self._detector_type == "hog":
            return self._detect_hog(frame)
        return [], 0

    def _detect_yolo(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], int]:
        results = self._yolo(
            frame,
            classes=[0],                    # COCO class 0 = person
            conf=self.cfg.conf_threshold,
            iou=self.cfg.nms_iou,
            verbose=False,
        )
        boxes: List[Tuple[int, int, int, int]] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
        return boxes, len(boxes)

    def _detect_dnn(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], int]:
        enhanced = enhance_frame(frame)
        h, w = enhanced.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(enhanced, (300, 300)),
            1.0, (300, 300), (104.0, 177.0, 123.0),
            swapRB=False, crop=False,
        )
        self._dnn.setInput(blob)
        det = self._dnn.forward()

        raw_boxes: List[Tuple[int, int, int, int]] = []
        raw_scores: List[float] = []
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < self.cfg.conf_threshold:
                continue
            x1 = max(0, int(det[0, 0, i, 3] * w))
            y1 = max(0, int(det[0, 0, i, 4] * h))
            x2 = min(w, int(det[0, 0, i, 5] * w))
            y2 = min(h, int(det[0, 0, i, 6] * h))
            if x2 > x1 and y2 > y1:
                raw_boxes.append((x1, y1, x2, y2))
                raw_scores.append(conf)

        clean = nms_xyxy(raw_boxes, raw_scores, self.cfg.nms_iou)
        return clean, len(clean)

    def _detect_hog(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], int]:
        enhanced = enhance_frame(frame)
        h0, w0 = enhanced.shape[:2]
        if w0 == 0:
            return [], 0
        scale = 640.0 / w0
        small = cv2.resize(enhanced, (640, int(h0 * scale)))

        rects, weights = self._hog.detectMultiScale(
            small,
            winStride=self.cfg.hog_win_stride,
            padding=self.cfg.hog_padding,
            scale=self.cfg.hog_scale,
            hitThreshold=self.cfg.hog_hit_thresh,
            useMeanshiftGrouping=False,
        )
        if len(rects) == 0:
            return [], 0

        inv = 1.0 / scale
        xywh = [
            [int(x * inv), int(y * inv), int(w * inv), int(h_ * inv)]
            for (x, y, w, h_) in rects
        ]
        scores = [
            float(weights[i]) if i < len(weights) else 0.5
            for i in range(len(rects))
        ]
        clean = nms_xywh(xywh, scores, 0.50)
        return clean, len(clean)

    # ── Frame annotation ──────────────────────────────────────────────────────

    def _draw_hud(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        raw_count: int,
        smoothed: int,
        occupancy: float,
    ) -> np.ndarray:
        """Annotate frame with bounding boxes and a semi-transparent HUD."""
        det_label = {
            "yolo": "YOLOv8n — full body",
            "dnn":  "DNN SSD — face / head",
            "hog":  "HOG+SVM — pedestrian",
        }.get(self._detector_type, self._detector_type.upper())

        # Bounding boxes
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)
            cv2.putText(
                frame, "person", (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 0), 2, cv2.LINE_AA,
            )

        # HUD background
        bar_h = 130
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

        color = (
            (0, 200, 80) if occupancy < 70 else
            (0, 165, 255) if occupancy < 90 else
            (0, 0, 255)
        )

        cv2.putText(frame, f"Detector  : {det_label}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Passengers (stable) : {smoothed}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Raw detection       : {raw_count}",
                    (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Occupancy           : {occupancy:.1f}%",
                    (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)

        # Occupancy progress bar
        bx, by, bw, bh2 = 10, 114, frame.shape[1] - 20, 10
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh2), (70, 70, 70), -1)
        fill = int(bw * min(occupancy / 100.0, 1.0))
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh2), color, -1)

        return frame

    # ── Backend API integration ───────────────────────────────────────────────

    def _send_to_backend(self, count: int, occupancy: float) -> None:
        """
        POST current sensor reading to the FastAPI backend.
        Runs in a daemon thread so it never blocks the camera loop.
        """
        now = datetime.datetime.now()
        payload = {
            "station_name":          self.cfg.station_name,
            "boarding":              self.cfg.boarding,
            "deboarding":            self.cfg.deboarding,
            "first_ticket_time":     now.strftime("%H:%M"),
            "day":                   now.strftime("%A"),
            "date":                  now.strftime("%Y-%m-%d"),
            "ticket_boarding_count": self.cfg.ticket_boarding,
            "occupancy_percentage":  round(occupancy, 2),
            "events_data":           {"has_event": self.cfg.has_event},
        }
        try:
            resp = requests.post(
                self.cfg.api_url, json=payload, timeout=self.cfg.api_timeout
            )
            resp.raise_for_status()
            data = resp.json()
            adj = data.get("adjustments", {})
            log.info(
                "API ← stop=%-28s  base=%5.1f  final=%3s  occ=%5.1f%%  "
                "adj(ev=%+.1f rt=%+.1f cv=%+.1f)",
                data.get("stop_name", "?"),
                data.get("base_prediction", 0),
                data.get("final_prediction", "?"),
                occupancy,
                adj.get("event", 0),
                adj.get("realtime", 0),
                adj.get("opencv", 0),
            )
        except requests.exceptions.ConnectionError:
            log.warning("Backend unreachable at %s", self.cfg.api_url)
        except requests.exceptions.Timeout:
            log.warning("Backend request timed out (%.1fs)", self.cfg.api_timeout)
        except Exception as exc:
            log.warning("Backend error: %s", exc)

    # ── Main capture loop ─────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Open camera and run the detection/display loop.
        Exits on: stop() call, Q key, or camera failure.
        """
        cap = open_camera(self.cfg.webcam_index)

        log.info("=" * 56)
        log.info("  Bus Passenger Monitor — LIVE")
        log.info("=" * 56)
        log.info("  Station      : %s", self.cfg.station_name)
        log.info("  Bus capacity : %d passengers", self.cfg.bus_capacity)
        log.info("  Detector     : %s", self._detector_type.upper())
        log.info("  API endpoint : %s", self.cfg.api_url)
        log.info("  Send every   : %.0f s", self.cfg.send_interval)
        log.info("  Smoothing    : %d-frame rolling average", self.cfg.smooth_window)
        log.info("=" * 56)
        log.info("Press Q in the video window to quit.")

        frame_idx = 0
        cached_boxes: List[Tuple[int, int, int, int]] = []
        cached_count = 0
        consecutive_fail = 0
        MAX_CONSECUTIVE_FAILS = 30

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_fail += 1
                if consecutive_fail >= MAX_CONSECUTIVE_FAILS:
                    log.error(
                        "Camera stopped after %d consecutive failures — exiting.",
                        MAX_CONSECUTIVE_FAILS,
                    )
                    break
                time.sleep(0.05)
                continue

            consecutive_fail = 0
            frame_idx += 1

            # ── Inference (every N frames for performance) ─────────────────
            if frame_idx % self.cfg.detect_every_n == 0:
                try:
                    cached_boxes, cached_count = self.detect(frame)
                except Exception as exc:
                    log.warning("Detection error: %s", exc)
                    cached_boxes, cached_count = [], 0

            # ── Temporal smoothing ─────────────────────────────────────────
            self._history.append(cached_count)
            smoothed = round(sum(self._history) / max(len(self._history), 1))
            occupancy = min(smoothed / self.cfg.bus_capacity * 100.0, 100.0)

            # ── Annotate and display ───────────────────────────────────────
            display = self._draw_hud(
                frame.copy(), cached_boxes, cached_count, smoothed, occupancy
            )
            cv2.imshow(self.WINDOW_TITLE, display)

            # ── Send to backend ────────────────────────────────────────────
            now = time.time()
            if now - self._last_send >= self.cfg.send_interval:
                self._last_send = now
                threading.Thread(
                    target=self._send_to_backend,
                    args=(smoothed, occupancy),
                    daemon=True,
                ).start()

            # ── Quit on Q ─────────────────────────────────────────────────
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Q pressed — shutting down.")
                break

        cap.release()
        cv2.destroyAllWindows()
        log.info("Camera released. Goodbye.")

    def stop(self) -> None:
        """Signal the run loop to exit cleanly (called by SIGINT / SIGTERM)."""
        log.info("Stop signal received — finishing current frame …")
        self._stop.set()


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-time bus passenger detection — press Q to quit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--station", default="Yeshwanthapura T.T.M.C.",
        help="Bus station name (sent to backend with every API call)",
    )
    p.add_argument("--camera",   type=int,   default=0,
                   help="Webcam device index (0 = default camera)")
    p.add_argument("--capacity", type=int,   default=50,
                   help="Maximum bus passenger capacity")
    p.add_argument("--api",      default="http://127.0.0.1:8000/predict-demand",
                   help="FastAPI backend URL")
    p.add_argument("--conf",     type=float, default=0.35,
                   help="Detection confidence threshold (0–1)")
    p.add_argument("--interval", type=float, default=5.0,
                   help="Seconds between API calls")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    cfg = DetectorConfig(
        station_name   = args.station,
        webcam_index   = args.camera,
        bus_capacity   = args.capacity,
        api_url        = args.api,
        conf_threshold = args.conf,
        send_interval  = args.interval,
    )

    detector = BusPassengerDetector(cfg)

    # Graceful shutdown on Ctrl-C and SIGTERM
    def _sig(sig, _frame):
        detector.stop()

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    detector.run()


if __name__ == "__main__":
    main()
