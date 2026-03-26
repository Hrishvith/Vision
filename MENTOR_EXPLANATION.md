# Technical Explanation — Smart Bus Demand Prediction System

A complete explanation of how this project works, written for academic review.

---

## 1. Problem Statement

Public buses in cities like Bengaluru often run overcrowded on some routes while running near-empty on others. The goal of this project is to use **computer vision** and **machine learning** to count passengers in real time and predict demand at upcoming stops — so operators can deploy additional buses before overcrowding happens.

---

## 2. System Architecture

The system has two independent parts that talk to each other via a REST API:

```
┌─────────────────────────────────────────┐
│  Part A — Computer Vision (bus_camera)  │
│                                         │
│  Webcam → Frame → Detect persons        │
│        → Count → Smooth → Send to API   │
└─────────────────────┬───────────────────┘
                      │ HTTP POST every 5 s
                      ▼
┌─────────────────────────────────────────┐
│  Part B — ML Prediction (FastAPI)       │
│                                         │
│  Receive sensor data                    │
│      → Feature engineering             │
│      → RandomForestRegressor           │
│      → Return predicted demand         │
└─────────────────────────────────────────┘
```

---

## 3. Computer Vision Module (`bus_camera.py`)

### 3.1 Detector Selection (Priority Chain)

The code tries three detectors in order, using the best one available:

```
YOLOv8n  →  DNN SSD  →  HOG+SVM
(pip install needed)   (files downloaded)   (always available)
```

**Why three?** We want the system to work everywhere — even on low-spec devices without internet access.

### 3.2 YOLOv8 (Priority 1)

- **What it is**: You Only Look Once version 8, a convolutional neural network trained on the COCO dataset (80 object classes including "person")
- **How it works**: Divides the image into a grid; each cell predicts bounding boxes and class probabilities in a single forward pass — hence "you only look once"
- **Why lightweight**: We use `yolov8n.pt` (nano variant, ~6 MB) for real-time speed on CPU
- Code:
  ```python
  results = self._yolo(frame, classes=[0], conf=0.35, iou=0.45)
  ```

### 3.3 DNN SSD Face Detector (Priority 2)

- **Model**: ResNet-10 + Single Shot Detector, trained by OpenCV team
- **Files**: `deploy.prototxt` (network architecture) + `res10_300x300_ssd_iter_140000.caffemodel` (weights, 10 MB)
- **Why it fits buses**: Seated passengers show faces/heads even when bodies are obscured by seats
- **Preprocessing**: CLAHE on LAB colour space + bilateral filter before inference
- **Output**: Bounding boxes with confidence scores; filtered with NMS

### 3.4 HOG + SVM (Priority 3)

- **HOG**: Histogram of Oriented Gradients — counts how many pixels point in each direction inside small cells; produces a fixed-length feature vector per frame window
- **SVM**: Support Vector Machine classifier — trained by OpenCV on the [INRIA Person Dataset](https://pascal.inrialpes.fr/data/human/) to recognise HOG patterns that look like pedestrians
- Built directly into OpenCV; no download needed
- Slower and less accurate than YOLO/DNN, but always available as a fallback

### 3.5 Non-Maximum Suppression (NMS)

Without NMS, the same person is often detected by multiple overlapping windows:

```
Without NMS: [3 boxes around same head] → count = 3  ← WRONG
With NMS:    [1 best box per person]    → count = 1  ← CORRECT
```

We use `cv2.dnn.NMSBoxes()` with IoU threshold 0.45 — boxes that overlap by more than 45% are merged into the highest-confidence one.

### 3.6 Temporal Smoothing

Camera detections are noisy — a face briefly turns away and the count drops by 1. We use a **rolling average** over the last 10 frames:

```python
history.append(raw_count)
smoothed = round(sum(history) / len(history))
```

This gives a stable, jitter-free passenger count for the API and display.

### 3.7 CLAHE (Contrast Limited Adaptive Histogram Equalization)

Bus interiors have uneven lighting — bright near windows, dark in the aisle. Standard histogram equalisation stretches the entire image, losing local detail. CLAHE operates on small tiles independently:

```python
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l = clahe.apply(l)    # only equalise the luminance channel
```

Working in LAB colour space means colours are not distorted.

---

## 4. Machine Learning Model (`backend/predictor/predictor.py`)

### 4.1 Training Data

- **Source**: BMTC ticketing data, Jan–Mar 2025
- **Route**: Yeshwanthapura → Kengeri (43 stops)
- **Records**: 73,530 rows
- **Raw columns**: date, day, station_name, first_ticket_time, boarding, deboarding

### 4.2 Feature Engineering (`create_features`)

The model cannot use raw strings — everything must be numeric. We engineer 37 features:

| Feature group | Examples | How |
|---------------|----------|-----|
| Time | hour, minute, is_peak_hour | parse first_ticket_time |
| Day | day_of_week (0–6), is_weekend | map day name → int |
| Station | stop_sequence (1–43), station_encoded | ordinal map |
| Flow | net_flow = boarding − deboarding | arithmetic |
| Lag | boarding_lag_1, boarding_lag_2 | shift by stop index |
| Rolling | boarding_rolling_3, boarding_rolling_7 | window mean |

### 4.3 Model: RandomForestRegressor

A **random forest** is an ensemble of decision trees:
1. Each tree is trained on a random bootstrap sample of the data
2. Each split in a tree uses a random subset of features
3. Prediction = average of all tree outputs

**Why Random Forest?**
- Handles non-linear relationships (peak hour effects, station position interactions)
- Robust to outliers (e.g., holiday anomalies)
- Feature importance available for interpretability
- No need to normalise features

### 4.4 Adjustments

After the model predicts a base count, three multiplier-based adjustments are applied:

| Adjustment | Trigger | Multiplier |
|-----------|---------|-----------|
| Event | high-footfall event nearby | +1.5× |
| Real-time | ticket count significantly deviates from average | ±0.2× |
| OpenCV | occupancy % from camera | +0.05 per 10% above 50% |

---

## 5. API Contract (FastAPI)

### POST `/predict-demand`

**Request body** (sent by `bus_camera.py` every 5 seconds):
```json
{
  "station_name": "Yeshwanthapura T.T.M.C.",
  "boarding": 5,
  "deboarding": 1,
  "first_ticket_time": "08:32",
  "day": "Saturday",
  "date": "2026-03-07",
  "ticket_boarding_count": 12,
  "occupancy_percentage": 24.0,
  "events_data": {"has_event": false}
}
```

**Response**:
```json
{
  "stop_name": "Yeshwanthapura T.T.M.C.",
  "base_prediction": 6.81,
  "final_prediction": 7,
  "adjustments": {"event": 0.0, "realtime": 0.0, "opencv": 0.0}
}
```

### Why REST / JSON?

- Language-agnostic: a frontend dashboard (React, Flutter) can consume the same endpoint
- Stateless: each request is self-contained; backend can be replicated horizontally
- Pydantic validates every field automatically, rejecting malformed inputs before they reach the model

---

## 6. Key Design Decisions

| Decision | Reason |
|----------|--------|
| Separate camera process from backend | Camera runs on the bus device; backend runs on a central server. They communicate over the network. |
| Daemon threads for API calls | API calls should never block the camera loop. If the backend is slow, frames keep processing. |
| Three-detector fallback | Makes deployment flexible: YOLOv8 on powerful hardware, DNN SSD on mid-range, HOG on a Raspberry Pi. |
| Rolling average smoothing | Detection is inherently noisy on a per-frame basis; smoothing gives operators a stable number to act on. |
| `DetectorConfig` dataclass | All parameters in one place; easy to override via CLI args or environment variables in production. |

---

## 7. Limitations and Future Work

| Limitation | Future improvement |
|------------|-------------------|
| Single camera angle | Use IP cameras mounted at door; multiple angles improve count accuracy |
| Haar/DNN misses profiles | YOLOv8 or a retrained model would handle turned-away faces |
| No persistent storage | Add PostgreSQL / TimescaleDB for historical data and capacity dashboards |
| Route hardcoded | Make route configurable; load station list from a database |
| No re-ID tracking | Same person counted twice if they move in/out of frame — add DeepSORT tracking |
