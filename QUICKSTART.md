# Quick Start Guide

Get the Bus Passenger Detection system running in 5 minutes.

---

## Prerequisites

- Python 3.9 or later
- A webcam (built-in or USB)
- Windows 10/11, macOS, or Linux

---

## Step 1 — Install Dependencies

```bash
cd c:\Users\anees\predictor

# Using the project virtual environment (recommended)
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Optional but recommended — adds YOLOv8 for best detection accuracy
.venv\Scripts\python.exe -m pip install ultralytics
```

---

## Step 2 — Verify Setup

```bash
.venv\Scripts\python.exe test_setup.py
```

Expected output:
```
✓  Python >= 3.9
✓  opencv-python
✓  numpy
✓  requests
...
✓  deploy.prototxt
✓  res10_300x300_ssd_iter_140000.caffemodel
✓  model/bus_model.pkl
✓  DNN Caffe model loads without error

All required checks passed! Run:
  python bus_camera.py
```

---

## Step 3 — Start the Backend

Open a terminal and run:

```bash
cd c:\Users\anees\predictor
.venv\Scripts\python.exe -m uvicorn backend.api.app:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Test it's working:
```
http://127.0.0.1:8000/health   → {"status": "backend running"}
http://127.0.0.1:8000/docs     → Interactive API documentation
```

---

## Step 4 — Run the Camera Module

Open a **second terminal** and run:

```bash
cd c:\Users\anees\predictor
.venv\Scripts\python.exe bus_camera.py
```

A video window opens showing:
- Live camera feed
- Green bounding boxes around detected persons
- HUD overlay: passenger count, raw detection, occupancy %
- Colour-coded occupancy bar (green → orange → red)

Console output every 5 seconds:
```
08:32:15 [INFO ] bus_camera — API ← stop=Yeshwanthapura T.T.M.C.  base=6.8  final= 7  occ= 4.0%
```

Press **Q** in the video window to quit.

---

## CLI Options

```bash
# Change station name
python bus_camera.py --station "Majestic Bus Stand"

# Use a different camera (e.g., external USB camera)
python bus_camera.py --camera 1

# Change bus capacity
python bus_camera.py --capacity 60

# Connect to a remote backend
python bus_camera.py --api http://192.168.1.100:8000/predict-demand

# Lower confidence threshold (more detections, may increase false positives)
python bus_camera.py --conf 0.25

# All options together
python bus_camera.py --station "Majestic" --camera 1 --capacity 60 --conf 0.3
```

---

## No ML Model? Use the Sample Backend

If you don't have `model/bus_model.pkl`, use the sample backend instead:

```bash
# Terminal 1 — sample backend (no model needed)
.venv\Scripts\python.exe backend_sample.py

# Terminal 2 — camera with sample backend
.venv\Scripts\python.exe bus_camera.py --api http://127.0.0.1:8001/predict-demand
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not opening | Try `--camera 1` or `--camera 2` |
| `Passengers detected: 0` | Move closer to the camera; reduce `--conf 0.2` |
| Backend unreachable | Start uvicorn first; check port 8000 is free |
| Low detection accuracy | Install ultralytics: `pip install ultralytics` |
| MSMF grab error | Already handled — code automatically falls back to DirectShow |
