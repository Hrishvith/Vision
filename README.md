<<<<<<< HEAD
# Vision

Solution for public transport.

Bus route simulation for Yeshwanthapura T.T.M.C. to Kengeri T.T.M.C.

## Files

- `ypr-kengeri-simulation.jsx`: React simulation component
- `bus-simulation.html`: Standalone runnable simulation page

## Run in React

Import `ypr-kengeri-simulation.jsx` into your React app and render the default export.

## Run standalone

Open `bus-simulation.html` in a browser.
=======
# Smart Bus Demand Prediction System

Real-time AI-powered passenger detection and demand forecasting for smart public transport.

## Overview

This system uses computer vision and machine learning to count passengers on buses in real time and predict future demand at each stop — helping operators deploy buses more efficiently and reduce overcrowding.

```
Live Camera Feed
      │
      ▼
┌─────────────────────────┐
│   bus_camera.py         │  ← OpenCV + AI detection
│   BusPassengerDetector  │    YOLOv8n / DNN SSD / HOG
└────────────┬────────────┘
             │  REST API (every 5 s)
             ▼
┌─────────────────────────┐
│   FastAPI Backend       │  ← backend/api/app.py
│   /predict-demand       │    RandomForestRegressor
└─────────────────────────┘
```

## Project Structure

```
predictor/
├── bus_camera.py              ← Main OpenCV detection module  ⭐
├── backend_sample.py          ← Standalone sample backend (no model needed)
├── opencv_bus_monitor.py      ← Original monitor (kept for reference)
│
├── backend/
│   ├── api/
│   │   └── app.py             ← Production FastAPI server
│   └── predictor/
│       ├── predictor.py       ← ML feature engineering + prediction
│       └── optimizer.py       ← Model optimisation utilities
│
├── model/
│   └── bus_model.pkl          ← Trained RandomForestRegressor (37 features)
│
├── deploy.prototxt            ← DNN SSD face detector architecture
├── res10_300x300_ssd_iter_140000.caffemodel  ← DNN weights (10 MB)
│
├── test_setup.py              ← Pre-flight dependency check
├── test_api.py                ← API endpoint integration tests
├── test_model.py              ← ML model smoke test
│
├── requirements.txt
├── QUICKSTART.md
├── DEPLOYMENT_GUIDE.md
└── MENTOR_EXPLANATION.md
```

## Detection Pipeline

| Priority | Detector | Accuracy | Requirements |
|----------|----------|----------|--------------|
| 1 | **YOLOv8n** | ⭐⭐⭐⭐⭐ | `pip install ultralytics` |
| 2 | **DNN SSD** (res10) | ⭐⭐⭐⭐ | model files downloaded ✓ |
| 3 | **HOG + SVM** | ⭐⭐⭐ | built into OpenCV ✓ |

The system automatically selects the best available detector at startup.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify setup
python test_setup.py

# 3. Start backend
python -m uvicorn backend.api.app:app --port 8000

# 4. Run camera monitor (new terminal)
python bus_camera.py
```

See [QUICKSTART.md](QUICKSTART.md) for detailed steps.

## Key Features

- **Real-time detection** — processes webcam or IP camera feed at full FPS
- **Multi-detector fallback** — works even without YOLOv8 installed
- **CLAHE + bilateral filter** — handles uneven bus lighting
- **Temporal smoothing** — rolling average eliminates frame jitter
- **REST API integration** — pushes sensor data to FastAPI for ML predictions
- **Graceful shutdown** — SIGINT, SIGTERM, and Q key all exit cleanly
- **CLI configurable** — station name, camera index, API URL all via args

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Computer Vision | OpenCV 4.x, YOLOv8, DNN SSD, HOG |
| Backend API | FastAPI + Pydantic v2 |
| ML Model | scikit-learn RandomForestRegressor |
| Data | Pandas, NumPy, Joblib |

## Route

**Yeshwanthapura → Kengeri** (43 stations, BMTC bus route)

Trained on Jan–Mar 2025 ticketing data (73,530 records).
>>>>>>> a22763c (BMTC Smart Bus Predictor - full project)
