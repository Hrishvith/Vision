BMTC Smart Bus Demand Prediction System
An end-to-end AI-powered system for predicting and monitoring passenger demand on BMTC (Bangalore Metropolitan Transport Corporation) bus routes. Built with a FastAPI backend, Streamlit frontend, and real-time computer-vision passenger detection.

Features
ML Demand Prediction — RandomForestRegressor (37 features) trained on YPR → Kengeri route data, with contextual adjustments for weather, events, and time-of-day.
Real-Time Dashboard — Streamlit app polling the backend every 10 seconds, showing network KPIs, crowd levels (LOW / MEDIUM / HIGH), and per-route stop flow charts.
Route Simulation — Interactive simulation of a bus traversing 43 stops with live boarding/deboarding, trip summaries, and speed control (1× / 2× / 4×).
Live Bus Camera — Webcam-based passenger detection using YOLOv8 (primary), DNN SSD (fallback), or HOG+SVM, with temporal smoothing and REST API integration.
FastAPI Backend — /health, /predict-demand, and /routes-status endpoints with CORS support and Pydantic validation.
Project Structure
predictor/
├── backend/
│   ├── api/app.py            # FastAPI application
│   └── predictor/predictor.py # ML model loading & prediction logic
├── frontend/
│   ├── dashboard.py           # Main Streamlit dashboard
│   └── pages/
│       ├── 1_simulation.py    # Route simulation page
│       └── 2_bus_camera.py    # Live camera detection page
├── model/
│   └── bus_model.pkl          # Trained RandomForest model
├── bus_camera.py              # Standalone camera detection script
├── deploy.prototxt            # DNN SSD model config
├── res10_300x300_ssd_iter_140000.caffemodel  # DNN SSD weights
└── requirements.txt
Prerequisites
Python 3.9+
Webcam (optional, for camera features)
Quick Start
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the backend (port 8000)
uvicorn backend.api.app:app --reload

# 4. Start the frontend (port 8501) — in a second terminal
streamlit run frontend/dashboard.py
Open http://localhost:8501 in your browser.

API Endpoints
Method	Path	Description
GET	/health	Liveness check
POST	/predict-demand	Predict passenger demand for a stop
GET	/routes-status	Current status of all monitored routes
Example Prediction Request
{
  "station_name": "Yeshwanthapura T.T.M.C.",
  "boarding": 5,
  "deboarding": 1,
  "first_ticket_time": "08:15",
  "day": "Monday",
  "date": "2024-01-01"
}
Standalone Camera
python bus_camera.py
python bus_camera.py --station "Majestic" --camera 1 --capacity 60
Press Q in the video window to quit.

Tech Stack
Backend: FastAPI, Uvicorn, scikit-learn, pandas, NumPy, joblib
Frontend: Streamlit, Plotly, Requests
Vision: OpenCV, YOLOv8 (ultralytics), HOG+SVM