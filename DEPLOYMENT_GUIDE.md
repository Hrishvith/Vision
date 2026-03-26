# Deployment Guide

Production deployment instructions for the Smart Bus Demand Prediction system.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Bus Camera (Raspberry Pi / NUC / Laptop)            │
│                                                      │
│   bus_camera.py                                      │
│   └── BusPassengerDetector                          │
│        ├── YOLOv8n / DNN SSD / HOG                  │
│        └── POST /predict-demand  every 5 s ──────►  │
└──────────────────────────────────────────────────────┘
                                                    │
                              ┌─────────────────────▼──────────────────────┐
                              │  Backend Server (VM / Cloud)               │
                              │                                            │
                              │  uvicorn backend.api.app:app               │
                              │  ├── GET  /health                          │
                              │  └── POST /predict-demand                  │
                              │       └── RandomForestRegressor            │
                              └────────────────────────────────────────────┘
```

---

## 1. System Requirements

### Camera Device (Edge)
- Python 3.9+, OpenCV 4.8+, requests
- RAM: 512 MB minimum (2 GB for YOLOv8)
- Works on: Raspberry Pi 4, Intel NUC, any laptop

### Backend Server
- Python 3.9+, FastAPI, scikit-learn
- RAM: 1 GB minimum
- OS: Ubuntu 22.04 LTS recommended

---

## 2. Backend Deployment (Linux Server)

### Install dependencies

```bash
git clone <your-repo>
cd predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run with Gunicorn + Uvicorn workers (production)

```bash
pip install gunicorn

gunicorn backend.api.app:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 2 \
  --bind 0.0.0.0:8000 \
  --timeout 30 \
  --access-logfile logs/access.log \
  --error-logfile  logs/error.log
```

### Systemd service (auto-restart on reboot)

Create `/etc/systemd/system/bus-backend.service`:

```ini
[Unit]
Description=Bus Demand Prediction Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/predictor
ExecStart=/home/ubuntu/predictor/.venv/bin/gunicorn \
    backend.api.app:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 2 \
    --bind 0.0.0.0:8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable bus-backend
sudo systemctl start  bus-backend
sudo systemctl status bus-backend
```

---

## 3. Camera Device Deployment

### Install (minimal — no YOLOv8)

```bash
pip install opencv-python numpy requests
```

### Install (with YOLOv8 — best accuracy)

```bash
pip install opencv-python numpy requests ultralytics
```

### Run at startup (Windows Task Scheduler)

```
Program:  C:\predictor\.venv\Scripts\python.exe
Args:     C:\predictor\bus_camera.py --station "Stop Name" --api http://<SERVER_IP>:8000/predict-demand
Start in: C:\predictor
Run:      Whether user is logged on or not
```

### Run at startup (Linux systemd)

```ini
[Unit]
Description=Bus Camera Passenger Detector

[Service]
User=pi
WorkingDirectory=/home/pi/predictor
ExecStart=/home/pi/predictor/.venv/bin/python bus_camera.py \
    --station "Yeshwanthapura T.T.M.C." \
    --api http://192.168.1.100:8000/predict-demand
Restart=always
Environment=DISPLAY=:0

[Install]
WantedBy=graphical.target
```

---

## 4. Network Security

### Firewall rules (UFW)

```bash
sudo ufw allow 8000/tcp    # API port
sudo ufw enable
```

### Reverse proxy with Nginx + HTTPS (recommended for public servers)

```nginx
server {
    listen 443 ssl;
    server_name your-server.com;

    ssl_certificate     /etc/letsencrypt/live/your-server.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-server.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Then update camera devices to use `--api https://your-server.com/predict-demand`.

---

## 5. Performance Tuning

| Scenario | Config |
|----------|--------|
| Best accuracy | Install `ultralytics`, set `--conf 0.35` |
| Low-power device | Use DNN SSD (default), set `--conf 0.40`, `detect_every_n=3` |
| High-traffic camera | Set `smooth_window=5`, `send_interval=3` |
| Multiple cameras | Run one `bus_camera.py` process per camera, each with different `--station` |

---

## 6. Monitoring & Logs

Backend logs to stdout; redirect to a file in production:

```bash
gunicorn ... 2>&1 | tee logs/backend.log
```

View live readings via the API:

```
GET http://your-server:8000/passenger-log   (if using backend_sample.py)
GET http://your-server:8000/docs            (interactive API docs)
```

---

## 7. Environment Variables (optional)

You can override defaults without editing source files:

```bash
export BUS_API_URL="http://192.168.1.100:8000/predict-demand"
export BUS_STATION="Majestic"
export BUS_CAPACITY=60
```

Then use in code:

```python
import os
cfg = DetectorConfig(
    api_url      = os.getenv("BUS_API_URL", "http://127.0.0.1:8000/predict-demand"),
    station_name = os.getenv("BUS_STATION", "Yeshwanthapura T.T.M.C."),
    bus_capacity = int(os.getenv("BUS_CAPACITY", 50)),
)
```
