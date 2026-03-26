"""
backend_sample.py
=================
Standalone sample backend — no ML model required.

Shows the exact API contract expected by bus_camera.py.
Use this to test the full pipeline without the trained model.

Endpoints
---------
  GET  /health           → {"status": "sample backend running"}
  POST /predict-demand   → PredictResponse (see schema below)
  GET  /passenger-log    → last 100 readings received from cameras

Run
---
    python backend_sample.py
    # or via uvicorn directly:
    python -m uvicorn backend_sample:app --port 8001 --reload

Then update bus_camera.py --api flag:
    python bus_camera.py --api http://127.0.0.1:8001/predict-demand
"""

from __future__ import annotations

import datetime
import logging
from collections import deque
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backend_sample")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bus Demand Predictor — Sample Backend",
    description=(
        "Lightweight rule-based backend compatible with bus_camera.py. "
        "Replace the heuristic in /predict-demand with a real model for production."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class EventsData(BaseModel):
    has_event: bool = False


class PredictRequest(BaseModel):
    station_name:          str
    boarding:              int   = 0
    deboarding:            int   = 0
    first_ticket_time:     str   = "08:00"
    day:                   str   = "Monday"
    date:                  str   = ""
    ticket_boarding_count: int   = 0
    occupancy_percentage:  float = 0.0
    events_data:           EventsData = EventsData()


class AdjustmentsResult(BaseModel):
    event:    float = 0.0
    realtime: float = 0.0
    opencv:   float = 0.0


class PredictResponse(BaseModel):
    stop_name:        str
    base_prediction:  float
    final_prediction: int
    adjustments:      AdjustmentsResult


class LogEntry(BaseModel):
    time:      str
    station:   str
    occupancy: float
    predicted: int
    boarding:  int
    deboarding: int


# ── In-memory reading log (last 100) ──────────────────────────────────────────
_log: deque = deque(maxlen=100)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health():
    """Health check — confirms backend is reachable."""
    return {"status": "sample backend running"}


@app.post("/predict-demand", response_model=PredictResponse, tags=["prediction"])
def predict_demand(req: PredictRequest):
    """
    Rule-based demand estimate (no ML model required).

    Logic
    -----
    base  = ticket_boarding_count × 1.2  +  boarding × 0.5
    final = base × occupancy_factor  +  event_bonus  (rounded)

    Replace this heuristic with a real model call for production use.
    """
    base = req.ticket_boarding_count * 1.2 + req.boarding * 0.5
    occ_factor = 1.0 + (req.occupancy_percentage / 200.0)   # 0% → ×1.0, 100% → ×1.5
    event_bonus = 2.5 if req.events_data.has_event else 0.0
    realtime_adj = round(base * occ_factor - base, 2)
    final = max(0, round(base * occ_factor + event_bonus))

    _log.append({
        "time":       datetime.datetime.now().isoformat(timespec="seconds"),
        "station":    req.station_name,
        "occupancy":  req.occupancy_percentage,
        "predicted":  final,
        "boarding":   req.boarding,
        "deboarding": req.deboarding,
    })

    log.info(
        "predict  station=%-30s  occ=%5.1f%%  base=%5.1f  final=%3d",
        req.station_name, req.occupancy_percentage, base, final,
    )

    return PredictResponse(
        stop_name        = req.station_name,
        base_prediction  = round(base, 2),
        final_prediction = final,
        adjustments      = AdjustmentsResult(
            event    = event_bonus,
            realtime = realtime_adj,
            opencv   = 0.0,
        ),
    )


@app.get("/passenger-log", tags=["monitoring"])
def passenger_log():
    """Return the last 100 readings received from bus cameras."""
    return {"total_readings": len(_log), "readings": list(_log)}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting sample backend on http://0.0.0.0:8001")
    log.info("Connect bus_camera.py with:  --api http://127.0.0.1:8001/predict-demand")
    uvicorn.run("backend_sample:app", host="0.0.0.0", port=8001, reload=False)
