"""
backend/api/app.py
------------------
FastAPI service for the Smart Bus Demand Prediction system.

Exposes two endpoints:
  GET  /health          – liveness check
  POST /predict-demand  – full prediction pipeline

Run with:
  uvicorn backend.api.app:app --reload

Install dependencies first:
  pip install fastapi uvicorn pandas numpy scikit-learn joblib
"""

import logging
import math as _math
import sys
from datetime import datetime as _datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so the relative import works
# whether the server is started from the project root or elsewhere.
# ---------------------------------------------------------------------------
from pathlib import Path
_PROJECT_ROOT = Path(__file__).parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.predictor.predictor import load_model, predict_from_raw_input  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Smart Bus Demand Prediction API",
    description=(
        "Predict passenger demand at a given bus stop using a trained "
        "RandomForestRegressor model with real-time contextual adjustments."
    ),
    version="1.0.0",
)

# Allow all origins so the frontend dashboard and OpenCV module can connect
# freely during development.  Tighten allow_origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load the ML model once at startup so every request reuses the same object.
# ---------------------------------------------------------------------------
_model = None


@app.on_event("startup")
def _startup():
    global _model
    _model = load_model()
    if _model is not None:
        logger.info("ML model loaded and ready.")
    else:
        logger.warning(
            "ML model could not be loaded — predictions will use random fallback."
        )


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class EventsData(BaseModel):
    has_event: bool = False
    event_popularity: str = Field(
        default="low",
        description="One of: low, medium, high",
    )
    distance_km: float = Field(
        default=999.0,
        description="Distance in km from the stop to the event venue",
    )


class PredictRequest(BaseModel):
    # Core stop data (required)
    station_name: str = Field(..., example="Yeshwanthapura T.T.M.C.")
    boarding: int = Field(..., ge=0, example=5)
    deboarding: int = Field(..., ge=0, example=1)
    first_ticket_time: str = Field(..., example="08:15")
    day: str = Field(..., example="Monday")
    date: str = Field(..., example="2024-01-01")

    # Optional real-time signals
    ticket_boarding_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of tickets issued in the last 15 minutes",
        example=12,
    )
    occupancy_percentage: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Current bus occupancy percentage (from OpenCV or sensors)",
        example=82.0,
    )
    events_data: Optional[EventsData] = Field(
        default=None,
        description="Nearby event information",
    )


class AdjustmentsResult(BaseModel):
    event: float
    realtime: float
    opencv: float


class PredictResponse(BaseModel):
    stop_name: str
    base_prediction: float
    final_prediction: int
    adjustments: AdjustmentsResult


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Liveness check — returns immediately if the server is running."""
    return {"status": "backend running"}


@app.post(
    "/predict-demand",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict passenger demand at a bus stop",
)
def predict_demand(request: PredictRequest):
    """
    Run the full prediction pipeline:

    1. Build the 37 model features from raw input via `create_features()`.
    2. Call `model.predict()` for a base passenger count.
    3. Apply contextual adjustments:
       - Nearby events → `adjust_with_events()`
       - Recent ticket activity → `adjust_with_realtime()`
       - Bus occupancy (OpenCV) → `adjust_with_opencv()`
    4. Return structured JSON with base and final predictions.
    """
    # Build the core raw-input dict consumed by predict_from_raw_input()
    raw_input = {
        "station_name":     request.station_name,
        "boarding":         request.boarding,
        "deboarding":       request.deboarding,
        "first_ticket_time": request.first_ticket_time,
        "day":              request.day,
        "date":             request.date,
    }

    # Convert the Pydantic EventsData model to a plain dict if provided
    events_dict = request.events_data.model_dump() if request.events_data else None

    logger.info(
        "Received /predict-demand request | station=%s | time=%s | day=%s",
        request.station_name,
        request.first_ticket_time,
        request.day,
    )

    try:
        result = predict_from_raw_input(
            raw_input=raw_input,
            model=_model,
            events_data=events_dict,
            ticket_boarding_count=request.ticket_boarding_count,
            occupancy_percentage=request.occupancy_percentage,
        )
    except ValueError as exc:
        # Missing / invalid input fields
        logger.warning("Bad request: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        # Unexpected prediction error
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed. See server logs.")

    logger.info(
        "Prediction complete | station=%s | base=%.2f | final=%d",
        result["stop_name"],
        result["base_prediction"],
        result["final_prediction"],
    )

    # Cache live occupancy so /routes-status can use real OpenCV data
    if request.occupancy_percentage is not None:
        _live_occupancy[request.station_name] = request.occupancy_percentage
        logger.info(
            "Live occupancy cached | station=%s | occupancy=%.1f%%",
            request.station_name, request.occupancy_percentage,
        )

    return PredictResponse(
        stop_name=result["stop_name"],
        base_prediction=result["base_prediction"],
        final_prediction=result["final_prediction"],
        adjustments=AdjustmentsResult(**result["adjustments"]),
    )


# ---------------------------------------------------------------------------
# Live occupancy cache — updated by /predict-demand (OpenCV), read by /routes-status
# Maps station_name → latest occupancy_percentage (float, 0–100)
# ---------------------------------------------------------------------------
_live_occupancy: Dict[str, float] = {}

# ---------------------------------------------------------------------------
# Route configuration – shared by /routes-status
# ---------------------------------------------------------------------------
_BUS_CAPACITY = 40

_ROUTE_STOPS: Dict[str, Dict[str, List[str]]] = {
    "YPR-Kengeri": {
        "stops": [
            "Yeshwanthapura T.T.M.C.", "Goraguntepalya", "Sumanahalli",
            "Mallathahalli Cross", "Kottigepalya", "Papareddypalya",
            "B.D.A. Complex Nagarabhavi", "Nagadevanahalli",
            "Kengeri Police Station", "Kengeri Bus Station",
        ],
        "labels": [
            "YPR", "Gorag.", "Sumana.", "Mallath.", "Kottige.",
            "Papar.", "BDA Nagar.", "Nagadeva.", "KGI Police", "KGI Bus Stn",
        ],
    },
    "Majestic-Uttarahalli": {
        "stops": [
            "Yeshwanthapura T.T.M.C.", "Goraguntepalya", "Sumanahalli",
            "Mallathahalli Cross", "Kottigepalya", "Kengeri Bus Station",
        ],
        "labels": ["Majestic", "Gorag.", "Sumana.", "Mallath.", "Kottige.", "Uttarahalli"],
    },
    "Hebbal-Silk Board": {
        "stops": [
            "Yeshwanthapura T.T.M.C.", "Goraguntepalya", "Mallathahalli Cross",
            "B.D.A. Complex Nagarabhavi", "Nagadevanahalli", "Kengeri T.T.M.C",
        ],
        "labels": ["Hebbal", "Gorag.", "Mallath.", "BDA Nagar.", "Nagadeva.", "Silk Board"],
    },
}


def _crowd_from_ratio(ratio: float) -> str:
    if ratio < 0.60:
        return "LOW"
    if ratio < 0.85:
        return "MEDIUM"
    return "HIGH"


# ---------------------------------------------------------------------------
# GET /routes-status
# ---------------------------------------------------------------------------

class StopDetail(BaseModel):
    name: str
    label: str
    predicted_pax: int
    crowd_level: str


class RouteStatus(BaseModel):
    route: str
    current_stop: str
    current_stop_idx: int
    occupancy_percent: int
    predicted_demand_next_hour: int
    current_buses: int
    required_buses: int
    crowd_level: str
    stops: List[StopDetail]


class RoutesStatusResponse(BaseModel):
    timestamp: str
    routes: List[RouteStatus]


@app.get(
    "/routes-status",
    response_model=RoutesStatusResponse,
    tags=["Operations"],
    summary="Real-time status snapshot for all monitored routes",
)
def routes_status():
    """
    For each route, predict demand at every stop using the ML model and derive:
    - current stop (simulates bus position based on minute-of-hour)
    - occupancy percentage (derived from prediction / bus capacity)
    - next-hour demand (sum of the next 4 stop predictions)
    - required buses

    In a production deployment the occupancy figure would be replaced by live
    data from onboard OpenCV sensors pushed to a shared store (Redis / DB).
    """
    now      = _datetime.now()
    time_str = now.strftime("%H:%M")
    day      = now.strftime("%A")
    date_str = now.strftime("%Y-%m-%d")

    route_results: List[Dict[str, Any]] = []

    for route_name, config in _ROUTE_STOPS.items():
        stops  = config["stops"]
        labels = config["labels"]
        n      = len(stops)

        # Derive current stop from the minute of the hour (bus cycles through stops)
        mins_per_stop = max(1, 60 // n)
        current_idx   = min(now.minute // mins_per_stop, n - 1)

        # Predict demand for every stop on the route
        stop_preds: List[int] = []
        for stop in stops:
            raw = {
                "station_name":      stop,
                "boarding":          5,
                "deboarding":        2,
                "first_ticket_time": time_str,
                "day":               day,
                "date":              date_str,
            }
            # Use live OpenCV occupancy if available for this stop
            live_occ = _live_occupancy.get(stop)
            try:
                res  = predict_from_raw_input(
                    raw_input=raw,
                    model=_model,
                    occupancy_percentage=live_occ,
                )
                pred = int(res["final_prediction"])
            except Exception:
                pred = 15  # safe fallback when model unavailable
            stop_preds.append(pred)

        # Next-hour demand = sum of predictions for the next 4 stops from current
        next_four   = [stop_preds[(current_idx + i) % n] for i in range(min(4, n))]
        next_hour   = sum(next_four)

        current_pred   = stop_preds[current_idx]
        current_stop_name = stops[current_idx]
        # Prefer live OpenCV occupancy for the current stop, otherwise derive from ML
        if current_stop_name in _live_occupancy:
            occupancy_pct = min(100, round(_live_occupancy[current_stop_name]))
        else:
            occupancy_pct = min(100, round((current_pred / _BUS_CAPACITY) * 100))
        crowd_level    = _crowd_from_ratio(occupancy_pct / 100)
        required_buses = _math.ceil(next_hour / _BUS_CAPACITY)
        # Current deployment is one bus fewer than required (reflects under-provisioning)
        current_buses  = max(1, required_buses - (1 if required_buses > 1 else 0))

        stops_detail = [
            {
                "name":          stop,
                "label":         label,
                "predicted_pax": pred,
                "crowd_level":   _crowd_from_ratio(pred / _BUS_CAPACITY),
            }
            for stop, label, pred in zip(stops, labels, stop_preds)
        ]

        route_results.append({
            "route":                      route_name,
            "current_stop":               stops[current_idx],
            "current_stop_idx":           current_idx,
            "occupancy_percent":          occupancy_pct,
            "predicted_demand_next_hour": next_hour,
            "current_buses":              current_buses,
            "required_buses":             required_buses,
            "crowd_level":                crowd_level,
            "stops":                      stops_detail,
        })

        logger.info(
            "routes-status | route=%s | current_stop=%s | next_hour=%d | crowd=%s",
            route_name, stops[current_idx], next_hour, crowd_level,
        )

    return {"timestamp": time_str, "routes": route_results}

