"""
Smart Bus Demand Prediction System - Predictor Module

This module serves as the brain of the backend, responsible for:
1. Loading the trained ML model (RandomForestRegressor, 37 features)
2. Feature engineering from raw input data
3. Generating base predictions via model.predict()
4. Applying contextual adjustments from various signals
5. Returning final passenger demand predictions
"""

import io
import pickle
import random
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parents[2] / "model" / "bus_model.pkl"
TEST_DATA_PATH = (
    Path(__file__).parents[2]
    / "model"
    / "test_data_ypr_kengeri_jan_mar_2025 (2).csv"
)

# ---------------------------------------------------------------------------
# Static mappings used by feature engineering
# ---------------------------------------------------------------------------

# Route order for the YPR → Kengeri bus route (BMTC).
# Index = stop_sequence (1-based).  All 43 stations in the dataset are listed.
_ROUTE_ORDER = [
    "Yeshwanthapura T.T.M.C.",
    "R.M.C. Yard (Yashawanthapura New Railway Station)",
    "Kanteerava Studio",
    "Kanteerava Studio (Stop 2)",
    "Goraguntepalya",
    "Modern Food Goraguntepalya",
    "Reliance Petrol Bunk Goraguntepalya",
    "Depot-31 Gate Summanahalli",
    "Sumanahalli",
    "Laggere Bridge",
    "Mariyappanapalya",
    "Govardhan Talkies",
    "Kempegowda Arch",
    "Kengunte Circle",
    "Kengunte Circle (Stop 2)",
    "Mallathahalli Cross",
    "Mallathahalli I.T.I. Layout",
    "M.E.I. Factory",
    "Kottigepalya",
    "Vokkaliga School Kottigepalya",
    "Papareddypalya",
    "Aladamara Papareddypalya",
    "Nandhini Layout",
    "Nandini Layout Ring Road",
    "B.D.A. Complex Nagarabhavi",
    "B.D.A. 2nd Block Nagarabhavi",
    "Bengaluru University Quarters",
    "Dr. Ambedkar Institute Of Technology",
    "Nagadevanahalli",
    "Kommaghatta Junction",
    "Kenchanapura Cross",
    "P.V.P. School",
    "Shirke K.H.B. Quarters",
    "Deepa Complex",
    "Lurdubayi Samudaya Bhavana",
    "Kengeri Police Station",
    "Kengeri Post Office",
    "Kengeri Church",
    "Kengeri Ganesha Temple",
    "Kengeri Railway Station",
    "Kengeri Satellite Town",
    "Kengeri T.T.M.C",
    "Kengeri Bus Station",
]
_TOTAL_STOPS = len(_ROUTE_ORDER)  # 43

# stop_sequence map: station name → 1-based integer
_STOP_SEQ_MAP: Dict[str, int] = {
    name: idx + 1 for idx, name in enumerate(_ROUTE_ORDER)
}

# station_encoded map: sklearn LabelEncoder encodes alphabetically (sorted).
_STATION_ENCODED_MAP: Dict[str, int] = {
    name: idx for idx, name in enumerate(sorted(_ROUTE_ORDER))
}

# Day-of-week mapping (matches Python's datetime.weekday())
_DAY_MAP: Dict[str, int] = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

# The 37 feature columns expected by the trained model (in exact order)
_MODEL_FEATURES = [
    "hour", "minute", "hour_sin", "hour_cos", "day_num", "time_of_day",
    "is_peak_am", "is_peak_pm", "is_weekend", "is_weekday", "is_early_morn",
    "is_night", "stop_sequence", "stop_position_ratio", "is_first_stop",
    "is_last_stop", "is_mid_stop", "station_encoded", "boarding", "deboarding",
    "net_flow", "total_flow", "board_ratio", "deboard_ratio", "passengers_in_bus",
    "prev_boarding", "prev_deboarding", "prev_flow", "prev2_flow", "prev3_flow",
    "prev_pax", "prev2_pax", "rolling_avg_3", "rolling_avg_5", "rolling_max_3",
    "cum_boarding", "cum_deboarding",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw input DataFrame into the 37-feature DataFrame expected
    by the trained RandomForestRegressor model.

    Raw input columns required:
        date             – date string (e.g. "1/1/2025" or "2024-01-01")
        day              – day name  (e.g. "Monday")
        station_name     – stop name (must match a known route station)
        first_ticket_time – time string "HH:MM" or "H:MM"
        boarding         – integer count of passengers boarding
        deboarding       – integer count of passengers deboarding

    The function is designed to work on both single-row and multi-row
    DataFrames.  Lag/rolling features are computed using .shift() so they
    are meaningful when a full trip's rows are passed in order; for a
    single-row call they are safely filled with 0.

    Returns:
        A new DataFrame containing exactly the 37 model feature columns.
    """
    df = df.copy()

    # ── 1. Time features ────────────────────────────────────────────────────
    # Parse "HH:MM" or "H:MM" from first_ticket_time
    time_parts = df["first_ticket_time"].astype(str).str.split(":", expand=True)
    df["hour"] = time_parts[0].astype(int)
    df["minute"] = time_parts[1].astype(int)

    # Cyclical encoding (captures the circular nature of the clock)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Time-of-day category: 0=early-morn, 1=morning, 2=afternoon, 3=evening, 4=night
    conditions = [
        df["hour"] < 6,
        df["hour"] < 12,
        df["hour"] < 17,
        df["hour"] < 21,
    ]
    choices = [0, 1, 2, 3]
    df["time_of_day"] = np.select(conditions, choices, default=4)

    df["is_peak_am"]    = df["hour"].between(7, 10).astype(int)
    df["is_peak_pm"]    = df["hour"].between(17, 20).astype(int)
    df["is_early_morn"] = (df["hour"] < 6).astype(int)
    df["is_night"]      = (df["hour"] >= 21).astype(int)

    # ── 2. Day features ──────────────────────────────────────────────────────
    df["day_num"]    = df["day"].map(_DAY_MAP).fillna(0).astype(int)
    df["is_weekend"] = df["day_num"].isin([5, 6]).astype(int)
    df["is_weekday"] = (1 - df["is_weekend"])

    # ── 3. Stop / station features ───────────────────────────────────────────
    df["stop_sequence"] = (
        df["station_name"].map(_STOP_SEQ_MAP).fillna(1).astype(int)
    )
    df["stop_position_ratio"] = df["stop_sequence"] / _TOTAL_STOPS
    df["is_first_stop"] = (df["stop_sequence"] == 1).astype(int)
    df["is_last_stop"]  = (df["stop_sequence"] == _TOTAL_STOPS).astype(int)
    df["is_mid_stop"]   = (
        (df["is_first_stop"] == 0) & (df["is_last_stop"] == 0)
    ).astype(int)
    df["station_encoded"] = (
        df["station_name"].map(_STATION_ENCODED_MAP).fillna(0).astype(int)
    )

    # ── 4. Flow features ─────────────────────────────────────────────────────
    df["boarding"]   = pd.to_numeric(df["boarding"],   errors="coerce").fillna(0)
    df["deboarding"] = pd.to_numeric(df["deboarding"], errors="coerce").fillna(0)
    df["net_flow"]   = df["boarding"] - df["deboarding"]
    df["total_flow"] = df["boarding"] + df["deboarding"]
    df["board_ratio"]   = df["boarding"]   / (df["total_flow"] + 1e-9)
    df["deboard_ratio"] = df["deboarding"] / (df["total_flow"] + 1e-9)

    # ── 5. In-trip cumulative features ───────────────────────────────────────
    df["passengers_in_bus"] = df["net_flow"].cumsum()
    df["cum_boarding"]      = df["boarding"].cumsum()
    df["cum_deboarding"]    = df["deboarding"].cumsum()

    # ── 6. Lag features (filled with 0 when no prior rows exist) ────────────
    df["prev_boarding"]   = df["boarding"].shift(1).fillna(0)
    df["prev_deboarding"] = df["deboarding"].shift(1).fillna(0)
    df["prev_flow"]       = df["net_flow"].shift(1).fillna(0)
    df["prev2_flow"]      = df["net_flow"].shift(2).fillna(0)
    df["prev3_flow"]      = df["net_flow"].shift(3).fillna(0)
    df["prev_pax"]        = df["passengers_in_bus"].shift(1).fillna(0)
    df["prev2_pax"]       = df["passengers_in_bus"].shift(2).fillna(0)

    # ── 7. Rolling features (filled with 0 when window not full) ────────────
    df["rolling_avg_3"] = (
        df["net_flow"].rolling(3, min_periods=1).mean().fillna(0)
    )
    df["rolling_avg_5"] = (
        df["net_flow"].rolling(5, min_periods=1).mean().fillna(0)
    )
    df["rolling_max_3"] = (
        df["net_flow"].rolling(3, min_periods=1).max().fillna(0)
    )

    return df[_MODEL_FEATURES]


# ---------------------------------------------------------------------------
# Model loading (with pickle compatibility shim for cross-version sklearn)
# ---------------------------------------------------------------------------

class _CompatUnpickler(pickle.Unpickler):
    """Rewrites moved sklearn module paths so older pkl files load in newer sklearn."""

    _RENAMES = {
        ("sklearn.ensemble.forest", "RandomForestRegressor"):
            ("sklearn.ensemble._forest", "RandomForestRegressor"),
        ("sklearn.ensemble.forest", "RandomForestClassifier"):
            ("sklearn.ensemble._forest", "RandomForestClassifier"),
        ("sklearn.tree.tree", "DecisionTreeRegressor"):
            ("sklearn.tree._classes", "DecisionTreeRegressor"),
        ("sklearn.tree.tree", "DecisionTreeClassifier"):
            ("sklearn.tree._classes", "DecisionTreeClassifier"),
    }

    def find_class(self, module, name):
        module, name = self._RENAMES.get((module, name), (module, name))
        return super().find_class(module, name)


def load_model() -> Optional[object]:
    """
    Load the trained ML model from disk.

    Tries joblib first (recommended for sklearn), then falls back to a
    compatibility-aware pickle unpickler that handles cross-version module renames.

    Returns:
        The loaded model object if available, None otherwise.
    """
    if not MODEL_PATH.exists():
        logger.warning(
            f"Model file not found at {MODEL_PATH}. "
            "Using fallback random predictions."
        )
        return None

    # 1) joblib — handles sklearn internal structures robustly
    if _HAS_JOBLIB:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded via joblib from {MODEL_PATH}")
            return model
        except Exception as exc:
            logger.debug(f"joblib load failed ({exc}), trying pickle shim…")

    # 2) Compatibility pickle unpickler
    try:
        with open(MODEL_PATH, "rb") as f:
            data = f.read()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _CompatUnpickler(io.BytesIO(data)).load()
        logger.info(f"Model loaded via compat-pickle from {MODEL_PATH}")
        return model
    except Exception as exc:
        logger.error(f"Error loading model: {exc}. Using fallback random predictions.")
        return None


def base_prediction(
    model: Optional[object],
    raw_input: Dict,
) -> float:
    """
    Generate base prediction from the trained model using a raw input dict.

    Args:
        model:     The loaded ML model (or None if not available).
        raw_input: Single-stop dict with keys:
                       station_name, boarding, deboarding,
                       first_ticket_time, day, date

    Returns:
        Base predicted passenger demand as a float.
        Falls back to a random value (50–200) when the model is unavailable
        or feature engineering fails.
    """
    if model is None:
        base = random.uniform(50, 200)
        logger.debug(f"No model — using fallback base prediction: {base:.1f}")
        return base

    try:
        df_raw = pd.DataFrame([raw_input])
        features = create_features(df_raw)
        prediction = model.predict(features)[0]
        logger.debug(
            f"Base prediction for {raw_input.get('station_name', '?')}: {prediction:.2f}"
        )
        return float(prediction)
    except Exception as exc:
        logger.warning(f"Model prediction failed: {exc}. Falling back to random.")
        return random.uniform(50, 200)


def adjust_with_events(
    base_demand: float,
    events_data: Optional[Dict] = None,
) -> Tuple[float, str]:
    """
    Adjust prediction based on nearby events.
    
    Args:
        base_demand: The base prediction to adjust
        events_data: Dictionary with event information:
            {
                "has_event": bool,
                "event_popularity": str,  # "low", "medium", "high"
                "distance_km": float,
            }
            
    Returns:
        Tuple of (adjusted_demand, adjustment_reason)
        
    Logic:
        - No event: no adjustment (1.0x multiplier)
        - Low popularity event: +5% (1.05x)
        - Medium popularity event: +15% (1.15x)
        - High popularity event: +30% (1.30x)
        - Nearby event (< 2 km): additional +10%
    """
    if events_data is None or not events_data.get("has_event", False):
        return base_demand, "no_event"
    
    multiplier = 1.0
    reason = "event_adjustment"
    
    # Adjust based on event popularity
    popularity = events_data.get("event_popularity", "low")
    if popularity == "high":
        multiplier *= 1.30
    elif popularity == "medium":
        multiplier *= 1.15
    elif popularity == "low":
        multiplier *= 1.05
    
    # Additional boost if event is nearby
    distance = events_data.get("distance_km", float("inf"))
    if distance < 2:
        multiplier *= 1.10
        reason = "nearby_event_adjustment"
    
    adjusted = base_demand * multiplier
    logger.debug(f"Event adjustment: {base_demand} -> {adjusted} ({reason})")
    return adjusted, reason


def adjust_with_realtime(
    base_demand: float,
    ticket_boarding_count: Optional[int] = None,
) -> Tuple[float, str]:
    """
    Adjust prediction based on recent real-time ticket boarding data.
    
    Args:
        base_demand: The base prediction to adjust
        ticket_boarding_count: Number of recent ticket boardings
        
    Returns:
        Tuple of (adjusted_demand, adjustment_reason)
        
    Logic:
        - Uses 15-minute rolling window of ticket data
        - Low activity (< 5): -20% (0.80x)
        - Medium activity (5-15): no change (1.0x)
        - High activity (> 15): +25% (1.25x)
        - Very high activity (> 30): +50% (1.50x)
    """
    if ticket_boarding_count is None:
        return base_demand, "no_realtime_data"
    
    multiplier = 1.0
    reason = "realtime_adjustment"
    
    if ticket_boarding_count < 5:
        multiplier = 0.80  # Low activity
        reason = "low_realtime_activity"
    elif ticket_boarding_count > 30:
        multiplier = 1.50  # Very high activity
        reason = "very_high_realtime_activity"
    elif ticket_boarding_count > 15:
        multiplier = 1.25  # High activity
        reason = "high_realtime_activity"
    # else: medium activity (5-15), no adjustment
    
    adjusted = base_demand * multiplier
    logger.debug(
        f"Real-time adjustment (boardings={ticket_boarding_count}): "
        f"{base_demand} -> {adjusted} ({reason})"
    )
    return adjusted, reason


def adjust_with_opencv(
    base_demand: float,
    occupancy_percentage: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Adjust prediction based on real-time bus occupancy via OpenCV detection.
    
    Args:
        base_demand: The base prediction to adjust
        occupancy_percentage: Current bus occupancy (0-100%)
        
    Returns:
        Tuple of (adjusted_demand, adjustment_reason)
        
    Logic:
        - Occupancy < 50%: no adjustment (1.0x)
        - Occupancy 50-80%: +10% (1.10x)
        - Occupancy > 80%: +35% (1.35x)
        - Occupancy > 95%: +60% (1.60x) - bus nearly full
        
    Intuition:
        High occupancy suggests strong demand, so next bus will likely
        have similarly high demand.
    """
    if occupancy_percentage is None:
        return base_demand, "no_occupancy_data"
    
    multiplier = 1.0
    reason = "occupancy_adjustment"
    
    if occupancy_percentage > 95:
        multiplier = 1.60  # Bus nearly full
        reason = "critical_occupancy"
    elif occupancy_percentage > 80:
        multiplier = 1.35  # High occupancy threshold
        reason = "high_occupancy"
    elif occupancy_percentage > 50:
        multiplier = 1.10  # Moderate occupancy
        reason = "moderate_occupancy"
    # else: low occupancy, no adjustment
    
    adjusted = base_demand * multiplier
    logger.debug(
        f"Occupancy adjustment ({occupancy_percentage}%): "
        f"{base_demand} -> {adjusted} ({reason})"
    )
    return adjusted, reason


def predict_from_raw_input(
    raw_input: Dict,
    model: Optional[object] = None,
    events_data: Optional[Dict] = None,
    ticket_boarding_count: Optional[int] = None,
    occupancy_percentage: Optional[float] = None,
) -> Dict:
    """
    Full prediction pipeline: raw input → feature engineering → model →
    adjustments → structured result.

    Args:
        raw_input: Dict with keys:
            station_name      – bus stop name
            boarding          – passengers boarding at this stop
            deboarding        – passengers deboarding at this stop
            first_ticket_time – time string "HH:MM"
            day               – day name (e.g. "Monday")
            date              – date string (e.g. "2024-01-01")
        model:                Loaded ML model (auto-loaded if None).
        events_data:          Optional event context dict (see adjust_with_events).
        ticket_boarding_count: Optional recent boarding count (see adjust_with_realtime).
        occupancy_percentage: Optional current bus occupancy 0–100 (see adjust_with_opencv).

    Returns:
        {
            "stop_name":        str,
            "base_prediction":  float,
            "final_prediction": int,
            "adjustments": {
                "event":    float,
                "realtime": float,
                "opencv":   float,
            }
        }

    Raises:
        ValueError: if required fields are missing from raw_input.
    """
    # ── Validate required fields ─────────────────────────────────────────────
    required = {"station_name", "boarding", "deboarding", "first_ticket_time", "day", "date"}
    missing = required - set(raw_input.keys())
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")

    # ── Auto-load model ───────────────────────────────────────────────────────
    if model is None:
        model = load_model()

    # ── Step 1: Base prediction via feature engineering + model ──────────────
    base = base_prediction(model, raw_input)

    # ── Step 2: Apply contextual adjustments ─────────────────────────────────
    after_events,   _ = adjust_with_events(base, events_data)
    after_realtime, _ = adjust_with_realtime(after_events, ticket_boarding_count)
    after_opencv,   _ = adjust_with_opencv(after_realtime, occupancy_percentage)

    final = max(0, round(after_opencv))

    logger.info(
        f"Prediction for {raw_input['station_name']}: "
        f"base={base:.1f} → final={final}"
    )

    return {
        "stop_name":        raw_input["station_name"],
        "base_prediction":  round(base, 2),
        "final_prediction": final,
        "adjustments": {
            "event":    round(after_events - base, 2),
            "realtime": round(after_realtime - after_events, 2),
            "opencv":   round(after_opencv - after_realtime, 2),
        },
    }


def final_prediction(
    stop_name: str,
    hour: int,
    day_of_week: int,
    model: Optional[object] = None,
    events_data: Optional[Dict] = None,
    ticket_boarding_count: Optional[int] = None,
    occupancy_percentage: Optional[float] = None,
) -> Dict:
    """
    Legacy convenience wrapper kept for backward compatibility.

    Builds a minimal raw_input dict from the simplified arguments and
    delegates to predict_from_raw_input().
    """
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    raw_input = {
        "station_name": stop_name,
        "boarding": 0,
        "deboarding": 0,
        "first_ticket_time": f"{hour:02d}:00",
        "day": day_names[day_of_week % 7],
        "date": "2024-01-01",
    }
    result = predict_from_raw_input(
        raw_input,
        model=model,
        events_data=events_data,
        ticket_boarding_count=ticket_boarding_count,
        occupancy_percentage=occupancy_percentage,
    )
    # Return legacy-compatible shape
    base = result["base_prediction"]
    adj  = result["adjustments"]
    return {
        "stop_name":        result["stop_name"],
        "predicted_demand": result["final_prediction"],
        "base_prediction":  base,
        "adjustments": {
            "event_adjustment":     adj["event"],
            "realtime_adjustment":  adj["realtime"],
            "occupancy_adjustment": adj["opencv"],
        },
        "adjustment_reasons": [],
    }


# ---------------------------------------------------------------------------
# Standalone test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Bus Demand Predictor — Integration Test")
    print("=" * 60)

    # Load the model once
    _model = load_model()

    # ── Test A: single dict input ─────────────────────────────────────────
    print("\n[ Test A ] Single raw-input prediction")
    sample_input = {
        "station_name": "Yeshwanthapura T.T.M.C.",
        "boarding": 5,
        "deboarding": 1,
        "first_ticket_time": "08:15",
        "day": "Monday",
        "date": "2024-01-01",
    }

    try:
        result = predict_from_raw_input(
            raw_input=sample_input,
            model=_model,
            events_data={"has_event": True, "event_popularity": "medium", "distance_km": 1.2},
            ticket_boarding_count=20,
            occupancy_percentage=75,
        )
        print(f"  Stop name        : {result['stop_name']}")
        print(f"  Base prediction  : {result['base_prediction']}")
        print(f"  Final prediction : {result['final_prediction']} passengers")
        print(f"  Adjustments      : {result['adjustments']}")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # ── Test B: load a row from the test CSV ──────────────────────────────
    print("\n[ Test B ] Prediction from test_data CSV row")
    if TEST_DATA_PATH.exists():
        try:
            df_test = pd.read_csv(TEST_DATA_PATH, nrows=5)
            row = df_test.iloc[0].to_dict()
            print(f"  Raw row: {row}")
            result_csv = predict_from_raw_input(raw_input=row, model=_model)
            print(f"  Stop name        : {result_csv['stop_name']}")
            print(f"  Base prediction  : {result_csv['base_prediction']}")
            print(f"  Final prediction : {result_csv['final_prediction']} passengers")
            print(f"  Adjustments      : {result_csv['adjustments']}")
        except Exception as exc:
            print(f"  ERROR loading/predicting from CSV: {exc}")
    else:
        print(f"  Test data not found at {TEST_DATA_PATH} — skipping CSV test.")

    print("\n" + "=" * 60)
