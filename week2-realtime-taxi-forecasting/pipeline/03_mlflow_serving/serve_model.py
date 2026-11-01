import json
import os
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SCRIPT_DIR = Path(__file__).resolve().parent
PLANNING_AREA_ID_MAP_PATH = Path(
    os.getenv(
        "PLANNING_AREA_ID_MAP_PATH",
        SCRIPT_DIR / "planning_area_id_map.csv",
    )
)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "taxi_availability_2h_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
PREDICTION_LOG_PATH = Path(
    os.getenv("PREDICTION_LOG_PATH", SCRIPT_DIR / "prediction_logs.db")
)

mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()
model = mlflow.pyfunc.load_model(MODEL_URI)


class PredictionRequest(BaseModel):
    features: Optional[Dict[str, Any]] = None
    records: Optional[List[Dict[str, Any]]] = None


app = FastAPI(title="Taxi Availability Prediction API")


def get_model_metadata():
    metadata = {
        "model_uri": MODEL_URI,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": None,
        "run_id": None,
        "expected_features": None,
    }

    try:
        model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        metadata["model_version"] = model_version.version
        metadata["run_id"] = model_version.run_id

        if model_version.run_id:
            run = client.get_run(model_version.run_id)
            features = run.data.params.get("features")
            if features:
                metadata["expected_features"] = features.split(",")
    except Exception:
        pass

    return metadata


MODEL_METADATA = get_model_metadata()


def load_planning_area_id_map():
    if not PLANNING_AREA_ID_MAP_PATH.exists():
        return {}

    area_map = pd.read_csv(PLANNING_AREA_ID_MAP_PATH)
    return {
        str(row.planning_area).upper(): int(row.planning_area_id)
        for row in area_map.itertuples(index=False)
    }


def prepare_records(payload: PredictionRequest):
    if payload.records:
        records = payload.records
    elif payload.features:
        records = [payload.features]
    else:
        raise HTTPException(
            status_code=400,
            detail="Send either 'features' for one prediction or 'records' for many predictions.",
        )

    df = pd.DataFrame(records)

    if "is_weekend" not in df.columns and "day_of_week" in df.columns:
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    if "planning_area_id" not in df.columns and "planning_area" in df.columns:
        area_id_map = load_planning_area_id_map()
        df["planning_area_id"] = (
            df["planning_area"].astype(str).str.upper().map(area_id_map)
        )

    if "is_rain_forecast_2h" not in df.columns and "forecast_2h" in df.columns:
        df["is_rain_forecast_2h"] = (
            df["forecast_2h"]
            .fillna("")
            .str.lower()
            .str.contains("rain|showers|thundery")
            .astype(int)
        )

    if "current_rainfall_5min_mm" not in df.columns:
        df["current_rainfall_5min_mm"] = 0.0

    expected_features = MODEL_METADATA.get("expected_features")
    if expected_features:
        missing = [name for name in expected_features if name not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required feature columns: {missing}",
            )
        df = df[expected_features]
        null_features = [
            name for name in expected_features if df[name].isna().any()
        ]
        if null_features:
            raise HTTPException(
                status_code=400,
                detail=f"Feature columns contain missing values: {null_features}",
            )

    return records, df


def ensure_log_table():
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(PREDICTION_LOG_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                model_uri TEXT NOT NULL,
                model_name TEXT,
                model_alias TEXT,
                model_version TEXT,
                input_json TEXT NOT NULL,
                prediction_json TEXT NOT NULL
            )
            """
        )


def log_prediction(input_records, predictions):
    ensure_log_table()
    timestamp = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(PREDICTION_LOG_PATH) as con:
        con.execute(
            """
            INSERT INTO prediction_logs (
                timestamp_utc,
                model_uri,
                model_name,
                model_alias,
                model_version,
                input_json,
                prediction_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                MODEL_METADATA["model_uri"],
                MODEL_METADATA["model_name"],
                MODEL_METADATA["model_alias"],
                MODEL_METADATA["model_version"],
                json.dumps(input_records),
                json.dumps(predictions),
            ),
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tracking_uri": TRACKING_URI,
        **MODEL_METADATA,
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    input_records, model_input = prepare_records(payload)
    raw_predictions = [float(value) for value in model.predict(model_input).tolist()]
    predictions = [int(round(value)) for value in raw_predictions]
    log_prediction(input_records, predictions)

    return {
        "predictions": predictions,
        "raw_predictions": [round(value, 3) for value in raw_predictions],
        "prediction_rounding": "nearest_integer",
        "model_uri": MODEL_METADATA["model_uri"],
        "model_name": MODEL_METADATA["model_name"],
        "model_alias": MODEL_METADATA["model_alias"],
        "model_version": MODEL_METADATA["model_version"],
        "logged_to": str(PREDICTION_LOG_PATH),
    }
