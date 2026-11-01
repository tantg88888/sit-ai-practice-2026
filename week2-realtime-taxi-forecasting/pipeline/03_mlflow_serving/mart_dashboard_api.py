from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


APP_DIR = Path(__file__).resolve().parent
DAY4_DIR = APP_DIR.parent
CLEAN_DIR = DAY4_DIR / "clean"
DEFAULT_LIVE_MART_DB = CLEAN_DIR / "dbt_taxi_features" / "outputs" / "taxi_features_live.db"
MART_DB_PATH = Path(
    os.getenv(
        "TAXI_FEATURE_MART_DB_PATH",
        DEFAULT_LIVE_MART_DB,
    )
)
PIPELINE_SCRIPT = CLEAN_DIR / "run_live_feature_pipeline.py"
DASHBOARD_PAGE = APP_DIR / "mart_dashboard.html"
PREDICTION_LOG_PATH = Path(
    os.getenv("MART_DASHBOARD_PREDICTION_LOG_PATH", APP_DIR / "mart_prediction_logs.jsonl")
)
API_LOG_PATH = Path(os.getenv("MART_DASHBOARD_API_LOG_PATH", APP_DIR / "mart_dashboard_api.log"))
PIPELINE_LOG_PATH = Path(
    os.getenv("MART_DASHBOARD_PIPELINE_LOG_PATH", APP_DIR / "mart_live_pipeline_last.log")
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/taxi_availability_2h_model@champion",
)

FEATURE_COLUMNS = [
    "planning_area_id",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_rain_forecast_2h",
    "current_rainfall_5min_mm",
    "current_taxi_count",
]
DEFAULT_AREA = "ANG MO KIO"

app = FastAPI(title="Day 4 Taxi Mart Dashboard API")
_model_cache: dict[str, Any] = {}

API_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(API_LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("mart_dashboard")


def live_mart_missing_error() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "message": "Live serving mart is not available yet.",
            "mart_db_path": str(MART_DB_PATH),
            "suggested_fix": "Click 'Collect + rebuild mart' or call POST /api/pipeline/run-once.",
        },
    )


def query_mart(sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    if not MART_DB_PATH.exists():
        logger.warning("live mart missing path=%s", MART_DB_PATH)
        raise live_mart_missing_error()

    with sqlite3.connect(f"file:{MART_DB_PATH}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params or []).fetchall()
    return [dict(row) for row in rows]


def table_exists(table_name: str) -> bool:
    rows = query_mart(
        """
        select 1 as exists_flag
        from sqlite_master
        where type = 'table' and name = ?
        limit 1
        """,
        [table_name],
    )
    return bool(rows)


def feature_relation() -> str:
    if not table_exists("fct_taxi_serving_features"):
        raise live_mart_missing_error()
    return "fct_taxi_serving_features"


def latest_feature_row(area: str) -> dict[str, Any]:
    area = area.strip() or DEFAULT_AREA
    rows = query_mart(
        f"""
        select *
        from {feature_relation()}
        where upper(planning_area) = upper(?)
        order by timestamp desc
        limit 1
        """,
        [area],
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"No serving feature row for area: {area}")
    return rows[0]


def load_model():
    if "model" not in _model_cache:
        import mlflow

        logger.info("loading model tracking_uri=%s model_uri=%s", MLFLOW_TRACKING_URI, MLFLOW_MODEL_URI)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        start = time.perf_counter()
        try:
            _model_cache["model"] = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
        except Exception as exc:
            logger.exception("model load failed model_uri=%s", MLFLOW_MODEL_URI)
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Failed to load MLflow champion model.",
                    "model_uri": MLFLOW_MODEL_URI,
                    "suggested_fix": (
                        "Train and register a model again, then reload. "
                        "For example: python train_linear_regression_mlflow.py "
                        "--data-version v1 --feature-version v1 --register-model "
                        "--model-name taxi_availability_2h_model --model-alias champion"
                    ),
                    "original_error": str(exc),
                },
            ) from exc
        _model_cache["loaded_at"] = datetime.now().isoformat()
        _model_cache["load_duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return _model_cache["model"]


def model_info(resolve_registry: bool = False) -> dict[str, Any]:
    info = {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_uri": MLFLOW_MODEL_URI,
        "loaded_at": _model_cache.get("loaded_at"),
        "load_duration_ms": _model_cache.get("load_duration_ms"),
    }

    if (
        not resolve_registry
        or not MLFLOW_MODEL_URI.startswith("models:/")
        or "@" not in MLFLOW_MODEL_URI
    ):
        return info

    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        model_ref = MLFLOW_MODEL_URI.removeprefix("models:/")
        model_name, alias = model_ref.split("@", 1)
        version = client.get_model_version_by_alias(model_name, alias)
        info.update(
            {
                "model_name": model_name,
                "model_alias": alias,
                "model_version": version.version,
                "run_id": version.run_id,
            }
        )
    except Exception as exc:
        info["registry_error"] = str(exc)

    return info


def append_prediction_log(row: dict[str, Any]) -> None:
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_prediction_logs(area: str, limit: int = 200) -> list[dict[str, Any]]:
    if not PREDICTION_LOG_PATH.exists():
        return []

    rows = []
    with PREDICTION_LOG_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("area", "")).upper() == area.upper():
                rows.append(row)

    return rows[-limit:]


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD_PAGE)


@app.get("/api/health")
def health() -> dict[str, Any]:
    relation = None
    relation_error = None
    if MART_DB_PATH.exists():
        try:
            relation = feature_relation()
        except HTTPException as exc:
            relation_error = exc.detail

    return {
        "status": "ok",
        "mart_db_path": str(MART_DB_PATH),
        "mart_db_exists": MART_DB_PATH.exists(),
        "feature_relation": relation,
        "feature_relation_error": relation_error,
        "api_log_path": str(API_LOG_PATH),
        "pipeline_log_path": str(PIPELINE_LOG_PATH),
        "feature_columns": FEATURE_COLUMNS,
        "model": model_info(),
    }


@app.get("/api/areas")
def areas() -> list[dict[str, Any]]:
    return query_mart(
        f"""
        select distinct planning_area
        from {feature_relation()}
        order by planning_area
        """
    )


@app.get("/api/dashboard-data")
def dashboard_data(area: str = "ANG MO KIO", limit: int = 36) -> dict[str, Any]:
    area = area.strip() or DEFAULT_AREA
    latest = latest_feature_row(area)
    history = query_mart(
        f"""
        select
            timestamp,
            prediction_target_timestamp,
            planning_area,
            current_taxi_count,
            forecast_2h,
            is_rain_forecast_2h,
            current_rainfall_5min_mm
        from {feature_relation()}
        where upper(planning_area) = upper(?)
        order by timestamp desc
        limit ?
        """,
        [area, limit],
    )
    history.reverse()
    predictions = read_prediction_logs(latest["planning_area"])
    return {
        "area": latest["planning_area"],
        "latest": latest,
        "history": history,
        "predictions": predictions,
        "model": model_info(),
    }


@app.get("/api/model/predict-2h")
def predict_2h(area: str = "ANG MO KIO") -> dict[str, Any]:
    area = area.strip() or DEFAULT_AREA
    source_row = latest_feature_row(area)
    features = {name: source_row[name] for name in FEATURE_COLUMNS}
    model_input = pd.DataFrame([features])

    start = time.perf_counter()
    model = load_model()
    raw_prediction = float(model.predict(model_input)[0])
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    rounded_prediction = int(round(raw_prediction))

    response = {
        "area": source_row["planning_area"],
        "planning_area_id": int(source_row["planning_area_id"]),
        "timestamp": source_row["timestamp"],
        "prediction_target_timestamp": source_row["prediction_target_timestamp"],
        "current_taxi_count": int(source_row["current_taxi_count"]),
        "forecast_2h": source_row["forecast_2h"],
        "current_rainfall_5min_mm": float(source_row["current_rainfall_5min_mm"]),
        "predicted_taxi_count_2h": rounded_prediction,
        "raw_predicted_taxi_count_2h": round(raw_prediction, 3),
        "prediction_rounding": "nearest_integer",
        "features": features,
        "inference_duration_ms": duration_ms,
        "logged_at": datetime.now().isoformat(),
        "model": model_info(resolve_registry=True),
    }
    append_prediction_log(response)
    return response


@app.post("/api/model/reload")
def reload_model() -> dict[str, Any]:
    _model_cache.clear()
    load_model()
    return model_info(resolve_registry=True)


@app.post("/api/pipeline/run-once")
def run_pipeline_once() -> dict[str, Any]:
    if not PIPELINE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail=f"Pipeline script not found: {PIPELINE_SCRIPT}")

    start = time.perf_counter()
    logger.info("live pipeline start script=%s", PIPELINE_SCRIPT)
    result = subprocess.run(
        [sys.executable, str(PIPELINE_SCRIPT)],
        cwd=str(CLEAN_DIR),
        capture_output=True,
        text=True,
        timeout=300,
    )
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    PIPELINE_LOG_PATH.write_text(
        "\n".join(
            [
                f"returncode={result.returncode}",
                f"duration_ms={duration_ms}",
                "===== STDOUT =====",
                result.stdout,
                "===== STDERR =====",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )

    if result.returncode != 0:
        logger.error("live pipeline failed duration_ms=%s log_path=%s", duration_ms, PIPELINE_LOG_PATH)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Live feature pipeline failed.",
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
                "duration_ms": duration_ms,
            },
        )

    logger.info("live pipeline done duration_ms=%s log_path=%s", duration_ms, PIPELINE_LOG_PATH)
    return {
        "status": "ok",
        "duration_ms": duration_ms,
        "mart_db_path": str(MART_DB_PATH),
        "pipeline_log_path": str(PIPELINE_LOG_PATH),
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }
