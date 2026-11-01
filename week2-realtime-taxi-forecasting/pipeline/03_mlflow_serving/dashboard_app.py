from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
import traceback

import duckdb
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR
# For the Day 4.2 demo, the scheduler writes a local SQLite database here.
# Keeping this default hard-coded makes the classroom commands shorter.
DB_PATH = Path(os.getenv("DASHBOARD_DB_PATH", APP_DIR / "live_dashboard_data.db"))
DB_BACKEND = "sqlite"
MLFLOW_WORK_DIR = APP_DIR

REVIEW_PAGE = APP_DIR / "dashboard_review.html"
PREDICTION_PAGE = APP_DIR / "dashboard_prediction.html"
PREDICTION_LOG_PATH = Path(
    os.getenv("PREDICTION_LOG_PATH", APP_DIR / "prediction_logs.jsonl")
)

SINGAPORE_TZ = timezone(timedelta(hours=8))

LOG_PATH = Path(os.getenv("DASHBOARD_LOG_PATH", APP_DIR / "dashboard_debug.log"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("day4.dashboard")

# Keep MLflow client failures reasonably quick when the local server is stopped.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")


def debug_print(message: str, **values: Any) -> None:
    """Print highly visible debug lines in the uvicorn terminal."""
    if values:
        details = " ".join(f"{key}={value!r}" for key, value in values.items())
        print(f"[DAY4_DEBUG] {message} | {details}", flush=True)
    else:
        print(f"[DAY4_DEBUG] {message}", flush=True)


# ============================================================================
# MLflow model loading configuration
# ============================================================================
#
# This dashboard is a model-serving app. It does not train a model.
# Instead, it asks MLflow:
#
#   1. Which Tracking Server should I talk to?
#   2. Which registered model should I load?
#   3. Which version of that model is currently selected for serving?
#
# MLFLOW_TRACKING_URI points to the MLflow Tracking Server / Model Registry.
# In the classroom setup, the MLflow server is usually running at localhost:5000.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# MLFLOW_MODEL_URI is the model pointer used by the application.
#
# models:/taxi_availability_2h_model@champion means:
#   - registered model name: taxi_availability_2h_model
#   - alias: champion
#   - actual version: whichever model version currently has the champion alias
#
# This is the key serving idea:
# the application code can stay stable while MLflow changes which model version
# is marked as champion.
MLFLOW_MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/taxi_availability_2h_model@champion",
)

debug_print(
    "dashboard_app imported",
    python=sys.executable,
    cwd=os.getcwd(),
    app_dir=str(APP_DIR),
    db_path=str(DB_PATH),
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_model_uri=MLFLOW_MODEL_URI,
)

# The model was trained with a specific feature contract.
# Day 4.2 still creates these features manually in the serving app.
# Day 4.3 improves this by introducing Feast as a feature store.
EXPECTED_FEATURE_VERSION = "v1"
EXPECTED_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rain_forecast_2h",
    "weather_category_id",
    "current_taxi_count",
]

WEATHER_CATEGORY_IDS = {
    "Unknown": 0,
    "Cloudy": 1,
    "Mist": 2,
    "Partly Cloudy (Day)": 3,
    "Partly Cloudy (Night)": 4,
    "Light Rain": 5,
    "Moderate Rain": 6,
    "Light Showers": 7,
    "Showers": 8,
    "Thundery Showers": 9,
    "Heavy Thundery Showers with Gusty Winds": 10,
    "Windy": 11,
}

app = FastAPI(title="Day 4.2 Taxi Dashboard With MLflow Model")

# ============================================================================
# MLflow model cache
# ============================================================================
#
# Loading a model from MLflow can take time because MLflow may need to:
#   - contact the Tracking Server / Model Registry;
#   - resolve the alias champion to a concrete model version;
#   - download/read model artifacts;
#   - reconstruct the Python model object.
#
# We cache the loaded model after the first prediction request.
# The /api/model/reload endpoint clears this cache when we intentionally move
# the champion alias to a new model version.
_model_cache: dict[str, Any] = {}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Write one log line for every API request so errors are easier to trace."""
    started = time.perf_counter()
    debug_print("request_start", path=request.url.path, query=request.url.query)
    logger.info("request_start path=%s query=%s", request.url.path, request.url.query)
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        debug_print(
            "request_failed",
            path=request.url.path,
            duration_ms=duration_ms,
        )
        logger.exception(
            "request_failed path=%s duration_ms=%s",
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "request_end path=%s status=%s duration_ms=%s",
        request.url.path,
        response.status_code,
        duration_ms,
    )
    debug_print(
        "request_end",
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return JSON for unexpected errors instead of plain text / HTML."""
    debug_print(
        "unhandled_exception",
        path=request.url.path,
        exception_type=type(exc).__name__,
        detail=str(exc),
    )
    logger.exception("unhandled_exception path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "path": request.url.path,
            "exception_type": type(exc).__name__,
            "detail": str(exc),
            "debug_log_path": str(LOG_PATH),
            "traceback_tail": traceback.format_exception(
                type(exc),
                exc,
                exc.__traceback__,
                limit=6,
            ),
        },
    )


def query_duckdb(sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    """Run a read-only query against the Day 1 DuckDB database."""
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        df = con.execute(sql, params or []).df()

    # FastAPI can return normal strings more easily than timestamp objects.
    for column in df.columns:
        if "timestamp" in column or column.startswith("valid_period"):
            df[column] = df[column].astype(str)

    return df.to_dict(orient="records")


def query_sqlite(sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    """Run a read-only query against the local live SQLite database."""
    if not DB_PATH.exists():
        debug_print("sqlite_db_missing", db_path=str(DB_PATH))
        raise HTTPException(status_code=500, detail=f"Database not found: {DB_PATH}")

    debug_print(
        "sqlite_query_start",
        db_path=str(DB_PATH),
        params=params,
        sql_preview=" ".join(sql.split())[:180],
    )
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params or []).fetchall()
    debug_print("sqlite_query_done", row_count=len(rows))
    return [dict(row) for row in rows]


def query_database(
    duckdb_sql: str,
    sqlite_sql: str | None = None,
    params: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Use DuckDB for the classroom snapshot or SQLite for live collection."""
    if DB_BACKEND == "sqlite":
        return query_sqlite(sqlite_sql or duckdb_sql, params)
    return query_duckdb(duckdb_sql, params)


def parse_timestamp(value: str) -> datetime:
    """Parse a timestamp and convert it to Singapore time."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SINGAPORE_TZ)
    return parsed.astimezone(SINGAPORE_TZ)


def is_rain_forecast(forecast: str) -> int:
    """Convert a weather text into a simple 0/1 rain feature."""
    text = (forecast or "").lower()
    return int(any(word in text for word in ["rain", "showers", "thundery"]))


def check_mlflow_server() -> None:
    """Fail fast with a useful message if the MLflow server is not reachable."""
    try:
        debug_print("mlflow_server_check_start", uri=MLFLOW_TRACKING_URI)
        response = requests.get(MLFLOW_TRACKING_URI, timeout=5)
        debug_print(
            "mlflow_server_check_done",
            uri=MLFLOW_TRACKING_URI,
            status=response.status_code,
            text_preview=response.text[:120],
        )
        logger.info(
            "mlflow_server_check status=%s uri=%s",
            response.status_code,
            MLFLOW_TRACKING_URI,
        )
    except requests.RequestException as exc:
        debug_print(
            "mlflow_server_unreachable",
            uri=MLFLOW_TRACKING_URI,
            error=str(exc),
        )
        logger.exception("mlflow_server_unreachable uri=%s", MLFLOW_TRACKING_URI)
        raise HTTPException(
            status_code=503,
            detail={
                "message": "MLflow Tracking Server is not reachable.",
                "tracking_uri": MLFLOW_TRACKING_URI,
                "suggested_fix": (
                    "Start the unified Day 4 container from day_4/day_4_mlflow "
                    "with: docker compose up --build"
                ),
                "original_error": str(exc),
            },
        ) from exc


def resolve_model_load_uri(metadata: dict[str, Any]) -> str:
    """Resolve a Registry source into a path this local dashboard can load."""
    registry_source = (
        metadata.get("model_source")
        or metadata.get("storage_location")
        or MLFLOW_MODEL_URI
    )

    debug_print("resolve_model_load_uri_start", registry_source=registry_source)

    if isinstance(registry_source, str) and (
        registry_source.startswith("/mlflow/") or registry_source.startswith("/workspace/")
    ):
        docker_path = Path(registry_source)
        debug_print(
            "resolve_model_load_uri_check_docker_path",
            registry_source=registry_source,
            exists=docker_path.exists(),
        )
        if docker_path.exists():
            return registry_source

        relative_path = (
            registry_source
            .removeprefix("/mlflow/")
            .removeprefix("/workspace/")
            .replace("/", os.sep)
        )
        local_path = MLFLOW_WORK_DIR / relative_path
        debug_print(
            "resolve_model_load_uri_local_fallback",
            registry_source=registry_source,
            local_path=str(local_path),
            exists=local_path.exists(),
        )
        if local_path.exists():
            return str(local_path)

    debug_print("resolve_model_load_uri_use_original", load_uri=MLFLOW_MODEL_URI)
    return MLFLOW_MODEL_URI


def load_mlflow_model():
    """Load the registered MLflow model once, then reuse it for predictions."""
    if "model" not in _model_cache:
        debug_print(
            "mlflow_load_start",
            tracking_uri=MLFLOW_TRACKING_URI,
            model_uri=MLFLOW_MODEL_URI,
            python=sys.executable,
        )
        logger.info(
            "mlflow_load_start tracking_uri=%s model_uri=%s",
            MLFLOW_TRACKING_URI,
            MLFLOW_MODEL_URI,
        )
        try:
            import mlflow
            debug_print("mlflow_import_ok", mlflow_version=getattr(mlflow, "__version__", None))
        except ModuleNotFoundError as exc:
            debug_print("mlflow_import_failed", python=sys.executable, error=str(exc))
            raise HTTPException(
                status_code=500,
                detail="MLflow is not installed in this Python environment.",
            ) from exc

        check_mlflow_server()

        # Step 1: point this Python process to the MLflow Tracking Server.
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Step 2a: ask the Registry which concrete model version champion means.
        metadata = get_model_registry_metadata()
        model_load_uri = resolve_model_load_uri(metadata)

        # Step 2: load the registered model through the MLflow Models URI.
        #
        # With the default URI, MLflow resolves:
        #
        #   models:/taxi_availability_2h_model@champion
        #
        # into a concrete model version, then loads the saved model artifact.
        debug_print(
            "mlflow_pyfunc_load_model_start",
            model_uri=MLFLOW_MODEL_URI,
            model_load_uri=model_load_uri,
        )
        _model_cache["model"] = mlflow.pyfunc.load_model(model_load_uri)
        debug_print("mlflow_pyfunc_load_model_done", model_type=type(_model_cache["model"]).__name__)

        # Step 3: keep serving metadata next to the cached model.
        # This lets the API response and prediction log show which model version
        # actually produced each prediction.
        _model_cache["model_uri"] = MLFLOW_MODEL_URI
        metadata["resolved_model_load_uri"] = model_load_uri
        _model_cache["metadata"] = metadata
        debug_print("mlflow_load_done", metadata=_model_cache["metadata"])
        logger.info("mlflow_load_done metadata=%s", _model_cache["metadata"])
    else:
        debug_print("mlflow_load_from_cache", cache_keys=sorted(_model_cache.keys()))

    return _model_cache["model"]


def get_model_registry_metadata() -> dict[str, Any]:
    """Describe the registry version currently selected by the model URI."""
    metadata = {
        "loaded_model_uri": _model_cache.get("model_uri"),
        "model_name": None,
        "model_alias": None,
        "model_version": None,
        "run_id": None,
        "run_name": None,
        "model_type": None,
        "model_source": None,
        "storage_location": None,
    }

    if not MLFLOW_MODEL_URI.startswith("models:/") or "@" not in MLFLOW_MODEL_URI:
        return metadata

    # Parse the model URI so we can ask MLflow Registry what alias points to.
    model_reference = MLFLOW_MODEL_URI.removeprefix("models:/")
    model_name, model_alias = model_reference.split("@", 1)
    metadata["model_name"] = model_name
    metadata["model_alias"] = model_alias

    try:
        import mlflow
    except ModuleNotFoundError:
        return metadata

    try:
        # The Registry lookup answers:
        # "For this registered model, which version currently has this alias?"
        logger.info(
            "mlflow_registry_lookup_start model_name=%s alias=%s",
            model_name,
            model_alias,
        )
        debug_print(
            "mlflow_registry_lookup_start",
            model_name=model_name,
            alias=model_alias,
        )
        check_mlflow_server()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version_by_alias(model_name, model_alias)
        debug_print(
            "mlflow_registry_lookup_done",
            model_name=model_name,
            alias=model_alias,
            version=model_version.version,
            run_id=model_version.run_id,
            source=model_version.source,
        )
        metadata["model_version"] = model_version.version
        metadata["run_id"] = model_version.run_id
        metadata["model_source"] = model_version.source
        metadata["storage_location"] = getattr(model_version, "storage_location", None)

        if model_version.run_id:
            run = client.get_run(model_version.run_id)
            metadata["run_name"] = run.data.tags.get("mlflow.runName")
            metadata["model_type"] = run.data.params.get("model_type")
    except Exception as exc:
        debug_print(
            "mlflow_registry_lookup_failed",
            exception_type=type(exc).__name__,
            error=str(exc),
        )
        logger.exception("mlflow_registry_lookup_failed")
        metadata["registry_error"] = str(exc)

    return metadata


def append_prediction_log(payload: dict[str, Any]) -> None:
    """Append one prediction event as JSON Lines for a simple monitoring log."""
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_recent_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read the newest JSON Lines records without loading the whole file."""
    if limit <= 0:
        return []

    rows: list[dict[str, Any]] = []
    with path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        position = file.tell()
        buffer = b""

        while position > 0 and len(rows) < limit:
            read_size = min(8192, position)
            position -= read_size
            file.seek(position)
            buffer = file.read(read_size) + buffer
            lines = buffer.splitlines()

            if position > 0:
                buffer = lines[0]
                lines = lines[1:]

            for line in reversed(lines):
                if not line.strip():
                    continue
                rows.append(json.loads(line.decode("utf-8")))
                if len(rows) >= limit:
                    break

        return list(reversed(rows))


def parse_input_features(row: dict[str, Any]) -> dict[str, Any]:
    features = row.get("input_features")
    if isinstance(features, dict):
        return features

    features_json = row.get("input_features_json")
    if isinstance(features_json, str) and features_json:
        try:
            parsed = json.loads(features_json)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {
        name: row.get(name)
        for name in EXPECTED_FEATURES
        if row.get(name) is not None
    }


def get_actual_taxi_count_near(area: str, target_timestamp: datetime) -> dict[str, Any] | None:
    rows = query_database(
        """
        SELECT api_timestamp, available_taxi_count
        FROM taxi_availability_typed
        WHERE UPPER(planning_area) = UPPER(?)
          AND api_timestamp >= CAST(? AS TIMESTAMPTZ)
        ORDER BY api_timestamp
        LIMIT 1
        """,
        """
        SELECT api_timestamp, available_taxi_count
        FROM taxi_availability
        WHERE UPPER(planning_area) = UPPER(?)
          AND api_timestamp >= ?
        ORDER BY api_timestamp
        LIMIT 1
        """,
        [area.upper(), target_timestamp.isoformat()],
    )
    if not rows:
        return None

    actual_timestamp = parse_timestamp(rows[0]["api_timestamp"])
    if actual_timestamp - target_timestamp > timedelta(minutes=10):
        return None

    return {
        "actual_timestamp": rows[0]["api_timestamp"],
        "actual_taxi_count": int(rows[0]["available_taxi_count"]),
    }


def get_latest_feature_row(area: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Collect the latest taxi/weather data and turn it into model features."""
    debug_print("feature_build_start", area=area)
    logger.info("feature_build_start area=%s", area)
    rows = query_database(
        """
        WITH latest_taxi AS (
            SELECT api_timestamp, planning_area, available_taxi_count
            FROM taxi_availability_typed
            WHERE UPPER(planning_area) = UPPER(?)
            ORDER BY api_timestamp DESC
            LIMIT 1
        ), matched_weather AS (
            SELECT
                t.api_timestamp,
                t.planning_area,
                t.available_taxi_count,
                w.forecast,
                ROW_NUMBER() OVER (
                    ORDER BY w.api_update_timestamp DESC NULLS LAST
                ) AS match_rank
            FROM latest_taxi AS t
            LEFT JOIN weather_forecasts_typed AS w
            ON UPPER(TRIM(t.planning_area)) = UPPER(TRIM(w.area))
            AND t.api_timestamp >= w.valid_period_start
            AND t.api_timestamp < w.valid_period_end
            AND w.api_update_timestamp <= t.api_timestamp
        )
        SELECT api_timestamp, planning_area, available_taxi_count, forecast
        FROM matched_weather
        WHERE match_rank = 1
        """,
        """
        WITH latest_taxi AS (
            SELECT api_timestamp, planning_area, available_taxi_count
            FROM taxi_availability
            WHERE UPPER(planning_area) = UPPER(?)
            ORDER BY api_timestamp DESC
            LIMIT 1
        ), matched_weather AS (
            SELECT
                t.api_timestamp,
                t.planning_area,
                t.available_taxi_count,
                w.forecast,
                ROW_NUMBER() OVER (
                    ORDER BY (w.api_update_timestamp IS NULL), w.api_update_timestamp DESC
                ) AS match_rank
            FROM latest_taxi AS t
            LEFT JOIN weather_forecasts AS w
            ON UPPER(TRIM(t.planning_area)) = UPPER(TRIM(w.area))
            AND t.api_timestamp >= w.valid_period_start
            AND t.api_timestamp < w.valid_period_end
            AND w.api_update_timestamp <= t.api_timestamp
        )
        SELECT api_timestamp, planning_area, available_taxi_count, forecast
        FROM matched_weather
        WHERE match_rank = 1
        """,
        [area.upper()],
    )

    if not rows:
        debug_print("feature_build_no_rows", area=area)
        raise HTTPException(status_code=404, detail=f"No taxi data found for area: {area}")

    row = rows[0]
    debug_print("feature_source_row", row=row)
    logger.info(
        "feature_source_row area=%s taxi_timestamp=%s forecast=%s current_taxi_count=%s",
        row.get("planning_area"),
        row.get("api_timestamp"),
        row.get("forecast"),
        row.get("available_taxi_count"),
    )
    timestamp = parse_timestamp(row["api_timestamp"])
    forecast = row.get("forecast") or "Unknown"

    # These columns match feature version v1 from train_linear_regression_mlflow.py.
    features = pd.DataFrame(
        [
            {
                "hour": timestamp.hour,
                "day_of_week": timestamp.weekday(),
                "is_weekend": int(timestamp.weekday() in [5, 6]),
                "is_rain_forecast_2h": is_rain_forecast(forecast),
                "weather_category_id": WEATHER_CATEGORY_IDS.get(forecast, 0),
                "current_taxi_count": int(row["available_taxi_count"]),
            }
        ]
    )

    debug_print(
        "feature_build_done",
        columns=list(features.columns),
        values=features.iloc[0].to_dict(),
    )
    logger.info(
        "feature_build_done area=%s columns=%s values=%s",
        area,
        list(features.columns),
        features.iloc[0].to_dict(),
    )
    return row, features


@app.get("/")
def index() -> FileResponse:
    return FileResponse(REVIEW_PAGE)


@app.get("/review")
def review_dashboard() -> FileResponse:
    return FileResponse(REVIEW_PAGE)


@app.get("/predict-dashboard")
def prediction_dashboard() -> FileResponse:
    return FileResponse(PREDICTION_PAGE)


@app.get("/api/areas")
def areas() -> list[dict[str, Any]]:
    return query_database(
        """
        SELECT DISTINCT planning_area
        FROM taxi_availability_typed
        ORDER BY planning_area
        """,
        """
        SELECT DISTINCT planning_area
        FROM taxi_availability
        ORDER BY planning_area
        """,
    )


@app.get("/api/taxi/current")
def current_taxi(timestamp: str | None = None) -> list[dict[str, Any]]:
    return query_database(
        """
        SELECT
            api_timestamp,
            planning_area,
            available_taxi_count
        FROM taxi_availability_typed
        WHERE api_timestamp = COALESCE(
            CAST(? AS TIMESTAMPTZ),
            (SELECT MAX(api_timestamp) FROM taxi_availability_typed)
        )
        ORDER BY available_taxi_count DESC, planning_area
        """,
        """
        SELECT
            api_timestamp,
            planning_area,
            available_taxi_count
        FROM taxi_availability
        WHERE api_timestamp = COALESCE(
            ?,
            (SELECT MAX(api_timestamp) FROM taxi_availability)
        )
        ORDER BY available_taxi_count DESC, planning_area
        """,
        [timestamp],
    )


@app.get("/api/weather/current")
def current_weather(update_timestamp: str | None = None) -> list[dict[str, Any]]:
    return query_database(
        """
        SELECT
            api_update_timestamp,
            area,
            forecast,
            valid_period_start,
            valid_period_end
        FROM weather_forecasts_typed
        WHERE api_update_timestamp = COALESCE(
            CAST(? AS TIMESTAMPTZ),
            (SELECT MAX(api_update_timestamp) FROM weather_forecasts_typed)
        )
        ORDER BY area
        """,
        """
        SELECT
            api_update_timestamp,
            area,
            forecast,
            valid_period_start,
            valid_period_end
        FROM weather_forecasts
        WHERE api_update_timestamp = COALESCE(
            ?,
            (SELECT MAX(api_update_timestamp) FROM weather_forecasts)
        )
        ORDER BY area
        """,
        [update_timestamp],
    )


@app.get("/api/taxi/timeseries")
def taxi_timeseries(area: str = "ANG MO KIO") -> list[dict[str, Any]]:
    taxi_rows = query_database(
        """
        SELECT
            api_timestamp,
            available_taxi_count,
            planning_area
        FROM taxi_availability_typed
        WHERE UPPER(planning_area) = UPPER(?)
        ORDER BY api_timestamp
        """,
        """
        SELECT
            api_timestamp,
            available_taxi_count,
            planning_area
        FROM taxi_availability
        WHERE UPPER(planning_area) = UPPER(?)
        ORDER BY api_timestamp
        """,
        [area.upper()],
    )

    weather_rows = query_database(
        """
        SELECT
            api_update_timestamp,
            valid_period_start,
            valid_period_end,
            area,
            forecast
        FROM weather_forecasts_typed
        WHERE UPPER(area) = UPPER(?)
        ORDER BY api_update_timestamp
        """,
        """
        SELECT
            api_update_timestamp,
            valid_period_start,
            valid_period_end,
            area,
            forecast
        FROM weather_forecasts
        WHERE UPPER(area) = UPPER(?)
        ORDER BY api_update_timestamp
        """,
        [area.upper()],
    )

    if not taxi_rows:
        return []

    weather_df = pd.DataFrame(weather_rows)
    if weather_df.empty:
        for row in taxi_rows:
            row["current_forecast"] = None
            row["forecast_2h"] = None
            row["forecast"] = None
        return taxi_rows

    weather_df["api_update_timestamp_ts"] = pd.to_datetime(
        weather_df["api_update_timestamp"],
        errors="coerce",
    )
    weather_df["valid_period_start_ts"] = pd.to_datetime(
        weather_df["valid_period_start"],
        errors="coerce",
    )
    weather_df["valid_period_end_ts"] = pd.to_datetime(
        weather_df["valid_period_end"],
        errors="coerce",
    )

    def match_forecast(target_time: datetime) -> str | None:
        matched = weather_df[
            (weather_df["valid_period_start_ts"] <= target_time)
            & (target_time < weather_df["valid_period_end_ts"])
            & (weather_df["api_update_timestamp_ts"] <= target_time)
        ].sort_values("api_update_timestamp_ts")
        if matched.empty:
            return None
        return str(matched.iloc[-1]["forecast"])

    enriched_rows = []
    for row in taxi_rows:
        timestamp = parse_timestamp(row["api_timestamp"])
        current_forecast = match_forecast(timestamp)
        forecast_2h = match_forecast(timestamp + timedelta(hours=2)) or current_forecast
        row = dict(row)
        row["current_forecast"] = current_forecast
        row["forecast_2h"] = forecast_2h
        # Keep the old field name so existing table code still works.
        row["forecast"] = current_forecast
        enriched_rows.append(row)

    return enriched_rows


@app.get("/api/freshness")
def freshness() -> list[dict[str, Any]]:
    return query_database(
        """
        SELECT
            'taxi' AS source,
            MAX(api_timestamp) AS latest_data_timestamp,
            COUNT(*) AS row_count
        FROM taxi_availability_typed
        UNION ALL
        SELECT
            'weather' AS source,
            MAX(api_update_timestamp) AS latest_data_timestamp,
            COUNT(*) AS row_count
        FROM weather_forecasts_typed
        ORDER BY source
        """,
        """
        SELECT
            'taxi' AS source,
            MAX(api_timestamp) AS latest_data_timestamp,
            COUNT(*) AS row_count
        FROM taxi_availability
        UNION ALL
        SELECT
            'weather' AS source,
            MAX(api_update_timestamp) AS latest_data_timestamp,
            COUNT(*) AS row_count
        FROM weather_forecasts
        UNION ALL
        SELECT
            'realtime_weather' AS source,
            MAX(reading_timestamp) AS latest_data_timestamp,
            COUNT(*) AS row_count
        FROM raw_realtime_weather_readings
        ORDER BY source
        """,
    )


@app.post("/api/data/refresh")
def refresh_live_data() -> dict[str, Any]:
    """Run one live data refresh inside the same Day 4 container."""
    command = [
        sys.executable,
        str(APP_DIR / "live_data_scheduler.py"),
        "--run-once",
        "--source",
        "all",
        "--verbose",
    ]
    env = os.environ.copy()
    env["DAY4_LIVE_DB_PATH"] = str(DB_PATH)

    debug_print("live_data_refresh_start", command=" ".join(command), db_path=str(DB_PATH))
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(APP_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        debug_print("live_data_refresh_timeout", timeout_seconds=120)
        raise HTTPException(
            status_code=504,
            detail={
                "message": "Live data refresh timed out.",
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            },
        ) from exc

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    debug_print(
        "live_data_refresh_done",
        returncode=completed.returncode,
        duration_ms=duration_ms,
        stdout_tail=completed.stdout[-500:],
        stderr_tail=completed.stderr[-500:],
    )

    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Live data refresh failed.",
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )

    return {
        "status": "refreshed",
        "duration_ms": duration_ms,
        "db_path": str(DB_PATH),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "freshness": freshness(),
    }


@app.get("/api/model/info")
def model_info() -> dict[str, Any]:
    # This endpoint is a quick inspection endpoint for students.
    # It shows which MLflow server/model URI the app is configured to use,
    # and which concrete model version the current alias resolves to.
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_uri": MLFLOW_MODEL_URI,
        **get_model_registry_metadata(),
        "expected_feature_version": EXPECTED_FEATURE_VERSION,
        "expected_features": EXPECTED_FEATURES,
    }


@app.get("/api/debug/status")
def debug_status() -> dict[str, Any]:
    """Small debugging endpoint for checking the local serving environment."""
    table_counts: dict[str, Any] = {}
    if DB_PATH.exists():
        try:
            with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
                for table in [
                    "taxi_availability",
                    "weather_forecasts",
                    "raw_realtime_weather_readings",
                    "collection_runs",
                ]:
                    table_counts[table] = con.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
        except Exception as exc:
            table_counts["error"] = str(exc)

    return {
        "app_dir": str(APP_DIR),
        "project_root": str(PROJECT_ROOT),
        "db_path": str(DB_PATH),
        "db_exists": DB_PATH.exists(),
        "db_size_bytes": DB_PATH.stat().st_size if DB_PATH.exists() else None,
        "db_backend": DB_BACKEND,
        "table_counts": table_counts,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "mlflow_model_uri": MLFLOW_MODEL_URI,
        "expected_feature_version": EXPECTED_FEATURE_VERSION,
        "expected_features": EXPECTED_FEATURES,
        "model_cache_keys": sorted(_model_cache.keys()),
        "debug_log_path": str(LOG_PATH),
    }


@app.post("/api/model/reload")
def reload_model() -> dict[str, Any]:
    # Use this after moving the champion alias in MLflow.
    #
    # Without reload:
    #   the app may keep using the model object already stored in _model_cache.
    #
    # With reload:
    #   clear the cache, resolve the model URI again, and load the current
    #   champion model version from MLflow.
    debug_print("model_reload_start", cache_keys_before=sorted(_model_cache.keys()))
    _model_cache.clear()
    load_mlflow_model()
    debug_print("model_reload_done", cache_keys_after=sorted(_model_cache.keys()))
    return {
        "status": "reloaded",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_uri": MLFLOW_MODEL_URI,
        **get_model_registry_metadata(),
    }


@app.get("/api/model/predict-2h")
def predict_taxi_2h(area: str = "ANG MO KIO") -> dict[str, Any]:
    # =========================================================================
    # Prediction flow
    # =========================================================================
    #
    # One prediction request does four MLOps-serving steps:
    #
    #   1. Read latest taxi/weather data from the local serving database.
    #   2. Convert those rows into the feature columns expected by the model.
    #   3. Load or reuse the MLflow champion model.
    #   4. Predict and write a prediction log for later monitoring.
    #
    # This is the bridge between "a trained model exists" and
    # "an application can use the model safely."
    request_started_at = datetime.now(SINGAPORE_TZ)
    total_start = time.perf_counter()
    debug_print("predict_start", area=area)

    try:
        # Step 1 and 2: create the model input DataFrame.
        # The columns must match the feature contract used during training.
        debug_print("predict_step_1_2_get_features_start", area=area)
        source_row, model_input = get_latest_feature_row(area)
        input_features = model_input.iloc[0].to_dict()
        debug_print(
            "predict_step_1_2_get_features_done",
            source_row=source_row,
            model_input=model_input.to_dict(orient="records"),
        )

        # Step 3: load the MLflow model.
        # If the model is already cached, load_mlflow_model() returns it quickly.
        # If not, it resolves MLFLOW_MODEL_URI and loads the model from MLflow.
        model_was_cached = "model" in _model_cache
        model_load_start = time.perf_counter()
        debug_print("predict_step_3_load_model_start", model_was_cached=model_was_cached)
        model = load_mlflow_model()
        model_load_duration_ms = round((time.perf_counter() - model_load_start) * 1000, 2)
        model_metadata = _model_cache.get("metadata", {})
        debug_print(
            "predict_step_3_load_model_done",
            duration_ms=model_load_duration_ms,
            metadata=model_metadata,
        )

        # Step 4: run inference.
        # mlflow.pyfunc models accept pandas DataFrames, so model_input is already
        # in the right shape for model.predict(...).
        inference_start = time.perf_counter()
        debug_print("predict_step_4_inference_start")
        prediction = float(model.predict(model_input)[0])
        inference_duration_ms = round((time.perf_counter() - inference_start) * 1000, 2)
        total_duration_ms = round((time.perf_counter() - total_start) * 1000, 2)
        rounded_prediction = int(round(prediction))
        debug_print(
            "predict_step_4_inference_done",
            raw_prediction=prediction,
            rounded_prediction=rounded_prediction,
            inference_duration_ms=inference_duration_ms,
            total_duration_ms=total_duration_ms,
        )
    except HTTPException:
        debug_print("predict_failed_http_exception", area=area)
        raise
    except Exception as exc:
        debug_print(
            "predict_failed_unexpected",
            area=area,
            exception_type=type(exc).__name__,
            error=str(exc),
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)),
        )
        raise

    response = {
        "area": source_row["planning_area"],
        "taxi_timestamp": source_row["api_timestamp"],
        "current_taxi_count": int(source_row["available_taxi_count"]),
        "forecast_2h": source_row.get("forecast") or "Unknown",
        "predicted_taxi_count_2h": rounded_prediction,
        "raw_predicted_taxi_count_2h": round(prediction, 3),
        "prediction_rounding": "nearest_integer",
        "model_uri": MLFLOW_MODEL_URI,
        "model_name": model_metadata.get("model_name"),
        "model_alias": model_metadata.get("model_alias"),
        "model_version": model_metadata.get("model_version"),
        "model_type": model_metadata.get("model_type"),
        "run_id": model_metadata.get("run_id"),
        "run_name": model_metadata.get("run_name"),
        "feature_version": EXPECTED_FEATURE_VERSION,
        "features": input_features,
        "model_was_cached": model_was_cached,
        "model_load_duration_ms": model_load_duration_ms,
        "inference_duration_ms": inference_duration_ms,
        "total_duration_ms": total_duration_ms,
        "logged_at": request_started_at.isoformat(),
    }

    # Monitoring starts with logs.
    # This JSONL record connects input features, model version, prediction, and
    # latency information. Later, when actual outcomes arrive, we can compare
    # prediction vs actual and watch for drift or performance degradation.
    append_prediction_log(
        {
            "logged_at": response["logged_at"],
            "area": response["area"],
            "taxi_timestamp": response["taxi_timestamp"],
            "current_taxi_count": response["current_taxi_count"],
            "forecast_2h": response["forecast_2h"],
            "predicted_taxi_count_2h": response["predicted_taxi_count_2h"],
            "raw_predicted_taxi_count_2h": response["raw_predicted_taxi_count_2h"],
            "prediction_rounding": response["prediction_rounding"],
            "model_uri": response["model_uri"],
            "model_name": response["model_name"],
            "model_alias": response["model_alias"],
            "model_version": response["model_version"],
            "model_type": response["model_type"],
            "run_id": response["run_id"],
            "run_name": response["run_name"],
            "feature_version": response["feature_version"],
            "input_features": input_features,
            "model_was_cached": response["model_was_cached"],
            "model_load_duration_ms": response["model_load_duration_ms"],
            "inference_duration_ms": response["inference_duration_ms"],
            "total_duration_ms": response["total_duration_ms"],
            **input_features,
        }
    )
    debug_print("predict_log_written", path=str(PREDICTION_LOG_PATH))

    debug_print("predict_done", response=response)
    return response


@app.get("/api/prediction-logs")
def prediction_logs(limit: int = 20) -> list[dict[str, Any]]:
    if not PREDICTION_LOG_PATH.exists():
        return []

    return read_recent_jsonl(PREDICTION_LOG_PATH, limit)


@app.get("/api/model/prediction-comparison")
def prediction_comparison(area: str = "ANG MO KIO", limit: int = 20) -> list[dict[str, Any]]:
    if not PREDICTION_LOG_PATH.exists():
        return []

    rows = [
        row
        for row in read_recent_jsonl(PREDICTION_LOG_PATH, 1000)
        if str(row.get("area", "")).upper() == area.upper()
    ][-limit:]

    comparisons = []
    for row in rows:
        taxi_timestamp = parse_timestamp(row["taxi_timestamp"])
        target_timestamp = taxi_timestamp + timedelta(hours=2)
        actual = get_actual_taxi_count_near(area, target_timestamp)
        actual_count = actual["actual_taxi_count"] if actual else None
        error = (
            actual_count - int(row["predicted_taxi_count_2h"])
            if actual_count is not None
            else None
        )

        comparisons.append(
            {
                "area": row.get("area"),
                "logged_at": row.get("logged_at"),
                "prediction_made_at": row.get("taxi_timestamp"),
                "target_timestamp": target_timestamp.isoformat(),
                "predicted_taxi_count_2h": int(row["predicted_taxi_count_2h"]),
                "raw_predicted_taxi_count_2h": row.get("raw_predicted_taxi_count_2h"),
                "actual_timestamp": actual["actual_timestamp"] if actual else None,
                "actual_taxi_count": actual_count,
                "prediction_error": error,
                "absolute_error": abs(error) if error is not None else None,
                "model_uri": row.get("model_uri"),
                "model_version": row.get("model_version"),
                "model_type": row.get("model_type"),
                "feature_version": row.get("feature_version"),
                "input_features": parse_input_features(row),
            }
        )

    return comparisons
