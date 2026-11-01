from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import pandas as pd
import requests


APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = Path(os.getenv("DAY4_SHARED_ASSETS_DIR", APP_DIR / "shared_assets"))
PERSISTED_DB_PATH = APP_DIR / "live_dashboard_data.db"
RUNNING_IN_DOCKER = Path("/.dockerenv").exists()

DEFAULT_DB_PATH = Path(
    os.getenv(
        "DAY4_LIVE_DB_PATH",
        "/tmp/live_dashboard_data.db" if RUNNING_IN_DOCKER else str(PERSISTED_DB_PATH),
    )
)
DEFAULT_GEOJSON_PATH = ASSETS_DIR / "MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson"

TAXI_API_URL = "https://api.data.gov.sg/v1/transport/taxi-availability"
WEATHER_API_URL = "https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast"
RAINFALL_API_URL = "https://api-open.data.gov.sg/v2/real-time/api/rainfall"
SINGAPORE_TZ = timezone(timedelta(hours=8))


def now_sgt() -> datetime:
    return datetime.now(SINGAPORE_TZ)


def to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=SINGAPORE_TZ)
    return value.astimezone(SINGAPORE_TZ).isoformat(timespec="seconds")


def parse_api_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SINGAPORE_TZ)
    return parsed.astimezone(SINGAPORE_TZ)


def payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_api_key() -> str | None:
    # Optional data.gov.sg key, read only from the environment (never from a file).
    env_key = os.getenv("DATA_GOV_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    return None


def request_headers() -> dict[str, str]:
    api_key = read_api_key()
    return {"X-API-Key": api_key} if api_key else {}


def fetch_json(url: str, timeout_seconds: int) -> dict[str, Any]:
    response = requests.get(url, headers=request_headers(), timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def init_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS collection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                scheduled_for TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                status TEXT NOT NULL,
                records_inserted INTEGER NOT NULL DEFAULT 0,
                message TEXT
            );

            CREATE TABLE IF NOT EXISTS raw_api_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                endpoint_url TEXT NOT NULL,
                collection_time TEXT NOT NULL,
                api_timestamp TEXT,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, payload_hash)
            );

            CREATE TABLE IF NOT EXISTS raw_taxi_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_time TEXT NOT NULL,
                api_timestamp TEXT NOT NULL,
                taxi_index INTEGER NOT NULL,
                longitude REAL NOT NULL,
                latitude REAL NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(api_timestamp, taxi_index, longitude, latitude)
            );

            CREATE TABLE IF NOT EXISTS taxi_availability (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_time TEXT NOT NULL,
                api_timestamp TEXT NOT NULL,
                planning_area TEXT NOT NULL,
                available_taxi_count INTEGER NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(api_timestamp, planning_area)
            );

            CREATE TABLE IF NOT EXISTS weather_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_time TEXT NOT NULL,
                api_update_timestamp TEXT NOT NULL,
                valid_period_start TEXT NOT NULL,
                valid_period_end TEXT NOT NULL,
                area TEXT NOT NULL,
                forecast TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(api_update_timestamp, valid_period_start, valid_period_end, area)
            );

            CREATE TABLE IF NOT EXISTS raw_weather_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_time TEXT NOT NULL,
                api_update_timestamp TEXT NOT NULL,
                valid_period_start TEXT NOT NULL,
                valid_period_end TEXT NOT NULL,
                area TEXT NOT NULL,
                forecast TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(api_update_timestamp, valid_period_start, valid_period_end, area)
            );

            CREATE TABLE IF NOT EXISTS raw_realtime_weather_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_time TEXT NOT NULL,
                reading_timestamp TEXT NOT NULL,
                station_id TEXT NOT NULL,
                station_name TEXT,
                latitude REAL,
                longitude REAL,
                reading_type TEXT NOT NULL,
                reading_unit TEXT,
                reading_value REAL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(reading_type, reading_timestamp, station_id)
            );
            """
        )


def record_run(
    db_path: Path,
    source: str,
    scheduled_for: datetime,
    started_at: datetime,
    status: str,
    records_inserted: int,
    message: str,
) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            INSERT INTO collection_runs (
                source, scheduled_for, started_at, ended_at,
                status, records_inserted, message
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                to_iso(scheduled_for),
                to_iso(started_at),
                to_iso(now_sgt()),
                status,
                records_inserted,
                message,
            ),
        )


def save_raw_snapshot(
    db_path: Path,
    source: str,
    endpoint_url: str,
    collection_time: str,
    api_timestamp: str | None,
    hash_value: str,
    payload: dict[str, Any],
) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            INSERT OR IGNORE INTO raw_api_snapshots (
                source, endpoint_url, collection_time, api_timestamp,
                payload_hash, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                endpoint_url,
                collection_time,
                api_timestamp,
                hash_value,
                json.dumps(payload, ensure_ascii=False),
            ),
        )


def load_subzones(geojson_path: Path) -> gpd.GeoDataFrame:
    if not geojson_path.exists():
        raise FileNotFoundError(f"Subzone GeoJSON not found: {geojson_path}")

    subzones = gpd.read_file(geojson_path)
    if "PLN_AREA_N" not in subzones.columns:
        raise ValueError("GeoJSON must contain a PLN_AREA_N column.")

    if subzones.crs is None:
        subzones = subzones.set_crs("EPSG:4326")
    else:
        subzones = subzones.to_crs("EPSG:4326")

    subzones = subzones[["PLN_AREA_N", "geometry"]].copy()
    subzones["PLN_AREA_N"] = subzones["PLN_AREA_N"].astype(str).str.strip().str.upper()
    return subzones


def extract_taxi_coordinates(payload: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    features = payload.get("features", [])
    if not features:
        raise ValueError("Taxi payload has no features.")

    first_feature = features[0]
    api_timestamp = first_feature.get("properties", {}).get("timestamp")
    coordinates = first_feature.get("geometry", {}).get("coordinates", [])
    if not api_timestamp:
        raise ValueError("Taxi payload has no properties.timestamp.")

    taxi_df = pd.DataFrame(coordinates, columns=["longitude", "latitude"])
    taxi_df["longitude"] = pd.to_numeric(taxi_df["longitude"], errors="coerce")
    taxi_df["latitude"] = pd.to_numeric(taxi_df["latitude"], errors="coerce")
    taxi_df = taxi_df.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)
    return to_iso(parse_api_timestamp(api_timestamp)), taxi_df


def aggregate_taxis_by_planning_area(
    taxi_df: pd.DataFrame,
    subzones: gpd.GeoDataFrame,
) -> pd.DataFrame:
    taxi_points = gpd.GeoDataFrame(
        taxi_df.copy(),
        geometry=gpd.points_from_xy(taxi_df["longitude"], taxi_df["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(taxi_points, subzones, how="inner", predicate="intersects")
    counts = (
        joined.groupby("PLN_AREA_N")
        .size()
        .rename("available_taxi_count")
        .reset_index()
        .rename(columns={"PLN_AREA_N": "planning_area"})
    )
    counts["available_taxi_count"] = counts["available_taxi_count"].astype(int)
    return counts.sort_values("planning_area").reset_index(drop=True)


def extract_weather_forecasts(payload: dict[str, Any]) -> pd.DataFrame:
    data = payload.get("data", payload)
    records = data.get("records") or data.get("items") or []
    if not records:
        raise ValueError("Weather payload has no records.")

    first_item = records[0]
    update_timestamp = (
        first_item.get("updatedTimestamp")
        or first_item.get("update_timestamp")
        or first_item.get("timestamp")
        or first_item.get("date")
    )
    valid_period = first_item.get("validPeriod") or first_item.get("valid_period") or {}
    valid_period_start = (
        valid_period.get("start")
        or first_item.get("validPeriodStart")
        or first_item.get("timestamp")
        or first_item.get("date")
    )
    valid_period_end = valid_period.get("end") or first_item.get("validPeriodEnd")
    if not update_timestamp or not valid_period_start or not valid_period_end:
        raise ValueError("Weather payload is missing update or valid-period timestamps.")

    rows = []
    for item in first_item.get("forecasts", []):
        area = item.get("area")
        forecast = item.get("forecast")
        if area and forecast:
            rows.append(
                {
                    "api_update_timestamp": to_iso(parse_api_timestamp(update_timestamp)),
                    "valid_period_start": to_iso(parse_api_timestamp(valid_period_start)),
                    "valid_period_end": to_iso(parse_api_timestamp(valid_period_end)),
                    "area": str(area).strip(),
                    "forecast": str(forecast).strip(),
                }
            )
    return pd.DataFrame(rows)


def extract_rainfall_readings(payload: dict[str, Any]) -> pd.DataFrame:
    data = payload.get("data", payload)
    stations = data.get("stations") or []
    station_lookup = {
        str(station.get("id") or station.get("deviceId")): station
        for station in stations
        if station.get("id") or station.get("deviceId")
    }
    records = data.get("readings") or data.get("items") or []
    if not records:
        raise ValueError("Rainfall payload has no readings.")

    rows = []
    for record in records:
        timestamp = record.get("timestamp") or record.get("date") or record.get("updatedTimestamp")
        readings = record.get("data") or record.get("readings") or []
        for reading in readings:
            station_id = str(
                reading.get("stationId")
                or reading.get("station_id")
                or reading.get("station")
                or ""
            )
            value = reading.get("value")
            if not station_id or value is None:
                continue
            station = station_lookup.get(station_id, {})
            location = station.get("location") or {}
            rows.append(
                {
                    "reading_timestamp": to_iso(parse_api_timestamp(timestamp)),
                    "station_id": station_id,
                    "station_name": station.get("name"),
                    "latitude": location.get("latitude"),
                    "longitude": location.get("longitude"),
                    "reading_type": "rainfall",
                    "reading_unit": "mm",
                    "reading_value": float(value),
                }
            )
    return pd.DataFrame(rows)


def collect_taxi(
    db_path: Path,
    geojson_path: Path,
    scheduled_for: datetime,
    timeout_seconds: int,
    verbose: bool,
) -> None:
    started_at = now_sgt()
    try:
        payload = fetch_json(TAXI_API_URL, timeout_seconds)
        hash_value = payload_hash(payload)
        collection_time = to_iso(started_at)
        api_timestamp, taxi_points = extract_taxi_coordinates(payload)
        subzones = load_subzones(geojson_path)
        counts = aggregate_taxis_by_planning_area(taxi_points, subzones)
        save_raw_snapshot(
            db_path,
            "taxi",
            TAXI_API_URL,
            collection_time,
            api_timestamp,
            hash_value,
            payload,
        )

        with sqlite3.connect(db_path) as con:
            before = con.execute("SELECT COUNT(*) FROM taxi_availability").fetchone()[0]
            con.executemany(
                """
                INSERT OR IGNORE INTO raw_taxi_points (
                    collection_time, api_timestamp, taxi_index,
                    longitude, latitude, payload_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        collection_time,
                        api_timestamp,
                        index,
                        row.longitude,
                        row.latitude,
                        hash_value,
                    )
                    for index, row in enumerate(taxi_points.itertuples(index=False))
                ],
            )
            con.executemany(
                """
                INSERT OR IGNORE INTO taxi_availability (
                    collection_time, api_timestamp, planning_area,
                    available_taxi_count, payload_hash
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        collection_time,
                        api_timestamp,
                        row.planning_area,
                        int(row.available_taxi_count),
                        hash_value,
                    )
                    for row in counts.itertuples(index=False)
                ],
            )
            after = con.execute("SELECT COUNT(*) FROM taxi_availability").fetchone()[0]

        inserted = after - before
        message = f"taxi points={len(taxi_points)}; planning-area rows={len(counts)}"
        record_run(db_path, "taxi", scheduled_for, started_at, "success", inserted, message)
        if verbose:
            print(f"[taxi] inserted={inserted}; {message}")
    except Exception as exc:
        record_run(db_path, "taxi", scheduled_for, started_at, "failed", 0, str(exc))
        raise


def collect_weather(
    db_path: Path,
    scheduled_for: datetime,
    timeout_seconds: int,
    verbose: bool,
) -> None:
    started_at = now_sgt()
    try:
        payload = fetch_json(WEATHER_API_URL, timeout_seconds)
        hash_value = payload_hash(payload)
        collection_time = to_iso(started_at)
        forecasts = extract_weather_forecasts(payload)
        api_timestamp = forecasts["api_update_timestamp"].iloc[0] if not forecasts.empty else None
        save_raw_snapshot(
            db_path,
            "weather",
            WEATHER_API_URL,
            collection_time,
            api_timestamp,
            hash_value,
            payload,
        )

        with sqlite3.connect(db_path) as con:
            before = con.execute("SELECT COUNT(*) FROM weather_forecasts").fetchone()[0]
            rows = [
                (
                    collection_time,
                    row.api_update_timestamp,
                    row.valid_period_start,
                    row.valid_period_end,
                    row.area,
                    row.forecast,
                    hash_value,
                )
                for row in forecasts.itertuples(index=False)
            ]
            con.executemany(
                """
                INSERT OR IGNORE INTO weather_forecasts (
                    collection_time, api_update_timestamp, valid_period_start,
                    valid_period_end, area, forecast, payload_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            con.executemany(
                """
                INSERT OR IGNORE INTO raw_weather_forecasts (
                    collection_time, api_update_timestamp, valid_period_start,
                    valid_period_end, area, forecast, payload_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            after = con.execute("SELECT COUNT(*) FROM weather_forecasts").fetchone()[0]

        inserted = after - before
        message = f"forecast rows={len(forecasts)}"
        record_run(db_path, "weather", scheduled_for, started_at, "success", inserted, message)
        if verbose:
            print(f"[weather] inserted={inserted}; {message}")
    except Exception as exc:
        record_run(db_path, "weather", scheduled_for, started_at, "failed", 0, str(exc))
        raise


def collect_realtime_weather(
    db_path: Path,
    scheduled_for: datetime,
    timeout_seconds: int,
    verbose: bool,
) -> None:
    started_at = now_sgt()
    try:
        payload = fetch_json(RAINFALL_API_URL, timeout_seconds)
        hash_value = payload_hash(payload)
        collection_time = to_iso(started_at)
        readings = extract_rainfall_readings(payload)
        api_timestamp = readings["reading_timestamp"].iloc[0] if not readings.empty else None
        save_raw_snapshot(
            db_path,
            "realtime_weather",
            RAINFALL_API_URL,
            collection_time,
            api_timestamp,
            hash_value,
            payload,
        )

        with sqlite3.connect(db_path) as con:
            before = con.execute(
                "SELECT COUNT(*) FROM raw_realtime_weather_readings"
            ).fetchone()[0]
            con.executemany(
                """
                INSERT OR IGNORE INTO raw_realtime_weather_readings (
                    collection_time, reading_timestamp, station_id,
                    station_name, latitude, longitude, reading_type,
                    reading_unit, reading_value, payload_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        collection_time,
                        row.reading_timestamp,
                        row.station_id,
                        row.station_name,
                        row.latitude,
                        row.longitude,
                        row.reading_type,
                        row.reading_unit,
                        row.reading_value,
                        hash_value,
                    )
                    for row in readings.itertuples(index=False)
                ],
            )
            after = con.execute(
                "SELECT COUNT(*) FROM raw_realtime_weather_readings"
            ).fetchone()[0]

        inserted = after - before
        message = f"rainfall rows={len(readings)}"
        record_run(
            db_path,
            "realtime_weather",
            scheduled_for,
            started_at,
            "success",
            inserted,
            message,
        )
        if verbose:
            print(f"[realtime_weather] inserted={inserted}; {message}")
    except Exception as exc:
        record_run(
            db_path,
            "realtime_weather",
            scheduled_for,
            started_at,
            "failed",
            0,
            str(exc),
        )
        raise


def run_source(
    source: str,
    db_path: Path,
    geojson_path: Path,
    scheduled_for: datetime,
    timeout_seconds: int,
    verbose: bool,
) -> bool:
    collectors: dict[str, Callable[[], None]] = {
        "taxi": lambda: collect_taxi(
            db_path, geojson_path, scheduled_for, timeout_seconds, verbose
        ),
        "weather": lambda: collect_weather(db_path, scheduled_for, timeout_seconds, verbose),
        "realtime_weather": lambda: collect_realtime_weather(
            db_path, scheduled_for, timeout_seconds, verbose
        ),
    }
    try:
        collectors[source]()
        return True
    except Exception as exc:
        print(f"[{source}] collection failed: {exc}")
        return False


def run_once(args: argparse.Namespace) -> None:
    scheduled_for = now_sgt()
    if args.source == "both":
        sources = ["taxi", "weather"]
    elif args.source == "all":
        sources = ["taxi", "weather", "realtime_weather"]
    else:
        sources = [args.source]

    succeeded = [
        run_source(
            source,
            args.db_path,
            args.geojson_path,
            scheduled_for,
            args.request_timeout_seconds,
            args.verbose,
        )
        for source in sources
    ]
    if not all(succeeded):
        raise RuntimeError("One or more collection sources failed.")
    sync_runtime_database(args.db_path)


def run_forever(args: argparse.Namespace) -> None:
    next_run = {
        "taxi": now_sgt(),
        "weather": now_sgt(),
        "realtime_weather": now_sgt(),
    }
    intervals = {
        "taxi": timedelta(minutes=args.taxi_interval_minutes),
        "weather": timedelta(minutes=args.weather_interval_minutes),
        "realtime_weather": timedelta(minutes=args.realtime_weather_interval_minutes),
    }
    print("Day 4.2 scheduler running. Press Ctrl+C to stop.")

    while True:
        current_time = now_sgt()
        for source in ("taxi", "weather", "realtime_weather"):
            if current_time >= next_run[source]:
                run_source(
                    source,
                    args.db_path,
                    args.geojson_path,
                    next_run[source],
                    args.request_timeout_seconds,
                    args.verbose,
                )
                next_run[source] = current_time + intervals[source]
                sync_runtime_database(args.db_path)
        time.sleep(1)


def prepare_runtime_database(db_path: Path) -> None:
    """Use /tmp inside Docker to avoid SQLite issues on Windows bind mounts."""
    if db_path == PERSISTED_DB_PATH:
        return
    if db_path.exists():
        return
    if PERSISTED_DB_PATH.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PERSISTED_DB_PATH, db_path)
        print(f"Copied existing dashboard database to runtime path: {db_path}")


def sync_runtime_database(db_path: Path) -> None:
    """Copy the runtime DB back to /workspace so students can inspect it."""
    if db_path == PERSISTED_DB_PATH:
        return
    if not db_path.exists():
        return
    shutil.copy2(db_path, PERSISTED_DB_PATH)
    print(f"Synced runtime database back to: {PERSISTED_DB_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Day 4.2 local scheduler. It writes live taxi/weather data into "
            "the SQLite database used by the model-serving dashboard."
        )
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--geojson-path", type=Path, default=DEFAULT_GEOJSON_PATH)
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument(
        "--source",
        choices=["taxi", "weather", "realtime_weather", "both", "all"],
        default="all",
    )
    parser.add_argument("--taxi-interval-minutes", type=int, default=2)
    parser.add_argument("--realtime-weather-interval-minutes", type=int, default=2)
    parser.add_argument("--weather-interval-minutes", type=int, default=30)
    parser.add_argument("--request-timeout-seconds", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.db_path = args.db_path.resolve()
    args.geojson_path = args.geojson_path.resolve()

    prepare_runtime_database(args.db_path)
    init_database(args.db_path)
    if args.run_once:
        run_once(args)
    else:
        try:
            run_forever(args)
        except KeyboardInterrupt:
            print("\nScheduler stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
