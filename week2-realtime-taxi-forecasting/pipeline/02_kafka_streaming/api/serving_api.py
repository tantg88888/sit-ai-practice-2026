"""Serve taxi snapshots written by the Kafka storage consumer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException
import pandas as pd


DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "data/streaming_ingestion.duckdb"))
app = FastAPI(title="Kafka Taxi Streaming Serving API")


def query_dataframe(sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    """Run a DuckDB query against the consumer-maintained database."""
    if not DATABASE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Database not found. Wait for the Kafka consumer to write data.",
        )

    with duckdb.connect(str(DATABASE_PATH), read_only=True) as connection:
        return connection.execute(sql, params or []).df()


def dataframe_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert dataframe rows into JSON-friendly records."""
    return dataframe.astype(object).where(pd.notna(dataframe), None).to_dict(
        orient="records"
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Kafka Taxi Streaming Serving API"}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "database_exists": DATABASE_PATH.exists(),
        "database_path": str(DATABASE_PATH),
    }


@app.get("/taxi-latest")
def taxi_latest() -> list[dict[str, object]]:
    """Return all coordinates from the latest taxi snapshot."""
    rows = query_dataframe(
        """
        SELECT
            CAST(api_timestamp AS VARCHAR) AS api_timestamp,
            longitude,
            latitude
        FROM taxi_availability
        WHERE api_timestamp = (SELECT MAX(api_timestamp) FROM taxi_availability)
        ORDER BY longitude, latitude
        """
    )
    return dataframe_records(rows)


@app.get("/taxi-summary")
def taxi_summary() -> dict[str, object]:
    """Return a small summary of the consumer-maintained taxi table."""
    rows = query_dataframe(
        """
        SELECT
            MAX(api_timestamp) AS latest_api_timestamp,
            COUNT(DISTINCT api_timestamp) AS snapshot_count,
            COUNT(*) FILTER (
                WHERE api_timestamp = (SELECT MAX(api_timestamp) FROM taxi_availability)
            ) AS latest_available_taxi_positions,
            COUNT(*) AS total_rows
        FROM taxi_availability
        """
    )
    return dataframe_records(rows)[0]


@app.get("/taxi-summary-1min")
def taxi_summary_1min() -> list[dict[str, object]]:
    """Return Flink-generated 1-minute taxi summary windows."""
    rows = query_dataframe(
        """
        SELECT
            CAST(window_start AS VARCHAR) AS window_start,
            CAST(window_end AS VARCHAR) AS window_end,
            snapshot_count,
            avg_available_taxi_count,
            max_available_taxi_count
        FROM taxi_summary_1min
        ORDER BY window_start DESC
        LIMIT 20
        """
    )
    return dataframe_records(rows)
