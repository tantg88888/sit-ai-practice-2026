import json
from datetime import datetime, timezone

import duckdb
import pandas as pd
import requests

from jobs.config import (
    WEATHER_API_URL,
    RAW_DIR,
    CLEAN_DIR,
    DB_PATH,
    ensure_dirs,
)


def fetch_weather_data():
    """
    Fetch raw 2-hour weather forecast data from data.gov.sg.
    Save the raw JSON file.
    """
    ensure_dirs()

    collected_at = datetime.now(timezone.utc)
    timestamp_str = collected_at.strftime("%Y%m%d_%H%M%S")

    response = requests.get(WEATHER_API_URL, timeout=10)
    response.raise_for_status()

    data = response.json()

    raw_path = RAW_DIR / f"weather_raw_{timestamp_str}.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(raw_path)


def clean_weather_data(raw_path: str):
    """
    Convert raw weather JSON into a clean table.

    Output columns:
    - update_timestamp
    - area
    - forecast
    """
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Day 1 now uses the data.gov.sg v2 weather endpoint, where useful fields
    # live under payload["data"]. Keep this compatible with the older v1 shape
    # so the lesson is less fragile if the source changes again.
    data_block = data.get("data", data)
    item = data_block["items"][0]
    update_timestamp = item["update_timestamp"]
    forecasts = item["forecasts"]

    rows = []

    for forecast_item in forecasts:
        rows.append(
            {
                "update_timestamp": update_timestamp,
                "area": forecast_item["area"],
                "forecast": forecast_item["forecast"],
            }
        )

    df = pd.DataFrame(rows)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_path = CLEAN_DIR / f"weather_clean_{timestamp_str}.csv"

    df.to_csv(clean_path, index=False)

    return str(clean_path)


def load_weather_to_duckdb(clean_path: str):
    """
    Load cleaned weather data into DuckDB.
    """
    df = pd.read_csv(clean_path)

    conn = duckdb.connect(str(DB_PATH))

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_forecasts (
            update_timestamp TIMESTAMP,
            area VARCHAR,
            forecast VARCHAR
        )
        """
    )

    conn.register("weather_df", df)

    conn.execute(
        """
        DELETE FROM weather_forecasts
        WHERE update_timestamp IN (
            SELECT DISTINCT CAST(update_timestamp AS TIMESTAMP)
            FROM weather_df
        )
        """
    )
    conn.execute(
        """
        INSERT INTO weather_forecasts (
            update_timestamp,
            area,
            forecast
        )
        SELECT
            CAST(update_timestamp AS TIMESTAMP),
            area,
            forecast
        FROM weather_df
        """
    )

    conn.close()

    print(f"Loaded weather data into DuckDB: {clean_path}")


def run_weather_pipeline():
    """
    Main function called by Airflow DAG.
    """
    raw_path = fetch_weather_data()
    clean_path = clean_weather_data(raw_path)
    load_weather_to_duckdb(clean_path)
