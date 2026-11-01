import argparse
import sqlite3
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a 2-hour taxi availability training dataset from a Day 1 SQLite database."
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to the source SQLite database.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to the output CSV file.",
    )
    return parser.parse_args()


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def build_training_table(db_path):
    with sqlite3.connect(db_path) as con:
        taxi = pd.read_sql_query(
            """
            SELECT api_timestamp, planning_area, available_taxi_count
            FROM taxi_availability
            ORDER BY api_timestamp, planning_area
            """,
            con,
        )

        weather = pd.read_sql_query(
            """
            SELECT valid_period_start, valid_period_end, UPPER(area) AS planning_area, forecast
            FROM weather_forecasts
            ORDER BY valid_period_start, planning_area
            """,
            con,
        )

    taxi["timestamp"] = pd.to_datetime(taxi["api_timestamp"])
    weather["valid_period_start_ts"] = pd.to_datetime(weather["valid_period_start"])
    weather["valid_period_end_ts"] = pd.to_datetime(weather["valid_period_end"])

    current = taxi.rename(columns={"available_taxi_count": "current_taxi_count"}).copy()
    current["desired_target_timestamp"] = current["timestamp"] + pd.Timedelta(hours=2)

    current["hour_of_day"] = current["timestamp"].dt.hour
    current["day_of_week"] = current["timestamp"].dt.dayofweek
    current["is_weekend"] = current["day_of_week"].isin([5, 6]).astype(int)
    current["is_peak_hour"] = current["hour_of_day"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    current["is_late_night"] = current["hour_of_day"].between(0, 5).astype(int)

    target = taxi[["timestamp", "planning_area", "available_taxi_count"]].rename(
        columns={
            "timestamp": "actual_target_timestamp",
            "available_taxi_count": "target_taxi_count_2h",
        }
    )

    matched_parts = []
    for area, cur_area in current.groupby("planning_area", sort=False):
        tgt_area = target[target["planning_area"] == area].sort_values(
            "actual_target_timestamp"
        )
        cur_area = cur_area.sort_values("desired_target_timestamp")

        matched = pd.merge_asof(
            cur_area,
            tgt_area[["actual_target_timestamp", "target_taxi_count_2h"]],
            left_on="desired_target_timestamp",
            right_on="actual_target_timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=10),
        )

        matched_parts.append(matched.dropna(subset=["target_taxi_count_2h"]))

    model_df = pd.concat(matched_parts, ignore_index=True)
    model_df["target_taxi_count_2h"] = model_df["target_taxi_count_2h"].astype(int)

    weather_parts = []
    for area, area_rows in model_df.groupby("planning_area", sort=False):
        weather_area = weather[weather["planning_area"] == area]
        merged = area_rows.merge(weather_area, on="planning_area", how="left")

        matched = merged[
            (merged["desired_target_timestamp"] >= merged["valid_period_start_ts"])
            & (merged["desired_target_timestamp"] < merged["valid_period_end_ts"])
        ].copy()

        matched["forecast_2h"] = matched["forecast"].fillna("Unknown")
        weather_parts.append(matched)

    model_df = pd.concat(weather_parts, ignore_index=True)
    model_df = model_df.drop_duplicates(subset=["timestamp", "planning_area"])
    model_df["forecast_2h"] = model_df["forecast_2h"].fillna("Unknown")
    model_df["is_rain_forecast_2h"] = model_df["forecast_2h"].str.lower().str.contains(
        "rain|showers|thundery"
    ).astype(int)

    return model_df[
        [
            "timestamp",
            "desired_target_timestamp",
            "actual_target_timestamp",
            "planning_area",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_peak_hour",
            "is_late_night",
            "forecast_2h",
            "is_rain_forecast_2h",
            "current_taxi_count",
            "target_taxi_count_2h",
        ]
    ].sort_values(["timestamp", "planning_area"])


def main():
    args = parse_args()
    db_path = resolve_path(args.db_path)
    output_path = resolve_path(args.output_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    training_table = build_training_table(db_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_table.to_csv(output_path, index=False)

    print(f"Source database: {db_path}")
    print(f"Output CSV: {output_path}")
    print(f"Rows: {len(training_table)}")
    print("Forecast values:")
    print(training_table["forecast_2h"].value_counts().to_string())


if __name__ == "__main__":
    main()
