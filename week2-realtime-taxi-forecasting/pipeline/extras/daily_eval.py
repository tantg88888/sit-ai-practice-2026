"""Daily evaluation of the 2-hour taxi prediction model.

Scheduled by the Airflow DAG taxi_model_daily_eval (schedule 0 8 * * *); it can
also be triggered manually.

It splits the dataset by day, trains on earlier days, evaluates on the most recent
day (predicted vs actual taxis 2h later), reports MAE/RMSE/R2 + per-area error, and
flags drift if MAE exceeds a threshold (which would trigger a retrain).

    python pipeline/extras/daily_eval.py          (from the project root)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Default: <project root>/data/... ; override with TAXI_DATASET_PATH (e.g. inside Airflow).
DATA = Path(os.getenv(
    "TAXI_DATASET_PATH",
    Path(__file__).resolve().parents[2] / "data" / "taxi_weather_2h_prediction_dataset.csv",
))
TARGET = "target_taxi_count_2h"
MAE_DRIFT_THRESHOLD = 25  # taxis


def main():
    df = pd.read_csv(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["area_id"] = df["planning_area"].astype("category").cat.codes
    features = ["current_taxi_count", "hour_of_day", "day_of_week", "is_weekend",
                "is_peak_hour", "is_late_night", "is_rain_forecast_2h", "area_id"]
    df = df.dropna(subset=features + [TARGET])

    days = sorted(df["date"].unique())
    eval_day = days[-1]
    train = df[df["date"] < eval_day]
    test = df[df["date"] == eval_day]
    print(f"Daily evaluation for {eval_day}  (train on {len(days)-1} earlier day(s), "
          f"{len(train):,} rows; evaluate on {len(test):,} rows)")

    model = RandomForestRegressor(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
    model.fit(train[features], train[TARGET])
    pred = model.predict(test[features])

    mae = mean_absolute_error(test[TARGET], pred)
    rmse = np.sqrt(mean_squared_error(test[TARGET], pred))
    r2 = r2_score(test[TARGET], pred)
    print(f"  MAE  = {mae:.2f} taxis")
    print(f"  RMSE = {rmse:.2f} taxis")
    print(f"  R2   = {r2:.3f}")

    test = test.assign(abs_err=np.abs(test[TARGET].values - pred))
    worst = test.groupby("planning_area")["abs_err"].mean().sort_values(ascending=False).head(5)
    print("  Worst areas by mean abs error:")
    for area, err in worst.items():
        print(f"    {area}: {err:.1f}")

    if mae > MAE_DRIFT_THRESHOLD:
        print(f"  DRIFT: MAE {mae:.1f} > {MAE_DRIFT_THRESHOLD} -> would trigger a retrain.")
    else:
        print(f"  OK: MAE within threshold ({MAE_DRIFT_THRESHOLD}); no retrain needed.")


if __name__ == "__main__":
    main()
