"""Detect unusually high/low taxi availability per planning area and raise alerts.

Method (hour-aware, so a normal peak/trough is NOT flagged): for each reading,
compare an area's taxi count to the distribution of that area's counts at the SAME
hour of day. If |z-score| >= THRESHOLD, it is an anomaly and we raise an alert.

    python pipeline/extras/anomaly_alert.py       (from the project root)
"""
import os
from pathlib import Path
import pandas as pd

# Default: <project root>/data/... ; override with TAXI_DATASET_PATH.
DATA = Path(os.getenv(
    "TAXI_DATASET_PATH",
    Path(__file__).resolve().parents[2] / "data" / "taxi_weather_2h_prediction_dataset.csv",
))
THRESHOLD = 2.5


def main():
    df = pd.read_csv(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Baseline mean/std per (area, hour), then a z-score for every reading.
    grp = df.groupby(["planning_area", "hour_of_day"])["current_taxi_count"]
    df["baseline_mean"] = grp.transform("mean")
    df["baseline_std"] = grp.transform("std").fillna(0)
    df["z"] = (df["current_taxi_count"] - df["baseline_mean"]) / df["baseline_std"].replace(0, pd.NA)

    # 1) Alerts in the most recent snapshot (what a live run would flag "now").
    latest_ts = df["timestamp"].max()
    latest = df[df["timestamp"] == latest_ts]
    live_alerts = latest[latest["z"].abs() >= THRESHOLD]
    print(f"Latest snapshot @ {latest_ts}: checked {len(latest)} areas, "
          f"{len(live_alerts)} alert(s).")
    for _, r in live_alerts.sort_values("z", key=abs, ascending=False).iterrows():
        direction = "HIGH" if r["z"] > 0 else "LOW"
        print(f"  [ALERT] {r['planning_area']}: unusually {direction} "
              f"taxis={int(r['current_taxi_count'])} "
              f"(typical {r['baseline_mean']:.0f} at {int(r['hour_of_day'])}:00, z={r['z']:+.2f})")

    # 2) Top anomalies across all history (examples from the historical data).
    top = df.dropna(subset=["z"]).reindex(df["z"].abs().sort_values(ascending=False).index).head(5)
    print("\nTop 5 anomalies detected in the dataset (demo examples):")
    for _, r in top.iterrows():
        direction = "HIGH" if r["z"] > 0 else "LOW"
        print(f"  [ALERT] {r['timestamp']:%Y-%m-%d %H:%M} {r['planning_area']}: {direction} "
              f"taxis={int(r['current_taxi_count'])} (typical {r['baseline_mean']:.0f}, z={r['z']:+.2f})")


if __name__ == "__main__":
    main()
