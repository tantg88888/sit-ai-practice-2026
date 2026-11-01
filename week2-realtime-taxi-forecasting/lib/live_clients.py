"""Live data.gov.sg clients for the taxi + weather demo (with cached fallback).

Endpoints (same as the Airflow / Kafka pipeline):
  - Taxi availability : https://api.data.gov.sg/v1/transport/taxi-availability         (v1, keyless)
  - 2-hour weather    : https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast  (v2, optional X-API-Key)

The optional data.gov.sg API key is read ONLY from the environment variable
DATA_GOV_API_KEY and sent in the `X-API-Key` header (a key raises the rate limit;
the endpoints also answer without one). Never commit a key file. If a live call
fails (offline, rate-limited, blocked), each function transparently falls back to a
bundled sample with the identical schema.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TAXI_URL = "https://api.data.gov.sg/v1/transport/taxi-availability"
WEATHER_URL = "https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast"
TAXI_FALLBACK = DATA_DIR / "sample_taxi_availability.json"
WEATHER_FALLBACK = DATA_DIR / "sample_2h_weather_forecast.json"


def read_api_key() -> str | None:
    """Return the data.gov.sg API key from the DATA_GOV_API_KEY env var, else None."""
    env = os.getenv("DATA_GOV_API_KEY")
    if env and env.strip():
        return env.strip()
    return None


def _headers() -> dict[str, str]:
    key = read_api_key()
    return {"X-API-Key": key} if key else {}


def key_status() -> str:
    """Human-readable key status (never prints any part of the key)."""
    if not read_api_key():
        return ("not set - calling the APIs keyless (lower rate limit); "
                "any failed call falls back to the bundled cached samples")
    return "set via DATA_GOV_API_KEY (value hidden)"


def _get_json(url: str, fallback: Path) -> tuple[dict[str, Any], str]:
    try:
        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        return resp.json(), "LIVE data.gov.sg"
    except Exception as exc:  # noqa: BLE001
        print(f"[fallback] live call failed ({type(exc).__name__}); using cached sample.")
        return json.loads(fallback.read_text(encoding="utf-8")), "CACHED sample"


def fetch_taxi() -> dict[str, Any]:
    """Return one taxi-availability snapshot: tidy fields + a points DataFrame."""
    payload, source = _get_json(TAXI_URL, TAXI_FALLBACK)
    feat = payload["features"][0]
    coords = feat["geometry"]["coordinates"]
    points = pd.DataFrame(coords, columns=["longitude", "latitude"])
    return {
        "source": source,
        "api_timestamp": feat["properties"]["timestamp"],
        "fetched_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "available_taxis": len(coords),
        "points": points,
    }


def _extract_weather(payload: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    """Handle BOTH the v2 (data.records[]) and v1 (items[]) weather shapes."""
    data = payload.get("data", payload)
    records = data.get("records") or data.get("items") or payload.get("items")
    first = records[0]
    update_ts = (first.get("updatedTimestamp") or first.get("update_timestamp")
                 or first.get("timestamp") or first.get("date") or "")
    rows = [{"area": f.get("area"), "forecast": f.get("forecast")}
            for f in first.get("forecasts", []) if f.get("area") and f.get("forecast")]
    return update_ts, pd.DataFrame(rows)


def fetch_weather() -> dict[str, Any]:
    """Return one 2-hour weather snapshot: tidy fields + a forecast DataFrame."""
    payload, source = _get_json(WEATHER_URL, WEATHER_FALLBACK)
    update_ts, forecasts = _extract_weather(payload)
    rainy_words = ("Rain", "Showers", "Thundery")
    forecasts["is_rain"] = forecasts["forecast"].str.contains("|".join(rainy_words)).astype(int)
    forecasts["area_key"] = forecasts["area"].str.strip().str.upper()
    return {
        "source": source,
        "update_timestamp": update_ts,
        "fetched_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "n_areas": len(forecasts),
        "n_rain_areas": int(forecasts["is_rain"].sum()),
        "forecasts": forecasts,
    }
