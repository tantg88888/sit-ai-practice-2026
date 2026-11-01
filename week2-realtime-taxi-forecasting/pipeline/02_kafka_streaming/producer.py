"""Poll Singapore taxi/weather APIs and publish snapshot events to Kafka."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from typing import Any

from confluent_kafka import Producer
import requests


# Kafka bootstrap server.
#
# In docker-compose.yml, the producer service receives:
#   KAFKA_BOOTSTRAP_SERVERS=day_3_kafka:19092
#
# os.getenv(...) reads that container environment variable. If we run this file
# outside Docker and the variable is missing, the second argument is the default.
#
# This value is not the topic name. It is the Kafka broker address the producer
# contacts first so it can discover the Kafka cluster.
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "day_3_kafka:19092")

# Same public data sources as the Day 2 Docker lesson.
TAXI_API_URL = os.getenv(
    "TAXI_API_URL",
    "https://api.data.gov.sg/v1/transport/taxi-availability",
)
WEATHER_API_URL = os.getenv(
    "WEATHER_API_URL",
    "https://api.data.gov.sg/v1/environment/2-hour-weather-forecast",
)

# Kafka topic.
#
# docker-compose.yml only passes this topic name into the container:
#   KAFKA_TOPIC=taxi_weather_events
#
# The producer code reads the name here, then uses it later in
# producer.produce(topic=TOPIC, ...). Kafka is the system that stores messages
# under this topic.
TOPIC = os.getenv("KAFKA_TOPIC", "taxi_weather_events")

POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "15"))


def utc_now_iso() -> str:
    """Return the producer ingestion timestamp."""
    return datetime.now(timezone.utc).isoformat()


def fetch_json(url: str) -> dict[str, Any]:
    """Fetch one JSON payload from a public API."""
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


def build_taxi_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert the taxi availability API response into one Kafka event."""
    feature = payload["features"][0]
    coordinates = feature["geometry"]["coordinates"]

    # This returned dictionary is the event schema for a taxi snapshot.
    # Kafka will receive this dictionary as JSON bytes in send_event(...).
    return {
        "event_type": "taxi_snapshot",
        "source": "data.gov.sg/taxi-availability",
        "event_time": feature["properties"]["timestamp"],
        "ingested_at": utc_now_iso(),
        "available_taxi_count": len(coordinates),
        "coordinates": coordinates,
        # Keep a small sample so students can inspect the shape without sending
        # hundreds of coordinates in every teaching dashboard row.
        "sample_coordinates": coordinates[:5],
    }


def build_weather_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert the 2-hour weather forecast API response into one Kafka event."""
    item = payload["items"][0]
    forecasts = item["forecasts"]
    forecast_counts: dict[str, int] = {}
    for row in forecasts:
        forecast = row["forecast"]
        forecast_counts[forecast] = forecast_counts.get(forecast, 0) + 1

    # This returned dictionary is the event schema for a weather snapshot.
    # It uses the same core fields as taxi_snapshot, especially event_type and
    # event_time, so the consumer can route both event types consistently.
    return {
        "event_type": "weather_snapshot",
        "source": "data.gov.sg/2-hour-weather-forecast",
        "event_time": item["update_timestamp"],
        "ingested_at": utc_now_iso(),
        "forecast_area_count": len(forecasts),
        "forecast_counts": forecast_counts,
        "forecasts": forecasts,
        "sample_forecasts": forecasts[:5],
    }


def delivery_report(error, message) -> None:
    """Print Kafka delivery metadata after Kafka accepts or rejects a message."""
    if error is not None:
        print(f"Delivery failed: {error}", flush=True)
        return

    print(
        f"Delivered key={message.key().decode()} "
        f"partition={message.partition()} offset={message.offset()}",
        flush=True,
    )


def send_event(producer: Producer, event: dict[str, Any]) -> None:
    """Serialize and publish one event to Kafka."""
    # The event_type is the key so related taxi/weather snapshots stay ordered
    # within their own partition.
    key = event["event_type"]

    # This is the actual Kafka write.
    #
    # topic: which named stream in Kafka receives the message.
    # key:   a small routing/ordering label. Here it is taxi_snapshot or
    #        weather_snapshot.
    # value: Kafka stores bytes, so we convert the Python dict to JSON text,
    #        then encode that text as UTF-8 bytes.
    # callback: called later with Kafka's delivery result.
    producer.produce(
        topic=TOPIC,
        key=key.encode("utf-8"),
        value=json.dumps(event).encode("utf-8"),
        callback=delivery_report,
    )
    producer.poll(0)


def main() -> None:
    # This creates the Kafka Producer client. It opens the path to Kafka using
    # the bootstrap server above, but no event is sent until producer.produce(...)
    # is called inside send_event(...).
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
    print(f"Producing to {TOPIC} via {KAFKA_BOOTSTRAP_SERVERS}", flush=True)

    while True:
        # This loop is the streaming version of the Day 2 pipeline:
        # instead of writing directly to DuckDB, it emits snapshot events first.
        taxi_event = build_taxi_event(fetch_json(TAXI_API_URL))
        weather_event = build_weather_event(fetch_json(WEATHER_API_URL))

        send_event(producer, taxi_event)
        send_event(producer, weather_event)
        producer.flush(10)

        print(
            f"Published taxi_count={taxi_event['available_taxi_count']} "
            f"weather_areas={weather_event['forecast_area_count']}",
            flush=True,
        )
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
