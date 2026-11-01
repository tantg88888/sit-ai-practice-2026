"""Consume taxi/weather Kafka events and load clean tables into DuckDB."""

from __future__ import annotations

import json
import os
from pathlib import Path

from confluent_kafka import Consumer
import duckdb
import pandas as pd


# These settings come from docker-compose.yml when the service runs in Docker.
#
# KAFKA_BOOTSTRAP_SERVERS is the Kafka broker address.
# KAFKA_TOPIC is the default raw topic.
# KAFKA_TOPICS can contain several topics because this storage consumer can read
# both raw producer events and optional Flink summary events.
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "day_3_kafka:19092")
TOPIC = os.getenv("KAFKA_TOPIC", "taxi_weather_events")
TOPICS = [
    topic.strip()
    for topic in os.getenv("KAFKA_TOPICS", TOPIC).split(",")
    if topic.strip()
]
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "data/streaming_ingestion.duckdb"))


def ensure_database() -> None:
    """Create the DuckDB tables maintained by this storage consumer."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DATABASE_PATH)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS taxi_availability (
                api_timestamp TIMESTAMP,
                longitude DOUBLE,
                latitude DOUBLE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_forecasts (
                update_timestamp TIMESTAMP,
                area VARCHAR,
                forecast VARCHAR
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS taxi_summary_1min (
                window_start TIMESTAMP,
                window_end TIMESTAMP,
                snapshot_count BIGINT,
                avg_available_taxi_count DOUBLE,
                max_available_taxi_count INTEGER
            )
            """
        )


def load_taxi_snapshot(event: dict) -> int:
    """Load one taxi snapshot event into DuckDB."""
    # producer.py creates taxi_snapshot events with a coordinates list. The
    # consumer turns that nested event payload into normal database rows.
    dataframe = pd.DataFrame(event["coordinates"], columns=["longitude", "latitude"])
    dataframe.insert(0, "api_timestamp", event["event_time"])

    with duckdb.connect(str(DATABASE_PATH)) as connection:
        # Public APIs can return the same timestamp for several polls. Delete
        # first so repeated snapshots do not inflate the latest taxi count.
        connection.execute(
            "DELETE FROM taxi_availability WHERE api_timestamp = CAST(? AS TIMESTAMP)",
            [event["event_time"]],
        )
        connection.register("taxi_df", dataframe)
        connection.execute(
            """
            INSERT INTO taxi_availability
            SELECT
                CAST(api_timestamp AS TIMESTAMP),
                longitude,
                latitude
            FROM taxi_df
            """
        )
    return len(dataframe)


def load_weather_snapshot(event: dict) -> int:
    """Load one weather snapshot event into DuckDB."""
    # producer.py creates weather_snapshot events with a forecasts list. The
    # consumer keeps the fields the dashboard/API need: area and forecast.
    dataframe = pd.DataFrame(event["forecasts"])[["area", "forecast"]]
    dataframe.insert(0, "update_timestamp", event["event_time"])

    with duckdb.connect(str(DATABASE_PATH)) as connection:
        connection.execute(
            "DELETE FROM weather_forecasts WHERE update_timestamp = CAST(? AS TIMESTAMP)",
            [event["event_time"]],
        )
        connection.register("weather_df", dataframe)
        connection.execute(
            """
            INSERT INTO weather_forecasts
            SELECT
                CAST(update_timestamp AS TIMESTAMP),
                area,
                forecast
            FROM weather_df
            """
        )
    return len(dataframe)


def load_taxi_summary_1min(event: dict) -> int:
    """Load one Flink-generated incremental taxi summary into DuckDB."""
    with duckdb.connect(str(DATABASE_PATH)) as connection:
        connection.execute(
            """
            DELETE FROM taxi_summary_1min
            WHERE
                window_start = CAST(? AS TIMESTAMP)
                AND window_end = CAST(? AS TIMESTAMP)
            """,
            [event["window_start"], event["window_end"]],
        )
        connection.execute(
            """
            INSERT INTO taxi_summary_1min
            VALUES (
                CAST(? AS TIMESTAMP),
                CAST(? AS TIMESTAMP),
                ?,
                ?,
                ?
            )
            """,
            [
                event["window_start"],
                event["window_end"],
                event["snapshot_count"],
                event["avg_available_taxi_count"],
                event["max_available_taxi_count"],
            ],
        )
    return 1


def handle_event(event: dict) -> str:
    """Route an event by type and return a compact log message."""
    # event_type is the small field that lets one Kafka topic carry multiple
    # shapes of events. The producer sets it; the consumer uses it to decide
    # which loader should handle the event.
    if event["event_type"] == "taxi_snapshot":
        row_count = load_taxi_snapshot(event)
        return f"loaded taxi_snapshot rows={row_count} event_time={event['event_time']}"
    if event["event_type"] == "weather_snapshot":
        row_count = load_weather_snapshot(event)
        return f"loaded weather_snapshot rows={row_count} event_time={event['event_time']}"
    if event["event_type"] == "taxi_summary_1min":
        row_count = load_taxi_summary_1min(event)
        return (
            f"loaded taxi_summary_1min rows={row_count} "
            f"window={event['window_start']}..{event['window_end']}"
        )
    return f"ignored unknown event_type={event.get('event_type')}"


def main() -> None:
    ensure_database()

    # This creates the Kafka Consumer client.
    #
    # bootstrap.servers: where to reach Kafka.
    # group.id: the name Kafka uses to track this consumer's progress.
    # auto.offset.reset: if this group is new, start from the earliest message.
    # enable.auto.commit: False means we will commit manually after DuckDB write.
    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "group.id": "day_3_storage-consumer",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )

    # subscribe(...) tells Kafka which topic streams this consumer wants to read.
    # It does not return messages yet; the poll loop below does that.
    consumer.subscribe(TOPICS)
    print(
        f"Storage consumer reading {', '.join(TOPICS)} via {KAFKA_BOOTSTRAP_SERVERS} "
        f"and writing {DATABASE_PATH}",
        flush=True,
    )

    try:
        while True:
            # poll(...) asks Kafka for one available message. If no message is
            # ready within 1 second, it returns None and the loop tries again.
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                print(f"Consumer error: {message.error()}", flush=True)
                continue

            # Kafka message values are bytes. The producer wrote JSON bytes, so
            # the consumer decodes them back into a Python dictionary event.
            event = json.loads(message.value())
            log_message = handle_event(event)

            # Commit after handle_event(...) succeeds. That way Kafka progress
            # only moves forward after the event has been written to DuckDB.
            consumer.commit(message=message, asynchronous=False)
            key = message.key().decode() if message.key() else event.get("event_type")
            print(
                f"partition={message.partition()} offset={message.offset()} "
                f"key={key} {log_message}",
                flush=True,
            )
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
