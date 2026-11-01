-- These ADD JAR statements make the lesson easier to run from the SQL client:
-- the SQL job carries the connector jars even if the Flink cluster was already
-- started before the jars were added to /opt/flink/lib.
ADD JAR 'file:///opt/flink/lib/flink-sql-connector-kafka-3.2.0-1.19.jar';
ADD JAR 'file:///opt/flink/lib/flink-connector-kafka-3.2.0-1.19.jar';
ADD JAR 'file:///opt/flink/lib/kafka-clients-3.4.0.jar';
ADD JAR 'file:///opt/flink/lib/flink-sql-json-1.19.1.jar';

CREATE TABLE raw_taxi_weather_events (
  event_type STRING,
  source STRING,
  event_time STRING,
  ingested_at STRING,
  available_taxi_count INT,
  forecast_area_count INT,
  proc_time AS PROCTIME()
) WITH (
  'connector' = 'kafka',
  'topic' = 'taxi_weather_events',
  'properties.bootstrap.servers' = 'day_3_kafka:19092',
  'properties.group.id' = 'day_3_flink-incremental-summary',
  'scan.startup.mode' = 'earliest-offset',
  'format' = 'json',
  'json.ignore-parse-errors' = 'true'
);

CREATE TABLE taxi_weather_incremental_events (
  event_type STRING,
  window_start STRING,
  window_end STRING,
  snapshot_count BIGINT,
  avg_available_taxi_count DOUBLE,
  max_available_taxi_count INT
) WITH (
  'connector' = 'kafka',
  'topic' = 'taxi_weather_incremental_events',
  'properties.bootstrap.servers' = 'day_3_kafka:19092',
  'format' = 'json'
);

INSERT INTO taxi_weather_incremental_events
SELECT
  'taxi_summary_1min' AS event_type,
  CAST(window_start AS STRING) AS window_start,
  CAST(window_end AS STRING) AS window_end,
  COUNT(*) AS snapshot_count,
  AVG(available_taxi_count) AS avg_available_taxi_count,
  MAX(available_taxi_count) AS max_available_taxi_count
FROM TABLE(
  TUMBLE(TABLE raw_taxi_weather_events, DESCRIPTOR(proc_time), INTERVAL '1' MINUTE)
)
WHERE event_type = 'taxi_snapshot'
GROUP BY window_start, window_end;
