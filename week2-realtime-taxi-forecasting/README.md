# Week 2 · Real-time taxi availability forecasting pipeline (Singapore)

**Area:** Data engineering & MLOps · **Format:** Individual project · SIT Graduate Certificate in AI Practice, 2026

## Problem
Predict how many taxis will be available in each Singapore planning area two hours ahead, using live open-government data.

## Data
Live [data.gov.sg](https://data.gov.sg) APIs (taxi availability, 2-hour weather forecast, rainfall) and URA Master Plan 2019 subzone boundaries. Public data.

## What I built
- Airflow DAG (fetch → clean → join → load) every 2 minutes, with retries, exponential backoff and a single-writer pool, landing in DuckDB
- Geopandas spatial join mapping taxi GPS points to 54 planning areas
- Streaming path: Kafka producer → topic → manual-commit consumer → DuckDB, plus a Flink 1-minute windowed aggregate, all on Docker Compose
- MLP regressor tracked and registered in MLflow, benchmarked against linear regression, served through FastAPI
- Live dashboard comparing current vs predicted (+2 h) availability, with anomaly alerts

## Results
- R² ≈ 0.75, MAE ≈ 20 taxis across 54 planning areas

## Stack
Airflow · Kafka · Flink · DuckDB · geopandas · scikit-learn · MLflow · FastAPI · Docker Compose

## Code
This project ran as a multi-container stack (Airflow, Kafka, Flink, MLflow, FastAPI) on Docker Compose in the course lab environment, not in Colab, so there is no one-click demo. The code is not published here.
