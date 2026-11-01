# Week 2 · Real-time taxi availability forecasting pipeline (Singapore)

Live Singapore taxi GPS and weather data are joined to URA planning areas and used to forecast available taxis per area two hours ahead, with anomaly alerts. The project includes a Colab demo and the Docker Compose pipeline code (Airflow, Kafka/Flink, MLflow, FastAPI).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tantg88888/sit-ai-practice-2026/blob/main/week2-realtime-taxi-forecasting/notebook.ipynb)

**Individual project** · SIT Graduate Certificate in AI Practice, *AI Foundations & Engineering Practice*, Week 2 (2026) · Area: data engineering & MLOps

## Run the demo

1. Click **Open in Colab** above. A **CPU** runtime is enough.
2. Optional: add a Colab secret named `DATA_GOV_API_KEY` (key icon in the left sidebar) holding your own [data.gov.sg](https://data.gov.sg) API key. The endpoints also work without a key, at a lower rate limit.
3. **Runtime → Run all.** The first cell installs `geopandas`, `duckdb` and `gradio` if they are missing and downloads the support files (`lib/`, `data/`, `shared_assets/`) from this repo.
4. After about 2–4 minutes a public **Gradio** link appears at the bottom. Click **Refresh live data** to pull a new snapshot and redraw the map. Pick a planning area to see its taxis now, the 2-hour rain forecast, the predicted taxis in 2 hours and an anomaly check. The **Pipeline** tab maps each stage to the production stack.

If a live API call fails (offline or rate-limited), the notebook switches to the cached samples in `data/` and says so in its output.

## Results

| What | Result | Source |
|---|---|---|
| GeoJSON spatial join | 1,851 / 1,852 live taxi points matched into 51 planning areas (332 subzone polygons) | recorded run of `notebook.ipynb` (live snapshot, 27 Jun 2026 15:56 SGT) |
| RandomForest, 2-hour taxi count, **random 80/20 split** | MAE 13.37 taxis · RMSE 23.53 · R² 0.908 | recorded run of `notebook.ipynb` |
| Anomaly rule (hour-aware z-score, alert if \|z\| ≥ 2.5) | Changi 775 taxis vs typical 634.5 → z +1.12 (normal); injected drop to 193 → z −3.51 (alert) | recorded run of `notebook.ipynb` |
| MLflow: MLPRegressor (best tracked run) | MAE 18.75 taxis · RMSE 33.09 (R² not logged) | project MLflow tracking DB (`mlflow.db`, not published), random 80/20 split, rainfall feature set |
| MLflow: Linear Regression (same data and features) | MAE 19.52 taxis · RMSE 34.31 | same MLflow tracking DB |

The random split flatters the model, because rows a few minutes apart from the same area end up in both train and test. The notebook therefore also scores the model on a **chronological hold-out** against a naive "same as now" baseline. That number is printed when you run it and is not recorded here. The notebook RandomForest and the MLflow models use different feature sets and dataset versions, so their numbers are not directly comparable.

## How it works

- **Ingest:** the data.gov.sg taxi-availability API (v1) returns raw GPS points for every free taxi. The 2-hour weather forecast comes from the v2 real-time API.
- **GeoJSON join:** `geopandas.sjoin` assigns each taxi point to a URA Master Plan 2019 subzone polygon, and the points are counted per planning area (`PLN_AREA_N`).
- **Store:** area counts are merged with the weather forecast and loaded into **DuckDB**.
- **Model:** a RandomForest uses current taxis, hour, day, peak/late-night flags, the rain forecast and the area to predict the taxi count in the same area about 2 hours later. The production stack tracks Linear Regression and MLPRegressor runs in **MLflow** and serves the `@champion` model with **FastAPI**.
- **Alerts:** an area's count is compared with its history at the same hour of day. If |z| ≥ 2.5, an alert fires.

## Architecture

```
data.gov.sg APIs (taxi availability · 2-h weather forecast · rainfall)
      │
      ├── BATCH  Airflow DAG every 2 min: fetch → clean → GeoJSON join → load ──┐
      │          (retries + exponential backoff, single-writer pool)            ├──► DuckDB
      └── LIVE   Kafka producer (15 s) → topic → manual-commit consumer ────────┘
                 Flink SQL 1-min tumbling window → summary topic → consumer
                                                                                │
      MLflow tracking + model registry (alias @champion) ◄── prepared 2-h dataset
                 │
                 └──► FastAPI serving ──► dashboards (map · now vs +2 h · anomaly alerts)
      extras/: 08:00 daily evaluation DAG + anomaly-alert script
```

- **Airflow batch** (`pipeline/01_airflow_batch`): DAG `dashboard_data_pipeline` runs every 2 minutes with two parallel chains, taxi and rainfall: fetch → clean → join to planning area → load. Fetch tasks use `retries=2`, a 30 s delay, exponential backoff and a 10-minute cap. Load tasks share the pool `data_ingestion_pool`, so only one task writes to DuckDB at a time. Loads are delete-then-insert, so reruns do not duplicate rows.
- **Kafka streaming** (`pipeline/02_kafka_streaming`): the producer polls the APIs every 15 s and publishes snapshot events. The consumer sets `enable.auto.commit=False` and commits each offset only after the DuckDB write succeeds. A **Flink SQL** job aggregates 1-minute tumbling windows into a summary topic. A FastAPI endpoint and a live map dashboard read from DuckDB.
- **MLflow + serving** (`pipeline/03_mlflow_serving`): training scripts for Linear Regression and MLPRegressor log params, metrics and models to MLflow and register `taxi_availability_2h_model@champion`. A FastAPI dashboard loads the champion and shows current vs predicted (+2 h) availability with alerts.
- **Airflow → MLflow** (`pipeline/04_airflow_mlflow`): DAG `mlflow_model_training` trains and registers the model against the MLflow server.
- **Extras** (`pipeline/extras`): `taxi_model_daily_eval_dag.py` (schedule `0 8 * * *`) runs `daily_eval.py`, which trains on earlier days, scores the latest day and flags drift if MAE exceeds 25. `anomaly_alert.py` applies the hour-aware z-score rule.

The pipeline stacks are adapted from the course's lab templates (Airflow, Kafka/Flink and MLflow labs) and wired together for this project. The notebook, `lib/` and `pipeline/extras/` were written for this project.

## Run the full stack locally

You need Docker Desktop and internet access. All paths are relative to `pipeline/`. Both Airflow stacks use port 8080, so run only one at a time.

```bash
# optional: a data.gov.sg key for the v2 endpoints (never commit it)
export DATA_GOV_API_KEY=...            # PowerShell: $env:DATA_GOV_API_KEY="..."

# 1) Batch: Airflow → DuckDB   (UI http://localhost:8080, login airflow / airflow)
cd pipeline/01_airflow_batch
cp .env.example .env                    # then put your own FERNET_KEY in .env
docker compose up -d                    # first start takes a few minutes
docker compose exec airflow-apiserver airflow pools set data_ingestion_pool 1 "single DuckDB writer"
# in the UI: un-pause dashboard_data_pipeline (runs every 2 min)
uvicorn serving_api:app --port 8052     # optional batch map dashboard (needs fastapi, duckdb, geopandas)
docker compose down

# 2) Streaming: Kafka (+ Flink)   (dashboard :8502, API :8003, Flink UI :8081)
cd ../02_kafka_streaming
docker compose up -d --build                     # Kafka, producer, consumer, API, dashboard
docker compose --profile flink up -d --build     # add the Flink 1-min window job
docker compose logs -f consumer

# 3) MLflow tracking + serving   (MLflow UI http://localhost:5000)
cd ../03_mlflow_serving
docker compose up -d --build
docker exec -w /workspace day4-mlops python train_mlp_regression_mlflow.py --data-path <your_dataset.csv> --register-model
docker exec -w /workspace day4-mlops python train_linear_regression_mlflow.py --data-path <your_dataset.csv>

# 4) Airflow-triggered training (stop stack 1 first, as it shares :8080)
cd ../04_airflow_mlflow
cp .env.example .env                    # own FERNET_KEY
docker compose up -d                    # then trigger DAG mlflow_model_training

# extras (from the folder root)
python pipeline/extras/daily_eval.py
python pipeline/extras/anomaly_alert.py
```

To generate a Fernet key, run `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` (see the Airflow docs on Fernet keys).

Notes:
- The MLflow training scripts expect the rainfall-enriched datasets (`taxi_weather_2h_prediction_dataset_0606_rainfall.csv` and `_0617_rainfall.csv`, 13 MB and 29 MB) with `planning_area_id` and `current_rainfall_5min_mm` columns. They are **not** in this repo because of size, so pass your own CSV with `--data-path`. The DAG in `04_airflow_mlflow` reads `data/taxi_weather_2h_prediction_dataset_0617_rainfall.csv`.
- `03_mlflow_serving/mart_dashboard_api.py` also needs the course's live feature-mart pipeline (a dbt project) mounted at `/clean`. That pipeline is not included here.
- Airflow writes its own `config/airflow.cfg` on first start. The generated file contains secrets, so it is not committed.

## Data

- [data.gov.sg](https://data.gov.sg) real-time APIs: taxi availability, 2-hour weather forecast and rainfall (Singapore Open Data Licence v1.0).
- URA **Master Plan 2019 Subzone Boundary (No Sea)** GeoJSON, via data.gov.sg (Singapore Open Data Licence v1.0). Stored in `shared_assets/` (3 MB).
- `data/taxi_weather_2h_prediction_dataset.csv`: area-level snapshots collected by the pipeline, each paired with the count about 2 hours later and the 2-hour forecast. `data/sample_*.json` are cached API responses used as the offline fallback.

## Limitations

- The training data covers only a few weekdays (late May 2026), so there is no weekend data, and weekend or holiday predictions are extrapolation.
- The headline notebook metrics use a random split, which is optimistic for time series. Use the chronological hold-out printed in the notebook as the fairer estimate.
- `current_taxi_count` dominates the feature importance. The model mostly learns "about what is there now, adjusted by hour and area", so check it against the naive baseline.
- Weather forecasts cover 47 named areas and are matched to planning areas by name. Unmatched areas fall back to the most common forecast island-wide.
- The anomaly baseline (mean and standard deviation per area and hour) is built from a few days of history and has no seasonality or holiday handling.
- The Colab demo runs in one process. The production behaviour (scheduling, retries, streaming, model registry) needs the Docker stack.
