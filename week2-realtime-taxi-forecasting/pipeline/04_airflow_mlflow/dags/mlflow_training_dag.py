from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


with DAG(
    dag_id="mlflow_model_training",
    description="Train and register the Day 4 taxi model with MLflow",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["mlflow", "model-training"],
) as dag:

    train_and_register_model = BashOperator(
        task_id="train_linear_regression_and_register",
        bash_command=(
            "cd /opt/airflow/jobs && "
            "python train_linear_regression_mlflow.py "
            "--data-version v1 "
            "--data-path /opt/airflow/data/taxi_weather_2h_prediction_dataset_0617_rainfall.csv "
            "--feature-version v1 "
            "--tracking-uri http://host.docker.internal:5000"
        ),
        retries=1,
        retry_delay=timedelta(minutes=1),
    )
