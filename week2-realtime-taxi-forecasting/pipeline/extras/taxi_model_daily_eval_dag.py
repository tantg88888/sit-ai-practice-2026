"""Airflow DAG: evaluate the 2-hour taxi model every day at 08:00.

Drop this file into an Airflow dags/ folder, mount pipeline/extras at
/opt/airflow/extras and set TAXI_DATASET_PATH to the mounted dataset CSV.
It can also be triggered manually from the UI.
"""
from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

with DAG(
    dag_id="taxi_model_daily_eval",
    description="Daily 08:00 evaluation of the 2-hour taxi prediction model",
    start_date=datetime(2026, 1, 1),
    schedule="0 8 * * *",   # every day at 08:00
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "evaluation"],
) as dag:
    evaluate = BashOperator(
        task_id="evaluate_2h_model",
        bash_command="cd /opt/airflow/extras && python daily_eval.py",
        retries=1,
    )
